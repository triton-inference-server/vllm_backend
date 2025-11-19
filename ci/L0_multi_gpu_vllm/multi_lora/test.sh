#!/bin/bash
# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

source ../../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./multi_lora_server.log"
CLIENT_LOG="./multi_lora_client.log"
TEST_RESULT_FILE='test_results.txt'
CLIENT_PY="./multi_lora_test.py"
DOWNLOAD_PY="./download.py"
SAMPLE_MODELS_REPO="../../../samples/model_repository"
EXPECTED_NUM_TESTS=2
GENERATE_ENDPOINT="localhost:8000/v2/models/vllm_llama_multi_lora/generate"
CHECK_FOR_ERROR=true

make_api_call() {
    local endpoint="$1"
    local data="$2"
    curl -X POST "$endpoint" --data-binary @- <<< "$data"
}

check_response() {
    local response="$1"
    local expected_response="$2"
    local error_message="$3"
    local check_error="${4:-false}"

    if [ -z "$response" ]; then
        echo -e "Expected a non-empty response from server"
        echo -e "\n***\n*** $error_message \n***"
        return 1
    fi

    local response_text=$(echo "$response" | jq '.text_output // empty')
    local response_error=$(echo "$response" | jq '.error // empty')

    if [ "$check_error" = true ]; then
        if [[ -n "$response_text" ]]; then
            echo -e "Server didn't return an error."
            echo "$response"
            echo -e "\n***\n*** $error_message \n***"
            return 1
        elif [[ "$expected_response" != "$response_error" ]]; then
            echo -e "Expected error message doesn't match actual response."
            echo "Expected: $expected_response."
            echo "Received: $response_error"
            echo -e "\n***\n*** $error_message\n***"
            return 1
        fi
    else
        if [[ ! -z "$response_error" ]]; then
            echo -e "Received an error from server."
            echo "$response"
            echo -e "\n***\n*** $error_message \n***"
            return 1
        elif [[ "$expected_response" != "$response_text" ]]; then
            echo "Expected response doesn't match actual"
            echo "Expected: $expected_response."
            echo "Received: $response_text"
            echo -e "\n***\n*** $error_message \n***"
            return 1
        fi
    fi

    return 0
}

# first we download weights
rm -rf weights && mkdir -p weights/loras/GemmaDoll && mkdir -p weights/loras/GemmaSheep
mkdir -p weights/backbone/gemma-2b

python3 $DOWNLOAD_PY -v > $CLIENT_LOG 2>&1

rm -rf models && mkdir -p models
cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/vllm_llama_multi_lora

export SERVER_ENABLE_LORA=true

# Check boolean flag value for `enable_lora`
model_json=$(cat <<EOF
{
    "model":"./weights/backbone/gemma-2b",
    "gpu_memory_utilization": 0.7,
    "tensor_parallel_size": 2,
    "block_size": 16,
    "enforce_eager": true,
    "enable_lora": true,
    "max_lora_rank": 32,
    "distributed_executor_backend":"ray"
}
EOF
)
echo "$model_json" > models/vllm_llama_multi_lora/1/model.json

multi_lora_json=$(cat <<EOF
{
    "doll": "./weights/loras/GemmaDoll",
    "sheep": "./weights/loras/GemmaSheep"
}
EOF
)
echo "$multi_lora_json" > models/vllm_llama_multi_lora/1/multi_lora.json

RET=0
# If it is the first time launching triton server with gemma-2b and multi-lora feature,
# it may take more than 1 minutes. Please wait.
SERVER_TIMEOUT=60000

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 $CLIENT_PY -v > $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Running $CLIENT_PY FAILED. \n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification FAILED.\n***"
        RET=1
    fi
fi

# Test generate endpoint + LoRA enabled (boolean flag)
EXPECTED_RESPONSE='" I love soccer. I play soccer every day.\nInstruct: Tell me"'
DATA='{
    "text_input": "Instruct: Tell me more about soccer\nOutput:",
    "parameters": {
        "stream": false,
        "temperature": 0,
        "top_p":1,
        "lora_name": "sheep",
        "exclude_input_in_output": true
    }
}'
RESPONSE=$(make_api_call "$GENERATE_ENDPOINT" "$DATA")
check_response "$RESPONSE" "$EXPECTED_RESPONSE" "Valid LoRA + Generate Endpoint Test FAILED." || RET=1

EXPECTED_RESPONSE="\"LoRA unavailable is not supported, we currently support ['doll', 'sheep']\""
DATA='{
    "text_input": "Instruct: Tell me more about soccer\nOutput:",
    "parameters": {
        "stream": false,
        "temperature": 0,
        "top_p":1,
        "lora_name": "unavailable",
        "exclude_input_in_output": true
    }
}'
RESPONSE=$(make_api_call "$GENERATE_ENDPOINT" "$DATA")
check_response "$RESPONSE" "$EXPECTED_RESPONSE" "Invalid LoRA + Generate Endpoint Test FAILED." $CHECK_FOR_ERROR || RET=1

unset EXPECTED_RESPONSE
unset RESPONSE
unset DATA
set -e

kill $SERVER_PID
wait $SERVER_PID

# Check string flag value for `enable_lora`
model_json=$(cat <<EOF
{
    "model":"./weights/backbone/gemma-2b",
    "gpu_memory_utilization": 0.7,
    "tensor_parallel_size": 2,
    "block_size": 16,
    "enforce_eager": true,
    "enable_lora": "true",
    "max_lora_rank": 32,
    "distributed_executor_backend":"ray"
}
EOF
)
echo "$model_json" > models/vllm_llama_multi_lora/1/model.json

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 $CLIENT_PY -v > $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Running $CLIENT_PY FAILED. \n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification FAILED.\n***"
        RET=1
    fi
fi

# Test generate endpoint + LoRA enabled (str flag)
EXPECTED_RESPONSE='" I think it is a very interesting subject.\n\nInstruct: What do you"'
DATA='{
    "text_input": "Instruct: What do you think of Computer Science?\nOutput:",
    "parameters": {
        "stream": false,
        "temperature": 0,
        "top_p":1,
        "lora_name": "doll",
        "exclude_input_in_output": true
    }
}'
RESPONSE=$(make_api_call "$GENERATE_ENDPOINT" "$DATA")
check_response "$RESPONSE" "$EXPECTED_RESPONSE" "Valid LoRA + Generate Endpoint Test FAILED." || RET=1

EXPECTED_RESPONSE="\"LoRA unavailable is not supported, we currently support ['doll', 'sheep']\""
DATA='{
    "text_input": "Instruct: What do you think of Computer Science?\nOutput:",
    "parameters": {
        "stream": false,
        "temperature": 0,
        "top_p":1,
        "lora_name": "unavailable",
        "exclude_input_in_output": true
    }
}'
RESPONSE=$(make_api_call "$GENERATE_ENDPOINT" "$DATA")
check_response "$RESPONSE" "$EXPECTED_RESPONSE" "Invalid LoRA + Generate Endpoint Test FAILED." $CHECK_FOR_ERROR || RET=1

unset EXPECTED_RESPONSE
unset RESPONSE
unset DATA
set -e

kill $SERVER_PID
wait $SERVER_PID

# disable lora
export SERVER_ENABLE_LORA=false
# check bool flag value for `enable_lora`
model_json=$(cat <<EOF
{
    "model":"./weights/backbone/gemma-2b",
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 2,
    "block_size": 16,
    "enforce_eager": true,
    "enable_lora": false,
    "distributed_executor_backend":"ray"
}
EOF
)
echo "$model_json" > models/vllm_llama_multi_lora/1/model.json

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 $CLIENT_PY -v >> $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Running $CLIENT_PY FAILED. \n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification FAILED.\n***"
        RET=1
    fi
fi

# Test generate endpoint + LoRA disabled (boolean flag)
EXPECTED_RESPONSE='"LoRA feature is not enabled."'
DATA='{
    "text_input": "Instruct: What do you think of Computer Science?\nOutput:",
    "parameters": {
        "stream": false,
        "temperature": 0,
        "top_p":1,
        "lora_name": "doll",
        "exclude_input_in_output": true
    }
}'
RESPONSE=$(make_api_call "$GENERATE_ENDPOINT" "$DATA")
check_response "$RESPONSE" "$EXPECTED_RESPONSE" "Disabled LoRA + Generate Endpoint Test FAILED." $CHECK_FOR_ERROR || RET=1

set -e

kill $SERVER_PID
wait $SERVER_PID

# disable lora
export SERVER_ENABLE_LORA=false
# check string flag value for `enable_lora`
model_json=$(cat <<EOF
{
    "model":"./weights/backbone/gemma-2b",
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 2,
    "block_size": 16,
    "enforce_eager": true,
    "enable_lora": "false",
    "distributed_executor_backend":"ray"
}
EOF
)
echo "$model_json" > models/vllm_llama_multi_lora/1/model.json

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 $CLIENT_PY -v >> $CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Running $CLIENT_PY FAILED. \n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification FAILED.\n***"
        RET=1
    fi
fi

# Test generate endpoint + LoRA disabled (str flag)
EXPECTED_RESPONSE='"LoRA feature is not enabled."'
DATA='{
    "text_input": "Instruct: What do you think of Computer Science?\nOutput:",
    "parameters": {
        "stream": false,
        "temperature": 0,
        "top_p":1,
        "lora_name": "doll",
        "exclude_input_in_output": true
    }
}'
RESPONSE=$(make_api_call "$GENERATE_ENDPOINT" "$DATA")
check_response "$RESPONSE" "$EXPECTED_RESPONSE" "Disabled LoRA + Generate Endpoint Test FAILED." $CHECK_FOR_ERROR > $CLIENT_LOG 2>&1 || RET=1

set -e

kill $SERVER_PID
wait $SERVER_PID

rm -rf models/
rm -rf weights/

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** Multi LoRA test FAILED. \n***"
else
    echo -e "\n***\n*** Multi LoRA test PASSED. \n***"
fi

collect_artifacts_from_subdir

exit $RET