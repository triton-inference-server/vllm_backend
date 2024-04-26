#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --log-verbose=1"
SERVER_LOG="./multi_lora_server.log"
CLIENT_LOG="./multi_lora_client.log"
TEST_RESULT_FILE='test_results.txt'
CLIENT_PY="./multi_lora_test.py"
DOWNLOAD_PY="./download.py"
SAMPLE_MODELS_REPO="../../../samples/model_repository"
EXPECTED_NUM_TESTS=2

# first we download weights
pip install -U huggingface_hub

rm -rf weights && mkdir -p weights/loras/GemmaDoll && mkdir -p weights/loras/GemmaSheep
mkdir -p weights/backbone/gemma-2b

python3 $DOWNLOAD_PY -v > $CLIENT_LOG 2>&1

rm -rf models && mkdir -p models
cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/vllm_llama_multi_lora

export SERVER_ENABLE_LORA=true

model_json=$(cat <<EOF
{
    "model":"./weights/backbone/gemma-2b",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.7,
    "tensor_parallel_size": 2,
    "block_size": 16,
    "enforce_eager": "true",
    "enable_lora": "true",
    "max_lora_rank": 32,
    "lora_extra_vocab_size": 256
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
set -e

kill $SERVER_PID
wait $SERVER_PID

# disable lora
export SERVER_ENABLE_LORA=false
model_json=$(cat <<EOF
{
    "model":"./weights/backbone/gemma-2b",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 2,
    "block_size": 16,
    "enforce_eager": "true",
    "enable_lora": "false",
    "lora_extra_vocab_size": 256
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