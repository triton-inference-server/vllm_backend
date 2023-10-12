#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --log-verbose=1"
SERVER_LOG="./vllm_backend_server.log"
CLIENT_LOG="./vllm_backend_client.log"
TEST_RESULT_FILE='test_results.txt'
CLIENT_PY="./vllm_backend_test.py"
EXPECTED_NUM_TESTS=1

mkdir -p models/vllm_opt/1/
cp ../qa_models/vllm_opt/model.json models/vllm_opt/1/
cp ../qa_models/vllm_opt/config.pbtxt models/vllm_opt

mkdir -p models/add_sub/1/
cp ../qa_models/add_sub/model.py models/add_sub/1/
cp ../qa_models/add_sub/config.pbtxt models/add_sub

# Invalid model attribute
mkdir -p models/vllm_invalid_1/1/
cp ../qa_models/vllm_opt/config.pbtxt models/vllm_invalid_1/1/
echo '{"model":"facebook/opt-125m", "invlaid_attribute": "test", "gpu_memory_utilization":0.3}' > models/vllm_invalid_1/1/model.json

# Invalid model name
mkdir -p models/vllm_invalid_2/1/
cp ../qa_models/vllm_opt/config.pbtxt models/vllm_invalid_2/1/
echo '{"model":"invalid_model/opt-125m", "disable_log_requests": "true", "gpu_memory_utilization":0.3}' > models/vllm_invalid_2/1/model.json

pip3 install tritonclient
pip3 install grpcio

RET=0

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 -m unittest -v $CLIENT_PY > $CLIENT_LOG 2>&1

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

# Test Python backend cmdline parameters are propagated to vllm backend
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --backend-config=python,default-max-batch-size=8"
SERVER_LOG="./vllm_test_cmdline_server.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID
rm -rf "./models"

COUNT=$(grep -c "default-max-batch-size\":\"8" "$SERVER_LOG")
if [[ "$COUNT" -ne 2 ]]; then
  echo "Cmdline parameters verification Failed"
fi

# Test loading multiple vllm models at the same time
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR}"
SERVER_LOG="./vllm_test_multi_model.log"

# Create two models, one is just a copy of the other, and make sure gpu
# utilization is low enough for multiple models to avoid OOM
MODEL1="vllm_one"
MODEL2="vllm_two"
mkdir -p models/${MODEL1}/1/
cp ../qa_models/vllm_opt/config.pbtxt models/${MODEL1}/
echo '{"model":"facebook/opt-125m", "disable_log_requests": "true", "gpu_memory_utilization":0.3}' > models/${MODEL1}/1/model.json
cp -r models/${MODEL1} models/${MODEL2}

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID
rm -rf "./models"

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** vLLM test FAILED. \n***"
else
    echo -e "\n***\n*** vLLM test PASSED. \n***"
fi

exit $RET
