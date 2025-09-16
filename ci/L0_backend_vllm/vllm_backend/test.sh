#!/bin/bash
# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SERVER_ARGS="--model-repository=$(pwd)/models --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --load-model=vllm_opt --log-verbose=1"
SERVER_LOG="./vllm_backend_server.log"
CLIENT_LOG="./vllm_backend_client.log"
TEST_RESULT_FILE='test_results.txt'
CLIENT_PY="./vllm_backend_test.py"
SAMPLE_MODELS_REPO="../../../samples/model_repository"
EXPECTED_NUM_TESTS=6

# Helpers =======================================
function assert_curl_success {
  message="${1}"
  if [ "$code" != "200" ]; then
    cat ./curl.out
    echo -e "\n***\n*** ${message} : line ${BASH_LINENO}\n***"
    RET=1
  fi
}

rm -rf models && mkdir -p models
cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/vllm_opt
# `vllm_opt` model will be loaded on server start and stay loaded throughout
# unittesting. To test vllm model load/unload we use a dedicated
# `vllm_load_test`. To ensure that vllm's memory profiler will not error out
# on `vllm_load_test` load, we reduce "gpu_memory_utilization" for `vllm_opt`,
# so that at least 60% of GPU memory was available for other models.
sed -i 's/"gpu_memory_utilization": 0.5/"gpu_memory_utilization": 0.4/' models/vllm_opt/1/model.json
cp -r models/vllm_opt models/vllm_load_test

mkdir -p models/add_sub/1/
wget -P models/add_sub/1/ https://raw.githubusercontent.com/triton-inference-server/python_backend/main/examples/add_sub/model.py
wget -P models/add_sub https://raw.githubusercontent.com/triton-inference-server/python_backend/main/examples/add_sub/config.pbtxt

# Invalid model attribute
cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/vllm_invalid_1/
sed -i 's/"enforce_eager"/"invalid_attribute"/' models/vllm_invalid_1/1/model.json

# Invalid model name
cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/vllm_invalid_2/
sed -i 's/"facebook\/opt-125m"/"invalid_model"/' models/vllm_invalid_2/1/model.json


# Sanity check ensembles are enabled and can successfully be loaded
mkdir -p models/ensemble_model/1
cp -r ensemble_config.pbtxt models/ensemble_model/config.pbtxt

RET=0

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

# Test Python backend cmdline parameters are propagated to vllm backend
SERVER_ARGS="--model-repository=$(pwd)/models --backend-directory=${BACKEND_DIR} --backend-config=python,default-max-batch-size=8"
SERVER_LOG="./vllm_test_cmdline_server.log"

rm -rf ./models/vllm_invalid_1 ./models/vllm_invalid_2 ./models/vllm_load_test

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

# Test loading multiple vllm models
SERVER_ARGS="--model-repository=$(pwd)/models --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --load-model=vllm_one"
SERVER_LOG="./vllm_test_multi_model.log"

# Create two models, one is just a copy of the other, and make sure gpu
# utilization is low enough for multiple models to avoid OOM.
# vLLM changed behavior of their GPU profiler from total to free memory,
# so to load two small models, we need to start
# triton server in explicit mode.
MODEL1="vllm_one"
MODEL2="vllm_two"
mkdir -p models
cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/${MODEL1}/
cp -r models/${MODEL1} models/${MODEL2}
sed -i 's/"gpu_memory_utilization": 0.5/"gpu_memory_utilization": 0.4/' models/${MODEL1}/1/model.json
sed -i 's/"gpu_memory_utilization": 0.5/"gpu_memory_utilization": 0.4/' models/${MODEL2}/1/model.json

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

# Explicitly load model
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out -X POST localhost:8000/v2/repository/models/vllm_two/load`
set -e
assert_curl_success "Failed to load 'vllm_two' model"

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

collect_artifacts_from_subdir

exit $RET