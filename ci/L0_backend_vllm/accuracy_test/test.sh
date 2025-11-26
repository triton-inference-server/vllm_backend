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
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./accuracy_test_server.log"
CLIENT_LOG="./accuracy_test_client.log"
TEST_RESULT_FILE='test_results.txt'
CLIENT_PY="./accuracy_test.py"
SAMPLE_MODELS_REPO="../../../samples/model_repository"
VLLM_ENGINE_LOG="vllm_engine.log"
EXPECTED_NUM_TESTS=2

rm -rf models && mkdir -p models
cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/vllm_opt
sed -i 's/"gpu_memory_utilization": 0.5/"gpu_memory_utilization": 0.3/' models/vllm_opt/1/model.json
[ -f vllm_baseline_output.pkl ] && rm vllm_baseline_output.pkl
RET=0

set +e
# Need to generate baseline first, since running 2 vLLM engines causes
# memory issues: https://github.com/vllm-project/vllm/issues/2248
python3 $CLIENT_PY --generate-baseline >> $VLLM_ENGINE_LOG 2>&1 & BASELINE_PID=$!
wait $BASELINE_PID

python3 $CLIENT_PY --generate-structured-baseline > $VLLM_ENGINE_LOG 2>&1 & BASELINE_PID=$!
wait $BASELINE_PID
set -e

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 $CLIENT_PY > $CLIENT_LOG 2>&1

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

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** Accuracy test FAILED. \n***"
else
    echo -e "\n***\n*** Accuracy test PASSED. \n***"
fi

collect_artifacts_from_subdir

exit $RET
