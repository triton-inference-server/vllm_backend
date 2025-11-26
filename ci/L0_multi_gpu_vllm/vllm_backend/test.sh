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
SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --log-verbose=1"
TEST_RESULT_FILE='test_results.txt'
CLIENT_PY="./vllm_multi_gpu_test.py"
SAMPLE_MODELS_REPO="../../../samples/model_repository"
EXPECTED_NUM_TESTS=1

### Helpers
function validate_file_contains() {
    local KEY="${1}"
    local FILE="${2}"

    if [ -z "${KEY}" ] || [ -z "${FILE}" ]; then
        echo "Error: KEY and FILE must be provided."
        return 1
    fi

    if [ ! -f "${FILE}" ]; then
        echo "Error: File '${FILE}' does not exist."
        return 1
    fi

    count=$(grep -o -w "${KEY}" "${FILE}" | wc -l)

    if [ "${count}" -ne 1 ]; then
        echo "Error: KEY '${KEY}' found ${count} times in '${FILE}'. Expected exactly once."
        return 1
    fi
}

function run_multi_gpu_test() {
    export KIND="${1}"
    export TENSOR_PARALLELISM="${2}"
    export INSTANCE_COUNT="${3}"
    export DISTRIBUTED_EXECUTOR_BACKEND="${4}"

    # Setup a clean model repository
    export TEST_MODEL="vllm_opt_${KIND}_tp${TENSOR_PARALLELISM}_count${INSTANCE_COUNT}"
    local TEST_MODEL_TRITON_CONFIG="models/${TEST_MODEL}/config.pbtxt"
    local TEST_MODEL_VLLM_CONFIG="models/${TEST_MODEL}/1/model.json"

    rm -rf models && mkdir -p models
    cp -r "${SAMPLE_MODELS_REPO}/vllm_model" "models/${TEST_MODEL}"
    sed -i "s/KIND_MODEL/${KIND}/" "${TEST_MODEL_TRITON_CONFIG}"
    sed -i "3s/^/    \"tensor_parallel_size\": ${TENSOR_PARALLELISM},\n/" "${TEST_MODEL_VLLM_CONFIG}"
    if [ $TENSOR_PARALLELISM -ne "1" ]; then
        jq --arg backend $DISTRIBUTED_EXECUTOR_BACKEND '. += {"distributed_executor_backend":$backend}' "${TEST_MODEL_VLLM_CONFIG}" > "temp.json"
        mv temp.json "${TEST_MODEL_VLLM_CONFIG}"
    fi
    # Assert the correct kind is set in case the template config changes in the future
    validate_file_contains "${KIND}" "${TEST_MODEL_TRITON_CONFIG}"

    # Start server
    echo "Running multi-GPU test with kind=${KIND}, tp=${TENSOR_PARALLELISM}, instance_count=${INSTANCE_COUNT}"
    SERVER_LOG="./vllm_multi_gpu_test--${KIND}_tp${TENSOR_PARALLELISM}_count${INSTANCE_COUNT}--server.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        exit 1
    fi

    # Run unit tests
    set +e
    CLIENT_LOG="./vllm_multi_gpu_test--${KIND}_tp${TENSOR_PARALLELISM}_count${INSTANCE_COUNT}--client.log"
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

    # Cleanup
    kill $SERVER_PID
    wait $SERVER_PID
}

### Test
rm -f *.log
RET=0

# Test the various cases of kind, tensor parallelism, and instance count
# for different ways to run multi-GPU models with vLLM on Triton
KINDS="KIND_MODEL KIND_GPU"
TPS="1 2"
INSTANCE_COUNTS="1 2"
DISTRIBUTED_EXECUTOR_BACKEND="ray"
for kind in ${KINDS}; do
  for tp in ${TPS}; do
    for count in ${INSTANCE_COUNTS}; do
        run_multi_gpu_test "${kind}" "${tp}" "${count}" "${DISTRIBUTED_EXECUTOR_BACKEND}"
    done
  done
done

### Results
if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** Multi GPU Utilization test FAILED. \n***"
else
    echo -e "\n***\n*** Multi GPU Utilization test PASSED. \n***"
fi

exit $RET
