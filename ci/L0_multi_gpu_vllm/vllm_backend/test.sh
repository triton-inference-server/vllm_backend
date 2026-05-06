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

# Helper: start server, run one Python test method, stop server.
function run_test_with_server() {
    local TEST_NAME="${1}"
    local TEST_METHOD="${2}"

    SERVER_LOG="./${TEST_NAME}.server.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        cat "$SERVER_LOG"
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        exit 1
    fi

    set +e
    CLIENT_LOG="./${TEST_NAME}.client.log"
    python3 "$CLIENT_PY" "${TEST_METHOD}" -v > "$CLIENT_LOG" 2>&1

    if [ $? -ne 0 ]; then
        cat "$CLIENT_LOG"
        echo -e "\n***\n*** ${TEST_NAME} FAILED.\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
        if [ $? -ne 0 ]; then
            cat "$CLIENT_LOG"
            echo -e "\n***\n*** Test Result Verification FAILED.\n***"
            RET=1
        fi
    fi
    set -e

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
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

    echo "Running multi-GPU test with kind=${KIND}, tp=${TENSOR_PARALLELISM}, instance_count=${INSTANCE_COUNT}"
    run_test_with_server \
        "vllm_multi_gpu_test--${KIND}_tp${TENSOR_PARALLELISM}_count${INSTANCE_COUNT}" \
        "VLLMMultiGPUTest.test_multi_gpu_model"
}

function create_gpu_device_ids_model() {
    local MODEL_NAME="${1}"
    local GPU_IDS="${2}"
    local TP="${3}"
    local DISTRIBUTED_EXECUTOR_BACKEND="${4}"
    local MODEL_DIR="models/${MODEL_NAME}"
    local TRITON_CONFIG="${MODEL_DIR}/config.pbtxt"
    local VLLM_CONFIG="${MODEL_DIR}/1/model.json"

    cp -r "${SAMPLE_MODELS_REPO}/vllm_model" "${MODEL_DIR}"
    validate_file_contains "KIND_MODEL" "${TRITON_CONFIG}"
    sed -i "3s/^/    \"tensor_parallel_size\": ${TP},\n/" "${VLLM_CONFIG}"
    if [ $TP -ne "1" ]; then
        jq --arg backend $DISTRIBUTED_EXECUTOR_BACKEND \
            '. += {"distributed_executor_backend":$backend}' \
            "${VLLM_CONFIG}" > "temp.json"
        mv temp.json "${VLLM_CONFIG}"
    fi
    cat >> "${TRITON_CONFIG}" << EOF
parameters {
  key: "GPU_DEVICE_IDS"
  value: { string_value: "${GPU_IDS}" }
}
EOF
}

function run_gpu_device_ids_test() {
    local GPU_IDS="${1}"
    local TP="${2}"
    local DISTRIBUTED_EXECUTOR_BACKEND="${3}"

    # Clear env vars from other tests
    unset KIND TENSOR_PARALLELISM INSTANCE_COUNT INVALID_GPU_DEVICE_IDS_MODELS

    export TEST_MODEL="vllm_gpu_device_ids_tp${TP}"
    export GPU_DEVICE_IDS="${GPU_IDS}"

    rm -rf models && mkdir -p models
    create_gpu_device_ids_model "${TEST_MODEL}" "${GPU_IDS}" "${TP}" "${DISTRIBUTED_EXECUTOR_BACKEND}"

    echo "Running valid GPU_DEVICE_IDS test with gpu_ids=${GPU_IDS}, tp=${TP}"
    run_test_with_server \
        "vllm_valid_gpu_device_ids_test--${GPU_IDS}_tp${TP}" \
        "VLLMMultiGPUTest.test_gpu_device_ids"

    # Verify that _validate_device_config logged the GPU_DEVICE_IDS specified.
    local EXPECTED_LOG_MSG="Detected KIND_MODEL instance with GPU_DEVICE_IDS specified"
    if ! grep -q "${EXPECTED_LOG_MSG}" "${SERVER_LOG}"; then
        echo -e "\n***\n*** ERROR: Expected log message not found in ${SERVER_LOG}:" \
                "\n***   '${EXPECTED_LOG_MSG}'\n***"
        RET=1
    fi

    unset GPU_DEVICE_IDS
}

function run_gpu_device_ids_validation_tests() {
    local DISTRIBUTED_EXECUTOR_BACKEND="${1}"
    local TP="2"

    # Clear env vars from other tests
    unset TEST_MODEL GPU_DEVICE_IDS KIND TENSOR_PARALLELISM INSTANCE_COUNT

    local INVALID_FORMAT="vllm_invalid_format_gpu_device_ids"
    local INVALID_WHITESPACE="vllm_invalid_whitespace_gpu_device_ids"
    local INVALID_NEGATIVE="vllm_invalid_negative_gpu_device_ids"
    local INVALID_DUPLICATE="vllm_invalid_duplicate_gpu_device_ids"
    local INVALID_COUNT="vllm_invalid_count_gpu_device_ids"

    rm -rf models && mkdir -p models
    # Non-integer string: triggers parse ValueError
    create_gpu_device_ids_model "${INVALID_FORMAT}" "abc" "${TP}" "${DISTRIBUTED_EXECUTOR_BACKEND}"
    # Whitespace-only: same parse ValueError path as invalid_format
    create_gpu_device_ids_model "${INVALID_WHITESPACE}" " " "${TP}" "${DISTRIBUTED_EXECUTOR_BACKEND}"
    # Negative GPU ID: triggers negative-ID check
    create_gpu_device_ids_model "${INVALID_NEGATIVE}" "-1" "${TP}" "${DISTRIBUTED_EXECUTOR_BACKEND}"
    # Duplicate GPU IDs: triggers duplicate check
    create_gpu_device_ids_model "${INVALID_DUPLICATE}" "0,0" "${TP}" "${DISTRIBUTED_EXECUTOR_BACKEND}"
    # Fewer IDs than world_size: triggers count-mismatch check
    create_gpu_device_ids_model "${INVALID_COUNT}" "0" "${TP}" "${DISTRIBUTED_EXECUTOR_BACKEND}"

    export INVALID_GPU_DEVICE_IDS_MODELS="${INVALID_FORMAT},${INVALID_WHITESPACE},${INVALID_NEGATIVE},${INVALID_DUPLICATE},${INVALID_COUNT}"

    echo "Running GPU_DEVICE_IDS validation tests with invalid configs"
    run_test_with_server \
        "vllm_gpu_device_ids_test--validation" \
        "VLLMMultiGPUTest.test_invalid_gpu_device_ids"
    unset INVALID_GPU_DEVICE_IDS_MODELS
}

### Test
rm -f *.log
RET=0
DISTRIBUTED_EXECUTOR_BACKEND="ray"

# Test the various cases of kind, tensor parallelism, and instance count
# for different ways to run multi-GPU models with vLLM on Triton
KINDS="KIND_MODEL KIND_GPU"
TPS="1 2"
INSTANCE_COUNTS="1 2"
for kind in ${KINDS}; do
  for tp in ${TPS}; do
    for count in ${INSTANCE_COUNTS}; do
        run_multi_gpu_test "${kind}" "${tp}" "${count}" "${DISTRIBUTED_EXECUTOR_BACKEND}"
    done
  done
done

# Test GPU_DEVICE_IDS parameter for per-model GPU pinning with KIND_MODEL
run_gpu_device_ids_test "0,1" "2" "${DISTRIBUTED_EXECUTOR_BACKEND}"
run_gpu_device_ids_test "1" "1" "${DISTRIBUTED_EXECUTOR_BACKEND}"
run_gpu_device_ids_validation_tests "${DISTRIBUTED_EXECUTOR_BACKEND}"

### Results
if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** Multi GPU Utilization test FAILED. \n***"
else
    echo -e "\n***\n*** Multi GPU Utilization test PASSED. \n***"
fi

exit $RET
