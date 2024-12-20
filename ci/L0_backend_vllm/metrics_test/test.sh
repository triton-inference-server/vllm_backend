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
SERVER_ARGS="--model-repository=$(pwd)/models --backend-directory=${BACKEND_DIR} --model-control-mode=explicit --load-model=vllm_opt --log-verbose=1"
SERVER_LOG="./vllm_metrics_server.log"
CLIENT_LOG="./vllm_metrics_client.log"
TEST_RESULT_FILE='test_results.txt'
CLIENT_PY="./vllm_metrics_test.py"
SAMPLE_MODELS_REPO="../../../samples/model_repository"
EXPECTED_NUM_TESTS=1

# Helpers =======================================
function copy_model_repository {
    rm -rf models && mkdir -p models
    cp -r ${SAMPLE_MODELS_REPO}/vllm_model models/vllm_opt
    # `vllm_opt` model will be loaded on server start and stay loaded throughout
    # unittesting. To ensure that vllm's memory profiler will not error out
    # on `vllm_load_test` load, we reduce "gpu_memory_utilization" for `vllm_opt`,
    # so that at least 60% of GPU memory was available for other models.
    sed -i 's/"gpu_memory_utilization": 0.5/"gpu_memory_utilization": 0.4/' models/vllm_opt/1/model.json
}

run_test() {
    local TEST_CASE=$1

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        exit 1
    fi

    set +e
    python3 $CLIENT_PY $TEST_CASE -v > $CLIENT_LOG 2>&1

    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Running $CLIENT_PY $TEST_CASE FAILED. \n***"
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

    # TODO: Non-graceful shutdown when metrics are enabled.
    kill $SERVER_PID
    wait $SERVER_PID
}

RET=0

# Test disabling vLLM metrics reporting without parameter "REPORT_CUSTOM_METRICS" in config.pbtxt
copy_model_repository
run_test VLLMTritonMetricsTest.test_vllm_metrics_disabled

# Test disabling vLLM metrics reporting with parameter "REPORT_CUSTOM_METRICS" set to "false" in config.pbtxt
copy_model_repository
echo -e "
parameters: {
  key: \"REPORT_CUSTOM_METRICS\"
  value: {
    string_value: \"false\"
  }
}
" >> models/vllm_opt/config.pbtxt
run_test VLLMTritonMetricsTest.test_vllm_metrics_disabled

# Test vLLM metrics reporting with parameter "REPORT_CUSTOM_METRICS" set to "true" in config.pbtxt
copy_model_repository
cp ${SAMPLE_MODELS_REPO}/vllm_model/config.pbtxt models/vllm_opt
echo -e "
parameters: {
  key: \"REPORT_CUSTOM_METRICS\"
  value: {
    string_value: \"true\"
  }
}
" >> models/vllm_opt/config.pbtxt
run_test VLLMTritonMetricsTest.test_vllm_metrics

# Test vLLM metrics custom sampling parameters
# Custom sampling parameters may result in different vLLM output depending
# on the platform. Therefore, these metrics are tests separately.
copy_model_repository
cp ${SAMPLE_MODELS_REPO}/vllm_model/config.pbtxt models/vllm_opt
echo -e "
parameters: {
  key: \"REPORT_CUSTOM_METRICS\"
  value: {
    string_value: \"true\"
  }
}
" >> models/vllm_opt/config.pbtxt
run_test VLLMTritonMetricsTest.test_custom_sampling_params

# Test enabling vLLM metrics reporting in config.pbtxt but disabling in model.json
copy_model_repository
jq '. += {"disable_log_stats" : true}' models/vllm_opt/1/model.json > "temp.json"
mv temp.json models/vllm_opt/1/model.json
echo -e "
parameters: {
  key: \"REPORT_CUSTOM_METRICS\"
  value: {
    string_value: \"true\"
  }
}
" >> models/vllm_opt/config.pbtxt
run_test VLLMTritonMetricsTest.test_vllm_metrics_disabled

# Test enabling vLLM metrics reporting in config.pbtxt while disabling in server option
copy_model_repository
echo -e "
parameters: {
  key: \"REPORT_CUSTOM_METRICS\"
  value: {
    string_value: \"true\"
  }
}
" >> models/vllm_opt/config.pbtxt
SERVER_ARGS="${SERVER_ARGS} --allow-metrics=false"
run_test VLLMTritonMetricsTest.test_vllm_metrics_refused

rm -rf "./models" "temp.json"

if [ $RET -eq 1 ]; then
    cat $CLIENT_LOG
    cat $SERVER_LOG
    echo -e "\n***\n*** vLLM test FAILED. \n***"
else
    echo -e "\n***\n*** vLLM test PASSED. \n***"
fi

collect_artifacts_from_subdir
exit $RET
