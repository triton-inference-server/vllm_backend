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

export CUDA_VISIBLE_DEVICES=0
source ../common/util.sh

pip3 install pytest==8.1.1
pip3 install tritonclient[grpc]

rm -f *.log *.report.xml
RET=0

function setup_model_repository {
    local sample_model_repo_path="../../samples/model_repository"
    rm -rf models && mkdir -p models
    cp -r $sample_model_repo_path/vllm_model models/vllm_opt
}

function enable_health_check {
    local enable_vllm_health_check="$1"
    echo -e "parameters: {" >> models/vllm_opt/config.pbtxt
    echo -e "  key: \"ENABLE_VLLM_HEALTH_CHECK\"" >> models/vllm_opt/config.pbtxt
    echo -e "  value: { string_value: \"$enable_vllm_health_check\" }" >> models/vllm_opt/config.pbtxt
    echo -e "}" >> models/vllm_opt/config.pbtxt
}

VLLM_INSTALL_PATH="/usr/local/lib/python3.12/dist-packages/vllm"
VLLM_V1_ENGINE_PATH="$VLLM_INSTALL_PATH/v1/engine"

function mock_vllm_async_llm_engine {
    # backup original file
    mv $VLLM_V1_ENGINE_PATH/async_llm.py $VLLM_V1_ENGINE_PATH/async_llm.py.backup
    cp $VLLM_V1_ENGINE_PATH/async_llm.py.backup $VLLM_V1_ENGINE_PATH/async_llm.py
    # overwrite the original check_health method
    echo -e "" >> $VLLM_V1_ENGINE_PATH/async_llm.py
    echo -e "    async def check_health(self, check_count=[0]):" >> $VLLM_V1_ENGINE_PATH/async_llm.py
    echo -e "        check_count[0] += 1" >> $VLLM_V1_ENGINE_PATH/async_llm.py
    echo -e "        if check_count[0] > 1:" >> $VLLM_V1_ENGINE_PATH/async_llm.py
    echo -e "            raise RuntimeError(\"Simulated vLLM check_health() failure\")" >> $VLLM_V1_ENGINE_PATH/async_llm.py
}

function unmock_vllm_async_llm_engine {
    # restore from backup
    rm -f $VLLM_V1_ENGINE_PATH/async_llm.py
    mv $VLLM_V1_ENGINE_PATH/async_llm.py.backup $VLLM_V1_ENGINE_PATH/async_llm.py
}

function test_check_health {
    local test_name="$1"
    local unit_test_name="$2"

    SERVER_LOG="$test_name.server.log"
    SERVER_ARGS="--model-repository=models --model-control-mode=explicit --load-model=*"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e
    python3 -m pytest --junitxml=$test_name.report.xml -s -v check_health_test.py::TestCheckHealth::$unit_test_name > $test_name.log
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** $test_name FAILED. \n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
}

# Test health check unspecified
# Cold start on SBSA device can take longer than default 120 seconds
PREV_SERVER_TIMEOUT=$SERVER_TIMEOUT
SERVER_TIMEOUT=240
setup_model_repository
test_check_health "health_check_unspecified" "test_vllm_is_healthy"
SERVER_TIMEOUT=$PREV_SERVER_TIMEOUT

# Test health check disabled
setup_model_repository
enable_health_check "false"
test_check_health "health_check_disabled" "test_vllm_is_healthy"

# Test health check enabled
setup_model_repository
enable_health_check "true"
test_check_health "health_check_enabled" "test_vllm_is_healthy"

# Mock check_health() from vLLM
mock_vllm_async_llm_engine

# Test health check unspecified with mocked vLLM check_health() failure
setup_model_repository
test_check_health "health_check_unspecified_mocked_failure" "test_vllm_is_healthy"

# Test health check disabled with mocked vLLM check_health() failure
setup_model_repository
enable_health_check "false"
test_check_health "health_check_disabled_mocked_failure" "test_vllm_is_healthy"

# Test health check enabled with mocked vLLM check_health() failure
setup_model_repository
enable_health_check "true"
test_check_health "health_check_enabled_mocked_failure" "test_vllm_not_healthy"

# Unmock check_health()
unmock_vllm_async_llm_engine

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $SERVER_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
