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

# Prepare Model
rm -rf models && mkdir -p models
SAMPLE_MODELS_REPO="../../samples/model_repository"
cp -r $SAMPLE_MODELS_REPO/vllm_model models/vllm_opt
sed -i 's/"gpu_memory_utilization": 0.5/"gpu_memory_utilization": 0.3/' models/vllm_opt/1/model.json

RET=0

# Test
SERVER_LOG="additional_outputs_test.server.log"
SERVER_ARGS="--model-repository=models"
# Cold start on SBSA device can take longer than default 120 seconds
PREV_SERVER_TIMEOUT=$SERVER_TIMEOUT
SERVER_TIMEOUT=240
run_server
SERVER_TIMEOUT=$PREV_SERVER_TIMEOUT
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e
python3 -m pytest --junitxml=test_additional_outputs.xml -s -v additional_outputs_test.py
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** additional_outputs_test FAILED. \n***"
    RET=1
fi
set -e
kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $SERVER_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
