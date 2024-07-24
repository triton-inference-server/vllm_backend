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

# Get latest VLLM RELEASED VERSION from https://github.com/triton-inference-server/vllm_backend/releases
TAG=$(curl https://api.github.com/repos/triton-inference-server/vllm_backend/releases/latest | grep -i "tag_name" | awk -F '"' '{print $4}')
export TRITON_CONTAINER_VERSION=${TAG#v} # example: 24.06
echo "TRITON_CONTAINER_VERSION = ${TRITON_CONTAINER_VERSION}" 

# Get latest VLLM RELEASED VERSION from https://github.com/vllm-project/vllm/releases
TAG=$(curl https://api.github.com/repos/triton-inference-server/server/releases/latest | grep -i "tag_name" | awk -F '"' '{print $4}')
export VLLM_VERSION=${TAG#v} # example: 0.5.3.post1

git clone -b mesharma-ci https://github.com/triton-inference-server/server.git 
set -x && python3 server/build.py -v  --enable-logging
                --enable-stats
                --enable-tracing
                --enable-metrics
                --enable-gpu-metrics
                --enable-cpu-metrics
                --enable-gpu
                --container-prebuild-command="docker login -u gitlab-ci-token -p ${CI_JOB_TOKEN} ${CI_REGISTRY}"
                --filesystem=gcs
                --filesystem=s3
                --filesystem=azure_storage
                --endpoint=http
                --endpoint=grpc
                --endpoint=sagemaker
                --endpoint=vertex-ai
                --upstream-container-version=${TRITON_CONTAINER_VERSION}
                --backend=python:r${TRITON_CONTAINER_VERSION}
                --backend=vllm:r${TRITON_CONTAINER_VERSION}
                --vllm-version=${VLLM_VERSION} || RV=$?; 2>&1
echo "RV=$RV"
