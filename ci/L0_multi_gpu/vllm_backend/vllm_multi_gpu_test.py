# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import unittest
from functools import partial

import nvidia_smi
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append("../../common")
from test_util import TestResultCollector, UserData, callback, create_vllm_request


class VLLMMultiGPUTest(TestResultCollector):
    def setUp(self):
        nvidia_smi.nvmlInit()
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.vllm_model_name = "vllm_opt"

    def get_gpu_memory_utilization(self, gpu_id):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.used

    def get_available_gpu_ids(self):
        device_count = nvidia_smi.nvmlDeviceGetCount()
        available_gpus = []
        for gpu_id in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
            if handle:
                available_gpus.append(gpu_id)
        return available_gpus

    def test_vllm_multi_gpu_utilization(self):
        gpu_ids = self.get_available_gpu_ids()
        self.assertGreaterEqual(len(gpu_ids), 2, "Error: Detected single GPU")

        print("\n\n=============== Before Loading vLLM Model ===============")
        mem_util_before_loading_model = {}
        for gpu_id in gpu_ids:
            memory_utilization = self.get_gpu_memory_utilization(gpu_id)
            print(f"GPU {gpu_id} Memory Utilization: {memory_utilization} bytes")
            mem_util_before_loading_model[gpu_id] = memory_utilization

        self.triton_client.load_model(self.vllm_model_name)
        self._test_vllm_model()

        print("=============== After Loading vLLM Model ===============")
        vllm_model_used_gpus = 0
        for gpu_id in gpu_ids:
            memory_utilization = self.get_gpu_memory_utilization(gpu_id)
            print(f"GPU {gpu_id} Memory Utilization: {memory_utilization} bytes")
            if memory_utilization > mem_util_before_loading_model[gpu_id]:
                vllm_model_used_gpus += 1

        self.assertGreaterEqual(vllm_model_used_gpus, 2)

    def _test_vllm_model(self, send_parameters_as_tensor=True):
        user_data = UserData()
        stream = False
        prompts = [
            "The most dangerous animal is",
            "The capital of France is",
            "The future of AI is",
        ]
        number_of_vllm_reqs = len(prompts)
        sampling_parameters = {"temperature": "0", "top_p": "1"}

        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(number_of_vllm_reqs):
            request_data = create_vllm_request(
                prompts[i],
                i,
                stream,
                sampling_parameters,
                self.vllm_model_name,
                send_parameters_as_tensor,
            )
            self.triton_client.async_stream_infer(
                model_name=self.vllm_model_name,
                request_id=request_data["request_id"],
                inputs=request_data["inputs"],
                outputs=request_data["outputs"],
                parameters=sampling_parameters,
            )

        for i in range(number_of_vllm_reqs):
            result = user_data._completed_requests.get()
            self.assertIsNot(type(result), InferenceServerException)

            output = result.as_numpy("text_output")
            self.assertIsNotNone(output)

        self.triton_client.stop_stream()

    def tearDown(self):
        nvidia_smi.nvmlShutdown()
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
