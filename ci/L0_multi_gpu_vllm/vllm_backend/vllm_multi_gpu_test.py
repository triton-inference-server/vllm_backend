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

import os
import sys
import unittest
from functools import partial

import tritonclient.grpc as grpcclient
from tritonclient.utils import *
from vllm.utils import import_pynvml

sys.path.append("../../common")
from test_util import TestResultCollector, UserData, callback, create_vllm_request

pynvml = import_pynvml()


class VLLMMultiGPUTest(TestResultCollector):
    def setUp(self):
        pynvml.nvmlInit()
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

    def get_gpu_memory_utilization(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used

    def get_available_gpu_ids(self):
        device_count = pynvml.nvmlDeviceGetCount()
        available_gpus = []
        for gpu_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            if handle:
                available_gpus.append(gpu_id)
        return available_gpus

    def _test_vllm_multi_gpu_utilization(self, model_name: str):
        """
        Test that loading a given vLLM model will increase GPU utilization
        across multiple GPUs, and run a sanity check inference to confirm
        that the loaded multi-gpu/multi-instance model is working as expected.
        """
        gpu_ids = self.get_available_gpu_ids()
        self.assertGreaterEqual(len(gpu_ids), 2, "Error: Detected single GPU")

        print("\n\n=============== Before Loading vLLM Model ===============")
        mem_util_before_loading_model = {}
        for gpu_id in gpu_ids:
            memory_utilization = self.get_gpu_memory_utilization(gpu_id)
            print(f"GPU {gpu_id} Memory Utilization: {memory_utilization} bytes")
            mem_util_before_loading_model[gpu_id] = memory_utilization

        self.triton_client.load_model(model_name)
        self._test_vllm_model(model_name)

        print("=============== After Loading vLLM Model ===============")
        vllm_model_used_gpus = 0
        gpu_memory_utilizations = []

        for gpu_id in gpu_ids:
            memory_utilization = self.get_gpu_memory_utilization(gpu_id)
            print(f"GPU {gpu_id} Memory Utilization: {memory_utilization} bytes")
            memory_delta = memory_utilization - mem_util_before_loading_model[gpu_id]
            if memory_delta > 0:
                vllm_model_used_gpus += 1
                gpu_memory_utilizations.append(memory_delta)

        self.assertGreaterEqual(vllm_model_used_gpus, 2)

        # Check if memory utilization is approximately equal across GPUs
        if len(gpu_memory_utilizations) >= 2:
            max_memory = max(gpu_memory_utilizations)
            min_memory = min(gpu_memory_utilizations)
            relative_diff = (max_memory - min_memory) / max_memory
            self.assertLessEqual(
                relative_diff,
                0.1,
                f"GPU memory utilization differs by {relative_diff:.2%} which exceeds the 10% threshold",
            )

    def _test_vllm_model(self, model_name: str, send_parameters_as_tensor: bool = True):
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
                model_name,
                send_parameters_as_tensor,
            )
            self.triton_client.async_stream_infer(
                model_name=model_name,
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

    def test_multi_gpu_model(self):
        """
        Tests that a multi-GPU vLLM model loads successfully on multiple GPUs
        and can handle a few sanity check inference requests.

        Multi-GPU models are currently defined here as either:
          - a single model instance with tensor parallelism > 1
          - multiple model instances each with tensor parallelism == 1

        FIXME: This test currently skips over a few combinations that may
        be enhanced in the future, such as:
          - tensor parallel models with multiple model instances
          - KIND_MODEL models with multiple model instances
        """
        model = os.environ.get("TEST_MODEL")
        kind = os.environ.get("KIND")
        tp = os.environ.get("TENSOR_PARALLELISM")
        instance_count = os.environ.get("INSTANCE_COUNT")
        for env_var in [model, kind, tp, instance_count]:
            self.assertIsNotNone(env_var)

        print(f"Test Matrix: {model=}, {kind=}, {tp=}, {instance_count=}")

        # Only support tensor parallelism or multiple instances for now, but not both.
        # Support for multi-instance tensor parallel models may require more
        # special handling in the backend to better handle device assignment.
        # NOTE: This eliminates the 1*1=1 and 2*2=4 test cases.
        if int(tp) * int(instance_count) != 2:
            msg = "TENSOR_PARALLELISM and INSTANCE_COUNT must have a product of 2 for this 2-GPU test"
            print("Skipping Test:", msg)
            self.skipTest(msg)

        # Loading a KIND_GPU model with Tensor Parallelism > 1 should fail and
        # recommend using KIND_MODEL instead for multi-gpu model instances.
        if kind == "KIND_GPU" and int(tp) > 1:
            with self.assertRaisesRegex(
                InferenceServerException, "please specify KIND_MODEL"
            ):
                self._test_vllm_multi_gpu_utilization(model)

            return

        # Loading a KIND_MODEL model with multiple instances can cause
        # oversubscription to specific GPUs and cause a CUDA OOM if the
        # gpu_memory_utilization settings are high without further handling
        # of device assignment in the backend.
        if kind == "KIND_MODEL" and int(instance_count) > 1:
            msg = "Testing multiple model instances of KIND_MODEL is not implemented at this time"
            print("Skipping Test:", msg)
            self.skipTest(msg)

        self._test_vllm_multi_gpu_utilization(model)

    def tearDown(self):
        pynvml.nvmlShutdown()
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
