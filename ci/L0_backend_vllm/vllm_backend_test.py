#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import queue
import sys
import unittest
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append("../common")
from test_util import TestResultCollector


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class VLLMTritonBackendTest(TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.vllm_model_name = "vllm_opt"
        self.python_model_name = "add_sub"

    def test_vllm_triton_backend(self):
        # Load both models
        self.triton_client.load_model(self.vllm_model_name)
        self.assertTrue(self.triton_client.is_model_ready(self.vllm_model_name))
        self.triton_client.load_model(self.python_model_name)
        self.assertTrue(self.triton_client.is_model_ready(self.python_model_name))

        # Unload vllm model and test add_sub model
        self.triton_client.unload_model(self.vllm_model_name)
        self.assertFalse(self.triton_client.is_model_ready(self.vllm_model_name))
        self._test_python_model()

        # Load vllm model and unload add_sub model
        self.triton_client.load_model(self.vllm_model_name)
        self.triton_client.unload_model(self.python_model_name)
        self.assertFalse(self.triton_client.is_model_ready(self.python_model_name))

        # Test vllm model and unload vllm model
        self._test_vllm_model(send_parameters_as_tensor=True)
        self._test_vllm_model(send_parameters_as_tensor=False)
        self.triton_client.unload_model(self.vllm_model_name)

    def _test_vllm_model(self, send_parameters_as_tensor):
        user_data = UserData()
        stream = False
        prompts = [
            "The most dangerous animal is",
            "The capital of France is",
            "The future of AI is",
        ]
        number_of_vllm_reqs = len(prompts)
        sampling_parameters = {"temperature": "0.1", "top_p": "0.95"}

        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(number_of_vllm_reqs):
            inputs, outputs = self._create_vllm_request_data(
                prompts[i], stream, sampling_parameters, send_parameters_as_tensor
            )
            self.triton_client.async_stream_infer(
                model_name=self.vllm_model_name,
                request_id=str(i),
                inputs=inputs,
                outputs=outputs,
                parameters=sampling_parameters,
            )

        for i in range(number_of_vllm_reqs):
            result = user_data._completed_requests.get()
            self.assertIsNot(type(result), InferenceServerException)

            output = result.as_numpy("TEXT")
            self.assertIsNotNone(output)

        self.triton_client.stop_stream()

    def _test_python_model(self):
        shape = [4]
        input0_data = np.random.rand(*shape).astype(np.float32)
        input1_data = np.random.rand(*shape).astype(np.float32)

        inputs = [
            grpcclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            grpcclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT0"),
            grpcclient.InferRequestedOutput("OUTPUT1"),
        ]

        response = self.triton_client.infer(
            self.python_model_name, inputs, request_id="10", outputs=outputs
        )
        self.assertTrue(
            np.allclose(input0_data + input1_data, response.as_numpy("OUTPUT0"))
        )
        self.assertTrue(
            np.allclose(input0_data - input1_data, response.as_numpy("OUTPUT1"))
        )

    def _create_vllm_request_data(
        self, prompt, stream, sampling_parameters, send_parameters_as_tensor
    ):
        inputs = []

        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        inputs.append(grpcclient.InferInput("PROMPT", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(prompt_data)

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("STREAM", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        if send_parameters_as_tensor:
            sampling_parameters_data = np.array(
                [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
            )
            inputs.append(grpcclient.InferInput("SAMPLING_PARAMETERS", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_parameters_data)

        outputs = [grpcclient.InferRequestedOutput("TEXT")]

        return inputs, outputs

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
