#!/usr/bin/env python3

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


class VLLMTest(TestResultCollector):
    def setUp(self):
        self.prompts = [
            "Hello, my name is",
            "The most dangerous animal is",
            "The capital of France is",
            "The future of AI is",
        ]
        self.model_name = "vllm"
        self.sampling_parameters = {"temperature": "0.1", "top_p": "0.95"}

    def _create_request_inputs_outputs(
        self, prompt, stream, send_parameters_as_tensor=True
    ):
        inputs = []

        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        inputs.append(grpcclient.InferInput("prompt", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(prompt_data)

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        if send_parameters_as_tensor:
            sampling_parameters_data = np.array(
                [json.dumps(self.sampling_parameters).encode("utf-8")], dtype=np.object_
            )
            inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_parameters_data)

        outputs = [grpcclient.InferRequestedOutput("text")]

        return inputs, outputs

    def test_vllm_with_sampling_parameters_as_tensor(self):
        user_data = UserData()
        stream = False
        number_of_requests = len(self.prompts)

        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))

            for i in range(number_of_requests):
                inputs, outputs = self._create_request_inputs_outputs(
                    self.prompts[i], stream
                )
                triton_client.async_stream_infer(
                    model_name=self.model_name,
                    request_id=str(i),
                    inputs=inputs,
                    outputs=outputs,
                    parameters=self.sampling_parameters,
                )

            for i in range(number_of_requests):
                result = user_data._completed_requests.get()
                self.assertIsNot(type(result), InferenceServerException)

                output = result.as_numpy("text")
                self.assertIsNotNone(output)

    def test_vllm_with_sampling_parameters_as_request_parameters(self):
        user_data = UserData()
        stream = False
        number_of_requests = len(self.prompts)

        with grpcclient.InferenceServerClient("localhost:8001") as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))

            for i in range(number_of_requests):
                inputs, outputs = self._create_request_inputs_outputs(
                    self.prompts[i], stream, False
                )
                triton_client.async_stream_infer(
                    model_name=self.model_name,
                    request_id=str(i),
                    inputs=inputs,
                    outputs=outputs,
                    parameters=self.sampling_parameters,
                )

            for i in range(number_of_requests):
                result = user_data._completed_requests.get()
                self.assertIsNot(type(result), InferenceServerException)

                output = result.as_numpy("text")
                self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
