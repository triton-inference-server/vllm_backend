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

import queue
import sys
import unittest
from functools import partial

import tritonclient.grpc as grpcclient
from tritonclient.utils import *

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid
from vllm import SamplingParams
import asyncio

sys.path.append("../common")
from test_util import TestResultCollector, create_vllm_request


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


async def generate_python_vllm_output(prompt, llm_engine):
    request_id = random_uuid()
    sampling_parameters = {"temperature": 0.1, "top_p": 0.95}
    sampling_params = SamplingParams(**sampling_parameters)

    python_vllm_output = None
    last_output = None

    async for vllm_output in llm_engine.generate(prompt, sampling_params, request_id):
        last_output = vllm_output

    if last_output:
        python_vllm_output = [
            (prompt + output.text).encode("utf-8") for output in last_output.outputs
        ]

    return python_vllm_output


class VLLMTritonAccuracyTest(TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        vllm_engine_config = {
            "model": "facebook/opt-125m",
            "gpu_memory_utilization": 0.25,
        }

        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**vllm_engine_config)
        )
        self.vllm_model_name = "vllm_opt"

    def test_vllm_model(self):
        user_data = UserData()
        stream = False
        prompts = [
            "The most dangerous animal is",
            "The capital of France is",
            "The future of AI is",
        ]
        number_of_vllm_reqs = len(prompts)
        sampling_parameters = {"temperature": "0.1", "top_p": "0.95"}
        python_vllm_output = []
        triton_vllm_output = []

        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(number_of_vllm_reqs):
            request_data = create_vllm_request(
                prompts[i], i, stream, sampling_parameters, self.vllm_model_name
            )
            self.triton_client.async_stream_infer(
                model_name=self.vllm_model_name,
                request_id=request_data["request_id"],
                inputs=request_data["inputs"],
                outputs=request_data["outputs"],
                parameters=sampling_parameters,
            )

            python_vllm_output.extend(
                asyncio.run(generate_python_vllm_output(prompts[i], self.llm_engine))
            )

        for i in range(number_of_vllm_reqs):
            result = user_data._completed_requests.get()
            self.assertIsNot(type(result), InferenceServerException)

            output = result.as_numpy("text_output")
            self.assertIsNotNone(output)

            triton_vllm_output.extend(output)

        print(triton_vllm_output)
        print(python_vllm_output)

        self.triton_client.stop_stream()

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
