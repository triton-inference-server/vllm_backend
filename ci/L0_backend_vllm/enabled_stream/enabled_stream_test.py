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


import sys
import unittest

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *

sys.path.append("../../common")
from test_util import AsyncTestResultCollector, create_vllm_request


class VLLMTritonStreamTest(AsyncTestResultCollector):
    async def test_vllm_model_enabled_stream(self):
        async with grpcclient.InferenceServerClient(
            url="localhost:8001"
        ) as triton_client:
            model_name = "vllm_opt"
            stream = True
            prompts = [
                "The most dangerous animal is",
                "The future of AI is",
            ]
            sampling_parameters = {"temperature": "0", "top_p": "1"}

            async def request_iterator():
                for i, prompt in enumerate(prompts):
                    yield create_vllm_request(
                        prompt, i, stream, sampling_parameters, model_name
                    )

            response_iterator = triton_client.stream_infer(
                inputs_iterator=request_iterator()
            )

            async for response in response_iterator:
                result, error = response
                self.assertIsNone(error, str(error))
                self.assertIsNotNone(result, str(result))

                output = result.as_numpy("text_output")
                self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
