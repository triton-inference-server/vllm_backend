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

PROMPTS = ["The most dangerous animal is"]
SAMPLING_PARAMETERS = {"temperature": "0", "top_p": "1"}


class VLLMTritonStreamTest(AsyncTestResultCollector):
    async def _test_vllm_model(
        self,
        prompts=PROMPTS,
        sampling_parameters=SAMPLING_PARAMETERS,
        stream=True,
        exclude_input_in_output=None,
        expected_output=None,
        expect_error=False,
    ):
        async with grpcclient.InferenceServerClient(
            url="localhost:8001"
        ) as triton_client:
            model_name = "vllm_opt"

            async def request_iterator():
                for i, prompt in enumerate(prompts):
                    yield create_vllm_request(
                        prompt,
                        i,
                        stream,
                        sampling_parameters,
                        model_name,
                        exclude_input_in_output=exclude_input_in_output,
                    )

            response_iterator = triton_client.stream_infer(
                inputs_iterator=request_iterator()
            )
            final_response = []
            async for response in response_iterator:
                result, error = response
                if expect_error:
                    self.assertIsInstance(error, InferenceServerException)
                    self.assertEqual(
                        error.message(),
                        "Error generating stream: When streaming, `exclude_input_in_output` = False is not allowed.",
                        error,
                    )
                    return

                self.assertIsNone(error, error)
                self.assertIsNotNone(result, result)
                output = result.as_numpy("text_output")
                self.assertIsNotNone(output, "`text_output` should not be None")
                final_response.append(str(output[0], encoding="utf-8"))
            if expected_output is not None:
                self.assertEqual(
                    final_response,
                    expected_output,
                    'Expected to receive the following response: "{}",\
                    but received "{}".'.format(
                        expected_output, final_response
                    ),
                )

    async def test_vllm_model_enabled_stream(self):
        """
        Verifying that request with multiple prompts runs successfully.
        """
        prompts = [
            "The most dangerous animal is",
            "The future of AI is",
        ]

        await self._test_vllm_model(prompts=prompts)

    async def test_vllm_model_enabled_stream_exclude_input_in_output_default(self):
        """
        Verifying that streaming request returns only generated diffs, which
        is default behaviour for `stream=True`.
        """
        expected_output = [
            " the",
            " one",
            " that",
            " is",
            " most",
            " likely",
            " to",
            " be",
            " killed",
            " by",
            " a",
            " car",
            ".",
            "\n",
            "I",
            "'m",
        ]
        await self._test_vllm_model(expected_output=expected_output)

    async def test_vllm_model_enabled_stream_exclude_input_in_output_false(self):
        """
        Verifying that streaming request returns only generated diffs even if
        `exclude_input_in_output` is set to False explicitly.
        """
        expected_output = "Error generating stream: When streaming, `exclude_input_in_output` = False is not allowed."
        await self._test_vllm_model(
            exclude_input_in_output=False,
            expected_output=expected_output,
            expect_error=True,
        )


if __name__ == "__main__":
    unittest.main()
