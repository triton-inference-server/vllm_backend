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

import os
import sys
import unittest
from functools import partial
from typing import List

import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append("../../common")
from test_util import AsyncTestResultCollector, UserData, callback, create_vllm_request

PROMPTS = ["Instruct: What do you think of Computer Science?\nOutput:"]
SAMPLING_PARAMETERS = {"temperature": "0", "top_p": "1"}

server_enable_lora = True


class VLLMTritonLoraTest(AsyncTestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.vllm_model_name = "vllm_llama_multi_lora"

    def _test_vllm_model(
        self,
        prompts: List[str],
        sampling_parameters,
        lora_name: List[str],
        server_enable_lora=True,
        stream=False,
        exclude_input_in_output=None,
        expected_output=None,
    ):
        assert len(prompts) == len(
            lora_name
        ), "The number of prompts and lora names should be the same"
        user_data = UserData()
        number_of_vllm_reqs = len(prompts)

        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(number_of_vllm_reqs):
            lora = lora_name[i] if lora_name else None
            sam_para_copy = sampling_parameters.copy()
            if lora is not None:
                sam_para_copy["lora_name"] = lora
            request_data = create_vllm_request(
                prompts[i],
                i,
                stream,
                sam_para_copy,
                self.vllm_model_name,
                exclude_input_in_output=exclude_input_in_output,
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
            if type(result) is InferenceServerException:
                print(result.message())
                if server_enable_lora:
                    self.assertEqual(
                        str(result.message()),
                        f"LoRA {lora_name[i]} is not supported, we currently support ['doll', 'sheep']",
                        "InferenceServerException",
                    )
                else:
                    self.assertEqual(
                        str(result.message()),
                        "LoRA feature is not enabled.",
                        "InferenceServerException",
                    )
                self.triton_client.stop_stream()
                return

            output = result.as_numpy("text_output")
            self.assertIsNotNone(output, "`text_output` should not be None")
            if expected_output is not None:
                self.assertEqual(
                    output,
                    expected_output[i],
                    'Actual and expected outputs do not match.\n \
                                  Expected "{}" \n Actual:"{}"'.format(
                        output, expected_output[i]
                    ),
                )

        self.triton_client.stop_stream()

    def test_multi_lora_requests(self):
        sampling_parameters = {"temperature": "0", "top_p": "1"}
        # make two requests separately to avoid the different arrival of response answers
        prompt_1 = ["Instruct: What do you think of Computer Science?\nOutput:"]
        lora_1 = ["doll"]
        expected_output = [
            b" I think it is a very interesting subject.\n\nInstruct: What do you"
        ]
        self._test_vllm_model(
            prompt_1,
            sampling_parameters,
            lora_name=lora_1,
            server_enable_lora=server_enable_lora,
            stream=False,
            exclude_input_in_output=True,
            expected_output=expected_output,
        )

        prompt_2 = ["Instruct: Tell me more about soccer\nOutput:"]
        lora_2 = ["sheep"]
        expected_output = [
            b" I love soccer. I play soccer every day.\nInstruct: Tell me"
        ]
        self._test_vllm_model(
            prompt_2,
            sampling_parameters,
            lora_name=lora_2,
            server_enable_lora=server_enable_lora,
            stream=False,
            exclude_input_in_output=True,
            expected_output=expected_output,
        )

    def test_none_exist_lora(self):
        prompts = [
            "Instruct: What is the capital city of France?\nOutput:",
        ]
        loras = ["bactrian"]
        sampling_parameters = {"temperature": "0", "top_p": "1"}
        self._test_vllm_model(
            prompts,
            sampling_parameters,
            lora_name=loras,
            server_enable_lora=server_enable_lora,
            stream=False,
            exclude_input_in_output=True,
            expected_output=None,  # this request will lead to lora not supported error, so there is no expected output
        )

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    server_enable_lora = os.environ.get("SERVER_ENABLE_LORA", "false").lower() == "true"

    unittest.main()
