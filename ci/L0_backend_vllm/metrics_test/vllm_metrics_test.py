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
import re
import sys
import unittest
from functools import partial

import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append("../../common")
from test_util import TestResultCollector, UserData, callback, create_vllm_request


class VLLMTritonMetricsTest(TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")
        self.vllm_model_name = "vllm_opt"
        self.prompts = [
            "The most dangerous animal is",
            "The capital of France is",
            "The future of AI is",
        ]
        self.sampling_parameters = {"temperature": "0", "top_p": "1"}

    def get_vllm_metrics(self):
        """
        Store vllm metrics in a dictionary.
        """
        r = requests.get(f"http://{self.tritonserver_ipaddr}:8002/metrics")
        r.raise_for_status()

        # Regular expression to match the pattern
        pattern = r"^(vllm:[^ {]+)(?:{.*})? ([0-9.-]+)$"
        vllm_dict = {}

        # Find all matches in the text
        matches = re.findall(pattern, r.text, re.MULTILINE)

        for match in matches:
            key, value = match
            vllm_dict[key] = float(value) if "." in value else int(value)

        return vllm_dict

    def vllm_infer(
        self,
        prompts,
        sampling_parameters,
        model_name,
    ):
        """
        Helper function to send async stream infer requests to vLLM.
        """
        user_data = UserData()
        number_of_vllm_reqs = len(prompts)

        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(number_of_vllm_reqs):
            request_data = create_vllm_request(
                prompts[i],
                i,
                False,
                sampling_parameters,
                model_name,
                True,
            )
            self.triton_client.async_stream_infer(
                model_name=model_name,
                inputs=request_data["inputs"],
                request_id=request_data["request_id"],
                outputs=request_data["outputs"],
                parameters=sampling_parameters,
            )

        for _ in range(number_of_vllm_reqs):
            result = user_data._completed_requests.get()
            if type(result) is InferenceServerException:
                print(result.message())
            self.assertIsNot(type(result), InferenceServerException, str(result))

            output = result.as_numpy("text_output")
            self.assertIsNotNone(output, "`text_output` should not be None")

        self.triton_client.stop_stream()

    def test_vllm_metrics(self):
        # Test vLLM metrics
        self.vllm_infer(
            prompts=self.prompts,
            sampling_parameters=self.sampling_parameters,
            model_name=self.vllm_model_name,
        )
        metrics_dict = self.get_vllm_metrics()

        # vllm:prompt_tokens_total
        self.assertEqual(metrics_dict["vllm:prompt_tokens_total"], 18)
        # vllm:generation_tokens_total
        self.assertEqual(metrics_dict["vllm:generation_tokens_total"], 48)

    def test_vllm_metrics_disabled(self):
        # Test vLLM metrics
        self.vllm_infer(
            prompts=self.prompts,
            sampling_parameters=self.sampling_parameters,
            model_name=self.vllm_model_name,
        )
        metrics_dict = self.get_vllm_metrics()

        # No vLLM metric found
        self.assertEqual(len(metrics_dict), 0)

    def test_vllm_metrics_refused(self):
        # Test vLLM metrics
        self.vllm_infer(
            prompts=self.prompts,
            sampling_parameters=self.sampling_parameters,
            model_name=self.vllm_model_name,
        )
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.get_vllm_metrics()

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
