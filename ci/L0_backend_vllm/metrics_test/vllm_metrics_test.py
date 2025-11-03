# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        self.inference_count = 2

    def parse_vllm_metrics(self):
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

        # Run the inference twice in case metrics are updated but engine crashes.
        for _ in range(self.inference_count):
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
        metrics_dict = self.parse_vllm_metrics()
        total_prompts = len(self.prompts)

        # vllm:prompt_tokens_total
        # (2, 133, 144, 2702, 3477, 16)
        # (2, 133, 812, 9, 1470, 16)
        # (2, 133, 499, 9, 4687, 16)
        self.assertEqual(
            metrics_dict["vllm:prompt_tokens_total"], 18 * self.inference_count
        )
        # vllm:generation_tokens_total
        # (5, 65, 14, 16, 144, 533, 7, 28, 848, 30, 10, 512, 4, 50118, 100, 437)
        # (5, 812, 9, 5, 1515, 3497, 4, 50118, 50118, 133, 812, 9, 1470, 16, 5, 812)
        # (11, 5, 1420, 9, 5, 82, 4, 50118, 50118, 133, 499, 9, 4687, 16, 11, 5)
        self.assertEqual(
            metrics_dict["vllm:generation_tokens_total"], 48 * self.inference_count
        )
        # vllm:time_to_first_token_seconds
        self.assertEqual(
            metrics_dict["vllm:time_to_first_token_seconds_count"],
            total_prompts * self.inference_count,
        )
        self.assertGreater(metrics_dict["vllm:time_to_first_token_seconds_sum"], 0)
        self.assertEqual(
            metrics_dict["vllm:time_to_first_token_seconds_bucket"],
            total_prompts * self.inference_count,
        )
        # vllm:time_per_output_token_seconds
        self.assertEqual(
            metrics_dict["vllm:time_per_output_token_seconds_count"],
            45 * self.inference_count,
        )
        self.assertGreater(metrics_dict["vllm:time_per_output_token_seconds_sum"], 0)
        self.assertEqual(
            metrics_dict["vllm:time_per_output_token_seconds_bucket"],
            45 * self.inference_count,
        )
        # vllm:e2e_request_latency_seconds
        self.assertEqual(
            metrics_dict["vllm:e2e_request_latency_seconds_count"],
            total_prompts * self.inference_count,
        )
        self.assertGreater(metrics_dict["vllm:e2e_request_latency_seconds_sum"], 0)
        self.assertEqual(
            metrics_dict["vllm:e2e_request_latency_seconds_bucket"],
            total_prompts * self.inference_count,
        )
        # vllm:request_prompt_tokens
        self.assertEqual(
            metrics_dict["vllm:request_prompt_tokens_count"],
            total_prompts * self.inference_count,
        )
        self.assertEqual(
            metrics_dict["vllm:request_prompt_tokens_sum"], 18 * self.inference_count
        )
        self.assertEqual(
            metrics_dict["vllm:request_prompt_tokens_bucket"],
            total_prompts * self.inference_count,
        )
        # vllm:request_generation_tokens
        self.assertEqual(
            metrics_dict["vllm:request_generation_tokens_count"],
            total_prompts * self.inference_count,
        )
        self.assertEqual(
            metrics_dict["vllm:request_generation_tokens_sum"],
            48 * self.inference_count,
        )
        self.assertEqual(
            metrics_dict["vllm:request_generation_tokens_bucket"],
            total_prompts * self.inference_count,
        )

    # TODO: Revisit this test due to the removal of best_of
    def test_custom_sampling_params(self):
        # Adding sampling parameters for testing metrics.
        # Definitions can be found here https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html
        n, temperature = 2, 1
        custom_sampling_parameters = self.sampling_parameters.copy()
        custom_sampling_parameters.update(
            {"n": str(n), "temperature": str(temperature)}
        )

        # Test vLLM metrics
        self.vllm_infer(
            prompts=self.prompts,
            sampling_parameters=custom_sampling_parameters,
            model_name=self.vllm_model_name,
        )
        metrics_dict = self.parse_vllm_metrics()
        # vllm:request_params_n
        self.assertIn("vllm:request_params_n_count", metrics_dict)
        self.assertIn("vllm:request_params_n_sum", metrics_dict)
        self.assertIn("vllm:request_params_n_bucket", metrics_dict)

    def test_vllm_metrics_disabled(self):
        # Test vLLM metrics
        self.vllm_infer(
            prompts=self.prompts,
            sampling_parameters=self.sampling_parameters,
            model_name=self.vllm_model_name,
        )
        metrics_dict = self.parse_vllm_metrics()

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
            self.parse_vllm_metrics()

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
