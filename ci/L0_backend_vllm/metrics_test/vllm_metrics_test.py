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
import re
import sys
import unittest
from functools import partial

import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append("../../common")
from test_util import TestResultCollector, UserData, callback, create_vllm_request

_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

PROMPTS = [
    "The most dangerous animal is",
    "The capital of France is",
    "The future of AI is",
]
SAMPLING_PARAMETERS = {"temperature": "0", "top_p": "1"}


def get_metrics():
    """
    Store vllm metrics in a dictionary.
    """
    r = requests.get(f"http://{_tritonserver_ipaddr}:8002/metrics")
    r.raise_for_status()

    # Regular expression to match the pattern
    pattern = r"^(vllm:.*){.*} (\d+)$"
    vllm_dict = {}

    # Find all matches in the text
    matches = re.findall(pattern, r.text, re.MULTILINE)

    for match in matches:
        key, value = match
        vllm_dict[key] = int(value)

    return vllm_dict


class VLLMTritonMetricsTest(TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.vllm_model_name = "vllm_opt"

    def test_vllm_metrics(self):
        # Supported vLLM metrics
        expected_metrics_dict = {
            "vllm:num_requests_running": 0,
            "vllm:num_requests_waiting": 0,
            "vllm:num_requests_swapped": 0,
            "vllm:gpu_cache_usage_perc": 0,
            "vllm:cpu_cache_usage_perc": 0,
            "vllm:num_preemptions_total": 0,
            "vllm:prompt_tokens_total": 0,
            "vllm:generation_tokens_total": 0,
        }

        # Test vLLM metrics
        self._test_vllm_model(
            prompts=PROMPTS,
            sampling_parameters=SAMPLING_PARAMETERS,
            stream=False,
            send_parameters_as_tensor=True,
            model_name=self.vllm_model_name,
        )
        expected_metrics_dict["vllm:prompt_tokens_total"] = 18
        expected_metrics_dict["vllm:generation_tokens_total"] = 48
        print(get_metrics())
        print(expected_metrics_dict)
        self.assertEqual(get_metrics(), expected_metrics_dict)

        self._test_vllm_model(
            prompts=PROMPTS,
            sampling_parameters=SAMPLING_PARAMETERS,
            stream=False,
            send_parameters_as_tensor=False,
            model_name=self.vllm_model_name,
        )
        expected_metrics_dict["vllm:prompt_tokens_total"] = 36
        expected_metrics_dict["vllm:generation_tokens_total"] = 96
        self.assertEqual(get_metrics(), expected_metrics_dict)

    def _test_vllm_model(
        self,
        prompts,
        sampling_parameters,
        stream,
        send_parameters_as_tensor,
        exclude_input_in_output=None,
        expected_output=None,
        model_name="vllm_opt",
    ):
        user_data = UserData()
        number_of_vllm_reqs = len(prompts)

        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(number_of_vllm_reqs):
            request_data = create_vllm_request(
                prompts[i],
                i,
                stream,
                sampling_parameters,
                model_name,
                send_parameters_as_tensor,
                exclude_input_in_output=exclude_input_in_output,
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
            if type(result) is InferenceServerException:
                print(result.message())
            self.assertIsNot(type(result), InferenceServerException, str(result))

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

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
