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

import sys
import unittest
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append("../../common")
from test_util import TestResultCollector, UserData, callback, create_vllm_request

PROMPTS = [
    "The most dangerous animal is",
    "The capital of France is",
    "The future of AI is",
]
SAMPLING_PARAMETERS = {"temperature": "0", "top_p": "1"}


class VLLMTritonBackendTest(TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.vllm_model_name = "vllm_opt"
        self.python_model_name = "add_sub"
        self.ensemble_model_name = "ensemble_model"
        self.vllm_load_test = "vllm_load_test"

    def test_vllm_triton_backend(self):
        # Load both vllm and add_sub models
        self.triton_client.load_model(self.vllm_load_test)
        self.assertTrue(self.triton_client.is_model_ready(self.vllm_load_test))
        self.triton_client.load_model(self.python_model_name)
        self.assertTrue(self.triton_client.is_model_ready(self.python_model_name))

        # Test to ensure that ensemble models are supported in vllm container.
        # If ensemble support not present, triton will error out at model loading stage.
        # Ensemble Model is a pipeline consisting of 1 model (vllm_opt)
        self.triton_client.load_model(self.ensemble_model_name)
        self.assertTrue(self.triton_client.is_model_ready(self.ensemble_model_name))
        self.triton_client.unload_model(self.ensemble_model_name)

        # Unload vllm model and test add_sub model
        self.triton_client.unload_model(self.vllm_load_test)
        self.assertFalse(self.triton_client.is_model_ready(self.vllm_load_test))
        self._test_python_model()

        # Load vllm model and unload add_sub model
        self.triton_client.load_model(self.vllm_load_test)
        self.assertTrue(self.triton_client.is_model_ready(self.vllm_load_test))
        self.triton_client.unload_model(self.python_model_name)
        self.assertFalse(self.triton_client.is_model_ready(self.python_model_name))

        # Test vllm model and unload vllm model
        self._test_vllm_model(
            prompts=PROMPTS,
            sampling_parameters=SAMPLING_PARAMETERS,
            stream=False,
            send_parameters_as_tensor=True,
            model_name=self.vllm_load_test,
        )
        self._test_vllm_model(
            prompts=PROMPTS,
            sampling_parameters=SAMPLING_PARAMETERS,
            stream=False,
            send_parameters_as_tensor=False,
            model_name=self.vllm_load_test,
        )
        self.triton_client.unload_model(self.vllm_load_test)
        self.assertFalse(self.triton_client.is_model_ready(self.vllm_load_test))

    def test_model_with_invalid_attributes(self):
        model_name = "vllm_invalid_1"
        with self.assertRaises(InferenceServerException):
            self.triton_client.load_model(model_name)

    def test_vllm_invalid_model_name(self):
        model_name = "vllm_invalid_2"
        with self.assertRaises(InferenceServerException):
            self.triton_client.load_model(model_name)

    def test_exclude_input_in_output_default(self):
        """
        Verifying default behavior for `exclude_input_in_output`
        in non-streaming mode.
        Expected result: prompt is returned with diffs.
        """
        prompts = [
            "The capital of France is",
        ]
        expected_output = [
            b"The capital of France is the capital of the French Republic.\n\nThe capital of France is the capital"
        ]
        sampling_parameters = {"temperature": "0", "top_p": "1"}
        self._test_vllm_model(
            prompts,
            sampling_parameters,
            stream=False,
            send_parameters_as_tensor=True,
            expected_output=expected_output,
        )

    def test_exclude_input_in_output_false(self):
        """
        Verifying behavior for `exclude_input_in_output` = False
        in non-streaming mode.
        Expected result: prompt is returned with diffs.
        """
        # Test vllm model and unload vllm model
        prompts = [
            "The capital of France is",
        ]
        expected_output = [
            b"The capital of France is the capital of the French Republic.\n\nThe capital of France is the capital"
        ]
        sampling_parameters = {"temperature": "0", "top_p": "1"}
        self._test_vllm_model(
            prompts,
            sampling_parameters,
            stream=False,
            send_parameters_as_tensor=True,
            exclude_input_in_output=False,
            expected_output=expected_output,
        )

    def test_exclude_input_in_output_true(self):
        """
        Verifying behavior for `exclude_input_in_output` = True
        in non-streaming mode.
        Expected result: only diffs are returned.
        """
        # Test vllm model and unload vllm model
        prompts = [
            "The capital of France is",
        ]
        expected_output = [
            b" the capital of the French Republic.\n\nThe capital of France is the capital"
        ]
        sampling_parameters = {"temperature": "0", "top_p": "1"}
        self._test_vllm_model(
            prompts,
            sampling_parameters,
            stream=False,
            send_parameters_as_tensor=True,
            exclude_input_in_output=True,
            expected_output=expected_output,
        )

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

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    unittest.main()
