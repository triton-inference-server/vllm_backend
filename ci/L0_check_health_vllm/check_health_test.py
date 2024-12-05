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

import json

import numpy as np
import tritonclient.grpc as grpcclient


class TestCheckHealth:
    _grpc_url = "localhost:8001"
    _model_name = "vllm_opt"
    _sampling_parameters = {"temperature": "0", "top_p": "1"}
    _prompt = "In this example,"

    def _get_inputs(self, prompt, stream=True, sampling_parameters=None):
        inputs = []

        inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(
            np.array([prompt.encode("utf-8")], dtype=np.object_)
        )

        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(np.array([stream], dtype=bool))

        if sampling_parameters is not None:
            inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(
                np.array(
                    [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
                )
            )

        return inputs

    def _callback(self, result, error):
        self._responses.append({"result": result, "error": error})

    def _llm_infer(self):
        inputs = self._get_inputs(
            self._prompt, stream=True, sampling_parameters=self._sampling_parameters
        )
        self._responses = []
        with grpcclient.InferenceServerClient(self._grpc_url) as client:
            client.start_stream(self._callback)
            client.async_stream_infer(
                self._model_name, inputs=inputs, parameters=self._sampling_parameters
            )
            client.stop_stream()

    def _assert_text_output_valid(self):
        text_output = ""
        for response in self._responses:
            result, error = response["result"], response["error"]
            assert error is None
            text_output += result.as_numpy(name="text_output")[0].decode("utf-8")
        assert len(text_output) > 0, "output is empty"
        assert text_output.count(" ") > 4, "output is not a sentence"

    def _assert_infer_exception(self, expected_exception_message):
        assert len(self._responses) == 1
        for response in self._responses:
            result, error = response["result"], response["error"]
            assert result is None
            assert str(error) == expected_exception_message

    def _assert_model_ready(self, expected_readiness):
        with grpcclient.InferenceServerClient(self._grpc_url) as client:
            # is_model_ready API
            assert client.is_model_ready(self._model_name) == expected_readiness
            # get_model_repository_index API
            model_state = None
            for model_index in client.get_model_repository_index().models:
                if model_index.name == self._model_name:
                    assert model_state is None, "duplicate model index found"
                    model_state = model_index.state == "READY"
            assert model_state == expected_readiness

    def test_vllm_is_healthy(self):
        num_repeats = 3
        for i in range(num_repeats):
            self._assert_model_ready(True)
            self._llm_infer()
            self._assert_text_output_valid()
        self._assert_model_ready(True)

    def test_vllm_not_healthy(self):
        self._assert_model_ready(True)
        # The 1st infer should complete successfully
        self._llm_infer()
        self._assert_text_output_valid()
        self._assert_model_ready(True)
        # The 2nd infer should begin with health check failed
        self._llm_infer()
        self._assert_infer_exception(
            "Model is unavailable due to unhealthy vLLM engine"
        )
        self._assert_model_ready(False)
        # The 3rd infer should have model not found
        self._llm_infer()
        self._assert_infer_exception(
            "Request for unknown model: 'vllm_opt' has no available versions"
        )
        self._assert_model_ready(False)
