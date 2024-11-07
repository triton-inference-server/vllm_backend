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
import pytest
import tritonclient.grpc as grpcclient


class TestAdditionalOutputs:
    _grpc_url = "localhost:8001"
    _model_name = "vllm_opt"
    _sampling_parameters = {"temperature": "0", "top_p": "1"}
    _prompt = "In this example,"

    def _get_inputs(
        self,
        prompt,
        stream=True,
        sampling_parameters=None,
        return_finish_reason=None,
        return_cumulative_logprob=None,
        return_num_token_ids=None,
    ):
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

        if return_finish_reason is not None:
            inputs.append(grpcclient.InferInput("return_finish_reason", [1], "BOOL"))
            inputs[-1].set_data_from_numpy(np.array([return_finish_reason], dtype=bool))

        if return_cumulative_logprob is not None:
            inputs.append(
                grpcclient.InferInput("return_cumulative_logprob", [1], "BOOL")
            )
            inputs[-1].set_data_from_numpy(
                np.array([return_cumulative_logprob], dtype=bool)
            )

        if return_num_token_ids is not None:
            inputs.append(grpcclient.InferInput("return_num_token_ids", [1], "BOOL"))
            inputs[-1].set_data_from_numpy(np.array([return_num_token_ids], dtype=bool))

        return inputs

    def _callback(self, result, error):
        self._responses.append({"result": result, "error": error})

    def _llm_infer(self, inputs):
        self._responses = []
        with grpcclient.InferenceServerClient(self._grpc_url) as client:
            client.start_stream(self._callback)
            client.async_stream_infer(
                self._model_name, inputs=inputs, parameters=self._sampling_parameters
            )
            client.stop_stream()
        assert len(self._responses) > 0

    def _assert_text_output_valid(self):
        text_output = ""
        for response in self._responses:
            result, error = response["result"], response["error"]
            assert error is None
            text_output += result.as_numpy(name="text_output")[0].decode("utf-8")
        assert len(text_output) > 0, "output is empty"
        assert text_output.count(" ") > 4, "output is not a sentence"

    def _assert_finish_reason(self, return_finish_reason):
        for i in range(len(self._responses)):
            result, error = self._responses[i]["result"], self._responses[i]["error"]
            assert error is None
            finish_reason_np = result.as_numpy(name="finish_reason")
            if return_finish_reason is None or return_finish_reason == False:
                assert finish_reason_np is None
                continue
            finish_reason = finish_reason_np[0].decode("utf-8")
            if i < len(self._responses) - 1:
                assert finish_reason == "None"
            else:
                assert finish_reason == "length"

    def _assert_cumulative_logprob(self, return_cumulative_logprob):
        prev_cumulative_logprob = 0.0
        for response in self._responses:
            result, error = response["result"], response["error"]
            assert error is None
            cumulative_logprob_np = result.as_numpy(name="cumulative_logprob")
            if return_cumulative_logprob is None or return_cumulative_logprob == False:
                assert cumulative_logprob_np is None
                continue
            cumulative_logprob = cumulative_logprob_np[0].astype(float)
            assert cumulative_logprob != prev_cumulative_logprob
            prev_cumulative_logprob = cumulative_logprob

    def _assert_num_token_ids(self, return_num_token_ids):
        for response in self._responses:
            result, error = response["result"], response["error"]
            assert error is None
            num_token_ids_np = result.as_numpy(name="num_token_ids")
            if return_num_token_ids is None or return_num_token_ids == False:
                assert num_token_ids_np is None
                continue
            num_token_ids = num_token_ids_np[0].astype(int)
            # TODO: vLLM may return token ids identical to the previous one when
            #       streaming, for example:
            #
            #       prev: None
            #       curr: text=' the', token_ids=array('l', [5])
            #
            #       prev: text=' the', token_ids=array('l', [5, 1385])
            #       curr: text=' the term', token_ids=array('l', [5, 1385])
            #
            #       prev: text=' the term', token_ids=array('l', [5, 1385, 44])
            #       curr: text=' the term', token_ids=array('l', [5, 1385, 44])
            #
            #       prev: text=' the term', token_ids=array('l', [5, 1385, 44, 48])
            #       curr: text=' the term “', token_ids=array('l', [5, 1385, 44, 48])
            #
            #       If this is no longer the case in a future release, change the assert
            #       to assert num_token_ids > 0.
            assert num_token_ids >= 0

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("return_finish_reason", [None, True, False])
    @pytest.mark.parametrize("return_cumulative_logprob", [None, True, False])
    @pytest.mark.parametrize("return_num_token_ids", [None, True, False])
    def test_additional_outputs(
        self,
        stream,
        return_finish_reason,
        return_cumulative_logprob,
        return_num_token_ids,
    ):
        inputs = self._get_inputs(
            self._prompt,
            stream=stream,
            sampling_parameters=self._sampling_parameters,
            return_finish_reason=return_finish_reason,
            return_cumulative_logprob=return_cumulative_logprob,
            return_num_token_ids=return_num_token_ids,
        )
        self._llm_infer(inputs)
        self._assert_text_output_valid()
        self._assert_finish_reason(return_finish_reason)
        self._assert_cumulative_logprob(return_cumulative_logprob)
        self._assert_num_token_ids(return_num_token_ids)
