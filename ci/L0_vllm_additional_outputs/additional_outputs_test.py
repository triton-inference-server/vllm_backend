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
import unittest

import numpy as np
import tritonclient.grpc as grpcclient


class InferTest(unittest.TestCase):
    _grpc_url = "localhost:8001"
    _model_name = "vllm_opt"
    _sampling_parameters = {"temperature": "0", "top_p": "1"}
    _prompt = "In this example,"

    def _get_inputs(
        self,
        prompt,
        stream=True,
        sampling_parameters=None,
        output_finish_reason=None,
        output_cumulative_logprob=None,
        output_num_token_ids=None,
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

        if output_finish_reason is not None:
            inputs.append(grpcclient.InferInput("output_finish_reason", [1], "BOOL"))
            inputs[-1].set_data_from_numpy(np.array([output_finish_reason], dtype=bool))

        if output_cumulative_logprob is not None:
            inputs.append(
                grpcclient.InferInput("output_cumulative_logprob", [1], "BOOL")
            )
            inputs[-1].set_data_from_numpy(
                np.array([output_cumulative_logprob], dtype=bool)
            )

        if output_num_token_ids is not None:
            inputs.append(grpcclient.InferInput("output_num_token_ids", [1], "BOOL"))
            inputs[-1].set_data_from_numpy(np.array([output_num_token_ids], dtype=bool))

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
        self.assertGreater(len(self._responses), 0)

    def _assert_text_output_valid(self):
        text_output = ""
        for response in self._responses:
            result, error = response["result"], response["error"]
            self.assertIsNone(error)
            text_output += result.as_numpy(name="text_output")[0].decode("utf-8")
        self.assertGreater(len(text_output), 0, "output is empty")
        self.assertGreater(text_output.count(" "), 4, "output is not a sentence")

    def _assert_finish_reason(self, output_finish_reason):
        for i in range(len(self._responses)):
            result, error = self._responses[i]["result"], self._responses[i]["error"]
            self.assertIsNone(error)
            finish_reason_np = result.as_numpy(name="finish_reason")
            if output_finish_reason is None or output_finish_reason == False:
                self.assertIsNone(finish_reason_np)
                continue
            finish_reason = finish_reason_np[0].decode("utf-8")
            if i < len(self._responses) - 1:
                self.assertEqual(finish_reason, "None")
            else:
                self.assertEqual(finish_reason, "length")

    def _assert_cumulative_logprob(self, output_cumulative_logprob):
        prev_cumulative_logprob = 0.0
        for response in self._responses:
            result, error = response["result"], response["error"]
            self.assertIsNone(error)
            cumulative_logprob_np = result.as_numpy(name="cumulative_logprob")
            if output_cumulative_logprob is None or output_cumulative_logprob == False:
                self.assertIsNone(cumulative_logprob_np)
                continue
            cumulative_logprob = cumulative_logprob_np[0].astype(float)
            self.assertNotEqual(cumulative_logprob, prev_cumulative_logprob)
            prev_cumulative_logprob = cumulative_logprob

    def _assert_num_token_ids(self, output_num_token_ids):
        for response in self._responses:
            result, error = response["result"], response["error"]
            self.assertIsNone(error)
            num_token_ids_np = result.as_numpy(name="num_token_ids")
            if output_num_token_ids is None or output_num_token_ids == False:
                self.assertIsNone(num_token_ids_np)
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
            #       to assertGreater().
            self.assertGreaterEqual(num_token_ids, 0)

    def _assert_additional_outputs_valid(
        self,
        stream,
        output_finish_reason,
        output_cumulative_logprob,
        output_num_token_ids,
    ):
        inputs = self._get_inputs(
            self._prompt,
            stream=stream,
            sampling_parameters=self._sampling_parameters,
            output_finish_reason=output_finish_reason,
            output_cumulative_logprob=output_cumulative_logprob,
            output_num_token_ids=output_num_token_ids,
        )
        self._llm_infer(inputs)
        self._assert_text_output_valid()
        self._assert_finish_reason(output_finish_reason)
        self._assert_cumulative_logprob(output_cumulative_logprob)
        self._assert_num_token_ids(output_num_token_ids)

    def test_additional_outputs(self):
        for stream in [True, False]:
            choices = [None, False, True]
            for output_finish_reason in choices:
                for output_cumulative_logprob in choices:
                    for output_num_token_ids in choices:
                        self._assert_additional_outputs_valid(
                            stream,
                            output_finish_reason,
                            output_cumulative_logprob,
                            output_num_token_ids,
                        )


if __name__ == "__main__":
    unittest.main()
