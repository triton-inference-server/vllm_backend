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
import time
import unittest
from functools import partial

import tritonclient.grpc as grpcclient
from tritonclient.utils import *

sys.path.append("../../common")
from test_util import TestResultCollector, UserData, callback, create_vllm_request


class VLLMRequestCancelTest(TestResultCollector):
    def test_request_cancellation(self, send_parameters_as_tensor=True):
        with grpcclient.InferenceServerClient(url="localhost:8001") as triton_client:
            log_file_path = "./request_cancellation_server.log"
            user_data = UserData()
            model_name = "vllm_opt"
            stream = False
            sampling_parameters = {
                "temperature": "0",
                "top_p": "1",
                "max_tokens": "1500",
            }
            prompt = f"Write an original and creative poem of at least 200 words."

            triton_client.start_stream(callback=partial(callback, user_data))

            request_data = create_vllm_request(
                prompt,
                "1",
                stream,
                sampling_parameters,
                model_name,
                send_parameters_as_tensor,
            )
            triton_client.async_stream_infer(
                model_name=model_name,
                request_id=request_data["request_id"],
                inputs=request_data["inputs"],
                outputs=request_data["outputs"],
                parameters=sampling_parameters,
            )
            time.sleep(1)

            triton_client.stop_stream(cancel_requests=True)
            time.sleep(1)
            self.assertFalse(user_data._completed_requests.empty())

            result = user_data._completed_requests.get()
            self.assertIsInstance(result, InferenceServerException)
            self.assertEqual(result.status(), "StatusCode.CANCELLED")
            self.assertTrue(user_data._completed_requests.empty())

            with open(log_file_path, mode="r") as log_file:
                log_text = log_file.read()
                self.assertIn("[vllm] Cancelling the request", log_text)
                self.assertIn("[vllm] Successfully cancelled the request", log_text)


if __name__ == "__main__":
    unittest.main()
