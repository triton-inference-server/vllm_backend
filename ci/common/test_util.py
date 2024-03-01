#!/usr/bin/env python3

# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import queue
import unittest

import numpy as np
import tritonclient.grpc as grpcclient


class TestResultCollector(unittest.TestCase):
    # TestResultCollector stores test result and prints it to stdout. In order
    # to use this class, unit tests must inherit this class. Use
    # `check_test_results` bash function from `common/util.sh` to verify the
    # expected number of tests produced by this class

    @classmethod
    def setResult(cls, total, errors, failures):
        cls.total, cls.errors, cls.failures = total, errors, failures

    @classmethod
    def tearDownClass(cls):
        # this method is called when all the unit tests in a class are
        # finished.
        json_res = {"total": cls.total, "errors": cls.errors, "failures": cls.failures}
        with open("test_results.txt", "w+") as f:
            f.write(json.dumps(json_res))

    def run(self, result=None):
        # result argument stores the accumulative test results
        test_result = super().run(result)
        total = test_result.testsRun
        errors = len(test_result.errors)
        failures = len(test_result.failures)
        self.setResult(total, errors, failures)


class AsyncTestResultCollector(unittest.IsolatedAsyncioTestCase):
    # TestResultCollector stores test result and prints it to stdout. In order
    # to use this class, unit tests must inherit this class. Use
    # `check_test_results` bash function from `common/util.sh` to verify the
    # expected number of tests produced by this class

    @classmethod
    def setResult(cls, total, errors, failures):
        cls.total, cls.errors, cls.failures = total, errors, failures

    @classmethod
    def tearDownClass(cls):
        # this method is called when all the unit tests in a class are
        # finished.
        json_res = {"total": cls.total, "errors": cls.errors, "failures": cls.failures}
        with open("test_results.txt", "w+") as f:
            f.write(json.dumps(json_res))

    def run(self, result=None):
        # result argument stores the accumulative test results
        test_result = super().run(result)
        total = test_result.testsRun
        errors = len(test_result.errors)
        failures = len(test_result.failures)
        self.setResult(total, errors, failures)


def create_vllm_request(
    prompt,
    request_id,
    stream,
    sampling_parameters,
    model_name,
    send_parameters_as_tensor=True,
    exclude_input_in_output=None,
):
    inputs = []

    inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
    inputs[-1].set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=np.object_))

    inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(np.array([stream], dtype=bool))

    if send_parameters_as_tensor:
        sampling_parameters_data = np.array(
            [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
        )
        inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(sampling_parameters_data)

    if exclude_input_in_output is not None:
        inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(np.array([exclude_input_in_output], dtype=bool))

    outputs = [grpcclient.InferRequestedOutput("text_output")]

    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": str(request_id),
        "parameters": sampling_parameters,
    }


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
