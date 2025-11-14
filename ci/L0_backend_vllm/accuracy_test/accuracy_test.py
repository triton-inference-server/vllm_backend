# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import asyncio
import json
import pickle
import sys
import unittest
from functools import partial

import tritonclient.grpc as grpcclient
from tritonclient.utils import *
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import StructuredOutputsParams
from vllm.utils import random_uuid

sys.path.append("../../common")
from test_util import TestResultCollector, UserData, callback, create_vllm_request

VLLM_ENGINE_CONFIG = {
    "model": "facebook/opt-125m",
    "gpu_memory_utilization": 0.3,
}


PROMPTS = [
    "The most dangerous animal is",
    "The capital of France is",
    "The future of AI is",
]

STRUCTURED_PROMPTS = ["Classify intent of the sentence: Harry Potter is underrated. "]

SAMPLING_PARAMETERS = {"temperature": 0, "top_p": 1}


async def generate_python_vllm_output(
    prompt,
    llm_engine,
    sampling_params=SamplingParams(**SAMPLING_PARAMETERS),
    structured_generation=None,
):
    request_id = random_uuid()
    python_vllm_output = None
    last_output = None
    if structured_generation:
        sampling_params.structured_outputs = structured_generation

    async for vllm_output in llm_engine.generate(prompt, sampling_params, request_id):
        last_output = vllm_output

    if last_output:
        python_vllm_output = [
            (prompt + output.text).encode("utf-8") for output in last_output.outputs
        ]
    return python_vllm_output


async def prepare_vllm_baseline_outputs(
    export_file="vllm_baseline_output.pkl", prompts=PROMPTS, structured_generation=None
):
    """
    Helper function that starts async vLLM engine and generates output for each
    prompt in `prompts`. Saves resulted baselines in `vllm_baseline_output.pkl`
    for further use.
    """
    llm_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**VLLM_ENGINE_CONFIG))
    python_vllm_output = []
    for i in range(len(prompts)):
        output = await generate_python_vllm_output(
            prompts[i], llm_engine, structured_generation=structured_generation
        )
        if output:
            python_vllm_output.extend(output)

    with open(export_file, "wb") as f:
        pickle.dump(python_vllm_output, f)

    return


class VLLMTritonAccuracyTest(TestResultCollector):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        self.vllm_model_name = "vllm_opt"

    def test_vllm_model(self):
        # Reading and verifying baseline data
        self.python_vllm_output = []
        with open("vllm_baseline_output.pkl", "rb") as f:
            self.python_vllm_output = pickle.load(f)

        self.assertNotEqual(
            self.python_vllm_output,
            [],
            "Loaded baseline outputs' list should not be empty",
        )
        self.assertIsNotNone(
            self.python_vllm_output, "Loaded baseline outputs' list should not be None"
        )
        self.assertEqual(
            len(self.python_vllm_output),
            len(PROMPTS),
            "Unexpected number of baseline outputs loaded, expected {}, but got {}".format(
                len(PROMPTS), len(self.python_vllm_output)
            ),
        )

        user_data = UserData()
        stream = False
        triton_vllm_output = []
        self.triton_client.start_stream(callback=partial(callback, user_data))
        for i in range(len(PROMPTS)):
            request_data = create_vllm_request(
                PROMPTS[i], i, stream, SAMPLING_PARAMETERS, self.vllm_model_name
            )
            self.triton_client.async_stream_infer(
                model_name=self.vllm_model_name,
                request_id=request_data["request_id"],
                inputs=request_data["inputs"],
                outputs=request_data["outputs"],
                parameters=request_data["parameters"],
            )

        for i in range(len(PROMPTS)):
            result = user_data._completed_requests.get()
            self.assertIsNot(type(result), InferenceServerException, str(result))

            output = result.as_numpy("text_output")
            self.assertIsNotNone(output, "`text_output` should not be None")

            triton_vllm_output.extend(output)

        self.triton_client.stop_stream()
        self.assertEqual(self.python_vllm_output.sort(), triton_vllm_output.sort())

    def test_structured_outputs(self):
        # Reading and verifying baseline data
        self.python_vllm_output = []
        with open("vllm_structured_baseline_output.pkl", "rb") as f:
            self.python_vllm_output = pickle.load(f)

        self.assertNotEqual(
            self.python_vllm_output,
            [],
            "Loaded baseline outputs' list should not be empty",
        )
        self.assertIsNotNone(
            self.python_vllm_output, "Loaded baseline outputs' list should not be None"
        )
        self.assertEqual(
            len(self.python_vllm_output),
            len(STRUCTURED_PROMPTS),
            "Unexpected number of baseline outputs loaded, expected {}, but got {}".format(
                len(STRUCTURED_PROMPTS), len(self.python_vllm_output)
            ),
        )

        user_data = UserData()
        stream = False
        triton_vllm_output = []

        self.triton_client.start_stream(callback=partial(callback, user_data))
        sampling_params = SAMPLING_PARAMETERS
        structured_outputs_params = {
            "choice": ["Positive", "Negative"],
        }
        sampling_params["structured_outputs"] = json.dumps(structured_outputs_params)
        for i in range(len(STRUCTURED_PROMPTS)):
            request_data = create_vllm_request(
                STRUCTURED_PROMPTS[i], i, stream, sampling_params, self.vllm_model_name
            )
            self.triton_client.async_stream_infer(
                model_name=self.vllm_model_name,
                request_id=request_data["request_id"],
                inputs=request_data["inputs"],
                outputs=request_data["outputs"],
                parameters=request_data["parameters"],
            )

        for i in range(len(STRUCTURED_PROMPTS)):
            result = user_data._completed_requests.get()
            self.assertIsNot(type(result), InferenceServerException, str(result))

            output = result.as_numpy("text_output")
            self.assertIsNotNone(output, "`text_output` should not be None")

            triton_vllm_output.extend(output)

        self.triton_client.stop_stream()
        self.assertEqual(self.python_vllm_output.sort(), triton_vllm_output.sort())

    def tearDown(self):
        self.triton_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        required=False,
        default=False,
        help="Generates baseline output for accuracy tests",
    )
    parser.add_argument(
        "--generate-structured-baseline",
        action="store_true",
        required=False,
        default=False,
        help="Generates baseline output for accuracy tests",
    )
    FLAGS = parser.parse_args()
    if FLAGS.generate_baseline:
        asyncio.run(prepare_vllm_baseline_outputs())
        exit(0)

    if FLAGS.generate_structured_baseline:
        structured_outputs_params = {
            "choice": ["Positive", "Negative"],
        }
        structured_generation = StructuredOutputsParams(**structured_outputs_params)
        asyncio.run(
            prepare_vllm_baseline_outputs(
                export_file="vllm_structured_baseline_output.pkl",
                prompts=STRUCTURED_PROMPTS,
                structured_generation=structured_generation,
            )
        )
        exit(0)

    unittest.main()
