#!/usr/bin/env python3

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

import argparse
import asyncio
import json
import sys

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *


class LLMClient:
    def __init__(self, flags: argparse.Namespace):
        self._client = grpcclient.InferenceServerClient(
            url=flags.url, verbose=flags.verbose
        )
        self._flags = flags
        self._loop = asyncio.get_event_loop()
        self._results_dict = {}

    async def async_request_iterator(
        self, prompts, sampling_parameters, exclude_input_in_output
    ):
        try:
            for iter in range(self._flags.iterations):
                for i, prompt in enumerate(prompts):
                    prompt_id = self._flags.offset + (len(prompts) * iter) + i
                    self._results_dict[str(prompt_id)] = []
                    yield self.create_request(
                        prompt,
                        self._flags.streaming_mode,
                        prompt_id,
                        sampling_parameters,
                        exclude_input_in_output,
                    )
        except Exception as error:
            print(f"Caught an error in the request iterator: {error}")

    async def stream_infer(self, prompts, sampling_parameters, exclude_input_in_output):
        try:
            # Start streaming
            response_iterator = self._client.stream_infer(
                inputs_iterator=self.async_request_iterator(
                    prompts, sampling_parameters, exclude_input_in_output
                ),
                stream_timeout=self._flags.stream_timeout,
            )
            async for response in response_iterator:
                yield response
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    async def process_stream(
        self, prompts, sampling_parameters, exclude_input_in_output
    ):
        # Clear results in between process_stream calls
        self.results_dict = []
        success = True
        # Read response from the stream
        async for response in self.stream_infer(
            prompts, sampling_parameters, exclude_input_in_output
        ):
            result, error = response
            if error:
                print(f"Encountered error while processing: {error}")
                success = False
            else:
                output = result.as_numpy("text_output")
                for i in output:
                    self._results_dict[result.get_response().id].append(i)
        return success

    async def run(self):
        # Sampling parameters for text generation
        # including `temperature`, `top_p`, top_k`, `max_tokens`, `early_stopping`.
        # Full list available at:
        # https://github.com/vllmproject/vllm/blob/5255d99dc595f9ae7647842242d6542aa4145a4f/vllm/sampling_params.py#L23
        sampling_parameters = {
            "temperature": "0.1",
            "top_p": "0.95",
            "max_tokens": "100",
        }
        exclude_input_in_output = self._flags.exclude_inputs_in_outputs
        if self._flags.lora_name is not None:
            sampling_parameters["lora_name"] = self._flags.lora_name
        with open(self._flags.input_prompts, "r") as file:
            print(f"Loading inputs from `{self._flags.input_prompts}`...")
            prompts = file.readlines()

        success = await self.process_stream(
            prompts, sampling_parameters, exclude_input_in_output
        )

        with open(self._flags.results_file, "w") as file:
            for id in self._results_dict.keys():
                for result in self._results_dict[id]:
                    file.write(result.decode("utf-8"))

                file.write("\n")
                file.write("\n=========\n\n")
            print(f"Storing results into `{self._flags.results_file}`...")

        if self._flags.verbose:
            with open(self._flags.results_file, "r") as file:
                print(f"\nContents of `{self._flags.results_file}` ===>")
                print(file.read())
        if success:
            print("PASS: vLLM example")
        else:
            print("FAIL: vLLM example")

    def run_async(self):
        self._loop.run_until_complete(self.run())

    def create_request(
        self,
        prompt,
        stream,
        request_id,
        sampling_parameters,
        exclude_input_in_output,
        send_parameters_as_tensor=True,
    ):
        inputs = []
        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        try:
            inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as error:
            print(f"Encountered an error during request creation: {error}")

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        # Request parameters are not yet supported via BLS. Provide an
        # optional mechanism to send serialized parameters as an input
        # tensor until support is added

        if send_parameters_as_tensor:
            sampling_parameters_data = np.array(
                [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
            )
            inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_parameters_data)

        inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(np.array([exclude_input_in_output], dtype=bool))

        # Add requested outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("text_output"))

        # Issue the asynchronous sequence inference.
        return {
            "model_name": self._flags.model,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
            "parameters": sampling_parameters,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="vllm_model",
        help="Model name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL and its gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        required=False,
        default=0,
        help="Add offset to request IDs used",
    )
    parser.add_argument(
        "--input-prompts",
        type=str,
        required=False,
        default="prompts.txt",
        help="Text file with input prompts",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=False,
        default="results.txt",
        help="The file with output results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=False,
        default=1,
        help="Number of iterations through the prompts file",
    )
    parser.add_argument(
        "-s",
        "--streaming-mode",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode",
    )
    parser.add_argument(
        "--exclude-inputs-in-outputs",
        action="store_true",
        required=False,
        default=False,
        help="Exclude prompt from outputs",
    )
    parser.add_argument(
        "-l",
        "--lora-name",
        type=str,
        required=False,
        default=None,
        help="The querying LoRA name",
    )
    FLAGS = parser.parse_args()

    client = LLMClient(FLAGS)
    client.run_async()
