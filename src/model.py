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

import gc
import json
import multiprocessing
import os
import pickle
import threading
import zmq

import numpy as np
import torch

import triton_python_backend_utils as pb_utils

from vllm.usage.usage_lib import UsageContext
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.multiprocessing import (
    RPCStartupRequest, RPCGenerateRequest,
    IPC_DATA_EXT, IPC_INPUT_EXT, IPC_OUTPUT_EXT,)
from vllm.usage.usage_lib import UsageContext

from vllm.sampling_params import SamplingParams, RequestOutputKind
from vllm.utils import random_uuid, get_open_zmq_ipc_path

# from utils.metrics import VllmStatLogger

_VLLM_ENGINE_ARGS_FILENAME = "model.json"

class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]},
            {
                "name": "stream",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "sampling_parameters",
                "data_type": "TYPE_STRING",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "exclude_input_in_output",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
        ]
        outputs = [{"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]}]

        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        # Add only missing inputs and output to the model configuration.
        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

        # We need to use decoupled transaction policy for saturating
        # vLLM engine for max throughtput.
        # TODO [DLIS:5233]: Allow asynchronous execution to lift this
        # restriction for cases there is exactly a single response to
        # a single request.
        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=True))

        # Disabling batching in Triton, let vLLM handle the batching on its own.
        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config

    def initialize(self, args):
        self.args = args
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "text_output"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        # Map: [ request_id -> sender ]
        self._sender_map = {}

        # IPC path
        ipc_path = get_open_zmq_ipc_path()

        # Make MQLLMEngine
        engine_args = self.make_and_validate_engine_args()
        self.engine_process = self.make_engine_process(engine_args, ipc_path)

        self.context = zmq.Context()
        
        socket = self.context.socket(zmq.constants.DEALER)
        socket.connect(f"{ipc_path}{IPC_DATA_EXT}")
        self.wait_for_engine(socket)
        
        # Send input to MQLLMEngine.
        self.input_socket = self.context.socket(zmq.constants.PUSH)
        self.input_socket.connect(f"{ipc_path}{IPC_INPUT_EXT}")

        # Get output from MQLLMEngine.
        self.output_socket = self.context.socket(zmq.constants.PULL)
        self.output_socket.connect(f"{ipc_path}{IPC_OUTPUT_EXT}") 

        # Thread for sending responses to client.
        self._output_thread = threading.Thread(target=self.output_loop)
        self._output_thread.start()       

    
    @staticmethod
    def wait_for_engine(socket):
        while True:
            try:
                # Wait for server to be ready.
                TritonPythonModel.send_rpc_request(
                    RPCStartupRequest.IS_SERVER_READY, socket)
                break
            except TimeoutError:
                print("RPC server timeout ... retrying")
        
        # Notify that client is ready.
        TritonPythonModel.send_rpc_request(
            RPCStartupRequest.CLIENT_IS_READY, socket, await_reply=False)

    @staticmethod
    def send_rpc_request(request, socket, await_reply=True):
        socket.send_multipart((pickle.dumps(request), ), copy=False)

        if await_reply:
            if socket.poll(timeout=10000.) == 0:
                raise TimeoutError
            frame = socket.recv(copy=False)
            data = pickle.loads(frame.buffer)
            if isinstance(data, BaseException):
                raise data

    def output_loop(self):
        while True:
            while self.output_socket.poll(5000.) == 0:
                print("Waiting for output")
            
            # Output of last engine step.
            message = self.output_socket.recv(copy=False)
            request_outputs = pickle.loads(message.buffer)

            # Loop through outputs, streaming responses back.
            for request_output in request_outputs:
                # Get sender.
                sender = self._sender_map[request_output.request_id]

                # Cleanup if finsihed.
                flag = 0
                if request_output.finished:
                    flag = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    self._sender_map.pop(request_output.request_id)

                # Send response back.
                stream_response = self.create_stream_response(request_output,
                                                              self.output_dtype)
                sender.send(stream_response, flag)


    def make_engine_process(self, engine_args: AsyncEngineArgs, ipc_path: str):
        context = multiprocessing.get_context("spawn")
        engine_process = context.Process(
            target=run_mp_engine,
            args=(engine_args, UsageContext.UNKNOWN_CONTEXT, ipc_path))
        engine_process.start()
    
        return engine_process        

    def make_and_validate_engine_args(self) -> AsyncEngineArgs:
        # Currently, Triton needs to use decoupled policy for asynchronously
        # forwarding requests to vLLM engine, so assert it.
        self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )
        assert (
            self.using_decoupled
        ), "vLLM Triton backend must be configured to use decoupled model transaction policy"

        engine_args_filepath = os.path.join(
            pb_utils.get_model_dir(), _VLLM_ENGINE_ARGS_FILENAME
        )
        assert os.path.isfile(
            engine_args_filepath
        ), f"'{_VLLM_ENGINE_ARGS_FILENAME}' containing vllm engine args must be provided in '{pb_utils.get_model_dir()}'"
        with open(engine_args_filepath) as file:
            self.vllm_engine_config = json.load(file)

        # Validate device and multi-processing settings are currently set based on model/configs.
        self.validate_device_config()

        # Make engine args from JSON.
        return AsyncEngineArgs(**self.vllm_engine_config)

    def validate_device_config(self):
        triton_kind = self.args["model_instance_kind"]
        triton_device_id = int(self.args["model_instance_device_id"])
        triton_instance = f"{self.args['model_name']}_{triton_device_id}"

        # Triton's current definition of KIND_GPU makes assumptions that
        # models only use a single GPU. For multi-GPU models, the recommendation
        # is to specify KIND_MODEL to acknowledge that the model will take control
        # of the devices made available to it.
        # NOTE: Consider other parameters that would indicate multi-GPU in the future.
        tp_size = int(self.vllm_engine_config.get("tensor_parallel_size", 1))
        if tp_size > 1 and triton_kind == "GPU":
            raise ValueError(
                "KIND_GPU is currently for single-GPU models, please specify KIND_MODEL "
                "in the model's config.pbtxt for multi-GPU models"
            )

        # If KIND_GPU is specified, specify the device ID assigned by Triton to ensure that
        # multiple model instances do not oversubscribe the same default device.
        if triton_kind == "GPU" and triton_device_id >= 0:
            self.logger.log_info(
                f"Detected KIND_GPU model instance, explicitly setting GPU device={triton_device_id} for {triton_instance}"
            )
            # vLLM doesn't currently (v0.4.2) expose device selection in the APIs
            torch.cuda.set_device(triton_device_id)


    def get_sampling_params_dict(self, params_json):
        """
        This functions parses the dictionary values into their
        expected format.
        """

        params_dict = json.loads(params_json)

        # Special parsing for the supported sampling parameters
        bool_keys = ["ignore_eos", "skip_special_tokens", "use_beam_search"]
        for k in bool_keys:
            if k in params_dict:
                params_dict[k] = bool(params_dict[k])

        float_keys = [
            "frequency_penalty",
            "length_penalty",
            "presence_penalty",
            "temperature",
            "top_p",
        ]
        for k in float_keys:
            if k in params_dict:
                params_dict[k] = float(params_dict[k])

        int_keys = ["best_of", "max_tokens", "min_tokens", "n", "top_k"]
        for k in int_keys:
            if k in params_dict:
                params_dict[k] = int(params_dict[k])

        return params_dict
                
    @staticmethod
    def create_response(vllm_output, prepend_input, output_dtype):
        """
        Parses the output from the vLLM engine into Triton
        response.
        """
        prompt = ""
        if prepend_input:
            prompt = vllm_output.prompt
        text_outputs = [
            (prompt + output.text).encode("utf-8") for output in vllm_output.outputs
        ]
        triton_output_tensor = pb_utils.Tensor(
            "text_output", np.asarray(text_outputs, dtype=output_dtype)
        )
        return pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])

    @staticmethod
    def create_stream_response(request_output, output_dtype):
        """
        Parses the output from the vLLM engine, extracts only newly generated
        text and packs it into Triton response.
        """

        # If Prompt is None, this is the first request, so send a normal response.
        if request_output.prompt is not None:
            return TritonPythonModel.create_response(request_output,
                                                     prepend_input=False,
                                                     output_dtype=output_dtype)
        
        # Otherwise, send the incremental outputs.
        text_outputs = [
            output.text.encode("utf-8") for output in request_output.outputs]
        triton_output_tensor = pb_utils.Tensor(
            "text_output", np.asarray(text_outputs, dtype=output_dtype))
        return pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])


    def add_request(self, request):
        """Make and send an RPCGenerateRequest to MQLLMEngine."""

        # Make request.
        request_id = random_uuid()
        self._sender_map[request_id] = request.get_response_sender()

        prompt = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        stream = pb_utils.get_input_tensor_by_name(request, "stream")
        if stream:
            stream = stream.as_numpy()[0]
        else:
            raise NotImplementedError
        prepend_input = pb_utils.get_input_tensor_by_name(request, "exclude_input_in_output")
        if prepend_input:
            # When `exclude_input_in_output` is False, we want to prepend
            # input prompt to output, thus prepend_input should be True,
            # and vice versa.
            prepend_input = not prepend_input.as_numpy()[0]
        elif prepend_input is None and stream:
            prepend_input = False
        else:
            prepend_input = True

        if prepend_input and stream:
            raise ValueError(
                "When streaming, `exclude_input_in_output` = False is not allowed."
            )

        # Request parameters are not yet supported via
        # BLS. Provide an optional mechanism to receive serialized
        # parameters as an input tensor until support is added

        parameters_input_tensor = pb_utils.get_input_tensor_by_name(
            request, "sampling_parameters"
        )
        if parameters_input_tensor:
            parameters = parameters_input_tensor.as_numpy()[0].decode("utf-8")
        else:
            parameters = request.parameters()

        sampling_params_dict = self.get_sampling_params_dict(parameters)
        sampling_params = SamplingParams(**sampling_params_dict,
                                         output_kind=RequestOutputKind.DELTA)

        # Send request to MQLLMEngine.
        request_bytes = pickle.dumps(
            RPCGenerateRequest(inputs=prompt,
                               sampling_params=sampling_params,
                               request_id=request_id))
        self.input_socket.send_multipart((request_bytes,), copy=False)        

    def execute(self, requests):
        for request in requests:
            if request is not None:
                self.add_request(request)

    def finalize(self):
        """
        Triton virtual method; called when the model is unloaded.
        """
        self.logger.log_info("[vllm] Issuing finalize to vllm backend")

        self.engine_process.kill()
        self.engine_process.join()

        self.context.destroy(linger=0)

        # Shutdown the event thread.
        # if self._engine_thread is not None:
        #     self._engine_thread.join()
        #     self._engine_thread = None

        # When using parallel tensors, the stub process may not shutdown due to
        # unreleased references, so manually run the garbage collector once.
        self.logger.log_info("[vllm] Running Garbage Collector on finalize...")
        gc.collect()
        self.logger.log_info("[vllm] Garbage Collector on finalize... done")
