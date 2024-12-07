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

import asyncio
import base64
import gc
import json
import os
import queue
import threading
from io import BytesIO
from typing import Dict, List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from utils.metrics import VllmStatLogger

_VLLM_ENGINE_ARGS_FILENAME = "model.json"
_MULTI_LORA_ARGS_FILENAME = "multi_lora.json"


class TritonPythonModel:
    @classmethod
    def auto_complete_config(cls, auto_complete_model_config):
        # Add inputs/outputs to the model config.
        cls._auto_complete_inputs_and_outputs(auto_complete_model_config)

        # We need to use decoupled transaction policy for saturating
        # vLLM engine for max throughtput.
        # TODO [DLIS:5233]: Allow asynchronous execution to lift this
        # restriction for cases there is exactly a single response to
        # a single request.
        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=True))

        # Disabling batching in Triton, let vLLM handle the batching on its own.
        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config

    @staticmethod
    def _auto_complete_inputs_and_outputs(auto_complete_model_config):
        # Inputs expected by the backend.
        inputs = [
            {"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]},
            {
                "name": "image",
                "data_type": "TYPE_STRING",
                "dims": [-1],  # can be multiple images as separate elements
                "optional": True,
            },
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
            {
                "name": "return_finish_reason",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_cumulative_logprob",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_logprobs",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_num_input_tokens",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_num_output_tokens",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
        ]
        # Outputs expected by the backend.
        outputs = [
            {"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "finish_reason", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "cumulative_logprob", "data_type": "TYPE_FP32", "dims": [-1]},
            {"name": "logprobs", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "num_input_tokens", "data_type": "TYPE_UINT32", "dims": [1]},
            {"name": "num_output_tokens", "data_type": "TYPE_UINT32", "dims": [-1]},
        ]

        # Collect input and output names from the provided model config.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        # Add missing inputs and outputs to the model config.
        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

    def initialize(self, args):
        self.args = args
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "text_output"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        # Setup vLLM engine health check
        self._enable_health_check = self._get_bool_config_param(
            "ENABLE_VLLM_HEALTH_CHECK"
        )
        self._is_healthy = True

        # Initialize engine arguments
        # TODO: Move this into _init_engine(), after moving check metrics enabled.
        self._init_engine_args()

        # Check if metrics are enabled. The ZMQ process cannot be used when metrics are
        # enabled.
        # TODO: Move the check into _setup_metrics().
        self._enable_metrics = (
            self._get_bool_config_param("REPORT_CUSTOM_METRICS")
            and not self._aync_engine_args.disable_log_stats
        )

        # Starting the vLLM engine and its event thread running the AsyncIO event loop.
        self._init_engine()

        # Setup vLLM metrics
        self._setup_metrics()

        # Starting the response thread. It allows vLLM to keep making progress while
        # response sender(s) are sending responses to server frontend.
        self._response_queue = queue.Queue()
        self._response_thread = threading.Thread(target=self._response_loop)
        self._response_thread.start()

    def _init_engine_args(self):
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
        self._validate_device_config()

        # Check for LoRA config and set it up if enabled
        self._setup_lora()

        # Create an AsyncEngineArgs from the config from JSON
        self._aync_engine_args = AsyncEngineArgs(**self.vllm_engine_config)

    def _init_engine(self):
        # Run the engine in a separate thread running the AsyncIO event loop.
        self._llm_engine = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(
            target=asyncio.run, args=(self._run_llm_engine(),)
        )
        self._event_thread.start()
        with self._llm_engine_start_cv:
            while self._llm_engine is None:
                self._llm_engine_start_cv.wait()

        # The 'threading.Thread()' will not raise the exception here should the engine
        # failed to start, so the exception is passed back via the engine variable.
        if isinstance(self._llm_engine, Exception):
            e = self._llm_engine
            self.logger.log_error(f"[vllm] Failed to start engine: {e}")
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e

    async def _run_llm_engine(self):
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        try:
            # Start the vLLM engine. The engine lives for the scope of this with
            # statement.
            # TODO: Metrics should work with ZMQ enabled.
            async with build_async_engine_client_from_engine_args(
                engine_args=self._aync_engine_args,
                disable_frontend_multiprocessing=self._enable_metrics,
            ) as engine:
                # Capture the engine event loop and make it visible to other threads.
                self._event_loop = asyncio.get_running_loop()

                # Signal the engine is started and make it visible to other threads.
                with self._llm_engine_start_cv:
                    self._llm_engine = engine
                    self._llm_engine_start_cv.notify_all()

                # Wait for the engine shutdown signal.
                await self._llm_engine_shutdown_event.wait()

                # Wait for the ongoing requests to complete.
                while self._ongoing_request_count > 0:
                    self.logger.log_info(
                        "[vllm] Awaiting remaining {} requests".format(
                            self._ongoing_request_count
                        )
                    )
                    await asyncio.sleep(1)

                # Cancel all tasks in the event loop.
                for task in asyncio.all_tasks(loop=self._event_loop):
                    if task is not asyncio.current_task():
                        task.cancel()
        except Exception as e:
            # Signal and pass the exception back via the engine variable if the engine
            # failed to start. If the engine has started, re-raise the exception.
            with self._llm_engine_start_cv:
                if self._llm_engine is None:
                    self._llm_engine = e
                    self._llm_engine_start_cv.notify_all()
                    return
            raise e

        self._llm_engine = None
        self.logger.log_info("[vllm] Shutdown complete")

    def _validate_device_config(self):
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

    def _setup_lora(self):
        self.enable_lora = False

        # Check if `enable_lora` field is in the `model.json`,
        # and if it is, read its contents, which can be string or bool.
        if (
            "enable_lora" in self.vllm_engine_config.keys()
            and str(self.vllm_engine_config["enable_lora"]).lower() == "true"
        ):
            # create Triton LoRA weights repository
            multi_lora_args_filepath = os.path.join(
                pb_utils.get_model_dir(), _MULTI_LORA_ARGS_FILENAME
            )
            try:
                with open(multi_lora_args_filepath) as lora_file:
                    lora_repository: Dict[str, str] = json.load(lora_file)
                self.lora_repository = lora_repository
                self.supported_loras: List[str] = list(self.lora_repository.keys())
                self.supported_loras_len = len(self.supported_loras)
                self.enable_lora = True
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Triton backend cannot find {multi_lora_args_filepath}."
                )

    def _setup_metrics(self):
        self._vllm_metrics = None
        # TODO: Do not read metrics directly from the vLLM engine, read from prometheus
        #       client to allow the use of ZMQ process when metrics are enabled. See
        #       https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/entrypoints/openai/api_server.py#L222-L245
        if self._enable_metrics:
            try:
                labels = {
                    "model": self.args["model_name"],
                    "version": self.args["model_version"],
                }
                # Add vLLM custom metrics
                engine_config = self._llm_engine.engine.model_config
                self._vllm_metrics = VllmStatLogger(
                    labels, engine_config.max_model_len, self.logger
                )
                self._llm_engine.add_logger("triton", self._vllm_metrics)
            except pb_utils.TritonModelException as e:
                if "metrics not supported" in str(e):
                    # Metrics are disabled at the server
                    self.logger.log_info("[vllm] Metrics not supported")
                else:
                    raise e

    def _get_bool_config_param(self, param_name: str) -> bool:
        return (param_name in self.model_config["parameters"]) and (
            self.model_config["parameters"][param_name]["string_value"].lower()
            == "true"
        )

    def _response_loop(self):
        while True:
            item = self._response_queue.get()
            # To signal shutdown a None item will be added to the queue.
            if item is None:
                break
            response_state, response, response_flag = item
            response_sender = response_state["response_sender"]
            try:
                response_sender.send(response, response_flag)
                # Stop checking for cancellation if the last response is generated.
                if not response_state["last_response_generated"]:
                    response_state["is_cancelled"] = response_sender.is_cancelled()
            except Exception as e:
                self.logger.log_error(
                    f"An error occurred while sending a response: {e}"
                )
            finally:
                if response_flag == pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL:
                    self._ongoing_request_count -= 1

    def execute(self, requests):
        if self._enable_health_check and not self._check_health(requests):
            return None
        for request in requests:
            request = self._verify_loras(request)
            if request is not None:
                assert (
                    self._llm_engine_shutdown_event.is_set() is False
                ), "Cannot create tasks after shutdown has been requested"
                coro = self._generate(request)
                asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return None

    async def _generate(self, request):
        response_sender = request.get_response_sender()
        response_state = {
            "response_sender": response_sender,
            "is_cancelled": False,
            "last_response_generated": False,  # last response ready but not yet sent
        }
        self._ongoing_request_count += 1
        decrement_ongoing_request_count = True
        try:
            request_id = random_uuid()
            (
                prompt,
                stream,
                prepend_input,
                parameters,
                additional_outputs,
            ) = self._get_input_tensors(request)

            sampling_params_dict = self._get_sampling_params_dict(parameters)
            lora_name = sampling_params_dict.pop("lora_name", None)
            sampling_params = SamplingParams(**sampling_params_dict)
            lora_request = None
            if lora_name is not None:
                lora_id = str(self.supported_loras.index(lora_name) + 1)
                lora_int_id = int(lora_id)
                lora_local_path = self.lora_repository[lora_name]
                lora_request = LoRARequest(lora_id, lora_int_id, lora_local_path)

            response_iterator = self._llm_engine.generate(
                prompt, sampling_params, request_id, lora_request=lora_request
            )

            request_output_state = {}
            async for request_output in response_iterator:
                # Cancellation state will be checked by the response loop and written to
                # the response state if streaming. If not streaming, cancellation state
                # needs to be checked here.
                is_cancelled = response_state["is_cancelled"]
                if not stream:
                    is_cancelled = response_sender.is_cancelled()
                if is_cancelled:
                    self.logger.log_info("[vllm] Cancelling the request")
                    await self._llm_engine.abort(request_id)
                    self.logger.log_info("[vllm] Successfully cancelled the request")

                    if stream:
                        # Add cancelled final response to response loop.
                        response_state["last_response_generated"] = True
                        response = pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(
                                message="Request was cancelled",
                                code=pb_utils.TritonError.CANCELLED,
                            )
                        )
                        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        decrement_ongoing_request_count = False
                        self._response_queue.put_nowait(
                            (response_state, response, flags)
                        )

                    break

                # Send each response if streaming.
                if stream:
                    response = self._create_response(
                        request_output_state,
                        request_output,
                        prepend_input=False,
                        additional_outputs=additional_outputs,
                    )
                    flags = 0
                    if request_output.finished:
                        response_state["last_response_generated"] = True
                        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        decrement_ongoing_request_count = False
                    self._response_queue.put_nowait((response_state, response, flags))

            # Send the last response which contains all the outputs if not streaming.
            if not stream:
                response_sender.send(
                    self._create_response(
                        request_output_state={},
                        request_output=request_output,
                        prepend_input=prepend_input,
                        additional_outputs=additional_outputs,
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

        except Exception as e:
            self.logger.log_error(f"[vllm] Error generating stream: {e}")
            error = pb_utils.TritonError(f"Error generating stream: {e}")
            text_output_tensor = pb_utils.Tensor(
                "text_output", np.asarray(["N/A"], dtype=self.output_dtype)
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[text_output_tensor], error=error
            )
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
            raise e

        finally:
            if decrement_ongoing_request_count:
                self._ongoing_request_count -= 1

    def _get_input_tensors(self, request):
        # prompt
        prompt = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()[0]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        # image
        images = pb_utils.get_input_tensor_by_name(request, "image")
        if images:
            images_vllm = []
            for image_np in images.as_numpy():
                image_b = base64.b64decode(image_np.decode("utf-8"))
                image_rgb = Image.open(BytesIO(image_b)).convert("RGB")
                images_vllm.append(image_rgb)
            if len(images_vllm) > 0:
                prompt = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": images_vllm},
                }

        # stream
        stream = pb_utils.get_input_tensor_by_name(request, "stream")
        if stream:
            stream = stream.as_numpy()[0]
        else:
            stream = False

        # prepend_input / exclude_input_in_output
        prepend_input = pb_utils.get_input_tensor_by_name(
            request, "exclude_input_in_output"
        )
        if prepend_input:
            # When `exclude_input_in_output` is False, we want to prepend input prompt
            # to output, thus prepend_input should be True, and vice versa.
            prepend_input = not prepend_input.as_numpy()[0]
        elif prepend_input is None and stream:
            prepend_input = False
        else:
            prepend_input = True
        if prepend_input and stream:
            raise ValueError(
                "When streaming, `exclude_input_in_output` = False is not allowed."
            )

        # parameters / sampling_parameters
        # An alternative mechanism to receive serialized parameters as an input tensor,
        # because request parameters are not yet supported via BLS.
        sampling_parameters = pb_utils.get_input_tensor_by_name(
            request, "sampling_parameters"
        )
        if sampling_parameters:
            parameters = sampling_parameters.as_numpy()[0].decode("utf-8")
        else:
            parameters = request.parameters()

        # additional outputs
        additional_outputs = {
            "return_finish_reason": None,
            "return_cumulative_logprob": None,
            "return_logprobs": None,
            "return_num_input_tokens": None,
            "return_num_output_tokens": None,
        }
        for tensor_name in additional_outputs.keys():
            tensor = pb_utils.get_input_tensor_by_name(request, tensor_name)
            if tensor:
                tensor = bool(tensor.as_numpy()[0])
            else:
                tensor = False
            additional_outputs[tensor_name] = tensor

        return prompt, stream, prepend_input, parameters, additional_outputs

    def _create_response(
        self, request_output_state, request_output, prepend_input, additional_outputs
    ):
        output_tensors = []

        # text_output
        prepend_prompt = ""
        if "prev_lens_text_output" not in request_output_state:
            # this is the first response
            if prepend_input:
                prepend_prompt = request_output.prompt
            request_output_state["prev_lens_text_output"] = [0] * len(
                request_output.outputs
            )
        prev_lens = request_output_state["prev_lens_text_output"]
        text_output = [
            (prepend_prompt + output.text[prev_len:]).encode("utf-8")
            for output, prev_len in zip(request_output.outputs, prev_lens)
        ]
        request_output_state["prev_lens_text_output"] = [
            len(output.text) for output in request_output.outputs
        ]
        output_tensors.append(
            pb_utils.Tensor(
                "text_output", np.asarray(text_output, dtype=self.output_dtype)
            )
        )

        # finish_reason
        if additional_outputs["return_finish_reason"]:
            finish_reason = [
                str(output.finish_reason) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "finish_reason", np.asarray(finish_reason, dtype=np.object_)
                )
            )

        # cumulative_logprob
        if additional_outputs["return_cumulative_logprob"]:
            cumulative_logprob = [
                output.cumulative_logprob for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "cumulative_logprob",
                    np.asarray(cumulative_logprob, dtype=np.float32),
                )
            )

        # logprobs
        # https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/sequence.py#L37-L58
        if additional_outputs["return_logprobs"]:
            if "prev_lens_logprobs" not in request_output_state:
                request_output_state["prev_lens_logprobs"] = [0] * len(
                    request_output.outputs
                )
            logprobs = []
            for i in range(len(request_output.outputs)):
                output = request_output.outputs[i]
                if output.logprobs is None:
                    logprobs.append("null".encode("utf-8"))
                    continue
                prev_len = request_output_state["prev_lens_logprobs"][i]
                request_output_state["prev_lens_logprobs"][i] = len(output.logprobs)
                logprobs_py = []
                for logprob_d_vllm in output.logprobs[prev_len:]:
                    logprob_d_py = {}
                    for token_id, logprob_vllm in logprob_d_vllm.items():
                        logprob_d_py[token_id] = {
                            "logprob": logprob_vllm.logprob,
                            "rank": logprob_vllm.rank,
                            "decoded_token": logprob_vllm.decoded_token,
                        }
                    logprobs_py.append(logprob_d_py)
                logprobs.append(json.dumps(logprobs_py).encode("utf-8"))
            output_tensors.append(
                pb_utils.Tensor("logprobs", np.asarray(logprobs, dtype=np.object_))
            )

        # num_input_tokens
        if additional_outputs["return_num_input_tokens"]:
            num_input_tokens = len(request_output.prompt_token_ids)
            output_tensors.append(
                pb_utils.Tensor(
                    "num_input_tokens", np.asarray(num_input_tokens, dtype=np.uint32)
                )
            )

        # num_output_tokens
        if additional_outputs["return_num_output_tokens"]:
            if "prev_lens_num_output_tokens" not in request_output_state:
                request_output_state["prev_lens_num_output_tokens"] = [0] * len(
                    request_output.outputs
                )
            prev_lens = request_output_state["prev_lens_num_output_tokens"]
            num_output_tokens = [
                (len(output.token_ids) - prev_len)
                for output, prev_len in zip(request_output.outputs, prev_lens)
            ]
            request_output_state["prev_lens_num_output_tokens"] = [
                len(output.token_ids) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "num_output_tokens", np.asarray(num_output_tokens, dtype=np.uint32)
                )
            )

        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def _get_sampling_params_dict(self, params_json):
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

    def _verify_loras(self, request):
        # We will check if the requested lora exists here, if not we will send a
        # response with `LoRA not found` information. In this way we may avoid
        # further processing.
        verified_request = None
        lora_error = None
        lora_name = None
        parameters_input_tensor = pb_utils.get_input_tensor_by_name(
            request, "sampling_parameters"
        )
        if parameters_input_tensor:
            parameters = parameters_input_tensor.as_numpy()[0].decode("utf-8")
            sampling_params_dict = self._get_sampling_params_dict(parameters)
            lora_name = sampling_params_dict.pop("lora_name", None)

        if lora_name is not None:
            if not self.enable_lora:
                lora_error = pb_utils.TritonError("LoRA feature is not enabled.")
                self.logger.log_info(
                    "[vllm] LoRA is not enabled, please restart the backend with LoRA enabled."
                )
            elif lora_name not in self.supported_loras:
                lora_error = pb_utils.TritonError(
                    f"LoRA {lora_name} is not supported, we currently support {self.supported_loras}"
                )
                self.logger.log_info(f"[vllm] LoRA {lora_name} not found.")

        if lora_error is not None:
            output_tensor = pb_utils.Tensor(
                "text_output",
                np.asarray(["[Error] Unsupported LoRA."], dtype=self.output_dtype),
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor], error=lora_error
            )
            response_sender = request.get_response_sender()
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
        else:
            verified_request = request
        return verified_request

    def _check_health(self, requests):
        coro = self._llm_engine.check_health()
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        try:
            future.result()
        except Exception as e:
            self.logger.log_error(
                f"[vllm] Engine is not healthy and model will be unloaded: {e}"
            )
            pb_utils.unload_model(self.model_config["name"])  # non-blocking
            self._is_healthy = False
        if not self._is_healthy:
            for request in requests:
                request.get_response_sender().send(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            message="Model is unavailable due to unhealthy vLLM engine",
                            code=pb_utils.TritonError.UNAVAILABLE,
                        )
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
        return self._is_healthy

    def finalize(self):
        self.logger.log_info("[vllm] Issuing finalize to vllm backend")
        self._llm_engine_shutdown_event.set()

        # Shutdown the event thread.
        if self._event_thread is not None:
            self._event_thread.join()
            self._event_thread = None

        # Shutdown the response thread.
        self._response_queue.put(None)
        if self._response_thread is not None:
            self._response_thread.join()
            self._response_thread = None

        # Shutdown the metrics thread.
        if self._vllm_metrics is not None:
            self._vllm_metrics.finalize()

        # When using parallel tensors, the stub process may not shutdown due to
        # unreleased references, so manually run the garbage collector once.
        self.logger.log_info("[vllm] Running Garbage Collector on finalize...")
        gc.collect()
        self.logger.log_info("[vllm] Garbage Collector on finalize... done")
