# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import gc
import json
import os
import queue
import threading
import traceback
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from vllm.engine.arg_utils import AsyncEngineArgs

from utils.metrics import VllmStatLoggerFactory
from utils.request import EmbedRequest, GenerateRequest
from utils.vllm_backend_utils import build_async_engine_client_from_engine_args

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
            # TODO: Support array input
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
            # Tentative input reserved for embedding requests in OpenAI-compatible frontend. Subject to change in the future.
            # WARN: Triton client should never set this input. It is reserved for embedding requests in OpenAI-compatible frontend.
            {
                "name": "embedding_request",
                "data_type": "TYPE_STRING",
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

        # Setup vLLM metrics
        self._setup_metrics()

        # Starting the vLLM engine and its event thread running the AsyncIO event loop.
        self._init_engine()

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
            self.logger.log_error(
                f"[vllm] Failed to start engine: {traceback.format_exc()}"
            )
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e

        # Get supported tasks from the engine running in another thread
        self.supported_tasks = asyncio.run_coroutine_threadsafe(
            self._llm_engine.get_supported_tasks(), self._event_loop
        ).result()

    async def _run_llm_engine(self):
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        try:
            # Start the vLLM engine. The engine lives for the scope of this with
            # statement.
            # TODO: Metrics should work with ZMQ enabled.
            async with build_async_engine_client_from_engine_args(
                engine_args=self._aync_engine_args,
                logger=self.logger,
                disable_frontend_multiprocessing=self._enable_metrics,
                stat_loggers=self._vllm_metrics,
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
        pp_size = int(self.vllm_engine_config.get("pipeline_parallel_size", 1))
        if (tp_size * pp_size) > 1 and triton_kind == "GPU":
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
            os.environ["CUDA_VISIBLE_DEVICES"] = str(triton_device_id)

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
                self.enable_lora = True
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Triton backend cannot find {multi_lora_args_filepath}."
                )

    def _setup_metrics(self):
        self._vllm_metrics = []
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
                factory = VllmStatLoggerFactory(labels, self.logger)
                self._vllm_metrics.append(factory)
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
                    f"An error occurred while sending a response: {traceback.format_exc()}"
                )
            finally:
                if response_flag == pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL:
                    self._ongoing_request_count -= 1

    def respond_error(self, request, error_message, triton_error):
        output_tensor = pb_utils.Tensor(
            "text_output",
            np.asarray([error_message], dtype=self.output_dtype),
        )
        response = pb_utils.InferenceResponse(
            output_tensors=[output_tensor], error=triton_error
        )
        response_sender = request.get_response_sender()
        response_sender.send(
            response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
        )

    def _validate_request_task_name(self, request):
        embedding_request = pb_utils.get_input_tensor_by_name(
            request, "embedding_request"
        )
        if embedding_request is None:
            request_task_name = "generate"
        else:
            request_task_name = "embed"

        if request_task_name not in self.supported_tasks:
            raise ValueError(
                f"Model {self.args['model_name']} does not support '{request_task_name}' request"
            )

        return request_task_name

    def execute(self, requests):
        if self._enable_health_check and not self._check_health(requests):
            return None
        for request in requests:
            request = self._verify_loras(request)
            if request is not None:
                assert (
                    self._llm_engine_shutdown_event.is_set() is False
                ), "Cannot create tasks after shutdown has been requested"
                coro = self._infer(request)
                asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return None

    async def _infer(self, request):
        response_sender = request.get_response_sender()
        response_state = {
            "response_sender": response_sender,
            "is_cancelled": False,
            "last_response_generated": False,  # last response ready but not yet sent
        }
        self._ongoing_request_count += 1
        decrement_ongoing_request_count = True
        try:
            request_task_name = self._validate_request_task_name(request)
            if request_task_name == "generate":
                if self.enable_lora:
                    request = GenerateRequest(
                        request,
                        self._llm_engine.generate,
                        self.output_dtype,
                        self.logger,
                        self.lora_repository,
                        self.supported_loras,
                    )
                else:
                    request = GenerateRequest(
                        request,
                        self._llm_engine.generate,
                        self.output_dtype,
                        self.logger,
                    )
            elif request_task_name == "embed":
                request = EmbedRequest(
                    request, self._llm_engine.encode, self.output_dtype, self.logger
                )
            else:
                raise ValueError(
                    f"VLLM backend does not support '{request_task_name}' request"
                )

            response_iterator = request.execute()

            request_output_state = {}
            async for request_output in response_iterator:
                # Cancellation state will be checked by the response loop and written to
                # the response state if streaming. If not streaming, cancellation state
                # needs to be checked here.
                is_cancelled = response_state["is_cancelled"]
                if not request.stream:
                    is_cancelled = response_sender.is_cancelled()
                if is_cancelled:
                    self.logger.log_info("[vllm] Cancelling the request")
                    await self._llm_engine.abort(request.id)
                    self.logger.log_info("[vllm] Successfully cancelled the request")

                    if request.stream:
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
                if request.stream:
                    response = request.create_response(
                        request_output,
                        request_output_state,
                        prepend_input=False,
                    )
                    flags = 0
                    if request_output.finished:
                        response_state["last_response_generated"] = True
                        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        decrement_ongoing_request_count = False
                    self._response_queue.put_nowait((response_state, response, flags))

            # Send the last response which contains all the outputs if not streaming.
            if not request.stream:
                if request_task_name == "generate":
                    response = request.create_response(
                        request_output=request_output,
                        request_output_state={},
                        prepend_input=request.prepend_input,
                    )
                else:
                    response = request.create_response(
                        request_output=request_output,
                    )
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        except Exception as e:
            self.logger.log_error(
                f"[vllm] Error generating stream: {traceback.format_exc()}"
            )
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
        else:
            parameters = request.parameters()

        lora_name = json.loads(parameters).pop("lora_name", None)
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
            self.respond_error(request, lora_error.message, lora_error)
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
                f"[vllm] Engine is not healthy and model will be unloaded: {traceback.format_exc()}"
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
        self._event_loop.call_soon_threadsafe(self._llm_engine_shutdown_event.set)

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
        for stat_logger_factory in self._vllm_metrics:
            stat_logger_factory.finalize()

        # When using parallel tensors, the stub process may not shutdown due to
        # unreleased references, so manually run the garbage collector once.
        self.logger.log_info("[vllm] Running Garbage Collector on finalize...")
        gc.collect()
        self.logger.log_info("[vllm] Garbage Collector on finalize... done")
