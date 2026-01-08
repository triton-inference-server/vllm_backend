# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.metrics.loggers import StatLoggerFactory


class TritonSamplingParams(SamplingParams):
    """
    Extended sampling parameters for text generation via
    Triton Inference Server and vLLM backend.

    Attributes:
        lora_name (Optional[str]): The name of the LoRA (Low-Rank Adaptation)
        to use for inference.
    """

    lora_name: Optional[str] = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the `TritonSamplingParams` object.

        This method overrides the `__repr__` method of the parent class
        to include additional attributes in the string representation.

        Returns:
            A string representation of the object.
        """
        base = super().__repr__()
        return f"{base}, lora_name={self.lora_name}"

    @staticmethod
    def from_dict(
        params_dict_str: str, logger: "pb_utils.Logger"
    ) -> "TritonSamplingParams":
        """
        Creates a `TritonSamplingParams` object from a dictionary string.

        This method parses a JSON string containing sampling parameters,
        converts the values to appropriate types, and creates a
        `TritonSamplingParams` object.

        Args:
            params_dict (str): A JSON string containing sampling parameters.
            logger (pb_utils.Logger): Triton Inference Server logger object.

        Returns:
            TritonSamplingParams: An instance of TritonSamplingParams.
        """
        try:
            params_dict = json.loads(params_dict_str)
            vllm_params_dict = SamplingParams.__annotations__
            type_mapping = {
                int: int,
                float: float,
                bool: bool,
                str: str,
                Optional[int]: int,
            }
            for key, value in params_dict.items():
                if key == "structured_outputs":
                    params_dict[key] = StructuredOutputsParams(**json.loads(value))
                elif key in vllm_params_dict:
                    vllm_type = vllm_params_dict[key]
                    if vllm_type in type_mapping:
                        params_dict[key] = type_mapping[vllm_type](params_dict[key])

            return TritonSamplingParams(**params_dict)

        except Exception as e:
            logger.log_error(
                f"[vllm] Was trying to create `TritonSamplingParams`, but got exception: {e}"
            )
            return None


# Copy from vllm/vllm/entrypoints/openai/api_server.py with custom stat_loggers
@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    logger: "pb_utils.Logger",
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool = False,
    stat_loggers: Optional[list[StatLoggerFactory]] = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if disable_frontend_multiprocessing:
        logger.log_warn("V1 is enabled, but got --disable-frontend-multiprocessing.")

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown()
