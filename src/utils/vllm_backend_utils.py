# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional, Union

try:
    from interegular.patterns import parse_pattern
except ImportError:
    parse_pattern = None

from vllm.sampling_params import SamplingParams, StructuredOutputsParams


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
            
            # Remove None values to let vLLM use defaults
            params_dict = {k: v for k, v in params_dict.items() if v is not None}

            for key, value in list(params_dict.items()):
                if key == "structured_outputs":
                    params_dict[key] = StructuredOutputsParams(**json.loads(value))
                elif key == "guided_decoding":
                    if isinstance(value, str):
                        value = json.loads(value)
                    
                    # Map guided_decoding to structured_outputs
                    # Remove backend if present as it is not supported in StructuredOutputsParams constructor
                    if "backend" in value:
                        backend = value.pop("backend")
                        if backend not in ["xgrammar", "xgrammar:no-fallback", "auto", "xgrammar:_auto"]:
                             raise ValueError(f"guided_decoding.backend is no longer supported request-level. Provided: {backend}")
                    
                    # If structured_outputs is not already set, use guided_decoding params
                    if "structured_outputs" not in params_dict:
                        params_dict["structured_outputs"] = StructuredOutputsParams(**value)
                    
            if "guided_decoding" in params_dict:
                del params_dict["guided_decoding"]

            for key, value in params_dict.items():
                if key in vllm_params_dict:
                    vllm_type = vllm_params_dict[key]
                    if vllm_type in type_mapping:
                        params_dict[key] = type_mapping[vllm_type](params_dict[key])

            return TritonSamplingParams(**params_dict)

        except Exception as e:
            logger.log_error(
                f"[vllm] Was trying to create `TritonSamplingParams`, but got exception: {e}"
            )
            return None

    def __post_init__(self):
        super().__post_init__()

        # Validate the structured outputs parameters.
        if self.structured_outputs:
            if not isinstance(self.structured_outputs, StructuredOutputsParams):
                raise ValueError(
                    "structured_outputs must be of type StructuredOutputsParams"
                )
            TritonSamplingParams._validate_guided_params(self.structured_outputs)

    @staticmethod
    def _validate_guided_params(params: StructuredOutputsParams):
        """
        Validates the structured outputs parameters.
        Raises an exception if the parameters are invalid.
        """
        if not params:
            return

        if not isinstance(params, StructuredOutputsParams):
            raise ValueError("structured_outputs must be of type StructuredOutputsParams")

        # Validate regex constraint if provided.
        if params.regex:
            if not isinstance(params.regex, str):
                raise ValueError("structured_outputs.regex must be a string")
            if parse_pattern:
                try:
                    parse_pattern(params.regex)
                except Exception as e:
                    raise ValueError(f"Invalid regex constraint: {e}") from e
            
        # backend validation is removed as it is not exposed in StructuredOutputsParams
        # and handled during construction/mapping.

        if params.grammar:
            if not isinstance(params.grammar, str):
                raise ValueError("grammar must be a string, describing a BNF grammar")
           
            try:
                from xgrammar import \
                    Grammar  # type: ignore[import]: do NOT move up to avoid premature CUDA init

                # Try to parse the converted grammar, to fail this request early
                Grammar.from_ebnf(params.grammar)
            except ImportError:
                pass
            except RuntimeError as e:
                raise ValueError(f"Invalid BNF grammar: {e}") from e

        # Validate choice constraint.
        if params.choice:
            if not isinstance(params.choice, list):
                raise ValueError("choice must be a list")
            for item in params.choice:
                if not isinstance(item, str):
                    raise ValueError("Each element in choice must be a string")

        # Validate JSON constraint.
        if params.json:
            if not isinstance(params.json, dict):
                raise ValueError("json must be a JSON schema dictionary")

        # Validate whitespace_pattern constraint.
        if params.whitespace_pattern:
            if not isinstance(params.whitespace_pattern, str):
                raise ValueError("whitespace_pattern must be a string")
