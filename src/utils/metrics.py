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

from typing import Dict, Union

import triton_python_backend_utils as pb_utils
from vllm.engine.metrics import StatLoggerBase as VllmStatLoggerBase
from vllm.engine.metrics import Stats as VllmStats
from vllm.engine.metrics import SupportsMetricsInfo


class TritonMetrics:
    def __init__(self, labels):
        # Initialize metric families
        # Iteration stats
        self.counter_prompt_tokens_family = pb_utils.MetricFamily(
            name="vllm:prompt_tokens_total",
            description="Number of prefill tokens processed.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.counter_generation_tokens_family = pb_utils.MetricFamily(
            name="vllm:generation_tokens_total",
            description="Number of generation tokens processed.",
            kind=pb_utils.MetricFamily.COUNTER,
        )

        # Initialize metrics
        # Iteration stats
        self.counter_prompt_tokens = self.counter_prompt_tokens_family.Metric(
            labels=labels
        )
        self.counter_generation_tokens = self.counter_generation_tokens_family.Metric(
            labels=labels
        )


class VllmStatLogger(VllmStatLoggerBase):
    """StatLogger is used as an adapter between vLLM stats collector and Triton metrics provider."""

    # local_interval not used here. It's for vLLM logs to stdout.
    def __init__(self, labels: Dict, local_interval: float = 0) -> None:
        # Tracked stats over current local logging interval.
        super().__init__(local_interval)
        self.metrics = TritonMetrics(labels=labels)

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        """Convenience function for logging to gauge.

        Args:
            gauge: A gauge metric instance.
            data: An int or float to set the gauge metric.

        Returns:
            None
        """
        gauge.set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        """Convenience function for logging to counter.

        Args:
            counter: A counter metric instance.
            data: An int or float to increment the count metric.

        Returns:
            None
        """
        if data != 0:
            counter.increment(data)

    def log(self, stats: VllmStats) -> None:
        """Logs tracked stats to triton metrics server every iteration.

        Args:
            stats: Created by LLMEngine for use by VllmStatLogger.

        Returns:
            None
        """
        self._log_counter(
            self.metrics.counter_prompt_tokens, stats.num_prompt_tokens_iter
        )
        self._log_counter(
            self.metrics.counter_generation_tokens, stats.num_generation_tokens_iter
        )
