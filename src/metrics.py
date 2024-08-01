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

from typing import Dict, List, Union

import triton_python_backend_utils as pb_utils
from vllm.engine.metrics import StatLoggerBase as VllmStatLoggerBase
from vllm.engine.metrics import Stats as VllmStats
from vllm.engine.metrics import SupportsMetricsInfo


class TritonMetrics:
    def __init__(self, labels):
        # System stats
        #   Scheduler State
        self.gauge_scheduler_running_family = pb_utils.MetricFamily(
            name="vllm:num_requests_running",
            description="Number of requests currently running on GPU.",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.gauge_scheduler_waiting_family = pb_utils.MetricFamily(
            name="vllm:num_requests_waiting",
            description="Number of requests waiting to be processed.",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.gauge_scheduler_swapped_family = pb_utils.MetricFamily(
            name="vllm:num_requests_swapped",
            description="Number of requests swapped to CPU.",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage_family = pb_utils.MetricFamily(
            name="vllm:gpu_cache_usage_perc",
            description="GPU KV-cache usage. 1 means 100 percent usage.",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.gauge_cpu_cache_usage_family = pb_utils.MetricFamily(
            name="vllm:cpu_cache_usage_perc",
            description="CPU KV-cache usage. 1 means 100 percent usage.",
            kind=pb_utils.MetricFamily.GAUGE,
        )

        # Iteration stats
        self.counter_num_preemption_family = pb_utils.MetricFamily(
            name="vllm:num_preemptions_total",
            description="Cumulative number of preemption from the engine.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
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

        # System stats
        #   Scheduler State
        self.gauge_scheduler_running = self.gauge_scheduler_running_family.Metric(
            labels=labels
        )
        self.gauge_scheduler_waiting = self.gauge_scheduler_waiting_family.Metric(
            labels=labels
        )
        self.gauge_scheduler_swapped = self.gauge_scheduler_swapped_family.Metric(
            labels=labels
        )
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage = self.gauge_gpu_cache_usage_family.Metric(
            labels=labels
        )
        self.gauge_cpu_cache_usage = self.gauge_cpu_cache_usage_family.Metric(
            labels=labels
        )

        # Iteration stats
        self.counter_num_preemption = self.counter_num_preemption_family.Metric(
            labels=labels
        )
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
        # Convenience function for logging to gauge.
        gauge.set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.increment(data)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.observe(datum)

    def log(self, stats: VllmStats) -> None:
        # System state data
        self._log_gauge(self.metrics.gauge_scheduler_running, stats.num_running_sys)
        self._log_gauge(self.metrics.gauge_scheduler_waiting, stats.num_waiting_sys)
        self._log_gauge(self.metrics.gauge_scheduler_swapped, stats.num_swapped_sys)
        self._log_gauge(self.metrics.gauge_gpu_cache_usage, stats.gpu_cache_usage_sys)
        self._log_gauge(self.metrics.gauge_cpu_cache_usage, stats.cpu_cache_usage_sys)

        # Iteration level data
        self._log_counter(
            self.metrics.counter_num_preemption, stats.num_preemption_iter
        )
        self._log_counter(
            self.metrics.counter_prompt_tokens, stats.num_prompt_tokens_iter
        )
        self._log_counter(
            self.metrics.counter_generation_tokens, stats.num_generation_tokens_iter
        )
