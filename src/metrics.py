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


# begin-metrics-definitions
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
        # self.histogram_time_to_first_token_family = pb_utils.MetricFamily(
        #     name="vllm:time_to_first_token_seconds",
        #     description="Histogram of time to first token in seconds.",
        #     kind=pb_utils.MetricFamily.HISTOGRAM,
        #     buckets=[
        #         0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
        #         0.75, 1.0, 2.5, 5.0, 7.5, 10.0
        #     ])
        # self.histogram_time_per_output_token_family = pb_utils.MetricFamily(
        #     name="vllm:time_per_output_token_seconds",
        #     description="Histogram of time per output token in seconds.",
        #     kind=pb_utils.MetricFamily.HISTOGRAM,
        #     buckets=[
        #         0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
        #         1.0, 2.5
        #     ])

        # Request stats
        #   Latency
        # self.histogram_e2e_time_request_family = pb_utils.MetricFamily(
        #     name="vllm:e2e_request_latency_seconds",
        #     description="Histogram of end to end request latency in seconds.",
        #     kind=pb_utils.MetricFamily.HISTOGRAM,
        #     buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        # #   Metadata
        # self.histogram_num_prompt_tokens_request_family = pb_utils.MetricFamily(
        #     name="vllm:request_prompt_tokens",
        #     description="Number of prefill tokens processed.",
        #     kind=pb_utils.MetricFamily.HISTOGRAM,
        #     buckets=build_1_2_5_buckets(max_model_len),
        # )
        # self.histogram_num_generation_tokens_request_family = \
        #     pb_utils.MetricFamily(
        #         name="vllm:request_generation_tokens",
        #         description="Number of generation tokens processed.",
        #         kind=pb_utils.MetricFamily.HISTOGRAM,
        #         buckets=build_1_2_5_buckets(max_model_len),
        #     )
        # self.histogram_best_of_request_family = pb_utils.MetricFamily(
        #     name="vllm:request_params_best_of",
        #     description="Histogram of the best_of request parameter.",
        #     kind=pb_utils.MetricFamily.HISTOGRAM,
        #     buckets=[1, 2, 5, 10, 20],
        # )
        # self.histogram_n_request_family = pb_utils.MetricFamily(
        #     name="vllm:request_params_n",
        #     description="Histogram of the n request parameter.",
        #     kind=pb_utils.MetricFamily.HISTOGRAM,
        #     buckets=[1, 2, 5, 10, 20],
        # )
        # self.counter_request_success_family = pb_utils.MetricFamily(
        #     name="vllm:request_success_total",
        #     description="Count of successfully processed requests.",
        #     kind=pb_utils.MetricFamily.COUNTER)

        # Speculatie decoding stats
        # self.gauge_spec_decode_draft_acceptance_rate_family = pb_utils.MetricFamily(
        #     name="vllm:spec_decode_draft_acceptance_rate",
        #     description="Speculative token acceptance rate.",
        #     kind=pb_utils.MetricFamily.GAUGE)
        # self.gauge_spec_decode_efficiency_family = pb_utils.MetricFamily(
        #     name="vllm:spec_decode_efficiency",
        #     description="Speculative decoding system efficiency.",
        #     kind=pb_utils.MetricFamily.GAUGE)
        # self.counter_spec_decode_num_accepted_tokens_family = pb_utils.MetricFamily(
        #     name="vllm:spec_decode_num_accepted_tokens_total",
        #     description="Number of accepted tokens.",
        #     kind=pb_utils.MetricFamily.COUNTER)
        # self.counter_spec_decode_num_draft_tokens_family = pb_utils.MetricFamily(
        #     name="vllm:spec_decode_num_draft_tokens_total",
        #     description="Number of draft tokens.",
        #     kind=pb_utils.MetricFamily.COUNTER)
        # self.counter_spec_decode_num_emitted_tokens_family = pb_utils.MetricFamily(
        #     name="vllm:spec_decode_num_emitted_tokens_total",
        #     description="Number of emitted tokens.",
        #     kind=pb_utils.MetricFamily.COUNTER)

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
        # self.histogram_time_to_first_token = self.histogram_time_to_first_token_family.Metric(
        #     labels=labels
        # )
        # self.histogram_time_per_output_token = self.histogram_time_per_output_token_family.Metric(
        #     labels=labels
        # )

        # Request stats
        #   Latency
        # self.histogram_e2e_time_request = self.histogram_e2e_time_request_family.Metric(
        # labels=labels
        # )
        # #   Metadata
        # self.histogram_num_prompt_tokens_request = self.histogram_num_prompt_tokens_request_family.Metric(
        # labels=labels
        # )
        # self.histogram_num_generation_tokens_request = self.histogram_num_generation_tokens_request_family.Metric(
        # labels=labels
        # )
        # self.histogram_best_of_request = self.histogram_best_of_request_family.Metric(
        # labels=labels
        # )
        # self.histogram_n_request = self.histogram_n_request_family.Metric(
        # labels=labels
        # )
        # self.counter_request_success = self.counter_request_success_family.Metric(
        #     labels=labels
        # )

        # Speculatie decoding stats
        # self.gauge_spec_decode_draft_acceptance_rate_ = self.gauge_spec_decode_draft_acceptance_rate_family.Metric(
        # labels=labels
        # )
        # self.gauge_spec_decode_efficiency = self.gauge_spec_decode_efficiency_family.Metric(
        # labels=labels
        # )
        # self.counter_spec_decode_num_accepted_tokens = self.counter_spec_decode_num_accepted_tokens_family.Metric(
        # labels=labels
        # )
        # self.counter_spec_decode_num_draft_tokens = self.counter_spec_decode_num_draft_tokens_family.Metric(
        # labels=labels
        # )
        # self.counter_spec_decode_num_emitted_tokens = self.counter_spec_decode_num_emitted_tokens_family.Metric(
        #     labels=labels
        # )


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

    # def _log_histogram(self, histogram, data: Union[List[int],
    #                                                 List[float]]) -> None:
    #     # Convenience function for logging list to histogram.
    #     for datum in data:
    #         histogram.labels(**self.labels).observe(datum)

    def log(self, stats: VllmStats) -> None:
        # self.maybe_update_spec_decode_metrics(stats)

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
        # self._log_histogram(self.metrics.histogram_time_to_first_token, stats.time_to_first_tokens_iter)
        # self._log_histogram(self.metrics.histogram_time_per_output_token, stats.time_per_output_tokens_iter)

        # Request level data
        # Latency
        # self._log_histogram(self.metrics.histogram_e2e_time_request, stats.time_e2e_requests)
        # Metadata
        # self._log_histogram(self.metrics.histogram_num_prompt_tokens_request, stats.num_prompt_tokens_requests)
        # self._log_histogram(self.metrics.histogram_num_generation_tokens_request, stats.num_generation_tokens_requests)
        # self._log_histogram(self.metrics.histogram_best_of_request, stats.best_of_requests)
        # self._log_histogram(self.metrics.histogram_n_request, stats.n_requests)
        # self._log_histogram(self.metrics.counter_request_success, stats.finished_reason_requests)

        # Speculatie decoding stats
        # if self.spec_decode_metrics is not None:
        #     self._log_gauge(self.metrics.gauge_spec_decode_draft_acceptance_rate, self.spec_decode_metrics.draft_acceptance_rate)
        #     self._log_gauge(self.metrics.gauge_spec_decode_efficiency, self.spec_decode_metrics.system_efficiency)
        #     self._log_counter(self.metrics.counter_spec_decode_num_accepted_tokens, self.spec_decode_metrics.accepted_tokens)
        #     self._log_counter(self.metrics.counter_spec_decode_num_draft_tokens, self.spec_decode_metrics.draft_tokens)
        #     self._log_counter(self.metrics.counter_spec_decode_num_emitted_tokens, self.spec_decode_metrics.emitted_tokens)
