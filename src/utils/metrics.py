# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue
import threading
from typing import Dict, List, Union

import triton_python_backend_utils as pb_utils
from vllm.config import VllmConfig
from vllm.engine.metrics import StatLoggerBase as VllmStatLoggerBase
from vllm.engine.metrics import Stats as VllmStats
from vllm.engine.metrics import SupportsMetricsInfo, build_1_2_5_buckets


class TritonMetrics:
    def __init__(self, labels: List[str], max_model_len: int):
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
        self.counter_num_preemption_family = pb_utils.MetricFamily(
            name="vllm:num_preemptions_total",
            description="Number of preemption tokens processed.",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.histogram_iteration_tokens_family = pb_utils.MetricFamily(
            name="vllm:iteration_tokens_total",
            description="Histogram of number of tokens per engine_step.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_time_to_first_token_family = pb_utils.MetricFamily(
            name="vllm:time_to_first_token_seconds",
            description="Histogram of time to first token in seconds.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_time_per_output_token_family = pb_utils.MetricFamily(
            name="vllm:time_per_output_token_seconds",
            description="Histogram of time per output token in seconds.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        # Request stats
        #   Latency
        self.histogram_e2e_time_request_family = pb_utils.MetricFamily(
            name="vllm:e2e_request_latency_seconds",
            description="Histogram of end to end request latency in seconds.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_queue_time_request_family = pb_utils.MetricFamily(
            name="vllm:request_queue_time_seconds",
            description="Histogram of time spent in WAITING phase for request.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_inference_time_request_family = pb_utils.MetricFamily(
            name="vllm:request_inference_time_seconds",
            description="Histogram of time spent in RUNNING phase for request.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_prefill_time_request_family = pb_utils.MetricFamily(
            name="vllm:request_prefill_time_seconds",
            description="Histogram of time spent in PREFILL phase for request.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_decode_time_request_family = pb_utils.MetricFamily(
            name="vllm:request_decode_time_seconds",
            description="Histogram of time spent in DECODE phase for request.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        #   Metadata
        self.histogram_num_prompt_tokens_request_family = pb_utils.MetricFamily(
            name="vllm:request_prompt_tokens",
            description="Number of prefill tokens processed.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_num_generation_tokens_request_family = pb_utils.MetricFamily(
            name="vllm:request_generation_tokens",
            description="Number of generation tokens processed.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.histogram_n_request_family = pb_utils.MetricFamily(
            name="vllm:request_params_n",
            description="Histogram of the n request parameter.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
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
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage_family = pb_utils.MetricFamily(
            name="vllm:gpu_cache_usage_perc",
            description="GPU KV-cache usage. 1 means 100 percent usage.",
            kind=pb_utils.MetricFamily.GAUGE,
        )

        # Initialize metrics
        # Iteration stats
        self.counter_prompt_tokens = self.counter_prompt_tokens_family.Metric(
            labels=labels
        )
        self.counter_generation_tokens = self.counter_generation_tokens_family.Metric(
            labels=labels
        )
        self.counter_num_preemption = self.counter_num_preemption_family.Metric(
            labels=labels
        )

        # Use the same bucket boundaries from vLLM sample metrics as an example.
        # https://github.com/vllm-project/vllm/blob/21313e09e3f9448817016290da20d0db1adf3664/vllm/engine/metrics.py#L81-L96
        self.histogram_iteration_tokens = self.histogram_iteration_tokens_family.Metric(
            labels=labels,
            buckets=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        )

        self.histogram_time_to_first_token = (
            self.histogram_time_to_first_token_family.Metric(
                labels=labels,
                buckets=[
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.04,
                    0.06,
                    0.08,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                    5.0,
                    7.5,
                    10.0,
                ],
            )
        )
        self.histogram_time_per_output_token = (
            self.histogram_time_per_output_token_family.Metric(
                labels=labels,
                buckets=[
                    0.01,
                    0.025,
                    0.05,
                    0.075,
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.75,
                    1.0,
                    2.5,
                ],
            )
        )
        # Request stats
        #   Latency
        request_latency_buckets = [
            0.3,
            0.5,
            0.8,
            1.0,
            1.5,
            2.0,
            2.5,
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            120.0,
            240.0,
            480.0,
            960.0,
            1920.0,
            7680.0,
        ]
        self.histogram_e2e_time_request = self.histogram_e2e_time_request_family.Metric(
            labels=labels,
            buckets=request_latency_buckets,
        )
        self.histogram_prefill_time_request = (
            self.histogram_prefill_time_request_family.Metric(
                labels=labels,
                buckets=request_latency_buckets,
            )
        )
        self.histogram_decode_time_request = (
            self.histogram_decode_time_request_family.Metric(
                labels=labels,
                buckets=request_latency_buckets,
            )
        )
        self.histogram_inference_time_request = (
            self.histogram_inference_time_request_family.Metric(
                labels=labels,
                buckets=request_latency_buckets,
            )
        )
        self.histogram_queue_time_request = (
            self.histogram_queue_time_request_family.Metric(
                labels=labels,
                buckets=request_latency_buckets,
            )
        )
        #   Metadata
        self.histogram_num_prompt_tokens_request = (
            self.histogram_num_prompt_tokens_request_family.Metric(
                labels=labels,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        )
        self.histogram_num_generation_tokens_request = (
            self.histogram_num_generation_tokens_request_family.Metric(
                labels=labels,
                buckets=build_1_2_5_buckets(max_model_len),
            )
        )
        self.histogram_n_request = self.histogram_n_request_family.Metric(
            labels=labels,
            buckets=[1, 2, 5, 10, 20],
        )
        # System stats
        #   Scheduler State
        self.gauge_scheduler_running = self.gauge_scheduler_running_family.Metric(
            labels=labels
        )
        self.gauge_scheduler_waiting = self.gauge_scheduler_waiting_family.Metric(
            labels=labels
        )
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage = self.gauge_gpu_cache_usage_family.Metric(
            labels=labels
        )


class VllmStatLogger(VllmStatLoggerBase):
    """StatLogger is used as an adapter between vLLM stats collector and Triton metrics provider."""

    def __init__(self, labels: Dict, vllm_config: VllmConfig, log_logger) -> None:
        # Tracked stats over current local logging interval.
        # local_interval not used here. It's for vLLM logs to stdout.
        super().__init__(local_interval=0, vllm_config=vllm_config)
        self.metrics = TritonMetrics(
            labels=labels, max_model_len=vllm_config.model_config.max_model_len
        )
        self.log_logger = log_logger

        # Starting the metrics thread. It allows vLLM to keep making progress
        # while reporting metrics to triton metrics service.
        self._logger_queue = queue.Queue()
        self._logger_thread = threading.Thread(target=self.logger_loop)
        self._logger_thread.start()

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        pass

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        """Convenience function for logging to gauge.

        Args:
            gauge: A gauge metric instance.
            data: An int or float to set as the current gauge value.

        Returns:
            None
        """
        self._logger_queue.put_nowait((gauge, "set", data))

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        """Convenience function for logging to counter.

        Args:
            counter: A counter metric instance.
            data: An int or float to increment the count metric.

        Returns:
            None
        """
        if data != 0:
            self._logger_queue.put_nowait((counter, "increment", data))

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        """Convenience function for logging list to histogram.

        Args:
            histogram: A histogram metric instance.
            data: A list of int or float data to observe into the histogram metric.

        Returns:
            None
        """
        for datum in data:
            self._logger_queue.put_nowait((histogram, "observe", datum))

    def log(self, stats: VllmStats) -> None:
        """Report stats to Triton metrics server.

        Args:
            stats: Created by LLMEngine for use by VllmStatLogger.

        Returns:
            None
        """
        # The list of vLLM metrics reporting to Triton is also documented here.
        # https://github.com/triton-inference-server/vllm_backend/blob/main/README.md#triton-metrics
        counter_metrics = [
            (self.metrics.counter_prompt_tokens, stats.num_prompt_tokens_iter),
            (self.metrics.counter_generation_tokens, stats.num_generation_tokens_iter),
            (self.metrics.counter_num_preemption, stats.num_preemption_iter),
        ]
        histogram_metrics = [
            (self.metrics.histogram_iteration_tokens, [stats.num_tokens_iter]),
            (
                self.metrics.histogram_time_to_first_token,
                stats.time_to_first_tokens_iter,
            ),
            (
                self.metrics.histogram_time_per_output_token,
                stats.time_per_output_tokens_iter,
            ),
            (self.metrics.histogram_e2e_time_request, stats.time_e2e_requests),
            (self.metrics.histogram_prefill_time_request, stats.time_prefill_requests),
            (self.metrics.histogram_decode_time_request, stats.time_decode_requests),
            (
                self.metrics.histogram_inference_time_request,
                stats.time_inference_requests,
            ),
            (self.metrics.histogram_queue_time_request, stats.time_queue_requests),
            (
                self.metrics.histogram_num_prompt_tokens_request,
                stats.num_prompt_tokens_requests,
            ),
            (
                self.metrics.histogram_num_generation_tokens_request,
                stats.num_generation_tokens_requests,
            ),
            (self.metrics.histogram_n_request, stats.n_requests),
        ]
        gauge_metrics = [
            (self.metrics.gauge_scheduler_running, stats.num_running_sys),
            (self.metrics.gauge_scheduler_waiting, stats.num_waiting_sys),
            (self.metrics.gauge_gpu_cache_usage, stats.gpu_cache_usage_sys),
        ]
        for metric, data in counter_metrics:
            self._log_counter(metric, data)
        for metric, data in histogram_metrics:
            self._log_histogram(metric, data)
        for metric, data in gauge_metrics:
            self._log_gauge(metric, data)

    def logger_loop(self):
        while True:
            item = self._logger_queue.get()
            # To signal shutdown a None item will be added to the queue.
            if item is None:
                break
            metric, command, data = item
            if command == "increment":
                metric.increment(data)
            elif command == "observe":
                metric.observe(data)
            elif command == "set":
                metric.set(data)
            else:
                self.log_logger.log_error(f"Undefined command name: {command}")

    def finalize(self):
        # Shutdown the logger thread.
        self._logger_queue.put(None)
        if self._logger_thread is not None:
            self._logger_thread.join()
            self._logger_thread = None
