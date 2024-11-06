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

import queue
import threading
from typing import Dict, List, Union

import triton_python_backend_utils as pb_utils
from vllm.engine.metrics import StatLoggerBase as VllmStatLoggerBase
from vllm.engine.metrics import Stats as VllmStats
from vllm.engine.metrics import SupportsMetricsInfo, build_1_2_5_buckets
from vllm.version import __version__ as _VLLM_VERSION

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
        # 'best_of' metric has been hidden since vllm 0.6.3
        # https://github.com/vllm-project/vllm/commit/cbc2ef55292b2af6ff742095c030e8425124c005
        if _VLLM_VERSION < "0.6.3":
            self.histogram_best_of_request_family = pb_utils.MetricFamily(
                name="vllm:request_params_best_of",
                description="Histogram of the best_of request parameter.",
                kind=pb_utils.MetricFamily.HISTOGRAM,
            )
        self.histogram_n_request_family = pb_utils.MetricFamily(
            name="vllm:request_params_n",
            description="Histogram of the n request parameter.",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )

        # Initialize metrics
        # Iteration stats
        self.counter_prompt_tokens = self.counter_prompt_tokens_family.Metric(
            labels=labels
        )
        self.counter_generation_tokens = self.counter_generation_tokens_family.Metric(
            labels=labels
        )
        # Use the same bucket boundaries from vLLM sample metrics as an example.
        # https://github.com/vllm-project/vllm/blob/21313e09e3f9448817016290da20d0db1adf3664/vllm/engine/metrics.py#L81-L96
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
        self.histogram_e2e_time_request = self.histogram_e2e_time_request_family.Metric(
            labels=labels,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
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
        if _VLLM_VERSION < "0.6.3":
            self.histogram_best_of_request = self.histogram_best_of_request_family.Metric(
                labels=labels,
                buckets=[1, 2, 5, 10, 20],
            )
        self.histogram_n_request = self.histogram_n_request_family.Metric(
            labels=labels,
            buckets=[1, 2, 5, 10, 20],
        )


class VllmStatLogger(VllmStatLoggerBase):
    """StatLogger is used as an adapter between vLLM stats collector and Triton metrics provider."""

    def __init__(self, labels: Dict, max_model_len: int, log_logger) -> None:
        # Tracked stats over current local logging interval.
        # local_interval not used here. It's for vLLM logs to stdout.
        super().__init__(local_interval=0)
        self.metrics = TritonMetrics(labels, max_model_len)
        self.log_logger = log_logger

        # Starting the metrics thread. It allows vLLM to keep making progress
        # while reporting metrics to triton metrics service.
        self._logger_queue = queue.Queue()
        self._logger_thread = threading.Thread(target=self.logger_loop)
        self._logger_thread.start()

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        pass

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
        ]
        histogram_metrics = [
            (
                self.metrics.histogram_time_to_first_token,
                stats.time_to_first_tokens_iter,
            ),
            (
                self.metrics.histogram_time_per_output_token,
                stats.time_per_output_tokens_iter,
            ),
            (self.metrics.histogram_e2e_time_request, stats.time_e2e_requests),
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
        if _VLLM_VERSION < "0.6.3":
            histogram_metrics.append((self.metrics.histogram_best_of_request, stats.best_of_requests))
        for metric, data in counter_metrics:
            self._log_counter(metric, data)
        for metric, data in histogram_metrics:
            self._log_histogram(metric, data)

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
            else:
                self.log_logger.log_error(f"Undefined command name: {command}")

    def finalize(self):
        # Shutdown the logger thread.
        self._logger_queue.put(None)
        if self._logger_thread is not None:
            self._logger_thread.join()
            self._logger_thread = None
