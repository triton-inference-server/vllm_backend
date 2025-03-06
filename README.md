<!--
# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# vLLM Backend

The Triton backend for [vLLM](https://github.com/vllm-project/vllm)
is designed to run
[supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)
on a
[vLLM engine](https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py).
You can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend).


This is a [Python-based backend](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md#python-based-backends).
When using this backend, all requests are placed on the
vLLM AsyncEngine as soon as they are received. Inflight batching and paged attention is handled
by the vLLM engine.

Where can I ask general questions about Triton and Triton backends?
Be sure to read all the information below as well as the [general
Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main [server](https://github.com/triton-inference-server/server)
repo. If you don't find your answer there you can ask questions on the
main Triton [issues page](https://github.com/triton-inference-server/server/issues).

## Installing the vLLM Backend

There are several ways to install and deploy the vLLM backend.

### Option 1. Use the Pre-Built Docker Container.

Pull a `tritonserver:<xx.yy>-vllm-python-py3` container with vLLM backend from the
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
registry. \<xx.yy\> is the version of Triton that you want to use. Please note,
that Triton's vLLM container has been introduced starting from 23.10 release.

```
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3
```

### Option 2. Build a Custom Container From Source
You can follow steps described in the
[Building With Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)
guide and use the
[build.py](https://github.com/triton-inference-server/server/blob/main/build.py)
script.

A sample command to build a Triton Server container with all options enabled is shown below. Feel free to customize flags according to your needs.

Please use [NGC registry](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)
to get the latest version of the Triton vLLM container, which corresponds to the
latest YY.MM (year.month) of [Triton release](https://github.com/triton-inference-server/server/releases).


```
# YY.MM is the version of Triton.
export TRITON_CONTAINER_VERSION=<YY.MM>
./build.py -v  --enable-logging
                --enable-stats
                --enable-tracing
                --enable-metrics
                --enable-gpu-metrics
                --enable-cpu-metrics
                --enable-gpu
                --filesystem=gcs
                --filesystem=s3
                --filesystem=azure_storage
                --endpoint=http
                --endpoint=grpc
                --endpoint=sagemaker
                --endpoint=vertex-ai
                --upstream-container-version=${TRITON_CONTAINER_VERSION}
                --backend=python:r${TRITON_CONTAINER_VERSION}
                --backend=vllm:r${TRITON_CONTAINER_VERSION}
                --backend=ensemble
```

### Option 3. Add the vLLM Backend to the Default Triton Container

You can install the vLLM backend directly into the NGC Triton container.
In this case, please install vLLM first. You can do so by running
`pip install vllm==<vLLM_version>`. Then, set up the vLLM backend in the
container with the following commands:

```
mkdir -p /opt/tritonserver/backends/vllm
git clone https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend
cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm
```

## Using the vLLM Backend

You can see an example
[model_repository](samples/model_repository)
in the [samples](samples) folder.
You can use this as is and change the model by changing the `model` value in `model.json`.
`model.json` represents a key-value dictionary that is fed to vLLM's AsyncLLMEngine when initializing the model.
You can see supported arguments in vLLM's
[arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py).
Specifically,
[here](https://github.com/vllm-project/vllm/blob/ee8217e5bee5860469204ee57077a91138c9af02/vllm/engine/arg_utils.py#L11)
and
[here](https://github.com/vllm-project/vllm/blob/ee8217e5bee5860469204ee57077a91138c9af02/vllm/engine/arg_utils.py#L201).

For multi-GPU support, EngineArgs like tensor_parallel_size can be specified in
[model.json](samples/model_repository/vllm_model/1/model.json).

Note: vLLM greedily consume up to 90% of the GPU's memory under default settings.
The sample model updates this behavior by setting gpu_memory_utilization to 50%.
You can tweak this behavior using fields like gpu_memory_utilization and other settings in
[model.json](samples/model_repository/vllm_model/1/model.json).

### Launching Triton Inference Server

Once you have the model repository set up, it is time to launch the Triton server.
We will use the [pre-built Triton container with vLLM backend](#option-1-use-the-pre-built-docker-container) from
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) in this example.

Run the following command inside the `vllm_backend` directory:
```
docker run --gpus all -it --net=host --rm -p 8001:8001 --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 tritonserver --model-repository ./samples/model_repository
```

Replace \<xx.yy\> with the version of Triton that you want to use.
Note that Triton's vLLM container was first published starting from
23.10 release.

After you start Triton you will see output on the console showing
the server starting up and loading the model. When you see output
like the following, Triton is ready to accept inference requests.

```
I1030 22:33:28.291908 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1030 22:33:28.292879 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1030 22:33:28.335154 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

### Sending Your First Inference

After you
[start Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
with the
[sample model_repository](samples/model_repository),
you can quickly run your first inference request with the
[generate endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md).

Try out the command below from another terminal:

```
curl -X POST localhost:8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
```

Upon success, you should see a response from the server like this one:
```
{"model_name":"vllm_model","model_version":"1","text_output":"What is Triton Inference Server?\n\nTriton Inference Server is a server that is used by many"}
```

In the [samples](samples) folder, you can also find a sample client,
[client.py](samples/client.py) which uses Triton's
[asyncio gRPC client library](https://github.com/triton-inference-server/client#python-asyncio-support-beta-1)
to run inference on Triton.

### Running the Latest vLLM Version

You can check the vLLM version included in Triton Inference Server from
[Framework Containers Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).
*Note:* The vLLM Triton Inference Server container has been introduced
starting from 23.10 release.

You can use  `pip install ...` within the container to upgrade vLLM version.

## Running Multiple Instances of Triton Server

If you are running multiple instances of Triton server with a Python-based backend,
you need to specify a different `shm-region-prefix-name` for each server. See
[here](https://github.com/triton-inference-server/python_backend#running-multiple-instances-of-triton-server)
for more information.

## Additional vLLM outputs

Additional vLLM outputs may be requested optionally on a per-request basis. See
[this docs](docs/additional_outputs.md) for more information.

## Triton Metrics
Starting with the 24.08 release of Triton, users can now obtain specific
vLLM metrics by querying the Triton metrics endpoint (see complete vLLM metrics
[here](https://docs.vllm.ai/en/latest/serving/metrics.html)). This can be
accomplished by launching a Triton server in any of the ways described above
(ensuring the build code / container is 24.08 or later) and querying the server.
Upon receiving a successful response, you can query the metrics endpoint by entering
the following:
```bash
curl localhost:8002/metrics
```
VLLM stats are reported by the metrics endpoint in fields that are prefixed with
`vllm:`. Triton currently supports reporting of the following metrics from vLLM.
```bash
# Number of prefill tokens processed.
counter_prompt_tokens
# Number of generation tokens processed.
counter_generation_tokens
# Histogram of time to first token in seconds.
histogram_time_to_first_token
# Histogram of time per output token in seconds.
histogram_time_per_output_token
# Histogram of end to end request latency in seconds.
histogram_e2e_time_request
# Number of prefill tokens processed.
histogram_num_prompt_tokens_request
# Number of generation tokens processed.
histogram_num_generation_tokens_request
# Histogram of the best_of request parameter.
histogram_best_of_request
# Histogram of the n request parameter.
histogram_n_request
```
Your output for these fields should look similar to the following:
```bash
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model="vllm_model",version="1"} 10
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model="vllm_model",version="1"} 16
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_count{model="vllm_model",version="1"} 1
vllm:time_to_first_token_seconds_sum{model="vllm_model",version="1"} 0.03233122825622559
vllm:time_to_first_token_seconds_bucket{model="vllm_model",version="1",le="0.001"} 0
...
vllm:time_to_first_token_seconds_bucket{model="vllm_model",version="1",le="+Inf"} 1
# HELP vllm:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_count{model="vllm_model",version="1"} 15
vllm:time_per_output_token_seconds_sum{model="vllm_model",version="1"} 0.04501533508300781
vllm:time_per_output_token_seconds_bucket{model="vllm_model",version="1",le="0.01"} 14
...
vllm:time_per_output_token_seconds_bucket{model="vllm_model",version="1",le="+Inf"} 15
# HELP vllm:e2e_request_latency_seconds Histogram of end to end request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_count{model="vllm_model",version="1"} 1
vllm:e2e_request_latency_seconds_sum{model="vllm_model",version="1"} 0.08686184883117676
vllm:e2e_request_latency_seconds_bucket{model="vllm_model",version="1",le="1"} 1
...
vllm:e2e_request_latency_seconds_bucket{model="vllm_model",version="1",le="+Inf"} 1
# HELP vllm:request_prompt_tokens Number of prefill tokens processed.
# TYPE vllm:request_prompt_tokens histogram
vllm:request_prompt_tokens_count{model="vllm_model",version="1"} 1
vllm:request_prompt_tokens_sum{model="vllm_model",version="1"} 10
vllm:request_prompt_tokens_bucket{model="vllm_model",version="1",le="1"} 0
...
vllm:request_prompt_tokens_bucket{model="vllm_model",version="1",le="+Inf"} 1
# HELP vllm:request_generation_tokens Number of generation tokens processed.
# TYPE vllm:request_generation_tokens histogram
vllm:request_generation_tokens_count{model="vllm_model",version="1"} 1
vllm:request_generation_tokens_sum{model="vllm_model",version="1"} 16
vllm:request_generation_tokens_bucket{model="vllm_model",version="1",le="1"} 0
...
vllm:request_generation_tokens_bucket{model="vllm_model",version="1",le="+Inf"} 1
# HELP vllm:request_params_n Histogram of the n request parameter.
# TYPE vllm:request_params_n histogram
vllm:request_params_n_count{model="vllm_model",version="1"} 1
vllm:request_params_n_sum{model="vllm_model",version="1"} 1
vllm:request_params_n_bucket{model="vllm_model",version="1",le="1"} 1
...
vllm:request_params_n_bucket{model="vllm_model",version="1",le="+Inf"} 1
```
To enable vLLM engine colleting metrics, "disable_log_stats" option need to be either false
or left empty (false by default) in [model.json](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json).
```bash
"disable_log_stats": false
```
*Note:* vLLM metrics are not reported to Triton metrics server by default
due to potential performance slowdowns. To enable vLLM model's metrics
reporting, please add following lines to its config.pbtxt as well.
```
parameters: {
  key: "REPORT_CUSTOM_METRICS"
  value: {
    string_value: "true"
  }
}
```

## vLLM Engine Health Check (BETA)

vLLM Engine Health Check may be enabled optionally, for more accurate model
state reported by the server. See [this docs](docs/health_check.md) for more
information.

## Referencing the Tutorial

You can read further in the
[vLLM Quick Deploy guide](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/vLLM)
in the
[tutorials](https://github.com/triton-inference-server/tutorials/) repository.
