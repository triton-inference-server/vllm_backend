<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

The Triton backend for [vLLM](https://github.com/vllm-project/vllm).
You can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems on the [issues
page](https://github.com/triton-inference-server/server/issues).
This backend is designed to run [vLLM](https://github.com/vllm-project/vllm)
with
[one of the HuggingFace models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)
it supports.

Where can I ask general questions about Triton and Triton backends?
Be sure to read all the information below as well as the [general
Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main [server](https://github.com/triton-inference-server/server)
repo. If you don't find your answer there you can ask questions on the
main Triton [issues page](https://github.com/triton-inference-server/server/issues).

## Build the vLLM Backend

As a Python-based backend, your Triton server just needs to have the [Python backend](https://github.com/triton-inference-server/python_backend)
located in the backends directory: `/opt/tritonserver/backends/python`. After that, you can save the vLLM backend in the backends folder as `/opt/tritonserver/backends/vllm`. The `model.py` file in the `src` directory should be in the vllm folder and will function as your Python-based backend.

In other words, there are no build steps. You only need to copy this to your Triton backends repository. If you use the official Triton vLLM container, this is already set up for you.

The backend repository should look like this:
```
/opt/tritonserver/backends/
`-- vllm
    |-- model.py
 -- python
    |-- libtriton_python.so
    |-- triton_python_backend_stub
    |-- triton_python_backend_utils.py
```

## Using the vLLM Backend

You can see an example model_repository in the `samples` folder.
You can use this as is and change the model by changing the `model` value in `model.json`.
You can change the GPU utilization and logging parameters in that file as well.

In the `samples` folder, you can also find a sample client, `client.py`.
This client is meant to function similarly to the Triton
[vLLM example](https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/vLLM).
By default, this will test `prompts.txt`, which we have included in the samples folder.


## Running the Latest vLLM Version

By default, the vLLM backend uses the version of vLLM that is available via Pip.
These are compatible with the newer versions of CUDA running in Triton.
If you would like to use a specific vLLM commit or the latest version of vLLM, you
will need to use a
[custom execution environment](https://github.com/triton-inference-server/python_backend#creating-custom-execution-environments).
Please see the
[conda](samples/conda) subdirectory of the `samples` folder for information on how to do so.

## Important Notes

* At present, Triton only supports one Python-based backend per server. If you try to start multiple vLLM models, you will get an error.

### Running Multiple Instances of Triton Server

Python-based backends use shared memory to transfer requests to the stub process. When running multiple instances of Triton Server on the same machine that use Python-based backend models, there would be shared memory region name conflicts that can result in segmentation faults or hangs. In order to avoid this issue, you need to specify different shm-region-prefix-name using the --backend-config flag.
```
# Triton instance 1
tritonserver --model-repository=/models --backend-config=python,shm-region-prefix-name=prefix1

# Triton instance 2
tritonserver --model-repository=/models --backend-config=python,shm-region-prefix-name=prefix2
```
Note that the hangs would only occur if the /dev/shm is shared between the two instances of the server. If you run the servers in different containers that do not share this location, you do not need to specify shm-region-prefix-name.