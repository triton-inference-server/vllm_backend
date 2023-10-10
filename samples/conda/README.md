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

If you would like to run conda with the latest version of vLLM, you will need to create a
a [custom execution environment](https://github.com/triton-inference-server/python_backend#creating-custom-execution-environments).
This is because vLLM currently does not support the latest versions of CUDA in the Triton environment.
Instructions for creating a custom execution environment with the latest vLLM version are below.

## Step 1: Build a Custom Execution Environment With vLLM and Other Dependencies

The provided script should build the package environment
for you which will be used to load the model in Triton.

Run the following command from this directory. You can use any version of Triton.
```
docker run --gpus all -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=8G --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/work -w /work nvcr.io/nvidia/tritonserver:23.09-py3 bash
./gen_vllm_env.sh
```

This step might take a while to build the environment packages. Once complete, the current folder will be populated with
`triton_python_backend_stub` and `vllm_env`.

## Step 2: Update Your Model Repository

You want to place the stub and environment in your model directory.
The model directory should look something like this:
```
model_repository/
`-- vllm_model
    |-- 1
    |   `-- model.json
    |-- config.pbtxt
    |-- triton_python_backend_stub
    `-- vllm_env
```

You also want to add this section to the config.pbtxt of your model:
```
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/vllm_env"}
}
```

## Step 3: Run Your Model

You can now start Triton server with your model!