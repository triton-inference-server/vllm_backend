<!--
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
-->

# vLLM Health Check (BETA)

> [!NOTE]
> The vLLM Health Check support is currently in BETA. Its features and
> functionality are subject to change as we collect feedback. We are excited to
> hear any thoughts you have!

The vLLM backend supports checking for
[vLLM Engine Health](https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/engine/async_llm_engine.py#L1177-L1185)
upon receiving each inference request. If the health check fails, the model
state will becomes NOT Ready at the server, which can be queried by the
[Repository Index](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md#index)
or
[Model Ready](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/library/http_client.h#L178-L192)
APIs.

The Health Check is disabled by default. To enable it, set the following
parameter on the model config to true
```
parameters: {
  key: "ENABLE_VLLM_HEALTH_CHECK"
  value: { string_value: "true" }
}
```
and select
[Model Control Mode EXPLICIT](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md#model-control-mode-explicit)
when the server is started.
