<!--
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
-->

# Security Policy

## Reporting a Vulnerability

To report a potential security vulnerability in this project or any other
NVIDIA product, please use one of the following channels. **Do not open a
public GitHub issue for a security vulnerability.**

1. **NVIDIA Vulnerability Disclosure Program** (preferred):
   <https://www.nvidia.com/en-us/security/>
2. **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt
   sensitive reports with NVIDIA's public PGP key
   (<https://www.nvidia.com/en-us/security/pgp-key>).
3. **GitHub Private Vulnerability Reporting**, if enabled, via the
   repository's **Security** tab.

**OEM partners should contact their NVIDIA Customer Program Manager.**

Please include:

1. Product name and version or branch that contains the vulnerability
2. Type of vulnerability (for example code execution, denial of service,
   information disclosure)
3. Instructions to reproduce the vulnerability
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit it

NVIDIA PSIRT acknowledges reports, triages them, and coordinates fixes and
disclosure with the reporter. See <https://www.nvidia.com/en-us/security/>
for past security bulletins and notices.

## Security Architecture and Context

**Project:** the Triton Inference Server backend for
[vLLM](https://github.com/vllm-project/vllm). It is a Python-based Triton
backend (`src/model.py`, `src/utils/`) that runs inside the Triton Python
backend stub process and forwards inference requests to a vLLM
`AsyncLLM` engine.

**Software classification:** Library (a plug-in loaded by Triton Inference
Server; it exposes no network listener of its own).

**Repository Exposure Classification:** Public (the repository is publicly
readable on GitHub).

**Service Exposure Classification:** Internal-Sensitive, medium confidence.
Basis: a deployment-dependent component whose exposure is determined by the
Triton Inference Server that hosts it; it processes user-supplied prompts and
may load third-party model weights. This is an informal descriptor, not an
official NVIDIA label.

**Primary security responsibility:** safely translate Triton request tensors
into vLLM engine calls and return the results, without widening the trust
granted to the host Triton process.

**Key interfaces and boundaries:**

- **Triton request tensors** (`text_input`, `image`, `sampling_parameters`,
  `stream`, `embedding_request`, and others), supplied by clients through
  Triton's HTTP/gRPC frontends. This is the untrusted-input boundary.
- **Model repository files** (`model.json` engine arguments and the optional
  `multi_lora.json` adapter map), read from the model directory at load time.
  These are trusted administrator-supplied configuration.
- **vLLM engine and model weights**, fetched or loaded by vLLM according to
  `model.json` (local path or a model hub identifier).
- **Triton metrics and logging APIs**, used to publish vLLM statistics and
  log messages.

## Threat Model

1. **Malformed or oversized image input:** the `image` tensor is
   base64-decoded and opened with Pillow (`src/utils/request.py`). A crafted
   or very large image can exhaust memory or trigger a parser defect in the
   image library, affecting the shared backend process.
2. **Untrusted `sampling_parameters` JSON:** clients provide a JSON string
   that is parsed and mapped onto vLLM sampling options, including
   `lora_name`. Extreme values (very large token counts, many sequences) can
   cause resource exhaustion in the shared engine. Parameters that fail to
   construct (for example, unsupported keys) fail in `src/utils/request.py`
   before the request is submitted to vLLM, so supported parameters with
   extreme values are the main concern.
3. **Malicious or tampered model artifacts:** `model.json` is passed to
   `AsyncEngineArgs`, and model weights or LoRA adapters are loaded by vLLM.
   Weights from an untrusted source, or an engine option that enables remote
   code execution in model loading, can lead to code execution in the Triton
   process.
4. **LoRA adapter selection:** `lora_name` is resolved through
   `multi_lora.json` to a filesystem path (`src/model.py`,
   `src/utils/request.py`). A writable or attacker-influenced adapter map or
   adapter directory lets an attacker load unintended weights.
5. **Reserved embedding input:** the `embedding_request` tensor is intended
   only for Triton's OpenAI-compatible frontend, but any client able to send
   it can supply arbitrary JSON (`input`, `pooling_params`) to the embedding
   path.
6. **Information disclosure through logs and errors:** tracebacks and request
   details are written to the Triton log (`self.logger.log_error`). Prompts,
   model paths, or stack traces could reach log consumers with weaker access
   controls than the inference API.
7. **Denial of service by request flooding or cancellation races:** all
   requests are placed on the vLLM engine as they arrive, and the engine runs
   in its own event-loop thread with a separate response thread. Floods or
   rapid cancellation can starve other tenants of GPU memory and
   throughput.

## Critical Security Assumptions

- **Authentication and authorization are external.** This backend performs
  none. It assumes Triton, a gateway, or network controls authenticate and
  authorize callers.
- **Transport security is external.** TLS termination and network isolation
  are provided by the Triton deployment or surrounding infrastructure.
- **Model repository contents are trusted.** `model.json`, `multi_lora.json`,
  model weights, and adapter files are assumed to come from a trusted source
  and to be writable only by administrators.
- **Input is not fully validated here.** Prompt text, image payloads, and
  sampling parameters are passed to vLLM and Pillow, which are assumed to
  handle hostile input safely; callers are expected to enforce size and rate
  limits upstream.
- **Resource isolation relies on the host.** Per-tenant quotas, GPU memory
  limits, and request limits are assumed to be enforced by Triton
  configuration and the deployment platform.
- **Dependencies are kept current.** The versions of vLLM and Pillow
  supplied by the Triton container are assumed to receive security updates
  through regular Triton releases.
