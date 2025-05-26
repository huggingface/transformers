<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Serving

Transformer models can be served for inference with specialized libraries such as Text Generation Inference (TGI) and vLLM. These libraries are specifically designed to optimize performance with LLMs and include many unique optimization features that may not be included in Transformers.

## TGI

[TGI](https://huggingface.co/docs/text-generation-inference/index) can serve models that aren't [natively implemented](https://huggingface.co/docs/text-generation-inference/supported_models) by falling back on the Transformers implementation of the model. Some of TGIs high-performance features aren't available in the Transformers implementation, but other features like continuous batching and streaming are still supported.

> [!TIP]
> Refer to the [Non-core model serving](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models) guide for more details.

Serve a Transformers implementation the same way you'd serve a TGI model.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2
```

Add `--trust-remote_code` to the command to serve a custom Transformers model.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <CUSTOM_MODEL_ID> --trust-remote-code
```

## vLLM

[vLLM](https://docs.vllm.ai/en/latest/index.html) can also serve a Transformers implementation of a model if it isn't [natively implemented](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models) in vLLM.

Many features like quantization, LoRA adapters, and distributed inference and serving are supported for the Transformers implementation.

> [!TIP]
> Refer to the [Transformers fallback](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers-fallback) section for more details.

By default, vLLM serves the native implementation and if it doesn't exist, it falls back on the Transformers implementation. But you can also set `--model-impl transformers` to explicitly use the Transformers model implementation.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers
```

Add the `trust-remote-code` parameter to enable loading a remote code model.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers \
    --trust-remote-code
```