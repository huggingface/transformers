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
    --model-impl transformers \
```

Add the `trust-remote-code` parameter to enable loading a remote code model.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers \
    --trust-remote-code \
```



# 모델 서빙하기

Transformer 모델은 추론(inference)을 위해 Text Generation Inference (TGI) 또는 vLLM 같은 특화된 라이브러리를 사용하여 서빙할 수 있습니다. 이러한 라이브러리들은 LLM의 성능 최적화를 위해 설계되었으며, Transformers 라이브러리에 기본으로 포함되지 않은 다양한 성능 최적화 기능을 제공합니다.

## TGI

[TGI](https://huggingface.co/docs/text-generation-inference)는 LLM 성능 최적화를 위해 설계된 라이브러리로, Transformers 라이브러리에서 지원되지 않을 수 있는 고유한 최적화 기능을 포함하고 있습니다.

TGI는 기본적으로 [TGI에서 공식 지원하지 않는 모델](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models)에 대해서도 Transformers의 구현체를 이용하여 서빙할 수 있습니다.

이 경우에도 TGI의 모델과 동일한 방법으로 Transformers 구현체를 서빙할 수 있습니다.


> [!TIP]
> 자세한 사항은 [Non-core model serving](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models) 가이드를 참고하세요.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <모델_아이디>
```

리모트 코드 모델을 로드하려면 `trust-remote-code` 옵션을 추가합니다.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <모델_아이디> --trust-remote-code
```



## vLLM

[vLLM](https://docs.vllm.ai/)은 Transformer 모델의 추론 성능 향상을 위해 최적화된 라이브러리로, [Transformers fallback](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers-fallback) 기능을 통해 기본적으로 지원하지 않는 모델도 Transformers 라이브러리를 통해 서빙할 수 있습니다.

Transformers 구현체에서도 양자화(quantization), LoRA 어댑터, 분산 추론 및 서빙(distributed inference and serving)과 같은 여러 기능이 지원됩니다.

> [!TIP]
> 자세한 사항은 [Transformers fallback](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers-fallback) 섹션을 참고하세요.

vLLM을 사용하면 기본적으로 네이티브 구현체를 우선 서빙하며, `model-impl transformers` 파라미터를 통해 Transformer 라이브러리 구현체로 서빙할 수 있습니다.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers
```

리모트 코드 모델을 사용하는 경우, `trust-remote-code` 옵션을 추가하여 사용합니다.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers \
    --trust-remote-code
```



