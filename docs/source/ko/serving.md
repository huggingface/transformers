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

# 모델 서빙 [[Serving]]

Text Generation Inference (TGI) 및 vLLM과 같은 특수한 라이브러리를 사용해 Transformer 모델을 추론에 사용할 수 있습니다. 이러한 라이브러리는 vLLM의 성능을 최적화하도록 설계되었으며, Transformers에는 포함되지 않은 고유한 최적화 기능을 다양하게 제공합니다.

## TGI [[TGI]]

[네이티브로 구현된 모델](https://huggingface.co/docs/text-generation-inference/supported_models)이 아니더라도 TGI로 Transformers 구현 모델을 서빙할 수 있습니다. TGI에서 제공하는 일부 고성능 기능은 지원하지 않을 수 있지만 연속 배칭이나 스트리밍과 같은 기능들은 사용할 수 있습니다.

> [!TIP]
> 더 자세한 내용은 [논-코어 모델 서빙](https://huggingface.co/docs/text-generation-inference/basic_tutorials/non_core_models) 가이드를 참고하세요.

TGI 모델을 서빙하는 방식과 동일한 방식으로 Transformer 구현 모델을 서빙할 수 있습니다.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id gpt2
```

커스텀 Transformers 모델을 서빙하려면 `--trust-remote_code`를 명령어에 추가하세요.

```docker
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id <CUSTOM_MODEL_ID> --trust-remote-code
```

## vLLM [[vLLM]]

[vLLM](https://docs.vllm.ai/en/latest/index.html)은 특정 모델이 vLLM에서 [네이티브로 구현된 모델](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models)이 아닐 경우, Transformers 구현 모델을 서빙할 수도 있습니다. 

Transformers 구현에서는 양자화, LoRA 어댑터, 분산 추론 및 서빙과 같은 다양한 기능이 지원됩니다.

> [!TIP]
> [Transformers fallback](https://docs.vllm.ai/en/latest/models/supported_models.html#transformers-fallback) 섹션에서 더 자세한 내용을 확인할 수 있습니다.

기본적으로 vLLM은 네이티브 구현을 서빙할 수 있지만, 해당 구현이 존재하지 않으면 Transformers 구현을 사용합니다. 하지만 `--model-impl transformers` 옵션을 설정하면 명시적으로 Transformers 모델 구현을 사용할 수 있습니다.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers \
```

`trust-remote-code` 파라미터를 추가해 원격 코드 모델 로드를 활성화할 수 있습니다.

```shell
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --task generate \
    --model-impl transformers \
    --trust-remote-code \
```