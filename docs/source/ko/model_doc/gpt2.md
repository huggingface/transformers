<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>


# GPT-2[[gpt-2]]

[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)는 GPT의 확장 버전으로, 인과적 트랜스포머 언어 모델이며, 10배 더 많은 매개변수와 학습 데이터를 가지고 있습니다. 이 모델은 이전의 모든 단어를 기반으로 다음 단어를 예측하도록 40GB 데이터 세트에서 사전 학습되었습니다. 이러한 접근 방식을 통해 이 모델은 제로샷 설정에서 많은 다운스트림 작업을 수행할 수 있게 되었습니다.

모델 아키텍처는 각 토큰이 이전 토큰에만 주의를 기울일 수 있는 단방향(인과적) 어텐션 메커니즘을 사용하므로, 텍스트 생성 작업에 특히 효과적입니다.

모든 원본 GPT-2 체크포인트는 [OpenAI community](https://huggingface.co/openai-community?search_models=gpt) 조직에서 찾을 수 있습니다.

> [!TIP]
> 오른쪽 사이드바의 GPT-2 모델을 클릭하여 GPT-2를 다양한 언어 작업에 적용하는 더 많은 예시를 확인하세요.

아래 예시는 [`Pipeline`] 또는 [`AutoModel`], 그리고 명령줄에서 GPT-2로 텍스트를 생성하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

# 텍스트 생성을 위한 파이프라인 생성
pipeline = pipeline(task="text-generation", model="openai-community/gpt2", dtype=torch.float16, device=0)
pipeline("Hello, I'm a language model")
```
</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 사전 학습된 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# 입력 텍스트를 토큰화하고 GPU로 이동
input_ids = tokenizer("Hello, I'm a language model", return_tensors="pt").to("cuda")

# 텍스트 생성
output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Hello, I'm a language model" | transformers run --task text-generation --model openai-community/gpt2 --device 0
```

</hfoption>
</hfoptions>

`transformers backend`를 사용하여 vLLM으로 모델을 서빙할 수도 있습니다.

```
vllm serve openai-community/gpt2 --model-imp transformers
```

양자화는 가중치를 더 낮은 정밀도로 표현하여 대형 모델의 메모리 부담을 줄입니다. 사용할 수 있는 더 많은 양자화 백엔드에 대해서는 [Quantization](../quantization/overview) 개요를 참조하세요.

아래 예시는 [bitsandbytes](../quantization/bitsandbytes)를 사용하여 가중치만 4비트로 양자화합니다.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# 양자화 설정 구성
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

# 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2-xl",
    quantization_config=quantization_config,
    device_map="auto"
)

# 토크나이저 로드 및 텍스트 생성
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
inputs = tokenizer("Once upon a time, there was a magical forest", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 참고사항[[notes]]

- GPT-2는 절대 위치 임베딩을 사용하므로 입력을 오른쪽에 패딩하세요.
- GPT-2는 이전에 계산된 키-값 어텐션 쌍을 재사용할 수 있습니다. [`GPT2Model.forward`]의 [past_key_values](https://huggingface.co/docs/transformers//en/model_doc/gpt2#transformers.GPT2Model.forward.past_key_values) 매개변수로 이 기능에 접근하세요.
- [Mistral](./mistral)의 학습 안정성 개선 사항을 적용하려면 [scale_attn_by_inverse_layer_idx](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.scale_attn_by_inverse_layer_idx)와 [reorder_and_upcast_attn](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.reorder_and_upcast_attn) 매개변수를 활성화하세요.

## GPT2Config

[[autodoc]] GPT2Config

## GPT2Tokenizer

[[autodoc]] GPT2Tokenizer
    - save_vocabulary

## GPT2TokenizerFast

[[autodoc]] GPT2TokenizerFast

## GPT2 특정 출력[[gpt2-specific-outputs]]

[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput

[[autodoc]] models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput

<frameworkcontent>
<pt>

## GPT2Model

[[autodoc]] GPT2Model
    - forward

## GPT2LMHeadModel

[[autodoc]] GPT2LMHeadModel
    - forward

## GPT2DoubleHeadsModel

[[autodoc]] GPT2DoubleHeadsModel
    - forward

## GPT2ForQuestionAnswering

[[autodoc]] GPT2ForQuestionAnswering
    - forward

## GPT2ForSequenceClassification

[[autodoc]] GPT2ForSequenceClassification
    - forward

## GPT2ForTokenClassification

[[autodoc]] GPT2ForTokenClassification
    - forward

</pt>
<tf>

## TFGPT2Model

[[autodoc]] TFGPT2Model
    - call

## TFGPT2LMHeadModel

[[autodoc]] TFGPT2LMHeadModel
    - call

## TFGPT2DoubleHeadsModel

[[autodoc]] TFGPT2DoubleHeadsModel
    - call

## TFGPT2ForSequenceClassification

[[autodoc]] TFGPT2ForSequenceClassification
    - call

## TFSequenceClassifierOutputWithPast

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutputWithPast

## TFGPT2Tokenizer

[[autodoc]] TFGPT2Tokenizer

</tf>
<jax>

## FlaxGPT2Model

[[autodoc]] FlaxGPT2Model
    - __call__

## FlaxGPT2LMHeadModel

[[autodoc]] FlaxGPT2LMHeadModel
    - __call__

</jax>
</frameworkcontent>