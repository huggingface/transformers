<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Assisted decoding

Assisted decoding은 주 모델이 토큰을 확정하기 전에 헬퍼가 후보 토큰을 제안하도록 허용하여 텍스트 생성 속도를 높입니다. 주 모델은 한 번의 순방향 패스에서 후보 토큰을 검증합니다. 헬퍼는 빠르고 저렴하며, 주 모델의 수십 번의 더 비싼 순방향 패스를 대체할 수 있습니다.

이 가이드는 Transformers의 assisted decoding 메소드를 다룹니다.

## Speculative decoding

[Speculative decoding](https://hf.co/papers/2211.17192)은 더 작은 assistant 모델을 사용하여 후보 토큰을 작성합니다. 주 모델은 이러한 토큰을 한 번의 패스에서 확인합니다. 검증된 토큰은 최종 출력에 들어가고 거부된 토큰은 표준 샘플링을 트리거합니다. 주 모델이 더 적은 비싼 순방향 패스를 실행하기 때문에 생성이 더 빠릅니다.

이 메소드는 assistant 모델이 주 모델보다 훨씬 작고 동일한 토크나이저를 사용할 때 가장 잘 작동합니다. Speculative decoding은 greedy search와 sampling은 지원하지만 배치 입력은 지원하지 않습니다.

`assistant_model`을 [`~GenerationMixin.generate`]에 전달하세요. 토큰 검증이 실패하면 재샘플링하려면 `do_sample=True`로 설정하세요.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine'
```

`assistant_model` 인수는 [`Pipeline`] API에서도 사용할 수 있습니다.

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B",
    assistant_model="meta-llama/Llama-3.2-1B",
    dtype="auto"
)
pipeline("Hugging Face is an open-source company, ", max_new_tokens=50, do_sample=False)
```

</hfoption>
<hfoption id="sampling">

무작위성을 제어하려면 `temperature`를 설정하세요. 더 낮은 temperature는 종종 지연 시간을 개선합니다.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

</hfoption>
</hfoptions>

## Prompt lookup decoding

Prompt lookup decoding은 assistant 모델이 필요하지 않습니다. 프롬프트에서 겹치는 n-gram을 찾아 후보 토큰을 제안합니다. 일치하는 것이 없으면 일반 자기회귀 디코딩으로 대체됩니다. 후보 토큰이 종종 소스 텍스트의 로컬 패턴을 반영하기 때문에 요약 및 번역과 같은 입력 기반 작업에 적합합니다.

`prompt_lookup_num_tokens`를 [`~GenerationMixin.generate`]에 전달하세요. 이것은 알고리즘이 반복되는 패턴을 감지할 때 프롬프트의 이전 부분에서 복사하려고 시도하는 토큰 수를 설정합니다.

<hfoptions id="prompt-lookup-decoding">
<hfoption id="greedy decoding">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, prompt_lookup_num_tokens=5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

</hfoption>
<hfoption id="sampling">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, prompt_lookup_num_tokens=5, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

</hfoption>
</hfoptions>

## Self-speculative decoding

Self-speculative decoding은 모델의 중간 레이어를 assistant로 사용하여 후보 토큰을 제안합니다. 제안이 일치하면 모델은 조기에 종료되고 나머지 레이어가 토큰을 검증하거나 수정합니다.

모두 하나의 모델이기 때문에 가중치와 캐시가 공유되어 추가 메모리 오버헤드 없이 속도가 향상됩니다. 이 기술은 중간 레이어에서 조기 종료 로짓을 지원하도록 훈련된 모델에서만 작동합니다.

종료 레이어를 설정하려면 `assistant_early_exit`를 [`~GenerationMixin.generate`]에 전달하세요.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/layerskip-llama3.2-1B")
model = AutoModelForCausalLM.from_pretrained("facebook/layerskip-llama3.2-1B", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_early_exit=4, do_sample=False, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Universal assisted decoding

Universal assisted decoding (UAD)은 주 모델과 assistant 모델이 서로 다른 토크나이저를 가지고 있을 때도 speculative decoding을 가능하게 합니다. 주 모델과 임의의 작은 assistant 모델을 페어링할 수 있습니다. 후보 토큰이 재인코딩되고 알고리즘이 가장 긴 공통 부분 시퀀스를 계산하여 연속이 정렬된 상태로 유지됩니다.

UAD를 활성화하려면 `tokenizer`, `assistant_tokenizer` 및 `assistant_model`을 [`~GenerationMixin.generate`]에 전달하세요.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assistant_tokenizer = AutoTokenizer.from_pretrained("double7/vicuna-68m")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", dtype="auto")
assistant_model = AutoModelForCausalLM.from_pretrained("double7/vicuna-68m", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

## Resources

- 텍스트 생성 지연 시간 및 assisted generation에 대한 더 많은 문맥은 [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation) 블로그 포스트를 읽어보세요.