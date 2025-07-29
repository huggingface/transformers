<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 어텐션 인터페이스 [[attention-interface]]

이 페이지는 지원되는 모델과 함께 사용자 정의 어텐션 함수를 등록하기 위해 `AttentionInterface`를 사용하는 방법을 설명합니다.

## 어텐션 함수 맞춤 설정하다 [[customizing-attention-function]]

최신 모델들은 이제 간단한 매핑 덕분에 어텐션 레이어에서 사용되는 하나의 어텐션 함수에서 다른 함수로 전환할 수 있습니다.
기본적으로 [`sdpa`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[`flash_attention_2`](https://github.com/Dao-AILab/flash-attention), [`flex_attention`](https://pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)
그리고 최적화 없이 단순한 행렬 곱셈인 `eager`의 구현을 제공합니다.
이는 모델을 인스턴스화할 때 일반적으로 선택할 수 있는 설정입니다:

```python
from transformers import AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B"

# 예시로 플래시 어텐션 사용
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
```

하지만 자신만의 어텐션 함수를 만들고 싶다면 어떨까요? 또는 기존 함수를 가지고 놀면서 여기저기 몇 줄의 명령문을 추가하려면? 이제 `AttentionInterface`로 이것이 가능합니다! 다음은 예시입니다:

```python
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch

model_id = "meta-llama/Llama-3.2-1B"

def my_new_sdpa(*args, **kwargs):
    print("어텐션 계산에 방금 들어갔습니다")
    return sdpa_attention_forward(*args, **kwargs)

AttentionInterface.register("my_new_sdpa", my_new_sdpa)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="my_new_sdpa")
# 새로운 어텐션 함수로 forward 실행 시도
model(torch.ones(1, 5, dtype=int))
```

모델의 레이어 수만큼 "어텐션 계산에 방금 들어갔습니다"가 출력되는 것을 볼 수 있습니다 (이 예시에서는 16번).

## 동적으로 어텐션 함수 전환 [[dynamically-switching-attention-function]]

모델의 어텐션 함수를 동적으로 변경할 수도 있습니다:

```python
# 원래 sdpa 구현 사용으로 되돌아가기
model.set_attn_implementation("sdpa")

model(torch.ones(1, 5, dtype=int))
```

이제 `sdpa` 어텐션을 사용하므로 명령문 출력이 중단됩니다.
이를 통해 모델을 다시 가져올 필요 없이 어텐션 함수를 빠르게 변경할 수 있습니다!

## 멀티모달 모델에서 백본별 다른 어텐션 [[different-attention-per-backbone-in-multimodal-models]]

멀티모달 모델의 경우 각 백본 모듈에 대해 다른 어텐션 함수가 더 잘 작동할 수 있습니다. 예를 들어, 일부 비전 백본은 fp32에서 더 나은 성능을 보이지만 FlashAttention과 호환되지 않습니다. 비전 인코더를 fp32로 유지하면서 FlashAttention을 계속 사용하려면, 딕셔너리를 생성하고 각 구성을 어텐션 구현에 매핑하세요:

```python
from transformers import AutoModelForImageTextToText

model_id = "facebook/chameleon-7b"

attention_implementation_per_backbone = {"vision_config": "sdpa", "text_config": "flash_attention_2"}
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation=attention_implementation_per_backbone)

# 참고: attention_implementation의 키는 서브 구성 이름과 동일해야 합니다
for key in attention_implementation_per_backbone:
    assert key in model.config.sub_configs, f"`attention_implementation`의 잘못된 키"

# 특정 백본을 생략할 수 있습니다 - 기본 어텐션 함수(SDPA)가 사용됩니다
# 이는 이전 예시와 동일합니다
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation={"text_config": "flash_attention_2"})


# 단일 문자열로 모든 백본에 동일한 어텐션 구현 설정, 비멀티모달 모델과 동일
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager")

# 또는 전역 구성을 위해 빈 키가 있는 딕셔너리 사용
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation={"": "eager"})
```

## 사용자 정의 어텐션 함수에 필요한 새 인수는 어떻게 처리하나요? [[what-about-new-args-needed-in-my-custom-attention-function]]

하지만 새 함수가 제대로 사용되기 위해 새로운 인수를 필요로 한다면 어떨까요? 문제없습니다! `AttentionInterface`를 지원하는 모델들은 kwargs를 어텐션 레이어와 사용되는 어텐션 함수까지 전파합니다. 이렇게 하면 모델의 forward에서 인수를 (kwargs로, 즉 인수의 이름을 명시해야 함) 전달할 수 있고, 어텐션에서 올바르게 사용됩니다. 하지만 사용자 정의 어텐션 함수에는 몇 가지 제한이 있습니다. 특히 다른 어텐션 함수의 시그니처와 반환 형식을 따라야 합니다:

```python
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch

def custom_attention(
    module: torch.nn.Module,  # 필수 인수
    query: torch.Tensor,  # 필수 인수
    key: torch.Tensor,  # 필수 인수
    value: torch.Tensor,  # 필수 인수
    attention_mask: Optional[torch.Tensor],  # 필수 인수
    a_new_kwargs = None,  # 이제 필요한 만큼 kwargs를 추가할 수 있습니다
    another_new_kwargs = None,  # 이제 필요한 만큼 kwargs를 추가할 수 있습니다
    **kwargs,  # 모델이 다른 인수들을 전달하므로 **kwargs를 받아야 합니다
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
    ...  # 마법을 부리세요!
    return attn_output, attn_weights  # attn_weights는 여기서 선택사항입니다

AttentionInterface.register("custom", custom_attention)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="custom")
# 새로운 kwargs로 Forward 패스
model(torch.ones(1, 5, dtype=int), a_new_kwargs=..., another_new_kwargs=...)
```

주어진 모델이 어텐션 함수에 어떤 인수/kwargs를 보내는지 확실하지 않다면, [GitHub](https://github.com/huggingface/transformers/tree/main/src/transformers/models)에서 해당 모델의 모델링 코드를 확인하세요!

## 현재 사용 가능한 구현에 접근하기 [[accessing-current-available-implementations]]

대부분의 경우 새 함수를 `register`하기만 하면 됩니다. 하지만 기존 함수에 접근하거나 몇 가지 확인을 수행해야 하는 경우, 선호되는 방법은 전역 `ALL_ATTENTION_FUNCTIONS`를 사용하는 것입니다. 이는 일반적인 Python 딕셔너리에서 기대하는 것과 같은 방식으로 작동합니다:

```python
>>> from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

>>> list(ALL_ATTENTION_FUNCTIONS.keys())
>>> ['flash_attention_2', 'flex_attention', 'sdpa']

>>> ALL_ATTENTION_FUNCTIONS["sdpa"]
>>> <function transformers.integrations.sdpa_attention.sdpa_attention_forward>

>>> ALL_ATTENTION_FUNCTIONS.get("sdpa", None)
>>> <function transformers.integrations.sdpa_attention.sdpa_attention_forward>

# 전역적으로 새 함수를 직접 `register`할 수도 있습니다
>>> ALL_ATTENTION_FUNCTIONS.register("new_func", new_func)
```

## 어텐션 마스크 인터페이스 [[attention-mask-interface]]

새로운 어텐션 함수를 사용한다는 것은 쿼리 토큰이 어떤 키와 값 토큰에 주의를 기울여야 하는지 결정하기 위해 새로운 형식의 어텐션 마스크가 필요할 수 있음을 의미합니다. 이제 `AttentionMaskInterface`로 이것이 가능합니다! `AttentionInterface`와 같은 방식으로 작동합니다:

```python
from transformers import AttentionMaskInterface
from transformers.masking_utils import sdpa_mask
import torch

def my_new_sdpa_mask(*args, **kwargs):
    print("어텐션 마스크 계산에 방금 들어갔습니다")
    return sdpa_mask(*args, **kwargs)

AttentionMaskInterface.register("my_new_sdpa_mask", my_new_sdpa_mask)
```

이를 등록해야 하는 이유는 어텐션 구현에 따라 마스크 형식을 자동으로 수정해야 하기 때문입니다 (예를 들어, flex attention은 BlockMask 형식을 사용하고 sdpa는 4D 텐서를 사용합니다).
기본적으로 어텐션 함수와 함께 어텐션 마스크 함수를 등록하지 않으면 마스크 생성이 건너뛰어지고 `attention_mask=None`이 어텐션 레이어에 전달됩니다.

어텐션 마스크 함수의 기본 시그니처는 다음과 같습니다:

```python
def custom_attention_mask(
    batch_size: int,  # 필수 인수
    cache_position: torch.Tensor,  # 필수 인수
    kv_length: int,  # 필수 인수
    kv_offset: int = 0,  # 필수 인수
    mask_function: Callable = causal_mask_function,  # 필수 인수
    attention_mask: Optional[torch.Tensor] = None,  # 필수 인수
    **kwargs,  # 몇 가지 추가 인수가 kwargs로 전달될 수 있으며, 특히 모델의 구성이 항상 전달됩니다
) -> Optional[torch.Tensor]:
```

이는 주로 `mask_function` 덕분에 작동하며, 이는 [torch의 mask_mod 함수](https://pytorch.org/blog/flexattention/) 형태의 `Callable`로 4개의 인덱스를 입력으로 받고 이 위치가 어텐션 계산에 참여해야 하는지를 나타내는 불린값을 반환합니다.

어떤 이유로 `mask_function`을 사용하여 마스크를 생성할 수 없다면, [torch export 해결방법](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py)과 유사한 작업을 수행하여 이를 해결할 수 있습니다.