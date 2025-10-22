<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 어텐션 인터페이스[[attention-interface]]

이 페이지에서는 `AttentionInterface`를 사용하여 지원되는 모델과 함께 사용할 사용자 지정 어텐션 함수를 등록하는 방법을 설명합니다.

## 어텐션 함수 사용자 지정[[customizing-attention-function]]

대부분의 최신 모델은 간단한 매핑 덕분에 어텐션 레이어에 사용되는 하나의 어텐션 함수에서 다른 어텐션 함수로 전환할 수 있습니다.
기본적으로 [`sdpa`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html%5D\(https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html\)),
[`flash_attention_2`](https://www.google.com/search?q=%5Bhttps://github.com/Dao-AILab/flash-attention%5D\(https://github.com/Dao-AILab/flash-attention\)) 및 [`flex_attention`](https://www.google.com/search?q=%5Bhttps://pytorch.org/docs/stable/nn.attention.flex_attention.html%23module-torch.nn.attention.flex_attention%5D\(https://pytorch.org/docs/stable/nn.attention.flex_attention.html%23module-torch.nn.attention.flex_attention\))
뿐만 아니라, 어떠한 최적화도 없는 간단한 행렬 곱셈인 `eager`에 대한 구현을 제공합니다.
이는 일반적으로 모델을 인스턴스화할 때 선택할 수 있는 설정입니다.

```python
from transformers import AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B"

# Here, using flash attention as an example
model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
```

하지만 자신만의 어텐션 함수를 만들고 싶거나, 단순히 기존 함수를 사용하여 몇 가지 구문을 추가하고 싶다면 어떻게 해야 할까요? `AttentionInterface`를 사용하여 할 수 있습니다\! 다음은 예시입니다.

```python
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch

model_id = "meta-llama/Llama-3.2-1B"

def my_new_sdpa(*args, **kwargs):
    print("I just entered the attention computation")
    return sdpa_attention_forward(*args, **kwargs)

AttentionInterface.register("my_new_sdpa", my_new_sdpa)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="my_new_sdpa")
# Try running the forward with the new attention function
model(torch.ones(1, 5, dtype=int))
```

모델의 레이어 수(이 예시에서는 16회)만큼 "I just entered the attention computation"이 출력되는 것을 보실 수 있습니다.

## 어텐션 함수 동적 전환[[dynamically-switching-attention-function]]

모델의 어텐션 함수를 동적으로 변경할 수도 있습니다.

```python
# Back to use original sdpa implementation
model.set_attn_implementation("sdpa")

model(torch.ones(1, 5, dtype=int))
```

이제 `sdpa` 어텐션을 사용하므로 문장 출력이 중지됩니다.
이를 통해 모델을 다시 로드할 필요 없이 어텐션 함수를 빠르게 변경할 수 있습니다\!

## 멀티모달 모델의 백본별 다른 어텐션[[different-attention-per-backbone-in-multimodal-models]]

멀티모달 모델에서는 각 백본 모듈에 따라 가장 효율적인 어텐션 함수가 다를 수 있습니다. 예를 들어, 일부 비전 백본은 fp32에서 더 잘 작동하지만 FlashAttention과 호환되지 않습니다. 비전 인코더를 fp32로 유지하면서 FlashAttention을 계속 사용하려면 아래와 같이 딕셔너리를 생성하고 각 config를 어텐션 구현에 매핑하세요.

```python
from transformers import AutoModelForImageTextToText

model_id = "facebook/chameleon-7b"

attention_implementation_per_backbone = {"vision_config": "sdpa", "text_config": "flash_attention_2"}
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation=attention_implementation_per_backbone)

# NOTE: keys in the attention implementation have to be the same as the sub-config names
for key in attention_implementation_per_backbone:
    assert key in model.config.sub_configs, f"Invalid key in `attention_implementation`"

# You can omit certain backbones - the default attention function (SDPA) will be used
# This is equivalent to the previous example
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation={"text_config": "flash_attention_2"})


# Set the same attention implementation for all backbones with single string, same as in non-multimodal models
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation="eager")

# Alternatively use a dict with an empty key for global configuration
model = AutoModelForImageTextToText.from_pretrained(model_id, attn_implementation={"": "eager"})
```

## 사용자 지정 어텐션 함수에 필요한 새 인수는 어떻게 처리하나요?[[what-about-new-args-needed-in-my-custom-attention-function]]

만일 새 함수가 새로운 인수가 있어야 올바르게 작동한다면 어떻게 해야 할까요? 문제 없습니다\! `AttentionInterface`를 지원하는 모델은 어텐션 레이어 및 사용되는 어텐션 함수로 kwargs를 전달합니다. kwargs(인수의 이름을 명시해야 함)를 전달함으로써 어텐션 함수에서 올바르게 사용할 수 있습니다. 그러나 사용자 지정 어텐션 함수에는 몇 가지 제한 사항이 있습니다. 그 중 하나는 다른 어텐션 함수의 시그니처와 반환 형식을 따라야 한다는 것입니다.

```python
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import torch

def custom_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: torch.Tensor,  # required arg
    value: torch.Tensor,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    a_new_kwargs = None,  # You can now add as many kwargs as you need
    another_new_kwargs = None,  # You can now add as many kwargs as you need
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    # do your magic!
    return attn_output, attn_weights  # attn_weights are optional here

AttentionInterface.register("custom", custom_attention)

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="custom")
# Forward pass with the new kwargs
model(torch.ones(1, 5, dtype=int), a_new_kwargs=..., another_new_kwargs=...)
```

주어진 모델이 어텐션 함수로 어떤 args/kwargs를 보내는지 확실하지 않다면 [GitHub](https://github.com/huggingface/transformers/tree/main/src/transformers/models)에서 해당 모델의 모델링 코드를 확인하세요\!

## 현재 사용 가능한 구현에 접근[[accessing-current-available-implementations]]

대부분의 경우 새 함수를 `register`하기만 하면 됩니다. 그러나 기존 함수에 접근하거나 몇 가지 검사를 수행해야 하는 경우, 전역 `ALL_ATTENTION_FUNCTIONS`를 사용하는 것이 좋습니다. 이는 일반적인 Python 딕셔너리와 동일하게 작동합니다.

```python
>>> from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

>>> list(ALL_ATTENTION_FUNCTIONS.keys())
>>> ['flash_attention_2', 'flex_attention', 'sdpa']

>>> ALL_ATTENTION_FUNCTIONS["sdpa"]
>>> <function transformers.integrations.sdpa_attention.sdpa_attention_forward>

>>> ALL_ATTENTION_FUNCTIONS.get("sdpa", None)
>>> <function transformers.integrations.sdpa_attention.sdpa_attention_forward>

# You can also globally `register` a new function directly on it
>>> ALL_ATTENTION_FUNCTIONS.register("new_func", new_func)
```

## 어텐션 마스크 인터페이스[[attention-mask-interface]]

새로운 어텐션 함수를 사용할 때, 쿼리 토큰이 어떤 키 및 토큰에 어텐션해야하는지 결정하기 위해 새로운 형식의 어텐션 마스크가 필요할 수 있습니다. `AttentionMaskInterface`를 통해 이것이 가능합니다\! 이는 `AttentionInterface`와 동일한 방식으로 작동합니다.



```python
from transformers import AttentionMaskInterface
from transformers.masking_utils import sdpa_mask
import torch

def my_new_sdpa_mask(*args, **kwargs):
    print("I just entered the attention mask computation")
    return sdpa_mask(*args, **kwargs)

AttentionMaskInterface.register("my_new_sdpa_mask", my_new_sdpa_mask)
```

AttentionMask를 등록해야 하는 이유는 어텐션 구현에 따라 마스크 형식을 자동으로 수정해야 하기 때문입니다(예를 들어, flex attention은 BlockMask 형식을 사용하는 반면 sdpa는 4D 텐서를 사용합니다).
어텐션 함수와 함께 어텐션 마스크 함수를 등록하지 않으면 마스크 생성이 건너뛰어지고 `attention_mask=None`이 어텐션 레이어로 전달됩니다.

어텐션 마스크 함수의 기본 시그니처는 다음과 같습니다.

```python
def custom_attention_mask(
    batch_size: int,  # required arg
    cache_position: torch.Tensor,  # required arg
    kv_length: int,  # required arg
    kv_offset: int = 0,  # required arg
    mask_function: Callable = causal_mask_function,  # required arg
    attention_mask: Optional[torch.Tensor] = None,  # required arg
    **kwargs,  # a few additional args may be passed as kwargs, especially the model's config is always passed
) -> Optional[torch.Tensor]:
```

이는 주로 `mask_function` 덕분에 작동하며, 이 함수는 [torch의 mask\_mod 함수](https://pytorch.org/blog/flexattention/) 형태의 `Callable`로, 4개의 인덱스를 입력으로 받아 해당 위치가 어텐션 계산에 참여해야 하는지 여부를 나타내는 boolean 값을 반환합니다.

어떤 이유로든 `mask_function`을 사용하여 마스크를 생성할 수 없는 경우, [torch export workaround](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py)와 유사한 방식으로 문제를 해결할 수 있습니다.