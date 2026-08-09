<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 모델 구성 요소 맞춤 설정하기[[customizing-model-components]]

모델을 완전히 새로 작성하는 대신 구성 요소를 수정하여 모델을 맞춤 설정하는 방법이 있습니다. 이 방법으로 모델을 특정 사용 사례에 맞게 모델을 조정할 수 있습니다. 예를 들어, 새로운 레이어를 추가하거나 아키텍처의 어텐션 메커니즘을 최적화할 수 있습니다. 이러한 맞춤 설정은 트랜스포머 모델에 직접 적용되므로, [`Trainer`], [`PreTrainedModel`] 및 [PEFT](https://huggingface.co/docs/peft/en/index) 라이브러리와 같은 기능을 계속 사용할 수 있습니다.

이 가이드에서는 모델의 어텐션 메커니즘을 맞춤 설정하여 [Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora)를 적용하는 방법을 설명합니다.

> [!TIP]
> 모델 코드를 반복적으로 수정하고 개발할 때 [clear_import_cache](https://github.com/huggingface/transformers/blob/9985d06add07a4cc691dc54a7e34f54205c04d40/src/transformers/utils/import_utils.py#L2286) 유틸리티가 매우 유용합니다. 이 기능은 캐시된 모든 트랜스포머 모듈을 제거하여 Python이 환경을 재시작하지 않고도 수정된 코드를 다시 가져올 수 있도록 합니다.
>
> ```py
> from transformers import AutoModel
> from transformers.utils.import_utils import clear_import_cache
>
> model = AutoModel.from_pretrained("bert-base-uncased")
> # 모델 코드 수정
> # 캐시를 지워 수정된 코드를 다시 가져오기
> clear_import_cache()
> # 업데이트된 코드를 사용하기 위해 다시 가져오기
> model = AutoModel.from_pretrained("bert-base-uncased")
> ```

## 어텐션 클래스[[attention-class]]

[Segment Anything](./model_doc/sam)은 이미지 분할 모델로, 어텐션 메커니즘에서 query-key-value(`qkv`) 프로젝션을 결합합니다. 학습 가능한 파라미터 수와 연산 부담을 줄이기 위해 `qkv` 프로젝션에 LoRA를 적용할 수 있습니다. 이를 위해서는 `qkv` 프로젝션을 분리하여 `q`와 `v`에 LoRA를 개별적으로 적용해야 합니다.

1. 원래의 `SamVisionAttention` 클래스를 상속하여 `SamVisionAttentionSplit`이라는 사용자 정의 어텐션 클래스를 만듭니다. `__init__`에서 결합된 `qkv`를 삭제하고, `q`, `k`, `v`를 위한 개별 선형 레이어를 생성합니다.

```py
import torch
import torch.nn as nn
from transformers.models.sam.modeling_sam import SamVisionAttention

class SamVisionAttentionSplit(SamVisionAttention, nn.Module):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        # 결합된 qkv 제거
        del self.qkv
        # q, k, v 개별 프로젝션 생성
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)
```

2. `_split_qkv_load_hook` 함수는 모델을 가져올 때, 사전 훈련된 `qkv` 가중치를 `q`, `k`, `v`로 분리하여 사전 훈련된 모델과의 호환성을 보장합니다.

```py
    def split_q_k_v_load_hook(self, state_dict, prefix, *args):
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if "qkv." in key:
                # 결합된 프로젝션에서 q, k, v 분리
                q, k, v = state_dict[key].chunk(3, dim=0)
                # 개별 q, k, v 프로젝션으로 대체
                state_dict[key.replace("qkv.", "q.")] = q
                state_dict[key.replace("qkv.", "k.")] = k
                state_dict[key.replace("qkv.", "v.")] = v
                # 기존 qkv 키를 삭제 대상으로 표시
                keys_to_delete.append(key)
        
        # 기존 qkv 키 제거
        for key in keys_to_delete:
            del state_dict[key]
```

3. `forward` 단계에서 `q`, `k`, `v`는 개별적으로 계산되며, 어텐션 메커니즘의 나머지 부분은 동일하게 유지됩니다.

```py
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        qkv_shapes = (batch_size *  self.num_attention_heads,  height * width, -1)
        query = self.q(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        key = self.k(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        value = self.v(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        return outputs
```

사용자 정의 `SamVisionAttentionSplit` 클래스를 원본 모델의 `SamVisionAttention` 모듈에 할당하여 교체합니다. 모델 내 모든 `SamVisionAttention` 인스턴스는 분리된 어텐션 버전으로 대체됩니다.

[`~PreTrainedModel.from_pretrained`]로 모델을 가져오세요.

```py
from transformers import SamModel

# 사전 훈련된 SAM 모델 가져오기
model = SamModel.from_pretrained("facebook/sam-vit-base")

# 비전-인코더 모듈에서 어텐션 클래스 교체
for layer in model.vision_encoder.layers:
    if hasattr(layer, "attn"):
        layer.attn = SamVisionAttentionSplit(model.config.vision_config, model.config.vision_config.window_size)
```

## LoRA[[lora]]

분리된 `q`, `k`, `v` 프로젝션을 사용할 때 , `q`와 `v`에 LoRA를 적용합니다.

[LoraConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig)를 생성하고, 랭크 `r`, `lora_alpha`, `lora_dropout`, `task_type`, 그리고 가장 중요한 적용될 모듈을 지정합니다.

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    # q와 v에 LoRA 적용
    target_modules=["q", "v"],
    lora_dropout=0.1,
    task_type="FEATURE_EXTRACTION"
)
```

모델과 [LoraConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig)를 [get\_peft\_model](https://huggingface.co/docs/peft/package_reference/peft_model#peft.get_peft_model)에 전달하여 모델에 LoRA를 적용합니다.

```py
model = get_peft_model(model, config)
```

[print_trainable_parameters](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftMixedModel.print_trainable_parameters)를 호출하여 전체 파라미터 수 대비 훈련되는 파라미터 수를 확인하세요.

```py
model.print_trainable_parameters()
"trainable params: 589,824 || all params: 94,274,096 || trainable%: 0.6256"
```
