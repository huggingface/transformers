<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 캐싱[[caching]]
누군가와 대화를 나누고 있는데, 상대방이 이전에 했던 말을 기억하지 못하고 당신이 대답할 때마다 처음부터 다시 시작해야 한다고 상상해 보세요. 이는 느리고 비효율적이겠죠?

이 비유를 트랜스포머 모델에도 적용할 수 있습니다. 자기회귀 모델의 생성은 한 번에 하나의 토큰씩 예측하기 때문에 느릴 수 있습니다. 각각의 새로운 예측은 이전의 모든 문맥에 의존합니다.

1000번째 토큰을 예측하려면, 모델은 이전 999개 토큰의 정보가 필요합니다. 이 정보는 각 토큰 표현들 사이의 행렬 곱을 통해 표현됩니다.

1001번째 토큰을 예측하려면, 이전 999개 토큰의 동일한 정보에 더하여 1000번째 토큰의 정보도 필요합니다. 이렇게 되면 토큰마다 모델은 반복적으로 많은 행렬 연산을 수행해야 합니다!

이러한 비효율성을 제거하기 위해 KV 캐시(Key-Value Cache)를 사용합니다. 어텐션 레이어에서 이전에 처리한 토큰으로부터 얻은 키와 값 쌍을 저장해두고, 이후 토큰 예측 시 이를 재사용하여 연산을 줄이는 방식입니다.

> [!WARNING]
> 캐싱은 **추론**에만 사용해야 합니다. 학습 중에 활성화되면 예상치 못한 오류가 발생할 수 있습니다.

캐싱이 어떻게 그리고 왜 작동하는지 더 잘 이해하기 위해, 어텐션 행렬의 구조를 자세히 살펴보겠습니다.

## 어텐션 행렬[[attention-matrices]]

**스케일드 닷-프로덕트 어텐션**은 배치 크기 `b`, 어텐션 헤드 수 `h`, 현재까지의 시퀀스 길이 `T`, 어텐션 헤드당 차원 `d_head`에 대해 아래와 같이 계산됩니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_{\text{head}}}} \times \text{mask} \right) V
$$

쿼리(`Q`), 키(`K`), 값(`V`) 행렬은 `(b, h, T, d_head)` 형태의 입력 임베딩에서의 투영입니다.

인과적 어텐션의 경우, 마스크는 모델이 미래 토큰에 어텐션 하는 것을 방지합니다. 토큰이 한 번 처리되면, 그 표현은 미래 토큰과 관련하여 절대 변하지 않습니다. 이는 \\( K_{\text{past}} \\)와 \\( V_{\text{past}} \\)를 캐시하여 마지막 토큰의 표현을 계산하는 데 재사용할 수 있음을 의미합니다.

$$
\text{Attention}(q_t, [\underbrace{k_1, k_2, \dots, k_{t-1}}_{\text{cached}}, k_{t}], [\underbrace{v_1, v_2, \dots, v_{t-1}}_{\text{cached}}, v_{t}])
$$

추론 시에는 다음 토큰 \\( t+1 \\)을 예측하는 표현 \\( x_t \\)를 계산하기 위해 마지막 토큰의 쿼리만 필요합니다. 단계에서 새로운 키와 값 벡터가 캐시에 **저장**되고 과거 키와 값에 **추가**됩니다.

$$
K_{\text{cache}} \leftarrow \text{concat}(K_{\text{past}}, k_t), \quad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{past}}, v_t)
$$

어텐션은 모델의 각 레이어에서 독립적으로 계산되며, 캐싱은 레이어별로 수행됩니다.

캐싱이 효율성을 어떻게 개선하는지 비교한 아래 표를 참조하세요.

| 캐싱 없음 | 캐싱 사용 |
|---|---|
| 단계마다 이전의 모든 `K`와 `V`를 재계산  | 단계마다 현재의 `K`와 `V`만 계산 |
| 단계당 어텐션 비용이 시퀀스 길이에 대해 **제곱** | 단계당 어텐션 비용이 시퀀스 길이에 대해 **선형** (메모리는 선형적으로 증가하지만, 토큰당 계산은 낮게 유지됨) |



## 캐시 클래스[[cache-class]]

기본 KV 캐시 인터페이스는 현재 토큰의 키와 값 텐서를 받아서 업데이트된 `K`와 `V` 텐서를 반환합니다. 이는 모델의 `forward` 메소드에 의해 내부적으로 관리됩니다.

```py
new_K, new_V = cache.update(k_t, v_t, layer_idx)
attn_output = attn_layer_idx_fn(q_t, new_K, new_V)
```

Transformers의 [`Cache`] 클래스를 사용할 때, 셀프 어텐션 모듈은 과거와 현재 정보를 통합하기 위해 몇 가지 중요한 단계를 수행합니다.

1. 어텐션 모듈은 현재 kv 쌍을 캐시에 저장된 과거 kv 쌍과 연결합니다. 이는 `(new_tokens_length, past_kv_length + new_tokens_length)` 형태의 어텐션 가중치를 생성합니다. 현재와 과거 kv 쌍이 본질적으로 결합해 어텐션 점수를 계산하며, 모델이 이전 문맥과 현재 입력을 인식하도록 보장합니다.

2. `forward` 메소드가 반복적으로 호출될 때, 어텐션 마스크 형태가 과거와 현재 kv 쌍의 결합된 길이와 일치하는 것이 중요합니다. 어텐션 마스크는 `(batch_size, past_kv_length + new_tokens_length)` 형태여야 합니다. 이는 일반적으로 [`~GenerationMixin.generate`]에서 내부적으로 처리되지만, [`Cache`]로 자체 생성 루프를 구현하고 싶다면 이를 염두에 두세요! 어텐션 마스크는 과거와 현재 토큰값을 보유해야 합니다.

3. `cache_position`을 인식하는 것도 중요합니다. 이는 유효한 `cache_position` 값을 전달해야 하므로 `forward` 메소드로 미리 채워진 [`Cache`]를 재사용하고 싶을 때 중요합니다. 이는 시퀀스에서의 입력 위치를 나타냅니다. `cache_position`은 패딩에 영향받지 않으며, 각 토큰에 대해 항상 하나씩 더 많은 위치를 추가합니다. 예를 들어, kv 캐시가 10개의 토큰을 포함하면 - 패드 토큰과 관계없이 - 다음 토큰의 캐시 위치는 `torch.tensor([10])`이어야 합니다.

## 캐시 저장소 구현[[cache-storage-implementation]]

캐시는 각 레이어가 key와 value 캐시를 포함하는 레이어 목록 형태로 구성되어 있습니다. key 및 value 캐시는 `[batch_size, num_heads, seq_len, head_dim]` 형태의 텐서입니다.

레이어는 서로 다른 타입일 수 있으며(예: `DynamicLayer`, `StaticLayer`, `StaticSlidingWindowLayer`), 이는 주로 시퀀스 길이를 어떻게 처리하고 캐시를 어떻게 갱신하는지에 따라 달라집니다.

가장 단순한 형태는 `DynamicLayer`로, 더 많은 토큰이 처리됨에 따라 점진적으로 확장됩니다. 시퀀스 길이 차원(`seq_len`)은 새로운 토큰이 추가될 때마다 증가합니다:

```py
cache.layers[idx].keys = torch.cat([cache.layers[idx].keys, key_states], dim=-2)
cache.layers[idx].values = torch.cat([cache.layers[idx].values, value_states], dim=-2)
```

`StaticLayer`나 `StaticSlidingWindowLayer`와 같은 다른 레이어 타입은 캐시가 생성될 때 고정된 시퀀스 길이를 가지며, 이는 `torch.compile`과 호환되도록 만듭니다. `StaticSlidingWindowLayer`의 경우, 새로운 토큰이 추가되면 기존 토큰은 캐시에서 제거됩니다.

아래 예제는 [`DynamicCache`]로 생성 루프를 만드는 방법을 보여줍니다. 논의된 바와 같이, 어텐션 마스크는 과거와 현재 토큰값의 연결이며 다음 토큰을 위해 캐시 위치에 `1`이 추가됩니다.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, infer_device

device = f"{infer_device()}:0"

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

past_key_values = DynamicCache(config=model.config)
messages = [{"role": "user", "content": "Hello, what's your name."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

generated_ids = inputs.input_ids
cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
    # 탐욕적 기법으로 다음 토큰 하나를 샘플링
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # 처리되지 않은 토큰을 남겨두어 다음 생성 단계를 위한 입력을 준비합니다. 우리의 경우 새로운 토큰 하나만 존재합니다.
    # 위에서 설명한 대로 새로운 토큰을 위해 어텐션 마스크를 확장합니다
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    cache_position = cache_position[-1:] + 1 # 다음 토큰을 위해 하나 더 위치 추가

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
```

## 캐시 위치[[cache-position]]

캐시 위치는 어텐션 캐시에서 새로운 토큰을 삽입할 위치를 추적합니다. 이는 패딩이나 배치 구조와 무관하게 컨텍스트 내에서 각 토큰의 절대적 위치를 나타냅니다. 이미 `N`개의 토큰을 캐시했고 현재 `K`개의 새로운 토큰을 처리하고 있다고 가정하겠습니다. 새로운 토큰에 대한 캐시 위치는 `N`부터 `N + K - 1`까지의 범위가 됩니다. 즉, `[N, N + 1, N + 2, ..., N + K - 1]` 위치의 토큰들을 처리하는 것입니다.

캐시 위치는 내부적으로 두 가지 목적으로 사용됩니다:

1. 입력 시퀀스에서 처리할 새로운 토큰을 선택하고, 아직 캐시되지 않은 토큰만 모델의 `forward`에 전달되도록 보장합니다.
2. 키/값 쌍을 캐시의 올바른 위치에 저장합니다. 이는 특정 캐시 길이를 미리 할당하는 [`StaticCache`]와 같은 고정 크기 캐시에서 특히 중요합니다.

생성 루프는 일반적으로 캐시 위치를 관리하지만, 사용자 정의 생성 메소드를 작성할 때는 캐시 위치가 정확해야 합니다. 캐시 위치는 고정된 슬롯에 키/값 상태를 읽고 쓰는 데 사용되기 때문입니다.


```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, infer_device

device = f"{infer_device()}:0"

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "You are a helpful assistant."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=10)

```


## 레거시 캐시 형식[[legacy-cache-format]]

[`Cache`] 클래스 이전에는 캐시가 텐서의 튜플의 튜플로 저장되었습니다. 이 형식은 텍스트가 생성됨에 따라 증가하기 때문에 동적이며, [`DynamicCache`]와 유사합니다.

레거시 형식은 본질적으로 동일한 데이터 구조이지만 다르게 조직화되었습니다.
- 각 내부 튜플은 레이어의 키와 값 텐서를 포함하는 튜플의 튜플입니다.
- 텐서는 동일한 형태 `[batch_size, num_heads, seq_len, head_dim]`를 갖습니다.
- 이 형식은 덜 유연하며 양자화나 오프로딩과 같은 기능을 지원하지 않습니다.

프로젝트가 이 레거시 형식에 의존한다면, [`~DynamicCache.from_legacy_cache`]를 사용하여 [`DynamicCache`]로 변환하는 것을 권장합니다. 레거시 캐시 형식은 사용이 중단되었으며 `Transformers`에서 더 이상 사용되지 않습니다. 특정 형식에서 캐시를 조작하는 커스텀 로직이 있는 경우 도움이 되는 [`DynamicCache.to_legacy_cache`] 함수를 사용하여 튜플 형식으로 다시 변환할 수 있습니다.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

# 캐시를 반환하려면 `return_dict_in_generate=True`가 필요하고 `return_legacy_cache`는 반환된 캐시를
# 레거시 형식으로 강제합니다
generation_outputs = model.generate(**inputs, return_dict_in_generate=True, return_legacy_cache=True, max_new_tokens=5)

cache = DynamicCache.from_legacy_cache(generation_outputs.past_key_values)
legacy_format_cache = cache.to_legacy_cache()
```
