<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DBRX[[dbrx]]

## 개요[[overview]]

DBRX는 [트랜스포머 기반의](https://www.isattentionallyouneed.com/) 다음 토큰을 예측하는 디코더 전용 LLM 모델입니다.
총 132B 매개변수를 가진 *세밀한* 전문가 혼합(MoE) 아키텍처를 사용하며, 이 중 36B 매개변수가 입력마다 활성화됩니다.
12T 토큰의 텍스트와 코드 데이터로 사전 학습되었습니다.

Mixtral-8x7B와 Grok-1과 같은 다른 공개 MoE 모델들과 비교했을 때, DBRX는 더 많은 수의 작은 전문가들을 사용하는 세밀한 구조를 가지고 있습니다. DBRX는 16개의 전문가 중 4개를 선택하는 반면, Mixtral-8x7B와 Grok-1은 8개의 전문가 중 2개를 선택합니다.

이는 65배 더 많은 전문가 조합을 가능하게 하며, 이를 통해 모델의 품질이 향상되는 것을 발견했습니다.
DBRX는 회전 위치 인코딩(RoPE), 게이트 선형 유닛(GLU), 그룹 쿼리 어텐션(GQA)을 사용합니다.
BPE 기반 모델이며 [tiktoken](https://github.com/openai/tiktoken) 저장소에 설명된 GPT-4 토크나이저를 사용합니다.
이러한 선택들은 철저한 평가와 스케일링 실험을 기반으로 이루어졌습니다.

DBRX는 신중하게 선별된 12T 토큰의 데이터로 사전 학습되었으며, 최대 문맥 길이는 32K 토큰입니다.
이 데이터는 토큰 대비 MPT 계열 모델 학습에 사용된 데이터보다 최소 2배 이상 더 좋은 것으로 추정됩니다.
이 새로운 데이터셋은 데이터 처리를 위한 Apache Spark™와 Databricks 노트북, 그리고 데이터 관리와 거버넌스를 위한 Unity Catalog를 포함한 Databricks 도구 전체를 활용하여 개발되었습니다.
우리는 사전 학습을 위해 커리큘럼 학습을 사용했으며, 학습 중 데이터 믹스를 변경하는 방식이 모델 품질을 상당히 개선한다는 것을 발견했습니다.


DBRX Instruct와 DBRX Base에 대한 더 자세한 정보는 이 [기술 블로그 포스트](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)에서 확인할 수 있습니다.

이 모델은 [eitan-turok](https://huggingface.co/eitanturok)와 [abhi-db](https://huggingface.co/abhi-db)가 기여했습니다. 원본 코드는 [이곳](https://github.com/databricks/dbrx-instruct)에서 찾을 수 있지만, 최신 버전이 아닐 수 있습니다.

## 사용 예[[usage-examples]]

`generate()` 메소드는 DBRX를 사용하여 텍스트를 생성하는 데 사용될 수 있습니다. 표준 어텐션 구현, 플래시 어텐션, PyTorch의 스케일된 내적 어텐션(Scaled Dot-Product Attention)을 사용하여 생성할 수 있습니다. 후자의 두 어텐션 구현 방식은 처리 속도를 크게 높여줍니다.

```python
from transformers import DbrxForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token="YOUR_HF_TOKEN")
model = DbrxForCausalLM.from_pretrained(
    "databricks/dbrx-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token="YOUR_HF_TOKEN",
    )

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

`pip install flash-attn`를 통해 플래시 어텐션을 설치하면, 더 빠른 생성이 가능합니다. (플래시 어텐션에 대한 HuggingFace 문서는 [이곳](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)에서 확인할 수 있습니다.)


```python
from transformers import DbrxForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token="YOUR_HF_TOKEN")
model = DbrxForCausalLM.from_pretrained(
    "databricks/dbrx-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token="YOUR_HF_TOKEN",
    attn_implementation="flash_attention_2",
    )

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

PyTorch의 스케일된 내적 어텐션을 사용하여도 더 빠른 생성이 가능합니다. (스케일된 내적 어텐션에 대한 HuggingFace 문서는 [이곳](https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)에서 확인할 수 있습니다.)


```python
from transformers import DbrxForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token="YOUR_HF_TOKEN")
model = DbrxForCausalLM.from_pretrained(
    "databricks/dbrx-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token="YOUR_HF_TOKEN",
    attn_implementation="sdpa",
    )

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## DbrxConfig[[transformers.DbrxConfig]]

[[autodoc]] DbrxConfig


## DbrxModel[[transformers.DbrxModel]]

[[autodoc]] DbrxModel
    - forward


## DbrxForCausalLM[[transformers.DbrxForCausalLM]]

[[autodoc]] DbrxForCausalLM
    - forward

