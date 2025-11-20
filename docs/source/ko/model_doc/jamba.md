<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# Jamba[[jamba]]

[Jamba](https://huggingface.co/papers/2403.19887)는 Transformer와 Mamba 기반의 하이브리드 전문가 혼합(MoE) 언어 모델로, 총 매개변수 수는 52B에서 398B까지 다양합니다. 이 모델은 Transformer 모델의 성능과 Mamba와 같은 상태 공간 모델의 효율성 및 긴 컨텍스트 처리 능력(256K 토큰)을 모두 활용하는 것을 목표로 합니다.

Jamba의 아키텍처는 블록과 레이어 기반 구조를 사용하여 Transformer와 Mamba 아키텍처를 통합할 수 있도록 설계되었습니다. 각 Jamba 블록은 어텐션 레이어 또는 Mamba 레이어 중 하나와 그 뒤를 잇는 다층 퍼셉트론(MLP)으로 구성되어 있습니다. Transformer 레이어는 8개의 레이어 중 하나의 비율로 주기적으로 배치됩니다. 또한 모델 용량을 확장하기 위해 MoE 레이어가 혼합되어 있습니다.

모든 원본 Jamba 체크포인트는 [AI21](https://huggingface.co/ai21labs) 조직에서 확인할 수 있습니다.

> [!TIP]
> 오른쪽 사이드바에 있는 Jamba 모델을 누르면 다양한 언어 작업에 Jamba를 적용하는 예제를 더 확인할 수 있습니다.

아래 예제는 [`Pipeline`]과 [`AutoModel`], 그리고 커맨드라인을 통해 텍스트를 생성하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
# 최적화된 Mamba 구현 설치
# !pip install mamba-ssm causal-conv1d>=1.2.0
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="ai21labs/AI21-Jamba-Mini-1.6",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "ai21labs/AI21-Jamba-Large-1.6",
)
model = AutoModelForCausalLM.from_pretrained(
    "ai21labs/AI21-Jamba-Large-1.6",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")

output = model.generate(**input_ids, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model ai21labs/AI21-Jamba-Mini-1.6 --device 0
```

</hfoption>
</hfoptions>

양자화는 가중치를 더 낮은 정밀도로 표현하여 대규모 모델의 메모리 부담을 줄여줍니다. 사용할 수 있는 다양한 양자화 백엔드에 대해서는 [Quantization](../quantization/overview)를 참고하세요.

아래 예제는 [bitsandbytes](../quantization/bitsandbytes)를 사용하여 가중치만 8비트로 양자화하는 방법을 보여줍니다.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_skip_modules=["mamba"])

# 모델을 8개의 GPU에 고르게 분산시키기 위한 디바이스 맵
device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 2, 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 3, 'model.layers.28': 3, 'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.layers.32': 3, 'model.layers.33': 3, 'model.layers.34': 3, 'model.layers.35': 3, 'model.layers.36': 4, 'model.layers.37': 4, 'model.layers.38': 4, 'model.layers.39': 4, 'model.layers.40': 4, 'model.layers.41': 4, 'model.layers.42': 4, 'model.layers.43': 4, 'model.layers.44': 4, 'model.layers.45': 5, 'model.layers.46': 5, 'model.layers.47': 5, 'model.layers.48': 5, 'model.layers.49': 5, 'model.layers.50': 5, 'model.layers.51': 5, 'model.layers.52': 5, 'model.layers.53': 5, 'model.layers.54': 6, 'model.layers.55': 6, 'model.layers.56': 6, 'model.layers.57': 6, 'model.layers.58': 6, 'model.layers.59': 6, 'model.layers.60': 6, 'model.layers.61': 6, 'model.layers.62': 6, 'model.layers.63': 7, 'model.layers.64': 7, 'model.layers.65': 7, 'model.layers.66': 7, 'model.layers.67': 7, 'model.layers.68': 7, 'model.layers.69': 7, 'model.layers.70': 7, 'model.layers.71': 7, 'model.final_layernorm': 7, 'lm_head': 7}
model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba-Large-1.6",
                                             dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                                             quantization_config=quantization_config,
                                             device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Large-1.6")

messages = [
   {"role": "system", "content": "You are an ancient oracle who speaks in cryptic but wise phrases, always hinting at deeper meanings."},
   {"role": "user", "content": "Hello!"},
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)

outputs = model.generate(input_ids, max_new_tokens=216)

# 출력 디코딩
conversation = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 어시스턴트의 응답만 추출
assistant_response = conversation.split(messages[-1]['content'])[1].strip()
print(assistant_response)
# 출력: Seek and you shall find. The path is winding, but the journey is enlightening. What wisdom do you seek from the ancient echoes?
```

## 참고[[notes]]

- 모델 성능 저하를 방지하기 위해 Mamba 블록은 양자화하지 마세요.
- 최적화된 Mamba 커널 없이 Mamba를 사용하면 지연 시간이 크게 증가하므로 권장되지 않습니다. 그래도 커널 없이 Mamba를 사용하고자 한다면 [`~AutoModel.from_pretrained`]에서 `use_mamba_kernels=False`로 설정하세요.

  ```py
  import torch
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba-1.5-Large",
                                               use_mamba_kernels=False)
  ```

## JambaConfig[[transformers.JambaConfig]]

[[autodoc]] JambaConfig


## JambaModel[[transformers.JambaModel]]

[[autodoc]] JambaModel
    - forward


## JambaForCausalLM[[transformers.JambaForCausalLM]]

[[autodoc]] JambaForCausalLM
    - forward


## JambaForSequenceClassification[[transformers.JambaForSequenceClassification]]

[[autodoc]] transformers.JambaForSequenceClassification
    - forward
