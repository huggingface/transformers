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
*Meta는 이 모델을 2025-04-05에 출시하고 같은 날 Hugging Face Transformers에 추가했습니다.*

# Llama4[[llama4]]


<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

Meta에서 개발한 [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)는 새로운 자기회귀 Mixture-of-Experts (MoE) 아키텍처를 도입합니다.
이 세대는 두 가지 모델로 나뉩니다:
- 128개의 전문가(expert)를 사용하여 총 약 400B 매개변수 중 17B 활성 매개변수를 갖는 고성능 Llama 4 Maverick
- 16개의 전문가만 사용하여 총 약 109B 매개변수 중 17B 활성 매개변수를 갖는 경량화된 Llama 4 Scout
-
두 모델 모두 네이티브 멀티모달을 위한 초기 융합(early fusion)을 활용하여 텍스트와 이미지 입력을 처리할 수 있습니다.
Maverick과 Scout 모두 200개 언어를 포함하는 데이터에서 최대 40조개의 토큰으로 훈련되었습니다.
(아랍어, 스페인어, 독일어, 힌디어를 포함한 12개 언어에 대한 특정 미세 조정 지원 포함)

Meta는 Llama 4 Scout을 누구나 쉽게 사용할 수 있도록 설계했습니다. Scout은 4비트 또는 8비트 양자화를 적용하면 단일 서버급 GPU에서도 실시간으로 실행할 수 있습니다. 반면, 더 대규모인 Llama 4 Maverick은 고성능 연산을 위해 BF16과 FP8 형식으로 제공합니다.
이 모델들은 모델 저장소에서 제공되는 사용자 지정 Llama 4 커뮤니티 라이선스 계약에 따라 출시됩니다.

모든 원본 Llama 체크포인트는 hugging face [meta-llama](https://huggingface.co/meta-llama) 페이지에서 확인하실 수 있습니다.

> [!TIP]
> Llama 4 모델 패밀리는 두 가지 형태로 제공됩니다: 109B와 402B 매개변수입니다. 이 두 형태 모두 매우 큰 모델이며
> 일반적인 기기에서는 실행할 수 없습니다. 아래에 메모리 사용량을 줄이는 방법 몇 가지를 정리했습니다.
>
> 더욱 빠르고 안정적인 다운로드를 위해 `hf_xet` 종속성 설치를 권장합니다:
> `pip install transformers[hf_xet]`

아래 예시들은 [`Pipeline`] 또는 [`AutoModel`]로 생성하는 방법을 보여줍니다. 또한 일부 Llama 4 변형이
최대 1천만 토큰의 컨텍스트 길이를 갖기 때문에, 매우 긴 컨텍스트 생성을 활성화하기 위해 올바른 속성을 토글하는 방법을 보여주는 예시도 추가했습니다.


<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

messages = [
    {"role": "user", "content": "마요네즈 레시피가 무엇인가요?"},
]

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    dtype=torch.bfloat16
)

output = pipe(messages, do_sample=False, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
```

</hfoption>
<hfoption id="AutoModel - Text only">

```py
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "당신은 누구신가요?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```

</hfoption>
<hfoption id="AutoModel - Multimodal">

```py
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_url},
            {"type": "text", "text": "이 이미지를 두 문장으로 설명해주세요."},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
```

</hfoption>
<hfoption id="AutoModel - Multimodal with multiple images">

```py
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {"type": "text", "text": "이 두 이미지가 어떻게 비슷하고, 어떻게 다른지 설명해주실 수 있나요?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
```

</hfoption>
<hfoption id="AutoModel - Long context">

주의: 아래 예시는 `device_map="auto"`와 flex-attention을 모두 사용합니다.
이 예시를 텐서 병렬 모드로 실행하려면 `torchrun`을 사용하세요.

향후 텐서 병렬 없이 `device_map="auto"`와 flex-attention을 함께 실행할 수 있도록
작업할 예정입니다.

```py
from transformers import Llama4ForConditionalGeneration, AutoTokenizer, infer_device
import torch
import time

file = "very_long_context_prompt.txt"
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

with open(file, "r") as f:
    very_long_text = "\n".join(f.readlines())

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flex_attention",
    dtype=torch.bfloat16
)

messages = [
    {"role": "user", "content": f"다음 텍스트들을 보세요: [{very_long_text}]\n\n\n\n책들은 무엇이며, 누가 썼나요? 좋은 목록을 만들어주세요."},
]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

device = infer_device()
torch_device_module = getattr(torch, device, torch.cuda)
torch_device_module.synchronize()
start = time.time()
out = model.generate(
    input_ids.to(model.device),
    prefill_chunk_size=2048*8,
    max_new_tokens=300,
    cache_implementation="hybrid",
)
print(time.time()-start)
print(tokenizer.batch_decode(out[:, input_ids.shape[-1]:]))
print(f"{torch_device_module.max_memory_allocated(model.device) / 1024**3:.2f} GiB")
```

</hfoption>
</hfoptions>

## 효율성; Llama 4의 최대 성능 활용하기[[efficiency-how-to-get-the-best-out-of-llama-4]]

### 어텐션 방법[[the-attention-methods]]

기본 설정으로 주어지는 어텐션 함수를 변경하면 계산 성능과 메모리 사용량을 크게 개선할 수 있습니다. 인터페이스에 대한 자세한 설명은 [어텐션 인터페이스](../attention_interface) 개요를 참조하세요.

Llama 4 모델은 처음 공개될 때부터 다음 어텐션 방식을 지원합니다: `eager`, `flex_attention`, `sdpa`. 최상의 결과를 위해 `flex_attention` 사용을 권장합니다.
어텐션 메커니즘 전환은 모델을 초기화할 때 이루어집니다:


<hfoptions id="Attention">
<hfoption id="Flex Attention">

Flex Attention은 모델이 긴 컨텍스트를 처리할 때 최적의 성능을 발휘합니다.

> [!TIP] 주의: 아래 예시는 `device_map="auto"`와 flex-attention을 모두 사용합니다.
> 이 예시를 텐서 병렬 모드로 실행하려면 `torchrun`을 사용하세요.
>
> 향후 텐서 병렬 없이 `device_map="auto"`와 flex-attention을 함께 실행할 수 있도록
> 작업할 예정입니다.

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    dtype=torch.bfloat16,
)
```
</hfoption>
<hfoption id="SDPA">
`sdpa` 어텐션 방법은 일반적으로 `eager` 방법보다 계산 효율적입니다.

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    device_map="auto",
    dtype=torch.bfloat16,
)
```
</hfoption>
<hfoption id="Eager">
`eager` 어텐션 방법이 기본으로 설정되어 있으므로 모델 로드 시 다른 설정이 필요하지 않습니다:

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)
```
</hfoption>
</hfoptions>


### 양자화[[quantization]]

양자화는 가중치를 더 낮은 정밀도로 바꿔 대형 모델의 메모리 부담을 줄입니다. 사용 가능한 양자화 백엔드에 대해서는 [양자화](../quantization/overview) 개요를 참조하세요.
현재는 FBGEMM과 LLM-Compressor를 지원하며, 곧 더 많은 방식이 추가될 예정입니다.

두 가지 방법을 사용하는 예시를 아래에서 확인하세요:



다음은 FBGEMM 접근법을 사용하여 BF16 모델을 FP8로 로드하는 예시입니다:

<hfoptions id="Quantization">
<hfoption id="FBGEMM">

```python
from transformers import AutoTokenizer, Llama4ForConditionalGeneration, FbgemmFp8Config
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "당신은 누구신가요?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=FbgemmFp8Config()
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```

</hfoption>
<hfoption id="LLM-Compressor">

LLLM-Compressor를 사용할 때는 함께 제공되는 사전 양자화된 FP8 체크포인트를 쓰는 것이 좋습니다:

```python
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "당신은 누구신가요?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    tp_plan="auto",
    dtype=torch.bfloat16,
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```
</hfoption>
</hfoptions>

### 오프로딩[[offloading]]

CPU 오프로딩을 활성화하면, GPU 메모리가 부족할 때 모델이 구성 요소를 CPU로 이동시킵니다.
추론 시 다양한 구성 요소들이 GPU와 CPU 간에 동적으로 로드되고 언로드됩니다. 이를 통해 CPU 메모리가 충분한 한 더 작은 머신에서도 모델을 로드할 수 있습니다.
다만 통신 오버헤드로 인해 추론 속도가 느려질 수 있습니다.

CPU 오프로딩을 활성화하려면 모델 로드 시 `device_map`을 `auto`로 지정하면 됩니다

```py
from transformers import Llama4ForConditionalGeneration
import torch

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)
```

## Llama4Config

[[autodoc]] Llama4Config

## Llama4TextConfig

[[autodoc]] Llama4TextConfig

## Llama4VisionConfig

[[autodoc]] Llama4VisionConfig

## Llama4Processor

[[autodoc]] Llama4Processor

## Llama4ImageProcessorFast

[[autodoc]] Llama4ImageProcessorFast

## Llama4ForConditionalGeneration

[[autodoc]] Llama4ForConditionalGeneration
- forward

## Llama4ForCausalLM

[[autodoc]] Llama4ForCausalLM
- forward

## Llama4TextModel

[[autodoc]] Llama4TextModel
- forward

## Llama4ForCausalLM

[[autodoc]] Llama4ForCausalLM
- forward

## Llama4VisionModel

[[autodoc]] Llama4VisionModel
- forward