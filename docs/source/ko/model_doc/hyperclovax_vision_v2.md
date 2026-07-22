<!--Copyright 2026 NAVER Corp. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-07-21.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# HyperCLOVAX Vision V2

HyperCLOVAX Vision V2는 NAVER가 개발한 비전-언어 멀티모달 모델입니다. [HyperClovaX](./hyperclovax.md) 언어 모델 백본과 [Qwen2.5-VL](./qwen2_5_vl) 비전 인코더를 결합한 구조입니다. 텍스트, 이미지, 비디오 입력을 지원하며, 내장된 thinking 토큰(`<think>...</think>`)을 통한 연쇄 추론(chain-of-thought reasoning) 기능을 제공합니다.

원본 HyperCLOVAX-SEED-Think-32B 체크포인트는 [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) 페이지에서 확인할 수 있습니다.

아래 예시는 [`HyperCLOVAXVisionV2ForConditionalGeneration`]을 사용하여 이미지를 기반으로 텍스트를 생성하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="이미지 입력">

```python
from transformers import AutoProcessor, HyperCLOVAXVisionV2ForConditionalGeneration

model = HyperCLOVAXVisionV2ForConditionalGeneration.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

messages = [
    {
        "role": "system",
        "content": "당신은 유능한 AI 어시스턴트입니다.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "이 이미지를 설명해 주세요."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</hfoption>
<hfoption id="비디오 입력">

```python
from transformers import AutoProcessor, HyperCLOVAXVisionV2ForConditionalGeneration

model = HyperCLOVAXVisionV2ForConditionalGeneration.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

messages = [
    {
        "role": "system",
        "content": "당신은 유능한 AI 어시스턴트입니다.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": "/path/to/video.mp4",
            },
            {"type": "text", "text": "이 비디오를 설명해 주세요."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</hfoption>
</hfoptions>

양자화는 가중치를 더 낮은 정밀도로 표현하여 큰 모델의 메모리 부담을 줄여줍니다. 사용 가능한 양자화 백엔드에 대한 자세한 내용은 [양자화](../quantization/overview) 개요를 참고하세요.

아래 예시는 [bitsandbytes](../quantization/bitsandbytes)를 사용하여 모델을 4-bit로 로드합니다.

```python
from transformers import AutoProcessor, BitsAndBytesConfig, HyperCLOVAXVisionV2ForConditionalGeneration

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = HyperCLOVAXVisionV2ForConditionalGeneration.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    device_map="auto",
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
```

## 노트 [[notes]]

- 이 모델은 연쇄 추론(chain-of-thought reasoning)을 지원합니다. 기본적으로 생성 프롬프트에 빈 `<think>\n\n</think>` 블록이 추가됩니다. `<think>...</think>` 태그 내에 명시적인 추론 과정을 생성하려면 `apply_chat_template`에 `thinking=True`를 전달하세요 (이미지/텍스트 입력 한정):

    ```python
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        thinking=True,
    ).to(model.device)
    ```

- 여러 번의 대화에서 혼합 미디어(이미지, 비디오)를 사용하는 멀티턴 대화를 지원합니다. 이미지와 비디오는 여러 턴에 걸쳐 나타날 수 있습니다.

    ```python
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://example.com/image1.jpg"},
                {"type": "text", "text": "이 이미지에서 무엇이 보이나요?"},
            ],
        },
        {
            "role": "assistant",
            "content": "고양이가 소파에 앉아 있는 모습이 보입니다.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://example.com/image2.jpg"},
                {"type": "text", "text": "첫 번째 이미지와 비교해서 어떻게 다른가요?"},
            ],
        },
    ]
    ```

- 함수/도구 호출(function/tool calling)을 지원합니다. `apply_chat_template`의 `tools` 파라미터로 도구를 전달하세요:

    ```python
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "특정 위치의 현재 날씨를 가져옵니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "도시 이름"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "서울의 날씨가 어떤가요?"}
    ]

    inputs = processor.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    ```

## HyperCLOVAXVisionV2Config

[[autodoc]] HyperCLOVAXVisionV2Config

## HyperCLOVAXVisionV2Processor

[[autodoc]] HyperCLOVAXVisionV2Processor
    - __call__

## HyperCLOVAXVisionV2Model

[[autodoc]] HyperCLOVAXVisionV2Model
    - forward
    - get_image_features
    - get_video_features

## HyperCLOVAXVisionV2ForConditionalGeneration

[[autodoc]] HyperCLOVAXVisionV2ForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features

