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
*이 모델은 2025년 2월 20일에 출시되었으며, 동시에 허깅페이스 `Transformer` 라이브러리에 추가되었습니다.*

# 소형 비전 언어 모델(SmolVLM)[[smolvlm]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## 개요[[overview]]
[SmolVLM2](https://huggingface.co/papers/2504.05299) ([블로그 글](https://huggingface.co/blog/smolvlm2)) 은 Idefics3 모델을 개선한 버전으로, 두 가지 주요 차이점이 있습니다:

- 텍스트 모델로 SmolLM2를 사용합니다.
- 한 장의 이미지뿐 아니라 여러 장의 이미지와 비디오 입력도 지원합니다.

## 사용 팁[[usage-tips]]

입력된 이미지는 설정에 따라 원본 해상도를 유지하거나 크기를 조절할 수 있습니다. 이때 이미지 크기 조절 여부와 방식은 `do_resize`와 `size` 파라미터로 결정됩니다.

비디오의 경우에는 업샘플링을 하면 안 됩니다.

만약 `do_resize`가 `True`일 경우, 모델은 기본적으로 이미지의 가장 긴 변을 4*512 픽셀이 되도록 크기를 조절합니다.
이 기본 동작은 `size` 파라미터에 딕셔너리를 전달하여 원하는 값으로 직접 설정할 수 있습니다. 예를 들어, 기본값은 `{"longest_edge": 4 * 512}` 이여도 사용자 필요에 따라 다른 값으로 변경할 수 있습니다.

다음은 리사이징을 제어하고 사용자 정의 크기로 변경하는 방법입니다:
```python
image_processor = SmolVLMImageProcessor(do_resize=True, size={"longest_edge": 2 * 512}, max_image_size=512)
```

또한, `max_image_size` 매개변수는 이미지를 분할하는 정사각형 패치의 크기를 제어합니다. 이 값은 기본적으로 512로 설정되어 있으며 필요에 따라 조정 가능합니다. 이미지 처리기는 리사이징을 마친 후, `max_image_size` 값을 기준으로 이미지를 여러 개의 정사각형 패치로 분할합니다.

이 모델의 기여자는 [orrzohar](https://huggingface.co/orrzohar) 입니다.



## 사용 예시[[usage-example]]

### 단일 미디어 추론[[single-media-inference]]

이 모델은 이미지와 비디오를 모두 입력으로 받을 수 있지만, 한 번에 사용할 수 있는 미디어는 반드시 하나의 종류여야 합니다. 관련 예시 코드는 다음과 같습니다.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)

conversation = [
    {
        "role": "user",
        "content":[
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "이 이미지에 대해 설명해주세요."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)


# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "이 비디오에 대해 자세히 설명해주세요."}
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

### 배치 다중 미디어 추론[[batch-mixed-media-inference]]

이 모델은 여러 이미지, 비디오, 텍스트로 구성된 입력을 한 번에 배치 형태로 처리할 수 있습니다. 관련 예시는 다음과 같습니다.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    dtype=torch.bfloat16,
    device_map="auto"
)

# 첫 번째 이미지에 대한 구성
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "이 이미지에 대해 설명해주세요."}
        ]
    }
]

# 두 장의 이미지를 포함한 구성
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "그림에 무엇이 적혀있나요?"}
        ]
    }
]

# 텍스트만 포함하고 있는 구성
conversation3 = [
    {"role": "user","content": "당신은 누구인가요?"}
]


conversations = [conversation1, conversation2, conversation3]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

## SmolVLMConfig[[transformers.SmolVLMConfig]]

[[autodoc]] SmolVLMConfig

## SmolVLMVisionConfig[[transformers.SmolVLMVisionConfig]]

[[autodoc]] SmolVLMVisionConfig

## Idefics3VisionTransformer[[transformers.SmolVLMVisionTransformer]]

[[autodoc]] SmolVLMVisionTransformer

## SmolVLMModel[[transformers.SmolVLMModel]]

[[autodoc]] SmolVLMModel
    - forward

## SmolVLMForConditionalGeneration[[transformers.SmolVLMForConditionalGeneration]]

[[autodoc]] SmolVLMForConditionalGeneration
    - forward

## SmolVLMImageProcessor[[transformers.SmolVLMImageProcessor]]
[[autodoc]] SmolVLMImageProcessor
    - preprocess

## SmolVLMImageProcessorFast[[transformers.SmolVLMImageProcessorFast]]
[[autodoc]] SmolVLMImageProcessorFast
    - preprocess

## SmolVLMVideoProcessor[[transformers.SmolVLMVideoProcessor]]
[[autodoc]] SmolVLMVideoProcessor
    - preprocess

## SmolVLMProcessor[[transformers.SmolVLMProcessor]]
[[autodoc]] SmolVLMProcessor
    - __call__
