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

# Qwen2-VL[[Qwen2-VL]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
</div>

## Overview[[Overview]]

[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) 모델은 알리바바 리서치의 Qwen팀에서 개발한 [Qwen-VL](https://huggingface.co/papers/2308.12966) 모델의 주요 업데이트 버전입니다.

블로그의 요약은 다음과 같습니다:

*이 블로그는 지난 몇 년간 Qwen-VL에서 중대한 개선을 거쳐 발전된 Qwen2-VL 모델을 소개합니다. 중요 개선 사항은 향상된 이미지 이해, 고급 비디오 이해, 통합 시각 에이전트 기능, 확장된 다언어 지원을 포함하고 있습니다.모델 아키텍처는 Naive Dynamic Resolution 지원을 통해 임의의 이미지 해상도를 처리할 수 있도록 최적화되었으며, 멀티모달 회전 위치 임베딩(M-ROPE)을 활용하여 1D 텍스트와 다차원 시각 데이터를 효과적으로 처리합니다. 이 업데이트된 모델은 시각 관련 작업에서 GPT-4o와 Claude 3.5 Sonnet 같은 선도적인 AI 시스템과 경쟁력 있는 성능을 보여주며, 텍스트 능력에서는 오픈소스 모델 중 상위권에 랭크되어 있습니다. 이러한 발전은 Qwen2-VL을 강력한 멀티모달 처리 및 추론 능력이 필요한 다양한 응용 분야에서 활용할 수 있는 다재다능한 도구로 만들어줍니다.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/qwen2_vl_architecture.jpeg"
alt="drawing" width="600"/>

<small> Qwen2-VL 구조. 출처: <a href="https://qwenlm.github.io/blog/qwen2-vl/">블로그 게시글</a> </small>

이 모델은 [simonJJJ](https://huggingface.co/simonJJJ)에 의해 기여되었습니다.

## 사용 예시[[Usage example]]

### 단일 미디어 추론[[Single Media inference]]

이 모델은 이미지와 비디오를 모두 인풋으로 받을 수 있습니다. 다음은 추론을 위한 예제 코드입니다.

```python

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# 사용 가능한 장치에서 모델을 반 정밀도(half-precision)로 로드
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# 추론: 아웃풋 생성
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)



# 비디오
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# 추론: 아웃풋 생성
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### 배치 혼합 미디어 추론[[Batch Mixed Media Inference]]

이 모델은 이미지, 비디오, 텍스트 등 다양한 유형의 데이터를 혼합하여 배치 입력으로 처리할 수 있습니다. 다음은 예제입니다.

```python

# 첫번째 이미지에 대한 대화
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image1.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# 두 개의 이미지에 대한 대화
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image2.jpg"},
            {"type": "image", "path": "/path/to/image3.jpg"},
            {"type": "text", "text": "What is written in the pictures?"}
        ]
    }
]

# 순수 텍스트로만 이루어진 대화
conversation3 = [
    {
        "role": "user",
        "content": "who are you?"
    }
]


# 혼합된 미디어로 이루어진 대화
conversation4 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image3.jpg"},
            {"type": "image", "path": "/path/to/image4.jpg"},
            {"type": "video", "path": "/path/to/video.jpg"},
            {"type": "text", "text": "What are the common elements in these medias?"},
        ],
    }
]

conversations = [conversation1, conversation2, conversation3, conversation4]
# 배치 추론을 위한 준비
ipnuts = processor.apply_chat_template(
    conversations,
    video_fps=1,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)


# 배치 추론
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

### 사용 팁[[Usage Tips]]

#### 이미지 해상도 트레이드오프[[Image Resolution trade-off]]

이 모델은 다양한 해상도의 입력을 지원합니다. 디폴트로 입력에 대해 네이티브(native) 해상도를 사용하지만, 더 높은 해상도를 적용하면 성능이 향상될 수 있습니다. 다만, 이는 더 많은 연산 비용을 초래합니다. 사용자는 최적의 설정을 위해 최소 및 최대 픽셀 수를 조정할 수 있습니다.

```python
min_pixels = 224*224
max_pixels = 2048*2048
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```

제한된 GPU RAM의 경우, 다음과 같이 해상도를 줄일 수 있습니다:

```python
min_pixels = 256*28*28
max_pixels = 1024*28*28 
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
```
이렇게 하면 각 이미지가 256~1024개의 토큰으로 인코딩됩니다. 여기서 28은 모델이 14 크기의 패치(patch)와 2의 시간 패치(temporal patch size)를 사용하기 때문에 나온 값입니다 (14 × 2 = 28).


#### 다중 이미지 인풋[[Multiple Image Inputs]]

기본적으로 이미지와 비디오 콘텐츠는 대화에 직접 포함됩니다. 여러 개의 이미지를 처리할 때는 이미지 및 비디오에 라벨을 추가하면 참조하기가 더 쉬워집니다. 사용자는 다음 설정을 통해 이 동작을 제어할 수 있습니다:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": "Hello, how are you?"}
        ]
    },
    {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. How can I assist you today?"
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you describe these images and video?"}, 
            {"type": "image"}, 
            {"type": "image"}, 
            {"type": "video"}, 
            {"type": "text", "text": "These are from my vacation."}
        ]
    },
    {
        "role": "assistant",
        "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"
    },
    {
        "role": "user",
        "content": "It was a trip to the mountains. Can you see the details in the images and video?"
    }
]

# 디폴트:
prompt_without_id = processor.apply_chat_template(conversation, add_generation_prompt=True)
# 예상 아웃풋: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'


# id 추가
prompt_with_id = processor.apply_chat_template(conversation, add_generation_prompt=True, add_vision_id=True)
# 예상 아웃풋: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?Picture 2: <|vision_start|><|image_pad|><|vision_end|>Picture 3: <|vision_start|><|image_pad|><|vision_end|>Video 1: <|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'

```

#### 빠른 생성을 위한 Flash-Attention 2[[Flash-Attention 2 to speed up generation]]

첫번째로, Flash Attention 2의 최신 버전을 설치합니다:

```bash
pip install -U flash-attn --no-build-isolation
```

또한, Flash-Attention 2를 지원하는 하드웨어가 필요합니다. 자세한 내용은 공식 문서인 [flash attention repository](https://github.com/Dao-AILab/flash-attention)에서 확인할 수 있습니다. FlashAttention-2는 모델이 `torch.float16` 또는 `torch.bfloat16` 형식으로 로드된 경우에만 사용할 수 있습니다.

Flash Attention-2를 사용하여 모델을 로드하고 실행하려면, 다음과 같이 모델을 로드할 때 `attn_implementation="flash_attention_2"` 옵션을 추가하면 됩니다:

```python
from transformers import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)
```

## Qwen2VLConfig

[[autodoc]] Qwen2VLConfig

## Qwen2VLImageProcessor

[[autodoc]] Qwen2VLImageProcessor
    - preprocess

## Qwen2VLImageProcessorFast

[[autodoc]] Qwen2VLImageProcessorFast
    - preprocess

## Qwen2VLProcessor

[[autodoc]] Qwen2VLProcessor

## Qwen2VLModel

[[autodoc]] Qwen2VLModel
    - forward

## Qwen2VLForConditionalGeneration

[[autodoc]] Qwen2VLForConditionalGeneration
    - forward
