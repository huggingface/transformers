
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
*이 모델은 2025년 5월 20일에 출시되었으며, 2025년 6월 26일에 Hugging Face Transformers에 추가되었습니다.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Gemma3n[[gemma3n]]

## 개요[[overview]]

[Gemma3n](https://developers.googleblog.com/en/introducing-gemma-3n/)은 사전 훈련된 버전과 명령어 기반 미세조정 버전이 제공되는 멀티모달 모델이며, 모델 크기는 E4B와 E2B 두 가지로 출시되었습니다. 언어 모델 아키텍처는 이전 Gemma 버전과 많은 부분을 공유하지만 이번 버전에는 여러 가지 새로운 기법이 추가되었습니다. 대표적으로 [교차 업데이트(AltUp)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html), [학습된 증강 잔여 레이어(LAuReL)](https://huggingface.co/papers/2411.07501), [MatFormer](https://huggingface.co/papers/2310.07707), 레이어별 임베딩, [통계적 Top-k를 이용한 활성화 희소성(SPARk-Transformer)](https://huggingface.co/papers/2506.06644), KV 캐시 공유 등이 있습니다. Gemma 3n은 [Gemma 3](./gemma3)와 유사한 어텐션 패턴을 사용합니다. 글로벌 셀프 어텐션 레이어 1개마다 로컬 슬라이딩 윈도우 셀프 어텐션 레이어 4개를 교차로 배치하며, 최대 컨텍스트 길이는 32k 토큰까지 지원합니다. 비전 모달리티에서는 MobileNet v5를 비전 인코더로 도입하여 기본 해상도를 768x768 픽셀로 처리합니다. 또한 오디오 모달리티에서는 [Universal Speech Model(USM)](https://huggingface.co/papers/2303.01037) 아키텍처를 기반으로 새롭게 학습된 오디오 인코더가 추가되었습니다.

명령어 기반 미세조정 버전은 지식 증류와 강화 학습을 통해 후처리 학습 되었습니다.

Gemma 3n의 원본 체크포인트는 [Gemma 3n][gemma3n-collection] 출시 페이지에서 확인할 수 있습니다.

> [!TIP]
> 오른쪽 사이드바에 있는 Gemma 3n 모델을 클릭하면, Gemma를 다양한 비전, 오디오, 
> 언어 작업에 적용하는 더 많은 예시를 확인할 수 있습니다.

아래 예시는 [`Pipeline`] 또는 [`AutoModel`] 클래스를 사용하여 이미지를 입력으로 받아 텍스트를 생성하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/gemma-3n-e4b",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="이 이미지에 무엇이 보이나요?"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model = Gemma3nForConditionalGeneration.from_pretrained(
    "google/gemma-3n-e4b-it",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-3n-e4b-it",
    padding_side="left"
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "당신은 도움이 되는 어시스턴트입니다."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "이 이미지에 무엇이 보이나요?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "식물은 특정 과정을 통해 에너지를 생성합니다." | transformers run --task text-generation --model google/gemma-3n-e2b --device 0
```

</hfoption>
</hfoptions>

## 참고사항[[notes]]

-   [`Gemma3nForConditionalGeneration`] 클래스를 사용하면 이미지-오디오-텍스트, 이미지-텍스트, 이미지-오디오, 오디오-텍스트, 이미지 단독, 오디오 단독 입력을 모두 처리할 수 있습니다.
-   Gemma 3n은 한 번의 입력에 여러 이미지를 지원합니다. 다만 프로세서에 전달하기 전에 이미지들이 배치 단위로 올바르게 묶여있는지 확인해야 합니다. 각 배치는 하나 이상의 이미지를 담은 리스트 형식입니다.

    ```py
    url_cow = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    messages =[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "당신은 도움이 되는 어시스턴트입니다."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": url_cow},
                {"type": "image", "url": url_cat},
                {"type": "text", "text": "어떤 이미지가 더 귀엽습니까?"},
            ]
        },
    ]
    ```
-   프로세서에 전달되는 텍스트에는 이미지를 삽입해야 하는 위치에 `<image_soft_token>` 토큰을 포함해야 합니다.
-   Gemma 3n은 입력당 최대 하나의 타깃 오디오 클립만 허용합니다. 다만 퓨샷 프롬프트에서는 여러 개의 오디오 클립을 함께 제공할 수 있습니다.
-   프로세서에 전달되는 텍스트에는 오디오 클립을 삽입해야 하는 위치에 `<audio_soft_token>` 토큰을 포함해야 합니다.
-   프로세서에는 채팅 메시지를 모델 입력 형식으로 변환하기 위한 자체 메서드인 [`~ProcessorMixin.apply_chat_template`]가 포함되어 있습니다.

## Gemma3nAudioFeatureExtractor[[transformers.Gemma3nAudioFeatureExtractor]]

[[autodoc]] Gemma3nAudioFeatureExtractor

## Gemma3nProcessor[[transformers.Gemma3nProcessor]]

[[autodoc]] Gemma3nProcessor

## Gemma3nTextConfig[[transformers.Gemma3nTextConfig]]

[[autodoc]] Gemma3nTextConfig

## Gemma3nVisionConfig[[transformers.Gemma3nVisionConfig]]

[[autodoc]] Gemma3nVisionConfig

## Gemma3nAudioConfig[[transformers.Gemma3nAudioConfig]]

[[autodoc]] Gemma3nAudioConfig

## Gemma3nConfig[[transformers.Gemma3nConfig]]

[[autodoc]] Gemma3nConfig

## Gemma3nTextModel[[transformers.Gemma3nTextModel]]

[[autodoc]] Gemma3nTextModel
    - forward

## Gemma3nModel[[transformers.Gemma3nModel]]

[[autodoc]] Gemma3nModel
    - forward

## Gemma3nForCausalLM[[transformers.Gemma3nForCausalLM]]

[[autodoc]] Gemma3nForCausalLM
    - forward

## Gemma3nForConditionalGeneration[[transformers.Gemma3nForConditionalGeneration]]

[[autodoc]] Gemma3nForConditionalGeneration
    - forward

