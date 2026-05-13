<!--Copyright 2026 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-04-09 and added to Hugging Face Transformers on 2026-04-30.*

# EXAONE 4.5

## 개요

[EXAONE 4.5](https://github.com/LG-AI-EXAONE/EXAONE-4.5) 모델은 LG AI연구원에서 공개한 최초의 오픈 웨이트(open-weight) 비전-자연어 모델(vision-language model)입니다. 
전용 비전 인코더를 기존 개발된 EXAONE 4.0 프레임워크에 통합하여 모델의 능력을 비전과 자연어를 고려한 멀티모달리티로 확장했습니다. EXAONE 4.5는 1.2B 크기의 비전 인코더를 포함해 총 33B 크기의 모델로 구성됩니다. 
EXAONE 4.5는 이전 EXAONE 모델군으로부터 이어져 온 강력한 언어 처리 능력 덕분에 범용 벤치마크에서 경쟁력 있는 성능을 달성함과 동시에, 동등 규모의 최신 SOTA 모델을 능가하는 문서 이해 능력과 한국 문화적 추론 능력을 갖추고 있습니다.

EXAONE 4.5는 EXAONE 4.0을 기반으로 몇 가지 핵심 개선 사항을 적용했습니다. 어휘 크기를 153,600으로 확장했으며, 컨텍스트 윈도우는 최대 256K 토큰까지 지원합니다. 또한 MTP(Multi-Token Prediction) 메커니즘을 도입해 모델 성능을 한층 더 높였습니다.

더 자세한 정보는 [기술 보고서](https://huggingface.co/papers/2604.08644), [블로그](https://www.lgresearch.ai/blog/view?seq=641), [공식 GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.5) 페이지를 참고해 주세요.

양자화된 버전을 포함한 공개된 모든 체크포인트는 [Huggingface 콜렉션](https://huggingface.co/collections/LGAI-EXAONE/exaone-45)에서 확인할 수 있습니다.

## 사용 팁

> 기대한 성능을 얻기 위해 다음 설정 사용을 권장합니다.
> - 범용 용도로는 `temperature=1.0`, `top_p=0.95`, `presence_penalty=1.5`를 권장합니다.
> - OCR/문서 관련 작업과 한국어 입력에는 `temperature=0.6`, `top_p=0.95`, `presence_penalty=1.5`, `top_k=20`을 권장합니다.
> - 텍스트 전용 입력에는 `temperature=1.0`, `top_p=0.95`를 권장합니다.
> - EXAONE-4.0과 달리 EXAONE 4.5는 기본값으로 `enable_thinking=True`를 사용합니다. 따라서 non-reasoning 모드를 사용할 때는 `enable_thinking=False`로 설정해야 합니다.
> - EXAONE 4.5는 질문에 답할 때 `\boxed{}` 형식을 선호합니다. 파싱 정확도를 높이려면 해당 형식 지시문과 함께 사용하는 것을 권장합니다.

정확한 결과가 중요한 작업에서는 EXAONE 4.5 모델을 reasoning 모드로 실행할 수 있습니다. 반면에 지연 시간이 정확도보다 중요한 작업에서는 EXAONE 4.5 모델을 non-reasoning 모드로 실행할 수 있습니다.

다음은 EXAONE 4.5 모델을 reasoning 모드로 사용하는 예제 코드입니다.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

model_id = "LGAI-EXAONE/EXAONE-4.5-33B"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
)

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(image_url)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": "이 이미지를 설명해 줘."},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,   # default: True
)
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_text = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)[0]
print(generated_text)
```


## Exaone4_5_Config

[[autodoc]] Exaone4_5_Config

## Exaone4_5_VisionConfig

[[autodoc]] Exaone4_5_VisionConfig

## Exaone4_5_Processor

[[autodoc]] Exaone4_5_Processor

## Exaone4_5_VisionModel

[[autodoc]] Exaone4_5_VisionModel
    - forward

## Exaone4_5_Model

[[autodoc]] Exaone4_5_Model
    - forward

## Exaone4_5_ForConditionalGeneration

[[autodoc]] Exaone4_5_ForConditionalGeneration
    - forward