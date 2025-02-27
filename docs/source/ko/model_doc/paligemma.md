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

# PaliGemma[[paligemma]]

## 개요[[overview]]

PaliGemma 모델은 구글이 제안한 [PaliGemma – Google의 최첨단 오픈 비전 언어 모델](https://huggingface.co/blog/paligemma)에서 소개 되었습니다. PaliGemma는 [SigLIP](siglip) 비전 인코더와 [Gemma](gemma) 언어 인코더로 구성된 3B 규모의 비전-언어 모델로, 두 인코더가 멀티모달 선형 프로젝션으로 연결되어 있습니다. 이 모델은 이미지를 고정된 수의 VIT토큰으로 분할하고 이를 선택적 프롬프트 앞에 추가 하며, 모든 이미지 토큰과 입력 텍스트 토큰에 대해 전체 블록 어텐션을 사용하는 특징을 가지고 있습니다.

PaliGemma는 224x224, 448x448, 896x896의 3가지 해상도로 제공되며, 3개의 기본 모델과 55개의 다양한 작업에 대해 미세 조정된 버전, 그리고 2개의 혼합 모델이 있습니다.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/paligemma_arch.png"
alt="drawing" width="600"/>

<small> PaliGemma 아키텍처 <a href="https://huggingface.co/blog/paligemma">블로그 포스트.</a> </small>

이 모델은 [Molbap](https://huggingface.co/Molbap)에 의해 기여 되었습니다.

## 사용 팁[[usage-tips]]

PaliGemma의 추론은 다음처럼 수행됩니다:

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "What is on the flower?"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(raw_image, prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
```

- PaliGemma는 대화용으로 설계되지 않았으며, 특정 사용 사례에 대해 미세 조정할 때 가장 잘 작동합니다. PaliGemma를 미세 조정할 수 있는 몇 가지 하위 작업에는 이미지 캡셔닝, 시각적 질문 답변(VQA), 오브젝트 디텍션, 참조 표현 분할 및 문서 이해가 포함됩니다.
- 모델에 필요한 이미지, 텍스트 및 선택적 레이블을 준비하는데 `PaliGemmaProcessor`를 사용할 수 있습니다. PaliGemma 모델을 미세 조정할 때는, 프로세서에 `suffix`인자를 전달하여 다음 처럼 모델의 `labels`를 생성할 수 있습니다:

```python
prompt = "What is on the flower?"
answer = "a bee"
inputs = processor(images=raw_image, text=prompt, suffix=answer, return_tensors="pt")
```

## 자료[[resources]]

PaliGemma를 시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다.여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.

- PaliGemma의 모든 기능을 소개하는 블로그 포스트는 [이곳](https://huggingface.co/blog/paligemma)에서 찾을 수 있습니다. 🌎
- Trainer API를 사용하여 VQA(Visual Question Answering)를 위해 PaliGemma를 미세 조정하는 방법과 추론에 대한 데모 노트북은 [이곳](https://github.com/huggingface/notebooks/tree/main/examples/paligemma)에서 찾을 수 있습니다. 🌎
- 사용자 정의 데이터셋(영수증 이미지 -> JSON)에 대해 PaliGemma를 미세 조정하는 방법과 추론에 대한 데모 노트북은 [이곳](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma)에서 찾을 수 있습니다. 🌎

## PaliGemmaConfig[[transformers.PaliGemmaConfig]]

[[autodoc]] PaliGemmaConfig

## PaliGemmaProcessor[[transformers.PaliGemmaProcessor]]

[[autodoc]] PaliGemmaProcessor

## PaliGemmaForConditionalGeneration[[transformers.PaliGemmaForConditionalGeneration]]

[[autodoc]] PaliGemmaForConditionalGeneration
    - forward
