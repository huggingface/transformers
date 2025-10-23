<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*이 모델은 2023-06-02에 발표되었으며 2025-04-28에 Hugging Face Transformers에 추가되었습니다.*

# SAM-HQ[[sam_hq]]

## 개요[[overview]]

SAM-HQ (High-Quality Segment Anything Model)는 Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu가 제안한 [Segment Anything in High Quality](https://huggingface.co/papers/2306.01567) 논문에서 소개되었습니다.

이 모델은 기존 SAM(Segment Anything Model)의 향상된 버전입니다. SAM-HQ는 SAM의 핵심 장점인 프롬프트 기반 설계, 효율성, 제로샷 일반화 능력을 그대로 유지하면서도 훨씬 더 높은 품질의 분할 마스크를 생성하는 것이 특징입니다.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

SAM-HQ는 기존 SAM 모델 대비 다음과 같은 5가지 핵심 개선 사항을 도입했습니다.

1. 고품질 출력 토큰: SAM-HQ는 SAM의 마스크 디코더에 학습 가능한 토큰을 주입합니다. 이 토큰은 모델이 더 높은 품질의 분할 마스크를 예측하도록 돕는 핵심적인 요소입니다.
2. 전역-지역 특징 융합: 모델의 서로 다른 단계에서 추출된 특징들을 결합하여 분할 마스크의 세부적인 정확도를 향상시킵니다. 이미지의 전체적인 맥락 정보와 객체의 미세한 경계 정보를 함께 활용하여 마스크 품질을 개선합니다.
3. 훈련 데이터 개선: SAM 모델이 SA-1B와 같은 대규모 데이터를 사용한 것과 달리, SAM-HQ는 신중하게 선별된 44,000개의 고품질 마스크로 구성된 데이터셋을 사용하여 훈련됩니다.
4. 높은 효율성: 마스크 품질을 상당히 개선했음에도 불구하고, 추가된 매개변수는 단 0.5%에 불과합니다.
5. 제로샷 성능: SAM-HQ는 성능이 개선되었음에도 불구하고, SAM 모델의 강력한 제로샷 일반화 능력을 그대로 유지합니다.

논문 초록 내용:

* 최근 발표된 SAM(Segment Anything Model)은 분할 모델의 규모를 확장하는 데 있어 획기적인 발전이며, 강력한 제로샷 기능과 유연한 프롬프트 기능을 제공합니다. 하지만 SAM은 11억 개의 마스크로 훈련되었음에도 불구하고, 특히 복잡하고 정교한 구조를 가진 객체를 분할할 때 마스크 예측 품질이 미흡한 경우가 많습니다. 저희는 HQ-SAM을 제안하며, SAM의 기존 장점인 프롬프트 기반 설계, 효율성, 제로샷 일반화 능력을 모두 유지하면서도 어떤 객체든 정확하게 분할할 수 있는 능력을 부여합니다. 저희는 신중한 설계를 통해 SAM의 사전 훈련된 모델 가중치를 재사용하고 보존하며 최소한의 추가적인 매개변수와 연산만을 도입했습니다. 핵심적으로 저희는 학습 가능한 고품질 출력 토큰을 설계했습니다. 이 토큰은 SAM의 마스크 디코더에 주입되어 고품질 마스크를 예측하는 역할을 담당합니다. 마스크의 세부 사항을 개선하기 위해 이 토큰을 마스크 디코더 특징에만 적용하는 것이 아니라 초기 및 최종 ViT 특징과 먼저 융합하여 사용합니다. 도입된 학습 가능한 매개변수를 훈련하기 위해 저희는 여러 출처에서 가져온 44,000개의 미세 조정된 마스크 데이터셋을 구성했습니다. HQ-SAM은 오직 이 44,000개 마스크 데이터셋만으로 훈련되며 GPU 8대를 사용했을 때 단 4시간이 소요됩니다.

SAM-HQ 사용 팁:

- SAM-HQ는 기존 SAM 모델보다 더 높은 품질의 마스크 생성하며, 특히 복잡한 구조와 미세한 세부 사항을 가진 객체에 대해 성능이 우수합니다.
- 이 모델은 더욱 정확한 경계와 얇은 구조에 대한 더 나은 처리 능력을 갖춘 이진 마스크를 예측합니다.
- SAM과 마찬가지로 모델은 입력으로 2차원 포인트 및 바운딩 박스를 사용할 때 더 좋은 성능을 보입니다.
- 하나의 이미지에 대해 다수의 포인트를 프롬프트로 입력하여 단일의 고품질 마스크를 예측할 수 있습니다.
- 이 모델은 SAM의 제로샷 일반화 능력을 그대로 유지합니다.
- SAM-HQ는 SAM 대비 약 0.5%의 추가 매개변수만을 가집니다.
- 현재 모델의 미세 조정은 지원되지 않습니다.

이 모델은 [sushmanth](https://huggingface.co/sushmanth)님께서 기여해주셨습니다.
원본 코드는 [여기](https://github.com/SysCV/SAM-HQ)에서 확인하실 수 있습니다.

아래는 이미지와 2차원 포인트가 주어졌을 때, 마스크를 생성하는 방법에 대한 예시입니다.

```python
import torch
from PIL import Image
import requests
from transformers import infer_device, SamHQModel, SamHQProcessor

device = infer_device()
model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base").to(device)
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 이미지 내 창문의 2차원 위치

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

또한, 프로세서에서 입력 이미지와 함께 사용자의 마스크를 직접 처리하여 모델에 전달할 수도 있습니다.

```python
import torch
from PIL import Image
import requests
from transformers import infer_device, SamHQModel, SamHQProcessor

device = infer_device()
model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base").to(device)
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
input_points = [[[450, 600]]]  # 이미지 내 창문의 2차원 위치

inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

## 자료[[resources]]

다음은 SAM-HQ 사용을 시작하는 데 도움이 되는 공식 Hugging Face 및 커뮤니티 (🌎로 표시) 자료 목록입니다.

- 모델 사용을 위한 데모 노트북 (출시 예정)
- 논문 구현 및 코드: [SAM-HQ 깃허브 저장소](https://github.com/SysCV/SAM-HQ)

## SamHQConfig[[transformers.SamHQConfig]]

[[autodoc]] SamHQConfig

## SamHQVisionConfig[[transformers.SamHQVisionConfig]]

[[autodoc]] SamHQVisionConfig

## SamHQMaskDecoderConfig[[transformers.SamHQMaskDecoderConfig]]

[[autodoc]] SamHQMaskDecoderConfig

## SamHQPromptEncoderConfig[[transformers.SamHQPromptEncoderConfig]]

[[autodoc]] SamHQPromptEncoderConfig

## SamHQProcessor[[transformers.SamHQProcessor]]

[[autodoc]] SamHQProcessor

## SamHQVisionModel[[transformers.SamHQVisionModel]]

[[autodoc]] SamHQVisionModel

## SamHQModel[[transformers.SamHQModel]]

[[autodoc]] SamHQModel
    - forward
