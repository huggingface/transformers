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

# 제로샷(zero-shot) 객체 탐지[[zeroshot-object-detection]]

[[open-in-colab]]

일반적으로 [객체 탐지](object_detection)에 사용되는 모델을 학습하기 위해서는 레이블이 지정된 이미지 데이터 세트가 필요합니다.
그리고 학습 데이터에 존재하는 클래스(레이블)만 탐지할 수 있다는 한계점이 있습니다.

다른 방식을 사용하는 [OWL-ViT](../model_doc/owlvit) 모델로 제로샷 객체 탐지가 가능합니다.
OWL-ViT는 개방형 어휘(open-vocabulary) 객체 탐지기입니다.
즉, 레이블이 지정된 데이터 세트에 미세 조정하지 않고 자유 텍스트 쿼리를 기반으로 이미지에서 객체를 탐지할 수 있습니다.

OWL-ViT 모델은 멀티 모달 표현을 활용해 개방형 어휘 탐지(open-vocabulary detection)를 수행합니다.
[CLIP](../model_doc/clip) 모델에 경량화(lightweight)된 객체 분류와 지역화(localization) 헤드를 결합합니다.
개방형 어휘 탐지는 CLIP의 텍스트 인코더로 free-text 쿼리를 임베딩하고, 객체 분류와 지역화 헤드의 입력으로 사용합니다.
이미지와 해당 텍스트 설명을 연결하면 ViT가 이미지 패치(image patches)를 입력으로 처리합니다.
OWL-ViT 모델의 저자들은 CLIP 모델을 처음부터 학습(scratch learning)한 후에, bipartite matching loss를 사용하여 표준 객체 인식 데이터셋으로 OWL-ViT 모델을 미세 조정했습니다.

이 접근 방식을 사용하면 모델은 레이블이 지정된 데이터 세트에 대한 사전 학습 없이도 텍스트 설명을 기반으로 객체를 탐지할 수 있습니다.

이번 가이드에서는 OWL-ViT 모델의 사용법을 다룰 것입니다:
- 텍스트 프롬프트 기반 객체 탐지
- 일괄 객체 탐지
- 이미지 가이드 객체 탐지

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:
```bash
pip install -q transformers
```

## 제로샷(zero-shot) 객체 탐지 파이프라인[[zeroshot-object-detection-pipeline]]

[`pipeline`]을 활용하면 가장 간단하게 OWL-ViT 모델을 추론해볼 수 있습니다.
[Hugging Face Hub에 업로드된 체크포인트](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads)에서 제로샷(zero-shot) 객체 탐지용 파이프라인을 인스턴스화합니다:

```python
>>> from transformers import pipeline

>>> checkpoint = "google/owlvit-base-patch32"
>>> detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
```

다음으로, 객체를 탐지하고 싶은 이미지를 선택하세요.
여기서는 [NASA](https://www.nasa.gov/multimedia/imagegallery/index.html) Great Images 데이터 세트의 일부인 우주비행사 에일린 콜린스(Eileen Collins) 사진을 사용하겠습니다.

```py
>>> import skimage
>>> import numpy as np
>>> from PIL import Image

>>> image = skimage.data.astronaut()
>>> image = Image.fromarray(np.uint8(image)).convert("RGB")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png" alt="Astronaut Eileen Collins"/>
</div>

이미지와 해당 이미지의 후보 레이블을 파이프라인으로 전달합니다.
여기서는 이미지를 직접 전달하지만, 컴퓨터에 저장된 이미지의 경로나 url로 전달할 수도 있습니다.
candidate_labels는 이 예시처럼 간단한 단어일 수도 있고 좀 더 설명적인 단어일 수도 있습니다.
또한, 이미지를 검색(query)하려는 모든 항목에 대한 텍스트 설명도 전달합니다.

```py
>>> predictions = detector(
...     image,
...     candidate_labels=["human face", "rocket", "nasa badge", "star-spangled banner"],
... )
>>> predictions
[{'score': 0.3571370542049408,
  'label': 'human face',
  'box': {'xmin': 180, 'ymin': 71, 'xmax': 271, 'ymax': 178}},
 {'score': 0.28099656105041504,
  'label': 'nasa badge',
  'box': {'xmin': 129, 'ymin': 348, 'xmax': 206, 'ymax': 427}},
 {'score': 0.2110239565372467,
  'label': 'rocket',
  'box': {'xmin': 350, 'ymin': -1, 'xmax': 468, 'ymax': 288}},
 {'score': 0.13790413737297058,
  'label': 'star-spangled banner',
  'box': {'xmin': 1, 'ymin': 1, 'xmax': 105, 'ymax': 509}},
 {'score': 0.11950037628412247,
  'label': 'nasa badge',
  'box': {'xmin': 277, 'ymin': 338, 'xmax': 327, 'ymax': 380}},
 {'score': 0.10649408400058746,
  'label': 'rocket',
  'box': {'xmin': 358, 'ymin': 64, 'xmax': 424, 'ymax': 280}}]
```

이제 예측값을 시각화해봅시다:

```py
>>> from PIL import ImageDraw

>>> draw = ImageDraw.Draw(image)

>>> for prediction in predictions:
...     box = prediction["box"]
...     label = prediction["label"]
...     score = prediction["score"]

...     xmin, ymin, xmax, ymax = box.values()
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_2.png" alt="Visualized predictions on NASA image"/>
</div>

## 텍스트 프롬프트 기반 객체 탐지[[textprompted-zeroshot-object-detection-by-hand]]

제로샷 객체 탐지 파이프라인 사용법에 대해 살펴보았으니, 이제 동일한 결과를 복제해보겠습니다.

[Hugging Face Hub에 업로드된 체크포인트](https://huggingface.co/models?other=owlvit)에서 관련 모델과 프로세서를 가져오는 것으로 시작합니다.
여기서는 이전과 동일한 체크포인트를 사용하겠습니다:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

다른 이미지를 사용해 보겠습니다:

```py
>>> import requests

>>> url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"
>>> im = Image.open(requests.get(url, stream=True).raw)
>>> im
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png" alt="Beach photo"/>
</div>

프로세서를 사용해 모델의 입력을 준비합니다.
프로세서는 모델의 입력으로 사용하기 위해 이미지 크기를 변환하고 정규화하는 이미지 프로세서와 텍스트 입력을 처리하는 [`CLIPTokenizer`]로 구성됩니다.

```py
>>> text_queries = ["hat", "book", "sunglasses", "camera"]
>>> inputs = processor(text=text_queries, images=im, return_tensors="pt")
```

모델에 입력을 전달하고 결과를 후처리 및 시각화합니다.
이미지 프로세서가 모델에 이미지를 입력하기 전에 이미지 크기를 조정했기 때문에, [`~OwlViTImageProcessor.post_process_object_detection`] 메소드를 사용해
예측값의 바운딩 박스(bounding box)가 원본 이미지의 좌표와 상대적으로 동일한지 확인해야 합니다.

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)
...     target_sizes = torch.tensor([im.size[::-1]])
...     results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(im)

>>> scores = results["scores"].tolist()
>>> labels = results["labels"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")

>>> im
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="Beach photo with detected objects"/>
</div>

## 일괄 처리[[batch-processing]]

여러 이미지와 텍스트 쿼리를 전달하여 여러 이미지에서 서로 다른(또는 동일한) 객체를 검색할 수 있습니다.
일괄 처리를 위해서 텍스트 쿼리는 이중 리스트로, 이미지는 PIL 이미지, PyTorch 텐서, 또는 NumPy 배열로 이루어진 리스트로 프로세서에 전달해야 합니다.

```py
>>> images = [image, im]
>>> text_queries = [
...     ["human face", "rocket", "nasa badge", "star-spangled banner"],
...     ["hat", "book", "sunglasses", "camera"],
... ]
>>> inputs = processor(text=text_queries, images=images, return_tensors="pt")
```

이전에는 후처리를 위해 단일 이미지의 크기를 텐서로 전달했지만, 튜플을 전달할 수 있고, 여러 이미지를 처리하는 경우에는 튜플로 이루어진 리스트를 전달할 수도 있습니다.
아래 두 예제에 대한 예측을 생성하고, 두 번째 이미지(`image_idx = 1`)를 시각화해 보겠습니다.

```py
>>> with torch.no_grad():
...     outputs = model(**inputs)
...     target_sizes = [x.size[::-1] for x in images]
...     results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)

>>> image_idx = 1
>>> draw = ImageDraw.Draw(images[image_idx])

>>> scores = results[image_idx]["scores"].tolist()
>>> labels = results[image_idx]["labels"].tolist()
>>> boxes = results[image_idx]["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_queries[image_idx][label]}: {round(score,2)}", fill="white")

>>> images[image_idx]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="Beach photo with detected objects"/>
</div>

## 이미지 가이드 객체 탐지[[imageguided-object-detection]]

텍스트 쿼리를 이용한 제로샷 객체 탐지 외에도 OWL-ViT 모델은 이미지 가이드 객체 탐지 기능을 제공합니다.
이미지를 쿼리로 사용해 대상 이미지에서 유사한 객체를 찾을 수 있다는 의미입니다.
텍스트 쿼리와 달리 하나의 예제 이미지에서만 가능합니다.

소파에 고양이 두 마리가 있는 이미지를 대상 이미지(target image)로, 고양이 한 마리가 있는 이미지를 쿼리로 사용해보겠습니다:

```py
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image_target = Image.open(requests.get(url, stream=True).raw)

>>> query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
```

다음 이미지를 살펴보겠습니다:

```py
>>> import matplotlib.pyplot as plt

>>> fig, ax = plt.subplots(1, 2)
>>> ax[0].imshow(image_target)
>>> ax[1].imshow(query_image)
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_5.png" alt="Cats"/>
</div>

전처리 단계에서 텍스트 쿼리 대신에 `query_images`를 사용합니다:

```py
>>> inputs = processor(images=image_target, query_images=query_image, return_tensors="pt")
```

예측의 경우, 모델에 입력을 전달하는 대신 [`~OwlViTForObjectDetection.image_guided_detection`]에 전달합니다.
레이블이 없다는 점을 제외하면 이전과 동일합니다.
이전과 동일하게 이미지를 시각화합니다.

```py
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)
...     target_sizes = torch.tensor([image_target.size[::-1]])
...     results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(image_target)

>>> scores = results["scores"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score, label in zip(boxes, scores, labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)

>>> image_target
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_6.png" alt="Cats with bounding boxes"/>
</div>

OWL-ViT 모델을 추론하고 싶다면 아래 데모를 확인하세요:

<iframe
	src="https://adirik-owl-vit.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
