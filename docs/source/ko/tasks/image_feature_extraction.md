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

# 이미지 특징 추출[[image-feature-extraction]]

[[open-in-colab]]

이미지 특징 추출은 주어진 이미지에서 의미론적으로 의미 있는 특징을 추출하는 작업입니다. 이는 이미지 유사성 및 이미지 검색 등 다양한 사용 사례가 있습니다.
게다가 대부분의 컴퓨터 비전 모델은 이미지 특징 추출에 사용할 수 있으며, 여기서 작업 특화 헤드(이미지 분류, 물체 감지 등)를 제거하고 특징을 얻을 수 있습니다. 이러한 특징은 가장자리 감지, 모서리 감지 등 고차원 수준에서 매우 유용합니다.
또한 모델의 깊이에 따라 실제 세계에 대한 정보(예: 고양이가 어떻게 생겼는지)를 포함할 수도 있습니다. 따라서 이러한 출력은 특정 데이터 세트에 대한 새로운 분류기를 훈련하는 데 사용할 수 있습니다.

이 가이드에서는:

- `image-feature-extraction` 파이프라인을 활용하여 간단한 이미지 유사성 시스템을 구축하는 방법을 배웁니다.
- 기본 모델 추론으로 동일한 작업을 수행합니다.

## `image-feature-extraction` 파이프라인을 이용한 이미지 유사성[[image-similarity-using-image-feature-extraction-pipeline]]

물고기 그물 위에 앉아 있는 두 장의 고양이 사진이 있습니다. 이 중 하나는 생성된 이미지입니다.

```python
from PIL import Image
import requests

img_urls = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"]
image_real = Image.open(requests.get(img_urls[0], stream=True).raw).convert("RGB")
image_gen = Image.open(requests.get(img_urls[1], stream=True).raw).convert("RGB")
```

파이프라인을 실행해 봅시다. 먼저 파이프라인을 초기화하세요. 모델을 지정하지 않으면, 파이프라인은 자동으로 [google/vit-base-patch16-224](google/vit-base-patch16-224) 모델로 초기화됩니다. 유사도를 계산하려면 `pool`을 True로 설정하세요. 


```python
import torch
from transformers import pipeline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device=DEVICE, pool=True)
```

`pipe`를 사용하여 추론하려면 두 이미지를 모두 전달하세요.

```python
outputs = pipe([image_real, image_gen])
```

출력에는 두 이미지의 풀링된(pooled) 임베딩이 포함되어 있습니다.

```python
# 단일 출력의 길이 구하기
print(len(outputs[0][0]))
# 출력 결과 표시하기
print(outputs)

# 768
# [[[-0.03909236937761307, 0.43381670117378235, -0.06913255900144577,
```

유사도 점수를 얻으려면, 이들을 유사도 함수에 전달해야 합니다.

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(torch.Tensor(outputs[0]),
                                     torch.Tensor(outputs[1]), dim=1)

print(similarity_score)

# tensor([0.6043])
```

풀링 이전의 마지막 은닉 상태를 얻고 싶다면, `pool` 매개변수에 아무 값도 전달하지 마세요. 또한, 기본값은 `False`로 설정되어 있습니다. 이 은닉 상태는 모델의 특징을 기반으로 새로운 분류기나 모델을 훈련시키는 데 유용합니다.

```python
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", device=DEVICE)
output = pipe(image_real)
```

아직 출력이 풀링되지 않았기 때문에, 첫 번째 차원은 배치 크기이고 마지막 두 차원은 임베딩 형태인 마지막 은닉 상태를 얻을 수 있습니다.

```python
import numpy as np
print(np.array(outputs).shape)
# (1, 197, 768)
```

## `AutoModel`을 사용하여 특징과 유사성 얻기[[getting-features-and-similarities-using-automodel]]

transformers의 `AutoModel` 클래스를 사용하여 특징을 얻을 수도 있습니다. `AutoModel`은 작업 특화 헤드 없이 모든 transformers 모델을 로드할 수 있으며, 이를 통해 특징을 추출할 수 있습니다.

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
```

추론을 위한 간단한 함수를 작성해 보겠습니다. 먼저 입력값을 `processor`에 전달한 다음, 그 출력값을 `model`에 전달할 것입니다.

```python
def infer(image):
  inputs = processor(image, return_tensors="pt").to(DEVICE)
  outputs = model(**inputs)
  return outputs.pooler_output
```

이 함수에 이미지를 직접 전달하여 임베딩을 얻을 수 있습니다.

```python
embed_real = infer(image_real)
embed_gen = infer(image_gen)
```

그리고 이 임베딩을 사용하여 다시 유사도를 계산할 수 있습니다.

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(embed_real, embed_gen, dim=1)
print(similarity_score)

# tensor([0.6061], device='cuda:0', grad_fn=<SumBackward1>)
```