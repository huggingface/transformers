<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLIP[[clip]]

## 개요[[overview]]

CLIP 모델은 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever가 제안한 [자연어 지도(supervision)를 통한 전이 가능한 시각 모델 학습](https://huggingface.co/papers/2103.00020)라는 논문에서 소개되었습니다. CLIP(Contrastive Language-Image Pre-Training)은 다양한 이미지와 텍스트 쌍으로 훈련된 신경망 입니다. GPT-2와 3의 제로샷 능력과 유사하게, 해당 작업에 직접적으로 최적화하지 않고도 주어진 이미지에 대해 가장 관련성 있는 텍스트 스니펫을 예측하도록 자연어로 지시할 수 있습니다.

해당 논문의 초록입니다.

*최신 컴퓨터 비전 시스템은 미리 정해진 고정된 객체 카테고리 집합을 예측하도록 훈련됩니다. 이러한 제한된 형태의 지도는 다른 시각적 개념을 지정하기 위해 추가적인 라벨링된 데이터가 필요하므로 그 일반성과 사용성을 제한합니다. 이미지 원시 텍스트에서 직접 학습하는 것은 훨씬 더 광범위한 지도 소스를 활용하는 아주 좋은 대안입니다. 이미지와 캡션을 맞추는 간단한 사전 학습 작업이, 인터넷에서 수집한 4억 쌍의 이미지-텍스트 데이터셋에서 SOTA 수준의 이미지 표현을 처음부터 효율적이고 확장 가능하게 학습하는 방법임을 확인할 수 있습니다. 사전 훈련 후, 자연어는 학습된 시각적 개념을 참조하거나 새로운 개념을 설명하는 데 사용되어 모델의 하위 작업으로의 제로샷 전이를 가능하게 합니다. 해당 논문에서는 OCR, 비디오 내 행동 인식, 지리적 위치 파악, 그리고 많은 종류의 세밀한 객체 분류 등 30개 이상의 다양한 기존 컴퓨터 비전 데이터셋에 대한 벤치마킹을 통해 이 접근 방식의 성능을 연구합니다. 이 모델은 대부분의 작업에 대해 의미 있게 전이되며, 종종 데이터셋별 훈련 없이도 완전 지도 학습 기준선과 경쟁력 있는 성능을 보입니다. 예를 들어, ImageNet에서 원래 ResNet-50의 정확도를 제로샷으로 일치시키는데, 이는 ResNet-50이 훈련된 128만 개의 훈련 예제를 전혀 사용할 필요가 없었습니다. 코드 및 사전 훈련된 모델 가중치는 이 https URL에서 공개합니다.*

이 모델은 [valhalla](https://huggingface.co/valhalla)에 의해 기여되었습니다.
원본 코드는 [이곳](https://github.com/openai/CLIP)에서 확인할 수 있습니다.

## 사용 팁과 예시[[usage-tips-and-example]]

CLIP은 멀티모달 비전 밒 언어 모델입니다. 이미지-텍스트 유사도 계산과 제로샷 이미지 분류에 사용될 수 있습니다. CLIP은 ViT와 유사한 트랜스포머를 사용하여 시각적 특징을 추출하고, 인과적 언어 모델을 사용하여 텍스트 특징을 추출합니다. 그 후 텍스트와 시각적 특징 모두 동일한 차원의 잠재(latent) 공간으로 투영됩니다. 투영된 이미지와 텍스트 특징 사이의 내적이 유사도 점수로 사용됩니다.

트랜스포머 인코더에 이미지를 입력하기 위해, 각 이미지는 고정 크기의 겹치지 않는 패치들의 시퀀스로 분할되고, 이후 선형 임베딩됩니다. [CLS]토큰이 전체 이미지의 표현으로 추가됩니다. 저자들은 또한 절대 위치 임베딩을 추가하고, 결과로 나온 벡터 시퀀스를 표준 트랜스포머 인토더에 입력합니다. [`CLIPImageProcessor`]는 모델을 위해 이미지를 리사이즈(또는 재스캐일링)하고 정규화하는데 사용될 수 있습니다.

[`CLIPTokenizer`]는 텍스트를 인코딩하는데 사용됩니다. [`CLIPProcessor`]는 [`CLIPImageProcessor`]와 [`CLIPTokenizer`]를 하나의 인스턴스로 감싸서 텍스트를 인코딩하고 이미지를 준비하는데 모두 사용됩니다.

다음 예시는 [`CLIPProcessor`]와 [`CLIPModel`]을 사용하여 이미지-텍스트 유사도 점수를 얻는 방법을 보여줍니다.


```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사성 점수
>>> probs = logits_per_image.softmax(dim=1)  # 확률을 레이블링 하기위해서 소프트맥스를 취합니다.
```


### CLIP과 플래시 어텐션2 결합[[combining-clip-and-flash-attention-2]]

먼저 최신버전의 플래시 어텐션2를 설치합니다.

```bash
pip install -U flash-attn --no-build-isolation
```

플래시 어텐션2와 호환되는 하드웨어를 가지고 있는지 확인하세요. 이에 대한 자세한 내용은 flash-attn 리포지토리의 공식문서에서 확인할 수 있습니다. 또한 모델을 반정밀도(`torch.float16`)로 로드하는 것을 잊지 마세요.

<Tip warning={true}>

작은 배치 크기를 사용할 때, 플래시 어텐션을 사용하면 모델이 느려지는 것을 느낄 수 있습니다.아래의 [플래시 어텐션과 SDPA를 사용한 예상 속도 향상](#Expected-speedups-with-Flash-Attention-and-SDPA) 섹션을 참조하여 적절한 어텐션 구현을 선택하세요.

</Tip>

플래시 어텐션2를 사용해서 모델을 로드하고 구동하기 위해서 다음 스니펫을 참고하세요:

```python
>>> import torch
>>> import requests
>>> from PIL import Image

>>> from transformers import CLIPProcessor, CLIPModel

>>> device = "cuda"
>>> dtype = torch.float16

>>> model = CLIPModel.from_pretrained(
...     "openai/clip-vit-base-patch32",
...     attn_implementation="flash_attention_2",
...     device_map=device,
...     dtype=dtype,
... )
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
>>> inputs.to(device)

>>> with torch.no_grad():
...     with torch.autocast(device):
...         outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사성 점수
>>> probs = logits_per_image.softmax(dim=1)  # 확률을 레이블링 하기위해서 소프트맥스를 취합니다.
>>> print(probs)
tensor([[0.9946, 0.0052]], device='cuda:0', dtype=torch.float16)
```


### 스케일된 내적 어텐션 (Scaled dot-product Attention(SDPA)) 사용하기[[using-scaled-dot-product-attention-sdpa]]

파이토치는 `torch.nn.functional`의 일부로 네이티브 스케일된 내적 어텐션(SPDA) 연산자를 포함하고 있습니다. 이 함수는 입력과 사용 중인 하드웨어에 따라 적용될 수 있는 여러 구현을 포함합니다. 자세한 정보는 [공식문서](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)나 [GPU 추론](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention) 페이지를 참조하세요.

`torch>=2.1.1`에서는 구현이 가능할 때 SDPA가 기본적으로 사용되지만, `from_pretrained()` 함수에서 `attn_implementation="sdpa"`를 설정하여 SDPA를 명시적으로 사용하도록 요청할 수도 있습니다.

```python
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.float16, attn_implementation="sdpa")
```

최고의 속도향상을 위해서, 반정밀도로 모델을 로드하는 것을 추천합니다. (예를들면 `torch.float16` 또는 `torch.bfloat16`).

### 플래시 어텐션과 스케일된 내적 어텐션(SDPA)으로 인해 예상되는 속도향상[[expected-speedups-with-flash-attention-and-sdpa]]

로컬 벤치마크(NVIDIA A10G, PyTorch 2.3.1+cu121)에서 `float16`을 사용하여 `"openai/clip-vit-large-patch14"` 체크포인트로 추론을 수행했을 때, 다음과 같은 속도 향상을 확인 했습니다.
[코드](https://gist.github.com/qubvel/ac691a54e54f9fae8144275f866a7ff8):

#### CLIPTextModel[[cliptextmodel]]

|   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                 4 |            0.009 |          0.012 |         0.737 |           0.007 |          1.269 |
|                16 |            0.009 |          0.014 |         0.659 |           0.008 |          1.187 |
|                32 |            0.018 |          0.021 |         0.862 |           0.016 |          1.142 |
|                64 |            0.034 |          0.034 |         1.001 |           0.03  |          1.163 |
|               128 |            0.063 |          0.058 |         1.09  |           0.054 |          1.174 |

![clip_text_model_viz_3](https://github.com/user-attachments/assets/e9826b43-4e66-4f4c-952b-af4d90bd38eb)

#### CLIPVisionModel[[clipvisionmodel]]

|   Image batch size |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |            0.016 |          0.013 |         1.247 |           0.012 |          1.318 |
|                  4 |            0.025 |          0.021 |         1.198 |           0.021 |          1.202 |
|                 16 |            0.093 |          0.075 |         1.234 |           0.075 |          1.24  |
|                 32 |            0.181 |          0.147 |         1.237 |           0.146 |          1.241 |

![clip_image_model_viz_3](https://github.com/user-attachments/assets/50a36206-e3b9-4adc-ac8e-926b8b071d63)

#### CLIPModel[[clipmodel]]

|   Image batch size |   Num text labels |   Eager (s/iter) |   FA2 (s/iter) |   FA2 speedup |   SDPA (s/iter) |   SDPA speedup |
|-------------------:|------------------:|-----------------:|---------------:|--------------:|----------------:|---------------:|
|                  1 |                 4 |            0.025 |          0.026 |         0.954 |           0.02  |          1.217 |
|                  1 |                16 |            0.026 |          0.028 |         0.918 |           0.02  |          1.287 |
|                  1 |                64 |            0.042 |          0.046 |         0.906 |           0.036 |          1.167 |
|                  4 |                 4 |            0.028 |          0.033 |         0.849 |           0.024 |          1.189 |
|                  4 |                16 |            0.034 |          0.035 |         0.955 |           0.029 |          1.169 |
|                  4 |                64 |            0.059 |          0.055 |         1.072 |           0.05  |          1.179 |
|                 16 |                 4 |            0.096 |          0.088 |         1.091 |           0.078 |          1.234 |
|                 16 |                16 |            0.102 |          0.09  |         1.129 |           0.083 |          1.224 |
|                 16 |                64 |            0.127 |          0.11  |         1.157 |           0.105 |          1.218 |
|                 32 |                 4 |            0.185 |          0.159 |         1.157 |           0.149 |          1.238 |
|                 32 |                16 |            0.19  |          0.162 |         1.177 |           0.154 |          1.233 |
|                 32 |                64 |            0.216 |          0.181 |         1.19  |           0.176 |          1.228 |

## 자료[[resources]]

CLIP을 시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다.

- [원격 센싱 (인공위성) 이미지와 캡션을 가지고 CLIP 미세조정하기](https://huggingface.co/blog/fine-tune-clip-rsicd):
[RSICD dataset](https://github.com/201528014227051/RSICD_optimal)을 가지고 CLIP을 미세조정 하는 방법과 데이터 증강에 대한 성능 비교에 대한 블로그 포스트

<PipelineTag pipeline="image-to-text"/>

- 사전학습된 CLIP모델을 이미지 캡셔닝을 위한 빔서치 추론에 어떻게 활용하는지에 관한 [노트북](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing)

**이미지 검색**

- 사전학습된 CLIP모델과 MRR(Mean Reciprocal Rank) 점수 연산을 사용한 이미지 검색에 대한 [노트북](https://colab.research.google.com/drive/1bLVwVKpAndpEDHqjzxVPr_9nGrSbuOQd?usp=sharing). 🌎
- 이미지 검색과 유사성 점수에 대해 보여주는 [노트북](https://colab.research.google.com/github/deep-diver/image_search_with_natural_language/blob/main/notebooks/Image_Search_CLIP.ipynb). 🌎
- Multilingual CLIP를 사용해서 이미지와 텍스트를 어떻게 같은 벡터 공간에 매핑 시키는지에 대한 [노트북](https://colab.research.google.com/drive/1xO-wC_m_GNzgjIBQ4a4znvQkvDoZJvH4?usp=sharing). 🌎
- [Unsplash](https://unsplash.com)와 [TMDB](https://www.themoviedb.org/) 데이터셋을 활용한 의미론적(semantic) 이미지 검색에서 CLIP을 구동하는 방법에 대한 [노트북](https://colab.research.google.com/github/vivien000/clip-demo/blob/master/clip.ipynb#scrollTo=uzdFhRGqiWkR). 🌎

**설명 가능성**

- 입력 토큰과 이미지 조각(segment) 사이의 유사성을 시각화 시키는 방법에 대한 [노트북](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb). 🌎

여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.

## CLIPConfig[[transformers.CLIPConfig]]

[[autodoc]] CLIPConfig

## CLIPTextConfig[[transformers.CLIPTextConfig]]

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig[[transformers.CLIPVisionConfig]]

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer[[transformers.CLIPTokenizer]]

[[autodoc]] CLIPTokenizer
    - get_special_tokens_mask
    - save_vocabulary

## CLIPTokenizerFast[[transformers.CLIPTokenizerFast]]

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor[[transformers.CLIPImageProcessor]]

[[autodoc]] CLIPImageProcessor
    - preprocess

## CLIPProcessor[[transformers.CLIPProcessor]]

[[autodoc]] CLIPProcessor


## CLIPModel[[transformers.CLIPModel]]

[[autodoc]] CLIPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPTextModel[[transformers.CLIPTextModel]]

[[autodoc]] CLIPTextModel
    - forward

## CLIPTextModelWithProjection[[transformers.CLIPTextModelWithProjection]]

[[autodoc]] CLIPTextModelWithProjection
    - forward

## CLIPVisionModelWithProjection[[transformers.CLIPVisionModelWithProjection]]

[[autodoc]] CLIPVisionModelWithProjection
    - forward

## CLIPVisionModel[[transformers.CLIPVisionModel]]

[[autodoc]] CLIPVisionModel
    - forward

## CLIPForImageClassification[[transformers.CLIPForImageClassification]]

[[autodoc]] CLIPForImageClassification
    - forward
