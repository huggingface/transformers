<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# VideoMAE[[videomae]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## 개요[[overview]]

VideoMAE 모델은 Zhan Tong, Yibing Song, Jue Wang, Limin Wang이 작성한 [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://huggingface.co/papers/2203.12602) 논문에서 제안되었습니다.
VideoMAE는 마스크드 오토인코더([MAE](vit_mae))를 비디오로 확장하여 여러 비디오 분류 벤치마크에서 최고 수준의 성능을 달성했다고 주장합니다.

논문의 초록은 다음과 같습니다:

*대규모 데이터셋에서 비디오 트랜스포머를 사전 학습하는 것은 일반적으로 상대적으로 작은 데이터셋에서 최고의 성능을 달성하는 데 필요합니다. 이 논문에서는 비디오 마스크드 오토인코더(VideoMAE)가 자기 지도 비디오 사전 학습(SSVP)을 위한 데이터 효율적인 학습기임을 보여줍니다. 우리는 최근의 ImageMAE에서 영감을 받아 맞춤형 비디오 튜브 마스킹과 재구성을 제안합니다. 이러한 간단한 디자인은 비디오 재구성 중 시간적 상관관계로 인한 정보 누출을 극복하는 데 효과적인 것으로 나타났습니다. SSVP에 대해 세 가지 중요한 발견을 했습니다: (1) 극도로 높은 마스킹 비율(즉, 90%~95%)에서도 VideoMAE는 양호한 성능을 보입니다. 시간적으로 중복된 비디오 콘텐츠는 이미지보다 높은 마스킹 비율을 가능하게 합니다. (2) VideoMAE는 추가 데이터 없이 매우 작은 데이터셋(약 3k-4k 비디오)에서 인상적인 결과를 달성합니다. 이는 부분적으로 고수준 구조 학습을 강제하는 비디오 재구성의 도전적인 작업 때문입니다. (3) VideoMAE는 SSVP에서 데이터 양보다 데이터 품질이 더 중요함을 보여줍니다. 사전 학습과 대상 데이터셋 간의 도메인 이동은 SSVP에서 중요한 문제입니다. 특히, 바닐라 ViT 백본을 사용한 우리의 VideoMAE는 추가 데이터 없이 Kinetics-400에서 83.9%, Something-Something V2에서 75.3%, UCF101에서 90.8%, HMDB51에서 61.1%를 달성할 수 있습니다.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/videomae_architecture.jpeg"
alt="drawing" width="600"/>

<small> VideoMAE 사전 학습. <a href="https://huggingface.co/papers/2203.12602">원본 논문</a>에서 발췌. </small>

이 모델은 [nielsr](https://huggingface.co/nielsr)이 기여했습니다.
원본 코드는 [여기](https://github.com/MCG-NJU/VideoMAE)에서 찾을 수 있습니다.

## Scaled Dot Product Attention (SDPA) 사용하기[[using-scaled-dot-product-attention-sdpa]]

PyTorch에는 `torch.nn.functional`의 일부로 네이티브 scaled dot-product attention (SDPA) 연산자가 포함되어 있습니다. 이 함수에는 
입력 및 사용 중인 하드웨어에 따라 적용할 수 있는 여러 가지 구현이 포함되어 있습니다. 
[공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
또는 [GPU 추론](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
페이지에서 자세한 정보를 확인하세요.

구현이 가능한 경우 SDPA는 기본적으로 `torch>=2.1.1`에서 사용되지만, 
`from_pretrained()`에서 `attn_implementation="sdpa"`를 설정하여 명시적으로 SDPA를 사용하도록 요청할 수도 있습니다.

```
from transformers import VideoMAEForVideoClassification
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", attn_implementation="sdpa", torch_dtype=torch.float16)
...
```

최상의 속도 향상을 위해 모델을 half-precision(예: `torch.float16` 또는 `torch.bfloat16`)으로 로드하는 것을 권장합니다.

로컬 벤치마크(A100-40GB, PyTorch 2.3.0, OS Ubuntu 22.04)에서 `float32`와 `MCG-NJU/videomae-base-finetuned-kinetics` 모델을 사용하여 추론 중 다음과 같은 속도 향상을 확인했습니다.

|   Batch size |   Average inference time (ms), eager mode |   Average inference time (ms), sdpa model |   Speed up, Sdpa / Eager (x) |
|--------------|-------------------------------------------|-------------------------------------------|------------------------------|
|            1 |                                        37 |                                        10 |                      3.7  |
|            2 |                                        24 |                                        18 |                      1.33 |
|            4 |                                        43 |                                        32 |                      1.34 |
|            8 |                                        84 |                                        60 |                      1.4  |

## 리소스[[resources]]

VideoMAE를 시작하는 데 도움이 되는 공식 Hugging Face 및 커뮤니티(🌎로 표시) 리소스 목록입니다.
여기에 포함될 리소스를 제출하고 싶으시다면, Pull Request를 열어주시면 검토하겠습니다! 리소스는 기존 리소스를 복제하는 대신 새로운 것을 보여주는 것이 이상적입니다.

**비디오 분류**
- 사용자 정의 데이터셋에서 VideoMAE 모델을 미세 조정하는 방법을 보여주는 [노트북](https://github.com/huggingface/notebooks/blob/main/examples/video_classification.ipynb).
- [비디오 분류 작업 가이드](../tasks/video_classification)
- 비디오 분류 모델로 추론을 수행하는 방법을 보여주는 [🤗 Space](https://huggingface.co/spaces/sayakpaul/video-classification-ucf101-subset).

## VideoMAEConfig[[transformers.VideoMAEConfig]]

[[autodoc]] VideoMAEConfig

## VideoMAEFeatureExtractor[[transformers.VideoMAEFeatureExtractor]]

[[autodoc]] VideoMAEFeatureExtractor
    - __call__

## VideoMAEImageProcessor[[transformers.VideoMAEImageProcessor]]

[[autodoc]] VideoMAEImageProcessor
    - preprocess

## VideoMAEModel[[transformers.VideoMAEModel]]

[[autodoc]] VideoMAEModel
    - forward

## VideoMAEForPreTraining[[transformers.VideoMAEForPreTraining]]

`VideoMAEForPreTraining`은 자기 지도 사전 학습을 위한 디코더를 상단에 포함합니다.

[[autodoc]] transformers.VideoMAEForPreTraining
    - forward

## VideoMAEForVideoClassification[[transformers.VideoMAEForVideoClassification]]

[[autodoc]] transformers.VideoMAEForVideoClassification
    - forward
