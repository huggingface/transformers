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

# 이미지 프로세서(Image processor) [[image-processors]]

이미지 프로세서는 이미지를 픽셀 값, 즉 이미지의 색상과 크기를 나타내는 텐서로 변환합니다. 이 픽셀 값은 비전 모델의 입력으로 사용됩니다. 이때 사전 학습된 모델이 새로운 이미지를 올바르게 인식하려면 입력되는 이미지의 형식이 학습 당시 사용했던 데이터와 똑같아야 합니다. 이미지 프로세서는 다음과 같은 작업을 통해 이미지 형식을 통일시켜주는 역할을 합니다.

- 이미지 크기를 조절하는 [`~BaseImageProcessor.center_crop`] 
- 픽셀 값을 정규화하는 [`~BaseImageProcessor.normalize`] 또는 크기를 재조정하는 [`~BaseImageProcessor.rescale`]

Hugging Face [Hub](https://hf.co)나 로컬 디렉토리에 있는 비전 모델에서 이미지 프로세서의 설정(이미지 크기, 정규화 및 리사이즈 여부 등)을 불러오려면 [`~ImageProcessingMixin.from_pretrained`]를 사용하세요. 각 사전 학습된 모델의 설정은 [preprocessor_config.json](https://huggingface.co/google/vit-base-patch16-224/blob/main/preprocessor_config.json) 파일에 저장되어 있습니다.

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

이미지를 이미지 프로세서에 전달하여 픽셀 값으로 변환하고, `return_tensors="pt"` 를 설정하여 PyTorch 텐서를 반환받으세요. 이미지가 텐서로 어떻게 보이는지 궁금하다면 입력값을 한번 출력해보시는걸 추천합니다!

```py
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/image_processor_example.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = image_processor(image, return_tensors="pt")
```

이 가이드에서는 이미지 프로세서 클래스와 비전 모델을 위한 이미지 전처리 방법에 대해 다룰 예정입니다.

## 이미지 프로세서 클래스(Image processor classes) [[image-processor-classes]]

이미지 프로세서들은 [`~BaseImageProcessor.center_crop`], [`~BaseImageProcessor.normalize`], [`~BaseImageProcessor.rescale`] 함수를 제공하는 [`BaseImageProcessor`] 클래스를 상속받습니다. 이미지 프로세서에는 두 가지 종류가 있습니다.

- [`BaseImageProcessor`]는 파이썬 기반 구현체입니다.
- [`BaseImageProcessorFast`]는 더 빠른 [torchvision-backed](https://pytorch.org/vision/stable/index.html) 버전입니다. [torch.Tensor](https://pytorch.org/docs/stable/tensors.html)입력의 배치 처리 시 최대 33배 더 빠를 수 있습니다. [`BaseImageProcessorFast`]는 현재 모든 비전 모델에서 사용할 수 있는 것은 아니기 때문에 모델의 API 문서를 참조하여 지원 여부를 확인해 주세요.

각 이미지 프로세서는 이미지 프로세서를 불러오고 저장하기 위한 [`~ImageProcessingMixin.from_pretrained`]와 [`~ImageProcessingMixin.save_pretrained`] 메소드를 제공하는 [`ImageProcessingMixin`] 클래스를 상속받아 기능을 확장시킵니다.

이미지 프로세서를 불러오는 방법은 [`AutoImageProcessor`]를 사용하거나 모델별 이미지 프로세서를 사용하는 방식 두 가지가 있습니다.

<hfoptions id="image-processor-classes">
<hfoption id="AutoImageProcessor">

[AutoClass](./model_doc/auto) API는 이미지 프로세서가 어떤 모델과 연관되어 있는지 직접 지정하지 않고도 편리하게 불러올 수 있는 방법을 제공합니다.

[`~AutoImageProcessor.from_pretrained`]를 사용해 이미지 프로세서를 불러옵니다. 만약 빠른 프로세서를 사용하고 싶다면 `use_fast=True`를 추가하세요.

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
```

</hfoption>
<hfoption id="model-specific image processor">

각 이미지 프로세서는 특정 비전 모델에 맞춰져 있습니다. 따라서 프로세서의 설정 파일에는 해당 모델이 필요로 하는 이미지 크기나 정규화, 리사이즈 적용 여부 같은 정보가 담겨있습니다.

이러한 이미지 프로세서는 모델별 클래스에서 직접 불러올 수 있으며, 더 빠른 버전의 지원 여부는 해당 모델의 API 문서에서 확인 가능합니다.

```py
from transformers import ViTImageProcessor

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

빠른 이미지 프로세서를 불러오기 위해 fast 구현 클래스를 사용해보세요.

```py
from transformers import ViTImageProcessorFast

image_processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")
```

</hfoption>
</hfoptions>

## 빠른 이미지 프로세서(Fast image processors) [[fast-image-processors]]

[`BaseImageProcessorFast`]는 [torchvision](https://pytorch.org/vision/stable/index.html)을 기반으로 하며, 특히 GPU에서 처리할 때 속도가 훨씬 빠릅니다. 이 클래스는 기존 [`BaseImageProcessor`]와 완전히 동일하게 설계되었기 때문에, 모델이 지원한다면 별도 수정 없이 바로 교체해서 사용할 수 있습니다. [torchvision](https://pytorch.org/get-started/locally/#mac-installation)을 설치한 뒤 `use_fast` 파라미터를 `True`로 지정해주시면 됩니다.


```py
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
```

`device` 파라미터를 사용해 어느 장치에서 처리할지 지정할 수 있습니다. 만약 입력값이 텐서(tensor)라면 그 텐서와 동일한 장치에서, 그렇지 않은 경우에는 기본적으로 CPU에서 처리됩니다. 아래는 빠른 프로세서를 GPU에서 사용하도록 설정하는 예제입니다.

```py
from torchvision.io import read_image
from transformers import DetrImageProcessorFast

images = read_image("image.jpg")
processor = DetrImageProcessorFast.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(images, return_tensors="pt", device="cuda")
```

<details>
<summary>Benchmarks</summary>

이 벤치마크는 NVIDIA A10G Tensor Core GPU가 장착된 [AWS EC2 g5.2xlarge](https://aws.amazon.com/ec2/instance-types/g5/) 인스턴스에서 측정된 결과입니다.

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_padded.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_batched_compiled.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_single.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_batched.png" />
</div>
</details>

## 전처리(Preprocess) [[preprocess]]

Transformers의 비전 모델은 입력값으로 PyTorch 텐서 형태의 픽셀 값을 받습니다. 이미지 프로세서는 이미지를 바로 이 픽셀 값 텐서(배치 크기, 채널 수, 높이, 너비)로 변환하는 역할을 합니다. 이 과정에서 모델이 요구하는 크기로 이미지를 조절하고, 픽셀 값 또한 모델 기준에 맞춰 정규화하거나 재조정합니다.

이러한 이미지 전처리는 이미지 증강과는 다른 개념입니다. 이미지 증강은 학습 데이터를 늘리거나 과적합을 막기 위해 이미지에 의도적인 변화(밝기, 색상, 회전 등)를 주는 기술입니다. 반면, 이미지 전처리는 이미지를 사전 학습된 모델이 요구하는 입력 형식에 정확히 맞춰주는 작업에만 집중합니다.

일반적으로 모델 성능을 높이기 위해, 이미지는 보통 증강 과정을 거친 뒤 전처리되어 모델에 입력됩니다. 이때 증강 작업은 [Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb), [Kornia](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)) 와 같은 라이브러리를 사용할 수 있으며, 이후 전처리 단계에서 이미지 프로세서를 사용하면 됩니다.

이번 가이드에서는 이미지 증강을 위해 torchvision의 [transforms](https://pytorch.org/vision/stable/transforms.html) 모듈을 사용하겠습니다.

우선 [food101](https://hf.co/datasets/food101) 데이터셋의 일부만 샘플로 불러와서 시작하겠습니다.

```py
from datasets import load_dataset

dataset = load_dataset("ethz/food101", split="train[:100]")
```

[transforms](https://pytorch.org/vision/stable/transforms.html) 모듈의 [Compose](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html)API는 여러 변환을 하나로 묶어주는 역할을 합니다. 여기서는 이미지를 무작위로 자르고 리사이즈하는 [RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)과 색상을 무작위로 바꾸는 [ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html)를 함께 사용해보겠습니다.

이때 잘라낼 이미지의 크기는 이미지 프로세서에서 가져올 수 있습니다. 모델에 따라 정확한 높이와 너비가 필요할 때도 있고, 가장 짧은 변 `shortest_edge` 값만 필요할 때도 있습니다.

```py
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

준비된 변환값 들을 이미지에 적용하고, RGB 형식으로 바꿔줍니다. 그 다음, 이렇게 증강된 이미지를 이미지 프로세서에 넣어 픽셀 값을 반환합니다.

여기서 `do_resize`파라미터를 `False`로 설정한 이유는, 앞선 증강 단계에서 [RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)을 통해 이미 이미지 크기를 조절했기 때문입니다. 만약 증강 과정을 생략한다면, 이미지 프로세서는 `image_mean`과 `image_std`값(전처리기 설정 파일에 저장됨)을 사용해 자동으로 리사이즈와 정규화를 수행하게 됩니다.

```py
def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples
```

[`~datasets.Dataset.set_transform`]을 사용하면 결합된 증강 및 전처리 기능을 전체 데이터셋에 실시간으로 적용됩니다.

```py
dataset.set_transform(transforms)
```

이제 처리된 픽셀 값을 다시 이미지로 변환하여 증강 및 전처리 결과가 어떻게 나왔는지 직접 확인해 봅시다.

```py
import numpy as np
import matplotlib.pyplot as plt

img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">이전</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">이후</figcaption>
  </div>
</div>

이미지 프로세서는 전처리뿐만 아니라, 객체 탐지나 분할과 같은 비전 작업에서 모델의 결과값을 바운딩 박스나 분할 맵처럼 의미 있는 예측으로 바꿔주는 후처리 기능도 갖추고 있습니다.

### 패딩(Padding) [[padding]]

[DETR](./model_doc/detr)과 같은 일부 모델은 훈련 중에 [scale augmentation](https://paperswithcode.com/method/image-scale-augmentation)을 사용하기 때문에 한 배치 내에 포함된 이미지들의 크기가 제각각 일 수 있습니다. 아시다시피 크기가 서로 다른 이미지들은 하나의 배치로 묶을 수 없죠.

이 문제를 해결하려면 이미지에 특수 패딩 토큰인 `0`을 채워 넣어 크기를 통일시켜주면 됩니다. [pad](https://github.com/huggingface/transformers/blob/9578c2597e2d88b6f0b304b5a05864fd613ddcc1/src/transformers/models/detr/image_processing_detr.py#L1151) 메소드로 패딩을 적용하고, 이렇게 크기가 통일된 이미지들을 배치로 묶기 위해 사용자 정의 `collate` 함수를 만들어 사용하세요.

```py
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
```
