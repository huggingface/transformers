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

# 이미지 분류[[image-classification]]

[[open-in-colab]]

<Youtube id="tjAIM7BOYhw"/>

이미지 분류는 이미지에 레이블 또는 클래스를 할당합니다. 텍스트 또는 오디오 분류와 달리 입력은
이미지를 구성하는 픽셀 값입니다. 이미지 분류에는 자연재해 후 피해 감지, 농작물 건강 모니터링, 의료 이미지에서 질병의 징후 검사 지원 등
다양한 응용 사례가 있습니다.

이 가이드에서는 다음을 설명합니다:

1. [Food-101](https://huggingface.co/datasets/food101) 데이터 세트에서 [ViT](model_doc/vit)를 미세 조정하여 이미지에서 식품 항목을 분류합니다.
2. 추론을 위해 미세 조정 모델을 사용합니다.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/image-classification)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에, 필요한 모든 라이브러리가 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate
```

Hugging Face 계정에 로그인하여 모델을 업로드하고 커뮤니티에 공유하는 것을 권장합니다. 메시지가 표시되면, 토큰을 입력하여 로그인하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Food-101 데이터 세트 가져오기[[load-food101-dataset]]

🤗 Datasets 라이브러리에서 Food-101 데이터 세트의 더 작은 부분 집합을 가져오는 것으로 시작합니다. 이렇게 하면 전체 데이터 세트에 대한
훈련에 많은 시간을 할애하기 전에 실험을 통해 모든 것이 제대로 작동하는지 확인할 수 있습니다.

```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

데이터 세트의 `train`을 [`~datasets.Dataset.train_test_split`] 메소드를 사용하여 훈련 및 테스트 세트로 분할하세요:

```py
>>> food = food.train_test_split(test_size=0.2)
```

그리고 예시를 살펴보세요:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

데이터 세트의 각 예제에는 두 개의 필드가 있습니다:

- `image`: 식품 항목의 PIL 이미지
- `label`: 식품 항목의 레이블 클래스

모델이 레이블 ID에서 레이블 이름을 쉽게 가져올 수 있도록
레이블 이름을 정수로 매핑하고, 정수를 레이블 이름으로 매핑하는 사전을 만드세요:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

이제 레이블 ID를 레이블 이름으로 변환할 수 있습니다:

```py
>>> id2label[str(79)]
'prime_rib'
```

## 전처리[[preprocess]]

다음 단계는 이미지를 텐서로 처리하기 위해 ViT 이미지 프로세서를 가져오는 것입니다:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

이미지에 몇 가지 이미지 변환을 적용하여 과적합에 대해 모델을 더 견고하게 만듭니다. 여기서 Torchvision의 [`transforms`](https://pytorch.org/vision/stable/transforms.html) 모듈을 사용하지만, 원하는 이미지 라이브러리를 사용할 수도 있습니다.

이미지의 임의 부분을 크롭하고 크기를 조정한 다음, 이미지 평균과 표준 편차로 정규화하세요:

```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )
>>> _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```

그런 다음 전처리 함수를 만들어 변환을 적용하고 이미지의 `pixel_values`(모델에 대한 입력)를 반환하세요:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

전체 데이터 세트에 전처리 기능을 적용하려면 🤗 Datasets [`~datasets.Dataset.with_transform`]을 사용합니다. 데이터 세트의 요소를 가져올 때 변환이 즉시 적용됩니다:

```py
>>> food = food.with_transform(transforms)
```

이제 [`DefaultDataCollator`]를 사용하여 예제 배치를 만듭니다. 🤗 Transformers의 다른 데이터 콜레이터와 달리, `DefaultDataCollator`는 패딩과 같은 추가적인 전처리를 적용하지 않습니다.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```


## 평가[[evaluate]]

훈련 중에 평가 지표를 포함하면 모델의 성능을 평가하는 데 도움이 되는 경우가 많습니다.
🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리로 평가 방법을 빠르게 가져올 수 있습니다. 이 작업에서는
[accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 평가 지표를 가져옵니다. (🤗 Evaluate [빠른 둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하여 평가 지표를 가져오고 계산하는 방법에 대해 자세히 알아보세요):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

그런 다음 예측과 레이블을 [`~evaluate.EvaluationModule.compute`]에 전달하여 정확도를 계산하는 함수를 만듭니다:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

이제 `compute_metrics` 함수를 사용할 준비가 되었으며, 훈련을 설정하면 이 함수로 되돌아올 것입니다.

## 훈련[[train]]

<Tip>

[`Trainer`]를 사용하여 모델을 미세 조정하는 방법에 익숙하지 않은 경우, [여기](../training#train-with-pytorch-trainer)에서 기본 튜토리얼을 확인하세요!

</Tip>

이제 모델을 훈련시킬 준비가 되었습니다! [`AutoModelForImageClassification`]로 ViT를 가져옵니다. 예상되는 레이블 수, 레이블 매핑 및 레이블 수를 지정하세요:

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

이제 세 단계만 거치면 끝입니다:

1. [`TrainingArguments`]에서 훈련 하이퍼파라미터를 정의하세요. `image` 열이 삭제되기 때문에 미사용 열을 제거하지 않는 것이 중요합니다. `image` 열이 없으면 `pixel_values`을 생성할 수 없습니다. 이 동작을 방지하려면 `remove_unused_columns=False`로 설정하세요! 다른 유일한 필수 매개변수는 모델 저장 위치를 지정하는 `output_dir`입니다. `push_to_hub=True`로 설정하면 이 모델을 허브에 푸시합니다(모델을 업로드하려면 Hugging Face에 로그인해야 합니다). 각 에폭이 끝날 때마다, [`Trainer`]가 정확도를 평가하고 훈련 체크포인트를 저장합니다.
2. [`Trainer`]에 모델, 데이터 세트, 토크나이저, 데이터 콜레이터 및 `compute_metrics` 함수와 함께 훈련 인수를 전달하세요.
3. [`~Trainer.train`]을 호출하여 모델을 미세 조정하세요.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model",
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"],
...     eval_dataset=food["test"],
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

훈련이 완료되면, 모든 사람이 모델을 사용할 수 있도록 [`~transformers.Trainer.push_to_hub`] 메소드로 모델을 허브에 공유하세요:

```py
>>> trainer.push_to_hub()
```


<Tip>

이미지 분류를 위한 모델을 미세 조정하는 자세한 예제는 다음 [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)을 참조하세요.

</Tip>

## 추론[[inference]]

좋아요, 이제 모델을 미세 조정했으니 추론에 사용할 수 있습니다!

추론을 수행하고자 하는 이미지를 가져와봅시다:

```py
>>> ds = load_dataset("food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="image of beignets"/>
</div>

미세 조정 모델로 추론을 시도하는 가장 간단한 방법은 [`pipeline`]을 사용하는 것입니다. 모델로 이미지 분류를 위한 `pipeline`을 인스턴스화하고 이미지를 전달합니다:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]
```

원한다면, `pipeline`의 결과를 수동으로 복제할 수도 있습니다:

이미지를 전처리하기 위해 이미지 프로세서를 가져오고 `input`을 PyTorch 텐서로 반환합니다:

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

입력을 모델에 전달하고 logits을 반환합니다:

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

확률이 가장 높은 예측 레이블을 가져오고, 모델의 `id2label` 매핑을 사용하여 레이블로 변환합니다:

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
