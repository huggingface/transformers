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

# 영상 분류 [[video-classification]]

[[open-in-colab]]


영상 분류는 영상 전체에 레이블 또는 클래스를 지정하는 작업입니다. 각 영상에는 하나의 클래스가 있을 것으로 예상됩니다. 영상 분류 모델은 영상을 입력으로 받아 어느 클래스에 속하는지에 대한 예측을 반환합니다. 이러한 모델은 영상이 어떤 내용인지 분류하는 데 사용될 수 있습니다. 영상 분류의 실제 응용 예는 피트니스 앱에서 유용한 동작 / 운동 인식 서비스가 있습니다. 이는 또한 시각 장애인이 이동할 때 보조하는데 사용될 수 있습니다

이 가이드에서는 다음을 수행하는 방법을 보여줍니다:

1. [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) 데이터 세트의 하위 집합을 통해 [VideoMAE](https://huggingface.co/docs/transformers/main/en/model_doc/videomae) 모델을 미세 조정하기.
2. 미세 조정한 모델을 추론에 사용하기.

> [!TIP]
> 이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/video-classification)를 확인하는 것이 좋습니다.


시작하기 전에 필요한 모든 라이브러리가 설치되었는지 확인하세요:
```bash
pip install -q pytorchvideo transformers evaluate
```

영상을 처리하고 준비하기 위해 [PyTorchVideo](https://pytorchvideo.org/)(이하 `pytorchvideo`)를 사용합니다.

커뮤니티에 모델을 업로드하고 공유할 수 있도록 Hugging Face 계정에 로그인하는 것을 권장합니다. 프롬프트가 나타나면 토큰을 입력하여 로그인하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## UCF101 데이터셋 불러오기 [[load-ufc101-dataset]]

[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) 데이터 세트의 하위 집합(subset)을 불러오는 것으로 시작할 수 있습니다. 전체 데이터 세트를 학습하는데 더 많은 시간을 할애하기 전에 데이터의 하위 집합을 불러와 모든 것이 잘 작동하는지 실험하고 확인할 수 있습니다.

```py
>>> from huggingface_hub import hf_hub_download

>>> hf_dataset_identifier = "sayakpaul/ucf101-subset"
>>> filename = "UCF101_subset.tar.gz"
>>> file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
```

데이터 세트의 하위 집합이 다운로드 되면, 압축된 파일의 압축을 해제해야 합니다:
```py
>>> import tarfile

>>> with tarfile.open(file_path) as t:
...      t.extractall(".")
```

전체 데이터 세트는 다음과 같이 구성되어 있습니다.

```bash
UCF101_subset/
    train/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    val/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    test/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
```


정렬된 영상의 경로는 다음과 같습니다:

```bash
...
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c04.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c06.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c02.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c06.avi'
...
```

동일한 그룹/장면에 속하는 영상 클립은 파일 경로에서 `g`로 표시되어 있습니다. 예를 들면, `v_ApplyEyeMakeup_g07_c04.avi`와 `v_ApplyEyeMakeup_g07_c06.avi` 이 있습니다. 이 둘은 같은 그룹입니다.

검증 및 평가 데이터 분할을 할 때, [데이터 누출(data leakage)](https://www.kaggle.com/code/alexisbcook/data-leakage)을 방지하기 위해 동일한 그룹 / 장면의 영상 클립을 사용하지 않아야 합니다. 이 튜토리얼에서 사용하는 하위 집합은 이러한 정보를 고려하고 있습니다.

그 다음으로, 데이터 세트에 존재하는 라벨을 추출합니다. 또한, 모델을 초기화할 때 도움이 될 딕셔너리(dictionary data type)를 생성합니다.

* `label2id`: 클래스 이름을 정수에 매핑합니다.
* `id2label`: 정수를 클래스 이름에 매핑합니다.

```py
>>> class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
>>> label2id = {label: i for i, label in enumerate(class_labels)}
>>> id2label = {i: label for label, i in label2id.items()}

>>> print(f"Unique classes: {list(label2id.keys())}.")

# Unique classes: ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress'].
```

이 데이터 세트에는 총 10개의 고유한 클래스가 있습니다. 각 클래스마다 30개의 영상이 훈련 세트에 있습니다

## 미세 조정하기 위해 모델 가져오기 [[load-a-model-to-fine-tune]]

사전 훈련된 체크포인트와 체크포인트에 연관된 이미지 프로세서를 사용하여 영상 분류 모델을 인스턴스화합니다. 모델의 인코더에는 미리 학습된 매개변수가 제공되며, 분류 헤드(데이터를 분류하는 마지막 레이어)는 무작위로 초기화됩니다. 데이터 세트의 전처리 파이프라인을 작성할 때는 이미지 프로세서가 유용합니다.

```py
>>> from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

>>> model_ckpt = "MCG-NJU/videomae-base"
>>> image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
>>> model = VideoMAEForVideoClassification.from_pretrained(
...     model_ckpt,
...     label2id=label2id,
...     id2label=id2label,
...     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
... )
```

모델을 가져오는 동안, 다음과 같은 경고를 마주칠 수 있습니다:

```bash
Some weights of the model checkpoint at MCG-NJU/videomae-base were not used when initializing VideoMAEForVideoClassification: [..., 'decoder.decoder_layers.1.attention.output.dense.bias', 'decoder.decoder_layers.2.attention.attention.key.weight']
- This IS expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of VideoMAEForVideoClassification were not initialized from the model checkpoint at MCG-NJU/videomae-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```


위 경고는 우리가 일부 가중치(예: `classifier` 층의 가중치와 편향)를 버리고 새로운 `classifier` 층의 가중치와 편향을 무작위로 초기화하고 있다는 것을 알려줍니다. 이 경우에는 미리 학습된 가중치가 없는 새로운 헤드를 추가하고 있으므로, 라이브러리가 모델을 추론에 사용하기 전에 미세 조정하라고 경고를 보내는 것은 당연합니다. 그리고 이제 우리는 이 모델을 미세 조정할 예정입니다.

**참고** 이 [체크포인트](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)는 도메인이 많이 중첩된 유사한 다운스트림 작업에 대해 미세 조정하여 얻은 체크포인트이므로 이 작업에서 더 나은 성능을 보일 수 있습니다. `MCG-NJU/videomae-base-finetuned-kinetics` 데이터 세트를 미세 조정하여 얻은 [체크포인트](https://huggingface.co/sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset)도 있습니다.

## 훈련을 위한 데이터 세트 준비하기[[prepare-the-datasets-for-training]]

영상 전처리를 위해 [PyTorchVideo 라이브러리](https://pytorchvideo.org/)를 활용할 것입니다. 필요한 종속성을 가져오는 것으로 시작하세요.

```py
>>> import pytorchvideo.data

>>> from pytorchvideo.transforms import (
...     ApplyTransformToKey,
...     Normalize,
...     RandomShortSideScale,
...     RemoveKey,
...     ShortSideScale,
...     UniformTemporalSubsample,
... )

>>> from torchvision.transforms import (
...     Compose,
...     Lambda,
...     RandomCrop,
...     RandomHorizontalFlip,
...     Resize,
... )
```

학습 데이터 세트 변환에는 '균일한 시간 샘플링(uniform temporal subsampling)', '픽셀 정규화(pixel normalization)', '랜덤 잘라내기(random cropping)' 및 '랜덤 수평 뒤집기(random horizontal flipping)'의 조합을 사용합니다. 검증 및 평가 데이터 세트 변환에는 '랜덤 잘라내기'와 '랜덤 뒤집기'를 제외한 동일한 변환 체인을 유지합니다. 이러한 변환에 대해 자세히 알아보려면 [PyTorchVideo 공식 문서](https://pytorchvideo.org)를 확인하세요.

사전 훈련된 모델과 관련된 이미지 프로세서를 사용하여 다음 정보를 얻을 수 있습니다:

* 영상 프레임 픽셀을 정규화하는 데 사용되는 이미지 평균과 표준 편차
* 영상 프레임이 조정될 공간 해상도


먼저, 몇 가지 상수를 정의합니다.

```py
>>> mean = image_processor.image_mean
>>> std = image_processor.image_std
>>> if "shortest_edge" in image_processor.size:
...     height = width = image_processor.size["shortest_edge"]
>>> else:
...     height = image_processor.size["height"]
...     width = image_processor.size["width"]
>>> resize_to = (height, width)

>>> num_frames_to_sample = model.config.num_frames
>>> sample_rate = 4
>>> fps = 30
>>> clip_duration = num_frames_to_sample * sample_rate / fps
```

이제 데이터 세트에 특화된 전처리(transform)과 데이터 세트 자체를 정의합니다. 먼저 훈련 데이터 세트로 시작합니다:

```py
>>> train_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     RandomShortSideScale(min_size=256, max_size=320),
...                     RandomCrop(resize_to),
...                     RandomHorizontalFlip(p=0.5),
...                 ]
...             ),
...         ),
...     ]
... )

>>> train_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "train"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
...     decode_audio=False,
...     transform=train_transform,
... )
```

같은 방식의 작업 흐름을 검증과 평가 세트에도 적용할 수 있습니다.

```py
>>> val_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     Resize(resize_to),
...                 ]
...             ),
...         ),
...     ]
... )

>>> val_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "val"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )

>>> test_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "test"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )
```


**참고**: 위의 데이터 세트의 파이프라인은 [공식 파이토치 예제](https://pytorchvideo.org/docs/tutorial_classification#dataset)에서 가져온 것입니다. 우리는 UCF-101 데이터셋에 맞게 [`pytorchvideo.data.Ucf101()`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.Ucf101) 함수를 사용하고 있습니다. 내부적으로 이 함수는 [`pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.LabeledVideoDataset) 객체를 반환합니다. `LabeledVideoDataset` 클래스는 PyTorchVideo 데이터셋에서 모든 영상 관련 작업의 기본 클래스입니다. 따라서 PyTorchVideo에서 미리 제공하지 않는 사용자 지정 데이터 세트를 사용하려면, 이 클래스를 적절하게 확장하면 됩니다. 더 자세한 사항이 알고 싶다면 `data` API [문서](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html) 를 참고하세요. 또한 위의 예시와 유사한 구조를 갖는 데이터 세트를 사용하고 있다면, `pytorchvideo.data.Ucf101()` 함수를 사용하는 데 문제가 없을 것입니다.

데이터 세트에 영상의 개수를 알기 위해 `num_videos` 인수에 접근할 수 있습니다.

```py
>>> print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
# (300, 30, 75)
```

## 더 나은 디버깅을 위해 전처리 영상 시각화하기[[visualize-the-preprocessed-video-for-better-debugging]]

```py
>>> import imageio
>>> import numpy as np
>>> from IPython.display import Image

>>> def unnormalize_img(img):
...     """Un-normalizes the image pixels."""
...     img = (img * std) + mean
...     img = (img * 255).astype("uint8")
...     return img.clip(0, 255)

>>> def create_gif(video_tensor, filename="sample.gif"):
...     """Prepares a GIF from a video tensor.
...
...     The video tensor is expected to have the following shape:
...     (num_frames, num_channels, height, width).
...     """
...     frames = []
...     for video_frame in video_tensor:
...         frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
...         frames.append(frame_unnormalized)
...     kargs = {"duration": 0.25}
...     imageio.mimsave(filename, frames, "GIF", **kargs)
...     return filename

>>> def display_gif(video_tensor, gif_name="sample.gif"):
...     """Prepares and displays a GIF from a video tensor."""
...     video_tensor = video_tensor.permute(1, 0, 2, 3)
...     gif_filename = create_gif(video_tensor, gif_name)
...     return Image(filename=gif_filename)

>>> sample_video = next(iter(train_dataset))
>>> video_tensor = sample_video["video"]
>>> display_gif(video_tensor)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif.gif" alt="Person playing basketball"/>
</div>

## 모델 훈련하기[[train-the-model]]

🤗 Transformers의 [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer)를 사용하여 모델을 훈련시켜보세요. `Trainer`를 인스턴스화하려면 훈련 설정과 평가 지표를 정의해야 합니다.  가장 중요한 것은 [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)입니다. 이 클래스는 훈련을 구성하는 모든 속성을 포함하며, 훈련 중 체크포인트를 저장할 출력 폴더 이름을 필요로 합니다. 또한 🤗 Hub의 모델 저장소의 모든 정보를 동기화하는 데 도움이 됩니다.

대부분의 훈련 인수는 따로 설명할 필요는 없습니다. 하지만 여기에서 중요한 인수는 `remove_unused_columns=False` 입니다. 이 인자는 모델의 호출 함수에서 사용되지 않는 모든 속성 열(columns)을 삭제합니다. 기본값은 일반적으로 True입니다. 이는 사용되지 않는 기능 열을 삭제하는 것이 이상적이며, 입력을 모델의 호출 함수로 풀기(unpack)가 쉬워지기 때문입니다. 하지만 이 경우에는 `pixel_values`(모델의 입력으로 필수적인 키)를 생성하기 위해 사용되지 않는 기능('video'가 특히 그렇습니다)이 필요합니다. 따라서 remove_unused_columns을 False로 설정해야 합니다.

```py
>>> from transformers import TrainingArguments, Trainer

>>> model_name = model_ckpt.split("/")[-1]
>>> new_model_name = f"{model_name}-finetuned-ucf101-subset"
>>> num_epochs = 4

>>> args = TrainingArguments(
...     new_model_name,
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=batch_size,
...     per_device_eval_batch_size=batch_size,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
...     max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
... )
```

`pytorchvideo.data.Ucf101()` 함수로 반환되는 데이터 세트는 `__len__` 메소드가 이식되어 있지 않습니다. 따라서,  `TrainingArguments`를 인스턴스화할 때 `max_steps`를 정의해야 합니다.

다음으로, 평가지표를 불러오고, 예측값에서 평가지표를 계산할 함수를 정의합니다. 필요한 전처리 작업은 예측된 로짓(logits)에 argmax 값을 취하는 것뿐입니다:

```py
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

**평가에 대한 참고사항**:

[VideoMAE 논문](https://huggingface.co/papers/2203.12602)에서 저자는 다음과 같은 평가 전략을 사용합니다. 테스트 영상에서 여러 클립을 선택하고 그 클립에 다양한 크롭을 적용하여 집계 점수를 보고합니다. 그러나 이번 튜토리얼에서는 간단함과 간결함을 위해 해당 전략을 고려하지 않습니다.

또한, 예제를 묶어서 배치를 형성하는 `collate_fn`을 정의해야합니다. 각 배치는 `pixel_values`와 `labels`라는 2개의 키로 구성됩니다.

```py
>>> def collate_fn(examples):
...     # permute to (num_frames, num_channels, height, width)
...     pixel_values = torch.stack(
...         [example["video"].permute(1, 0, 2, 3) for example in examples]
...     )
...     labels = torch.tensor([example["label"] for example in examples])
...     return {"pixel_values": pixel_values, "labels": labels}
```

그런 다음 이 모든 것을 데이터 세트와 함께 `Trainer`에 전달하기만 하면 됩니다:

```py
>>> trainer = Trainer(
...     model,
...     args,
...     train_dataset=train_dataset,
...     eval_dataset=val_dataset,
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
...     data_collator=collate_fn,
... )
```

데이터를 이미 처리했는데도 불구하고 `image_processor`를 토크나이저 인수로 넣은 이유는 JSON으로 저장되는 이미지 프로세서 구성 파일이 Hub의 저장소에 업로드되도록 하기 위함입니다.

`train` 메소드를 호출하여 모델을 미세 조정하세요:

```py
>>> train_results = trainer.train()
```

학습이 완료되면, 모델을 [`~transformers.Trainer.push_to_hub`] 메소드를 사용하여 허브에 공유하여 누구나 모델을 사용할 수 있도록 합니다:
```py
>>> trainer.push_to_hub()
```

## 추론하기[[inference]]

좋습니다. 이제 미세 조정된 모델을 추론하는 데 사용할 수 있습니다.

추론에 사용할 영상을 불러오세요:
```py
>>> sample_test_video = next(iter(test_dataset))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif_two.gif" alt="Teams playing basketball"/>
</div>

미세 조정된 모델을 추론에 사용하는 가장 간단한 방법은 [`pipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.VideoClassificationPipeline)에서 모델을 사용하는 것입니다. 모델로 영상 분류를 하기 위해 `pipeline`을 인스턴스화하고 영상을 전달하세요:

```py
>>> from transformers import pipeline

>>> video_cls = pipeline(model="my_awesome_video_cls_model")
>>> video_cls("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi")
[{'score': 0.9272987842559814, 'label': 'BasketballDunk'},
 {'score': 0.017777055501937866, 'label': 'BabyCrawling'},
 {'score': 0.01663011871278286, 'label': 'BalanceBeam'},
 {'score': 0.009560945443809032, 'label': 'BandMarching'},
 {'score': 0.0068979403004050255, 'label': 'BaseballPitch'}]
```

만약 원한다면 수동으로 `pipeline`의 결과를 재현할 수 있습니다:


```py
>>> def run_inference(model, video):
...     # (num_frames, num_channels, height, width)
...     perumuted_sample_test_video = video.permute(1, 0, 2, 3)
...     inputs = {
...         "pixel_values": perumuted_sample_test_video.unsqueeze(0),
...         "labels": torch.tensor(
...             [sample_test_video["label"]]
...         ),  # this can be skipped if you don't have labels available.
...     }

...     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...     inputs = {k: v.to(device) for k, v in inputs.items()}
...     model = model.to(device)

...     # forward pass
...     with torch.no_grad():
...         outputs = model(**inputs)
...         logits = outputs.logits

...     return logits
```

모델에 입력값을 넣고 `logits`을 반환받으세요:

```py
>>> logits = run_inference(trained_model, sample_test_video["video"])
```

`logits`을 디코딩하면, 우리는 다음 결과를 얻을 수 있습니다:

```py
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: BasketballDunk
```
