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

# 자동 음성 인식[[automatic-speech-recognition]]

[[open-in-colab]]

<Youtube id="TksaY_FDgnk"/>

자동 음성 인식(Automatic Speech Recognition, ASR)은 음성 신호를 텍스트로 변환하여 음성 입력 시퀀스를 텍스트 출력에 매핑합니다. 
Siri와 Alexa와 같은 가상 어시스턴트는 ASR 모델을 사용하여 일상적으로 사용자를 돕고 있으며, 회의 중 라이브 캡션 및 메모 작성과 같은 유용한 사용자 친화적 응용 프로그램도 많이 있습니다.

이 가이드에서 소개할 내용은 아래와 같습니다:

1. [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 데이터 세트에서 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)를 미세 조정하여 오디오를 텍스트로 변환합니다.
2. 미세 조정한 모델을 추론에 사용합니다.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/automatic-speech-recognition)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에 필요한 모든 라이브러리가 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate jiwer
```

Hugging Face 계정에 로그인하면 모델을 업로드하고 커뮤니티에 공유할 수 있습니다. 토큰을 입력하여 로그인하세요.

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## MInDS-14 데이터 세트 가져오기[[load-minds-14-dataset]]

먼저, 🤗 Datasets 라이브러리에서 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 데이터 세트의 일부분을 가져오세요. 
이렇게 하면 전체 데이터 세트에 대한 훈련에 시간을 들이기 전에 모든 것이 작동하는지 실험하고 검증할 수 있습니다.

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

[`~Dataset.train_test_split`] 메소드를 사용하여 데이터 세트의 `train`을 훈련 세트와 테스트 세트로 나누세요:

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

그리고 데이터 세트를 확인하세요:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

데이터 세트에는 `lang_id`와 `english_transcription`과 같은 유용한 정보가 많이 포함되어 있지만, 이 가이드에서는 `audio`와 `transcription`에 초점을 맞출 것입니다. 다른 열은 [`~datasets.Dataset.remove_columns`] 메소드를 사용하여 제거하세요:

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

예시를 다시 한번 확인해보세요:

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

두 개의 필드가 있습니다:

- `audio`: 오디오 파일을 가져오고 리샘플링하기 위해 호출해야 하는 음성 신호의 1차원 `array(배열)`
- `transcription`: 목표 텍스트

## 전처리[[preprocess]]

다음으로 오디오 신호를 처리하기 위한 Wav2Vec2 프로세서를 가져옵니다:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14 데이터 세트의 샘플링 레이트는 8000kHz이므로([데이터 세트 카드](https://huggingface.co/datasets/PolyAI/minds14)에서 확인), 사전 훈련된 Wav2Vec2 모델을 사용하려면 데이터 세트를 16000kHz로 리샘플링해야 합니다:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

위의 'transcription'에서 볼 수 있듯이 텍스트는 대문자와 소문자가 섞여 있습니다. Wav2Vec2 토크나이저는 대문자 문자에 대해서만 훈련되어 있으므로 텍스트가 토크나이저의 어휘와 일치하는지 확인해야 합니다:

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

이제 다음 작업을 수행할 전처리 함수를 만들어보겠습니다:

1. `audio` 열을 호출하여 오디오 파일을 가져오고 리샘플링합니다.
2. 오디오 파일에서 `input_values`를 추출하고 프로세서로 `transcription` 열을 토큰화합니다.

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

전체 데이터 세트에 전처리 함수를 적용하려면 🤗 Datasets [`~datasets.Dataset.map`] 함수를 사용하세요. `num_proc` 매개변수를 사용하여 프로세스 수를 늘리면 `map`의 속도를 높일 수 있습니다. [`~datasets.Dataset.remove_columns`] 메소드를 사용하여 필요하지 않은 열을 제거하세요:

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

🤗 Transformers에는 자동 음성 인식용 데이터 콜레이터가 없으므로 예제 배치를 생성하려면 [`DataCollatorWithPadding`]을 조정해야 합니다. 이렇게 하면 데이터 콜레이터는 텍스트와 레이블을 배치에서 가장 긴 요소의 길이에 동적으로 패딩하여 길이를 균일하게 합니다. `tokenizer` 함수에서 `padding=True`를 설정하여 텍스트를 패딩할 수 있지만, 동적 패딩이 더 효율적입니다.

다른 데이터 콜레이터와 달리 이 특정 데이터 콜레이터는 `input_values`와 `labels`에 대해 다른 패딩 방법을 적용해야 합니다.

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         # 입력과 레이블을 분할합니다
...         # 길이가 다르고, 각각 다른 패딩 방법을 사용해야 하기 때문입니다
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # 패딩에 대해 손실을 적용하지 않도록 -100으로 대체합니다
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

이제 `DataCollatorForCTCWithPadding`을 인스턴스화합니다:

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## 평가하기[[evaluate]]

훈련 중에 평가 지표를 포함하면 모델의 성능을 평가하는 데 도움이 되는 경우가 많습니다. 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리를 사용하면 평가 방법을 빠르게 불러올 수 있습니다. 
이 작업에서는 [단어 오류율(Word Error Rate, WER)](https://huggingface.co/spaces/evaluate-metric/wer) 평가 지표를 가져옵니다.
(평가 지표를 불러오고 계산하는 방법은 🤗 Evaluate [둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하세요):

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

그런 다음 예측값과 레이블을 [`~evaluate.EvaluationModule.compute`]에 전달하여 WER을 계산하는 함수를 만듭니다:

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

이제 `compute_metrics` 함수를 사용할 준비가 되었으며, 훈련을 설정할 때 이 함수로 되돌아올 것입니다.

## 훈련하기[[train]]

<frameworkcontent>
<pt>
<Tip>

[`Trainer`]로 모델을 미세 조정하는 것이 익숙하지 않다면, [여기](../training#train-with-pytorch-trainer)에서 기본 튜토리얼을 확인해보세요!

</Tip>

이제 모델 훈련을 시작할 준비가 되었습니다! [`AutoModelForCTC`]로 Wav2Vec2를 가져오세요. `ctc_loss_reduction` 매개변수로 CTC 손실에 적용할 축소(reduction) 방법을 지정하세요. 기본값인 합계 대신 평균을 사용하는 것이 더 좋은 경우가 많습니다:

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```

이제 세 단계만 남았습니다:

1. [`TrainingArguments`]에서 훈련 하이퍼파라미터를 정의하세요. `output_dir`은 모델을 저장할 경로를 지정하는 유일한 필수 매개변수입니다. `push_to_hub=True`를 설정하여 모델을 Hub에 업로드 할 수 있습니다(모델을 업로드하려면 Hugging Face에 로그인해야 합니다). [`Trainer`]는 각 에폭마다 WER을 평가하고 훈련 체크포인트를 저장합니다.
2. 모델, 데이터 세트, 토크나이저, 데이터 콜레이터, `compute_metrics` 함수와 함께 [`Trainer`]에 훈련 인수를 전달하세요.
3. [`~Trainer.train`]을 호출하여 모델을 미세 조정하세요.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     group_by_length=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     tokenizer=processor.feature_extractor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

훈련이 완료되면 모두가 모델을 사용할 수 있도록 [`~transformers.Trainer.push_to_hub`] 메소드를 사용하여 모델을 Hub에 공유하세요:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

자동 음성 인식을 위해 모델을 미세 조정하는 더 자세한 예제는 영어 자동 음성 인식을 위한 [블로그 포스트](https://huggingface.co/blog/fine-tune-wav2vec2-english)와 다국어 자동 음성 인식을 위한 [포스트](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)를 참조하세요.

</Tip>

## 추론하기[[inference]]

좋아요, 이제 모델을 미세 조정했으니 추론에 사용할 수 있습니다!

추론에 사용할 오디오 파일을 가져오세요. 필요한 경우 오디오 파일의 샘플링 비율을 모델의 샘플링 레이트에 맞게 리샘플링하는 것을 잊지 마세요!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

추론을 위해 미세 조정된 모델을 시험해보는 가장 간단한 방법은 [`pipeline`]을 사용하는 것입니다. 모델을 사용하여 자동 음성 인식을 위한 `pipeline`을 인스턴스화하고 오디오 파일을 전달하세요:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

<Tip>

텍스트로 변환된 결과가 꽤 괜찮지만 더 좋을 수도 있습니다! 더 나은 결과를 얻으려면 더 많은 예제로 모델을 미세 조정하세요!

</Tip>

`pipeline`의 결과를 수동으로 재현할 수도 있습니다:

<frameworkcontent>
<pt>
오디오 파일과 텍스트를 전처리하고 PyTorch 텐서로 `input`을 반환할 프로세서를 가져오세요:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

입력을 모델에 전달하고 로짓을 반환하세요:

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

가장 높은 확률의 `input_ids`를 예측하고, 프로세서를 사용하여 예측된 `input_ids`를 다시 텍스트로 디코딩하세요:

```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```
</pt>
</frameworkcontent>