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

# 텍스트 음성 변환[[text-to-speech]]

[[open-in-colab]]

텍스트 음성 변환 (TTS)는 텍스트에서 자연스러운 음성을 생성하는 작업으로, 여러 언어와 여러 화자에 대해 음성을 생성할 수 있습니다. 🤗 Transformers에는 현재 [Bark](../model_doc/bark), [MMS](../model_doc/mms), [VITS](../model_doc/vits), [SpeechT5](../model_doc/speecht5)와 같은 여러 텍스트 음성 변환 모델이 있습니다.

`"text-to-audio"` 파이프라인(또는 별칭인 `"text-to-speech"`)을 사용하여 쉽게 오디오를 생성할 수 있습니다. Bark와 같은 일부 모델은 웃음, 한숨, 울음과 같은 비언어적 의사소통을 생성하거나 음악을 추가하도록 조건을 설정할 수도 있습니다.
다음은 Bark와 함께 `"text-to-speech"` 파이프라인을 사용하는 방법의 예시입니다:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="suno/bark-small")
>>> text = "[clears throat] This is a test ... and I just took a long pause."
>>> output = pipe(text)
```

노트북에서 결과 오디오를 듣기 위해 사용할 수 있는 코드 스니펫입니다:

```python
>>> from IPython.display import Audio
>>> Audio(output["audio"], rate=output["sampling_rate"])
```

Bark와 다른 사전 학습된 TTS 모델이 할 수 있는 더 많은 예시를 보려면, 저희 [Audio course](https://huggingface.co/learn/audio-course/chapter6/pre-trained_models)를 참조하세요.

TTS 모델을 미세 조정하려는 경우, 현재 🤗 Transformers에서 사용할 수 있는 text-to-speech 모델은 [SpeechT5](model_doc/speecht5)와 [FastSpeech2Conformer](model_doc/fastspeech2_conformer)뿐이며, 향후 더 많은 모델이 추가될 예정입니다. SpeechT5는 speech-to-text와 text-to-speech 데이터의 조합으로 사전 학습되어 텍스트와 음성이 공유하는 숨겨진 표현의 통합된 공간을 학습할 수 있습니다. 이는 동일한 사전 학습된 모델을 다른 작업에 대해 미세 조정할 수 있음을 의미합니다. 또한 SpeechT5는 x-vector 화자 임베딩을 통해 여러 화자를 지원합니다.

이 가이드의 나머지 부분에서는 다음 방법을 설명합니다:

1. 원래 영어 음성으로 학습된 [SpeechT5](../model_doc/speecht5)를 [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 데이터 세트의 네덜란드어(`nl`) 언어 하위 집합에서 미세 조정합니다.
2. 파이프라인을 사용하거나 직접적으로 사용하는 두 가지 방법 중 하나로 개선된 모델을 추론에 사용합니다.

시작하기 전에 필요한 모든 라이브러리가 설치되어 있는지 확인하세요:

```bash
pip install datasets soundfile speechbrain accelerate
```

모든 SpeechT5 기능이 아직 공식 릴리스에 병합되지 않았으므로 소스에서 🤗Transformers를 설치하세요:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

이 가이드를 따르려면 GPU가 필요합니다. 노트북에서 작업하는 경우, 다음 명령을 실행하여 GPU를 사용할 수 있는지 확인하세요:

```bash
!nvidia-smi
```

또는 AMD GPU의 경우:

```bash
!rocm-smi
```

</Tip>

Hugging Face 계정에 로그인하여 모델을 업로드하고 커뮤니티와 공유하는 것을 권장합니다. 메시지가 표시되면 토큰을 입력하여 로그인하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 데이터 세트 로드하기[[load-the-dataset]]

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)는 2009-2020년 유럽 의회 행사 녹음에서 수집된 데이터로 구성된 대규모 다국어 음성 코퍼스입니다. 15개 유럽 언어에 대한 라벨링된 오디오-전사 데이터를 포함합니다. 이 가이드에서는 네덜란드어 언어 하위 집합을 사용하며, 다른 하위 집합을 선택해도 됩니다.

VoxPopuli나 다른 자동 음성 인식(ASR) 데이터 세트는 TTS 모델 학습에 가장 적합한 옵션이 아닐 수 있습니다. 과도한 배경 소음과 같이 ASR에 유익한 특성들은 일반적으로 TTS에서는 바람직하지 않습니다. 그러나 고품질의 다국어, 다중화자 TTS 데이터 세트를 찾는 것은 매우 어려울 수 있습니다.

데이터를 로드해보겠습니다:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

20,968개의 예시는 미세 조정에 충분할 것입니다. SpeechT5는 오디오 데이터가 16kHz의 샘플링 속도를 가질 것으로 예상하므로, 데이터 세트의 예시가 이 요구사항을 충족하는지 확인하세요:

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## 데이터 전처리하기[[preprocess-the-data]]

사용할 모델 체크포인트를 정의하고 적절한 프로세서를 로드하는 것부터 시작하겠습니다:

```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### SpeechT5 토큰화를 위한 텍스트 정리[[text-cleanup-for-speecht5-tokenization]]

텍스트 데이터를 정리하는 것부터 시작하겠습니다. 텍스트를 처리하기 위해 프로세서의 토크나이저 부분이 필요합니다:

```py
>>> tokenizer = processor.tokenizer
```

데이터 세트 예시에는 `raw_text`와 `normalized_text` 특성이 있습니다. 텍스트 입력으로 어떤 특성을 사용할지 결정할 때, SpeechT5 토크나이저에는 숫자에 대한 토큰이 없다는 점을 고려하세요. `normalized_text`에서는 숫자가 텍스트로 작성되어 있으므로 더 적합합니다. 따라서, 입력 텍스트로 `normalized_text`를 사용하는 것을 권장합니다.

SpeechT5는 영어로 학습되었기 때문에 네덜란드어 데이터 세트의 특정 문자를 인식하지 못할 수 있습니다. 그대로 두면 이러한 문자들은 `<unk>` 토큰으로 변환됩니다. 그러나 네덜란드어에서는 `à`와 같은 특정 문자들이 음절을 강조하는 데 사용됩니다. 텍스트의 의미를 보존하기 위해 이 문자를 일반적인 `a`로 바꿀 수 있습니다.

지원되지 않는 토큰을 식별하기 위해 문자를 토큰으로 사용하는 `SpeechT5Tokenizer`를 사용하여 데이터 세트의 모든 고유 문자를 추출하세요. 이를 위해 모든 예시의 전사를 하나의 문자열로 연결하고 문자 집합으로 변환하는 `extract_all_chars` 매핑 함수를 작성하세요.
매핑 함수에서 모든 전사를 한 번에 사용 가능하도록 `dataset.map()`에서 `batched=True`와 `batch_size=-1`을 설정하세요.

```py
>>> def extract_all_chars(batch):
...     all_text = " ".join(batch["normalized_text"])
...     vocab = list(set(all_text))
...     return {"vocab": [vocab], "all_text": [all_text]}


>>> vocabs = dataset.map(
...     extract_all_chars,
...     batched=True,
...     batch_size=-1,
...     keep_in_memory=True,
...     remove_columns=dataset.column_names,
... )

>>> dataset_vocab = set(vocabs["vocab"][0])
>>> tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```

이제 두 개의 문자 집합이 있습니다: 하나는 데이터 세트의 어휘이고 다른 하나는 토크나이저의 어휘입니다.
데이터 세트에서 지원되지 않는 문자를 식별하기 위해 이 두 집합의 차이를 구할 수 있습니다. 결과 집합에는 데이터 세트에는 있지만 토크나이저에는 없는 문자가 포함됩니다.

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

이전 단계에서 식별된 지원되지 않는 문자를 처리하기 위해 이러한 문자를 유효한 토큰에 매핑하는 함수를 정의하세요. 공백은 토크나이저에서 이미 `▁`로 대체되므로 별도로 처리할 필요가 없습니다.

```py
>>> replacements = [
...     ("à", "a"),
...     ("ç", "c"),
...     ("è", "e"),
...     ("ë", "e"),
...     ("í", "i"),
...     ("ï", "i"),
...     ("ö", "o"),
...     ("ü", "u"),
... ]


>>> def cleanup_text(inputs):
...     for src, dst in replacements:
...         inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
...     return inputs


>>> dataset = dataset.map(cleanup_text)
```

이제 텍스트의 특수 문자를 처리했으므로, 오디오 데이터에 집중할 차례입니다.

### 화자[[speakers]]

VoxPopuli 데이터 세트는 여러 화자의 음성을 포함하지만, 데이터 세트에는 몇 명의 화자가 표현되어 있을까요? 이를 확인하기 위해 고유 화자 수와 각 화자가 데이터 세트에 기여하는 예시 수를 세어볼 수 있습니다.
데이터 세트에 총 20,968개의 예시가 있으므로, 이 정보는 데이터의 화자와 예시 분포를 더 잘 이해하는 데 도움이 됩니다.

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

히스토그램을 그려서 각 화자에 대해 얼마나 많은 데이터가 있는지 감을 잡을 수 있습니다.

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.hist(speaker_counts.values(), bins=20)
>>> plt.ylabel("Speakers")
>>> plt.xlabel("Examples")
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_speakers_histogram.png" alt="Speakers histogram"/>
</div>

히스토그램은 데이터 세트 화자의 약 3분의 1이 100개 미만의 예시를 가지고 있으며, 약 10명의 화자가 500개 이상의 예시를 가지고 있음을 보여줍니다. 훈련 효율성을 개선하고 데이터 세트의 균형을 맞추기 위해 100개에서 400개 사이의 예시를 가진 화자로 데이터를 제한할 수 있습니다.

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

남은 화자가 몇 명인지 확인해 보겠습니다:

```py
>>> len(set(dataset["speaker_id"]))
42
```

남은 예시가 몇 개인지 살펴보겠습니다:

```py
>>> len(dataset)
9973
```

약 40명의 고유 화자로부터 10,000개 미만의 예시가 남았으며, 이는 충분해야 할 것입니다.

적은 예시를 가진 일부 화자들의 각 예시의 길이가 길면 실제로 더 많은 오디오를 사용할 수 있습니다. 그러나 각 화자의 총 오디오 양을 구하려면 전체 데이터 세트를 스캔해야 하며, 이는 각 오디오 파일을 로드하고 디코딩하는 시간이 오래 걸리는 과정입니다. 따라서 여기서는 이 단계를 건너뛰기로 했습니다.

### 화자 임베딩[[speaker-embeddings]]

TTS 모델이 여러 화자를 구별할 수 있도록 하려면 각 예시에 대한 화자 임베딩을 만들어야 합니다.
화자 임베딩은 특정 화자의 음성 특성을 포착하는 모델에 대한 추가 입력입니다.
이러한 화자 임베딩을 생성하기 위해 SpeechBrain의 사전 학습된 [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) 모델을 사용하세요.

입력 오디오 파형을 받아 해당 화자 임베딩이 포함된 512 요소 벡터를 출력하는 `create_speaker_embedding()` 함수를 만드세요.

```py
>>> import os
>>> import torch
>>> from speechbrain.inference.classifiers import EncoderClassifier
>>> from accelerate.test_utils.testing import get_backend

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
>>> device, _, _ = get_backend() # 기본 장치 유형(CUDA, CPU, XPU, MPS 등)을 자동으로 감지합니다.
>>> speaker_model = EncoderClassifier.from_hparams(
...     source=spk_model_name,
...     run_opts={"device": device},
...     savedir=os.path.join("/tmp", spk_model_name),
... )


>>> def create_speaker_embedding(waveform):
...     with torch.no_grad():
...         speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
...         speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
...         speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
...     return speaker_embeddings
```

`speechbrain/spkrec-xvect-voxceleb` 모델은 VoxCeleb 데이터 세트의 영어 음성으로 학습되었으나, 이 가이드의 훈련 예시는 네덜란드어라는 점에 주목하는 것이 중요합니다. 이 모델이 우리의 네덜란드어 데이터 세트에 대해 여전히 합리적인 화자 임베딩을 생성할 것이라고 믿지만, 이 가정이 모든 경우에 적용되지 않을 수 있습니다.

최적의 결과를 위해 먼저 목표 음성에서 X-vector 모델을 훈련하는 것을 권장합니다. 이렇게 하면 모델이 네덜란드어에 존재하는 고유한 음성 특성을 더 잘 포착할 수 있습니다.

### 데이터 세트 처리하기[[processing-the-dataset]]

마지막으로, 모델이 예상하는 형식으로 데이터를 처리해 보겠습니다. 단일 예시를 받아 `SpeechT5Processor` 객체를 사용하여 입력 텍스트를 토큰화하고 대상 오디오를 log-mel 스펙트로그램으로 로드하는 `prepare_dataset` 함수를 만드세요.
또한 화자 임베딩을 추가 입력으로 추가해야 합니다.

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example = processor(
...         text=example["normalized_text"],
...         audio_target=audio["array"],
...         sampling_rate=audio["sampling_rate"],
...         return_attention_mask=False,
...     )

...     # 배치 차원을 제거합니다.
...     example["labels"] = example["labels"][0]

...     # SpeechBrain을 사용하여 x-vector를 추출합니다.
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

단일 예시를 살펴보면서 처리가 올바른지 확인하세요:

```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

화자 임베딩은 512 요소 벡터여야 합니다:

```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

라벨은 80개의 mel bin이 있는 log-mel 스펙트로그램이어야 합니다.

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.imshow(processed_example["labels"].T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_1.png" alt="Log-mel spectrogram with 80 mel bins"/>
</div>

참고: 이 스펙트로그램이 혼란스럽다면, 낮은 주파수를 플롯의 아래쪽에, 높은 주파수를 위쪽에 배치하는 관례에 익숙하기 때문일 수 있습니다. 그러나 matplotlib 라이브러리를 사용하여 스펙트로그램을 이미지로 플롯할 때 y축이 뒤집어져 스펙트로그램이 거꾸로 나타납니다.

이제 처리 함수를 전체 데이터 세트에 적용하세요. 5분에서 10분 정도 걸립니다.

```py
>>> dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

데이터 세트의 일부 예시가 모델이 처리할 수 있는 최대 입력 길이(600 토큰)보다 길다는 경고가 표시됩니다.
해당 예시를 데이터 세트에서 제거하세요. 여기서는 더 나아가 더 큰 배치 크기를 허용하기 위해 200 토큰 이상인 모든 것을 제거합니다.

```py
>>> def is_not_too_long(input_ids):
...     input_length = len(input_ids)
...     return input_length < 200


>>> dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
>>> len(dataset)
8259
```

다음으로, 기본적인 train/test 분할을 만드세요:

```py
>>> dataset = dataset.train_test_split(test_size=0.1)
```

### 데이터 콜레이터[[data-collator]]

여러 예시를 배치로 결합하려면 사용자 정의 데이터 콜레이터를 정의해야 합니다. 이 콜레이터는 더 짧은 시퀀스를 패딩 토큰으로 패딩하여 모든 예시가 동일한 길이를 갖도록 합니다. 스펙트로그램 라벨의 경우, 패딩된 부분은 특수 값 `-100`으로 대체됩니다. 이 특수 값은 스펙트로그램 손실을 계산할 때 모델이 해당 부분을 무시하도록 지시합니다.

```py
>>> from dataclasses import dataclass
>>> from typing import Any, Dict, List, Union


>>> @dataclass
... class TTSDataCollatorWithPadding:
...     processor: Any

...     def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
...         input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
...         label_features = [{"input_values": feature["labels"]} for feature in features]
...         speaker_features = [feature["speaker_embeddings"] for feature in features]

...         # 입력과 타겟을 하나의 배치로 모읍니다.
...         batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

...         # 손실 계산에서 무시되도록 패딩 값을 -100으로 대체합니다.
...         batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

...         # 미세 조정 시에는 사용되지 않습니다.
...         del batch["decoder_attention_mask"]

...         # 타겟 길이를 감소 비율(reduction factor)의 배수로 내림(round down) 처리합니다.
...         if model.config.reduction_factor > 1:
...             target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
...             target_lengths = target_lengths.new(
...                 [length - length % model.config.reduction_factor for length in target_lengths]
...             )
...             max_length = max(target_lengths)
...             batch["labels"] = batch["labels"][:, :max_length]

...         # 또한 화자 임베딩(speaker embeddings)도 함께 추가합니다.
...         batch["speaker_embeddings"] = torch.tensor(speaker_features)

...         return batch
```

SpeechT5에서는 모델의 디코더 부분에 대한 입력이 2배로 줄어듭니다. 즉, 대상 시퀀스에서 다른 모든 타임스텝을 버립니다. 그런 다음 디코더는 두 배 길이의 시퀀스를 예측합니다. 원래 대상 시퀀스 길이가 홀수일 수 있으므로, 데이터 콜레이터는 배치의 최대 길이가 2의 배수가 되도록 반내림합니다.

```py
>>> data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## 모델 훈련하기[[train-the-model]]

프로세서를 로드하는 데 사용한 것과 동일한 체크포인트에서 사전 학습된 모델을 로드하세요:

```py
>>> from transformers import SpeechT5ForTextToSpeech

>>> model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

`use_cache=True` 옵션은 그래디언트 체크포인팅과 호환되지 않습니다. 훈련을 위해 비활성화하세요.

```py
>>> model.config.use_cache = False
```

훈련 인수를 정의하세요. 여기서는 훈련 과정에서 평가 메트릭을 계산하지 않습니다. 대신 손실만 살펴보겠습니다:

```python
>>> from transformers import Seq2SeqTrainingArguments

>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
...     per_device_train_batch_size=4,
...     gradient_accumulation_steps=8,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=4000,
...     gradient_checkpointing=True,
...     fp16=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=2,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     report_to=["tensorboard"],
...     load_best_model_at_end=True,
...     greater_is_better=False,
...     label_names=["labels"],
...     push_to_hub=True,
... )
```

`Trainer` 객체를 인스턴스화하고 모델, 데이터 세트, 데이터 콜레이터를 전달하세요.

```py
>>> from transformers import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
...     args=training_args,
...     model=model,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     data_collator=data_collator,
...     processing_class=processor,
... )
```

이제 훈련을 시작할 준비가 되었습니다! 훈련에는 몇 시간이 소요될 수 있습니다. GPU에 따라 훈련을 시작할 때 CUDA "out-of-memory" 오류가 발생할 수 있습니다. 이 경우 `per_device_train_batch_size`를 2배씩 점진적으로 줄이고 `gradient_accumulation_steps`를 2배로 늘려서 보상할 수 있습니다.

```py
>>> trainer.train()
```

파이프라인과 함께 체크포인트를 사용할 수 있도록 하려면, 체크포인트와 함께 프로세서를 저장하세요:

```py
>>> processor.save_pretrained("YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

최종 모델을 🤗 Hub에 푸시하세요:

```py
>>> trainer.push_to_hub()
```

## 추론[[inference]]

### 파이프라인으로 추론하기[[inference-with-a-pipeline]]

훌륭합니다. 이제 모델을 미세 조정했으므로 추론에 사용할 수 있습니다!
먼저 해당 파이프라인과 함께 사용하는 방법을 살펴보겠습니다. 체크포인트로 `"text-to-speech"` 파이프라인을 만들어보겠습니다:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

내레이션하고 싶은 네덜란드어 텍스트를 선택하세요. 예를 들어:

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

파이프라인과 함께 SpeechT5를 사용하려면 화자 임베딩이 필요합니다. 테스트 데이터 세트의 예시에서 가져와보겠습니다:

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

이제 텍스트와 화자 임베딩을 파이프라인에 전달할 수 있으며, 나머지는 파이프라인이 처리합니다:

```py
>>> forward_params = {"speaker_embeddings": speaker_embeddings}
>>> output = pipe(text, forward_params=forward_params)
>>> output
{'audio': array([-6.82714235e-05, -4.26525949e-04,  1.06134125e-04, ...,
        -1.22392643e-03, -7.76011671e-04,  3.29112721e-04], dtype=float32),
 'sampling_rate': 16000}
```

그런 다음 결과를 들어볼 수 있습니다:

```py
>>> from IPython.display import Audio
>>> Audio(output['audio'], rate=output['sampling_rate'])
```

### 수동으로 추론 실행하기[[run-inference-manually]]

파이프라인을 사용하지 않고도 동일한 추론 결과를 얻을 수 있지만, 더 많은 단계가 필요합니다.

🤗 Hub에서 모델을 로드하세요:

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

테스트 데이터 세트에서 예시를 선택하여 화자 임베딩을 얻으세요.

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

입력 텍스트를 정의하고 토큰화하세요.

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
>>> inputs = processor(text=text, return_tensors="pt")
```

모델로 스펙트로그램을 생성하세요:

```py
>>> spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

원한다면 스펙트로그램을 시각화하세요:

```py
>>> plt.figure()
>>> plt.imshow(spectrogram.T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_2.png" alt="Generated log-mel spectrogram"/>
</div>

마지막으로, 보코더를 사용하여 스펙트로그램을 소리로 변환하세요.

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

저희 경험상 이 모델에서 만족스러운 결과를 얻는 것은 어려울 수 있습니다. 화자 임베딩의 품질이 중요한 요소로 보입니다. SpeechT5는 영어 x-vector로 사전 학습되었기 때문에 영어 화자 임베딩을 사용할 때 가장 잘 작동합니다. 합성된 음성이 좋지 않게 들리면 다른 화자 임베딩을 사용해 보세요.

훈련 기간을 늘리는 것도 결과의 품질을 향상시킬 가능성이 높습니다. 그런데도 음성은 분명히 영어가 아닌 네덜란드어이며 화자의 음성 특성을 포착합니다(예시의 원본 오디오와 비교).
실험해 볼 수 있는 또 다른 부분은 모델의 구성입니다. 예를 들어, `config.reduction_factor = 1`을 사용하여 결과가 개선되는지 확인해보세요.

마지막으로, 윤리적 고려사항을 검토하는 것이 중요합니다. TTS 기술은 수많은 유용한 응용 프로그램을 가지고 있지만, 허락이나 동의 없이 타인의 음성을 흉내 내는 것과 같은 악의적인 목적으로도 사용될 수 있습니다. TTS를 신중하고 책임감 있게 사용해 주세요.
