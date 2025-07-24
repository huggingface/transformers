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

# 텍스트 음성 변환

[[open-in-colab]]

텍스트 음성 변환(TTS)은 텍스트에서 자연스러운 음성을 생성하는 것입니다. 이때, 음성은 다양한 언어로, 다양한 화자의 목소리로 생성될 수 있습니다. 현재 🤗 Transformers에서 [Bark](../model_doc/bark), [MMS](../model_doc/mms), [VITS](../model_doc/vits) and [SpeechT5](../model_doc/speecht5)와 같은 텍스트 음성 변환 모델들을 사용할 수 있습니다.

`"text-to-audio"` 파이프라인(또는 별칭인 `"text-to-speech"`)을 사용하여 손쉽게 오디오를 생성할 수 있습니다. Bark와 같은 일부 모델은 웃음, 한숨, 울음과 같은 비언어적 소통이나 음악도 조건에 따라 추가될 수 있습니다. Bark를 통해 이러한 방법으로 `"text-to-speech"` 파이프라인을 사용할 수 있습니다:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="suno/bark-small")
>>> text = "[clears throat] This is a test ... and I just took a long pause."
>>> output = pipe(text)
```

아래는 노트북에서 결과 오디오를 재생하는 데 사용할 수 있는 코드 예시입니다:

```python
>>> from IPython.display import Audio
>>> Audio(output["audio"], rate=output["sampling_rate"])
```

Bark 및 다른 사전 학습된 TTS 모델의 더 많은 활용 예시는 [오디오 강좌](https://huggingface.co/learn/audio-course/chapter6/pre-trained_models)를 참고하세요.

현재 🤗 Transformers에서 미세조정이 가능한 모델은 [SpeechT5](model_doc/speecht5) and [FastSpeech2Conformer](model_doc/fastspeech2_conformer) 뿐이지만, 차후에 더 많은 모델이 추가될 것입니다.

이 가이드의 나머지 부분은 아래 내용을 설명합니다:

1. 원래 영어 음성으로 학습된 [SpeechT5](../model_doc/speecht5) 모델을 [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 데이터세트의 네덜란드어 (`nl`) 부분을 사용하여 미세조정하기.

2. 파이프라인 사용 또는 모델 직접 호출을 통해 미세조정된 모델을 추론에 사용하기.

시작하기 전, 필요한 라이브러리들이 모두 설치되어있는지 확인하세요:

```bash
pip install datasets soundfile speechbrain accelerate
```

아직 SpeechT5의 모든 기능이 공식 릴리스에 통합되지 않았으므로 소스에서 🤗Transformers를 설치하세요:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

이 가이드를 따르기 위해서는 GPU가 필요합니다. 만약 노트북에서 작업 중이라면 아래 줄을 실행하여 GPU를 사용할 수 있는지 확인하세요:

```bash
!nvidia-smi
```

또는 AMD GPU의 경우:

```bash
!rocm-smi
```

</Tip>

Hugging Face 계정에 로그인하여 모델을 업로드하고 커뮤니티에 공유하는 것을 권장합니다. 메시지가 표시되면, 토큰을 입력하여 로그인하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 데이터세트 불러오기

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)는 2009년부터 2020년까지 유럽 의회 행사 녹음에서 추출한 데이터를 기반으로 하는 대규모 다국어 음성 코퍼스입니다. 15개 유럽 언어에 대한 레이블이 지정된 오디오 전사 데이터를 포함하고 있습니다. 이 가이드에서는 네덜란드어 부분을 사용하지만, 다른 다른 부분을 자유롭게 선택할 수 있습니다.

참고: VoxPopuli 또는 다른 자동 음성 인식(ASR) 데이터세트는 TTS(Text-to-Speech) 모델 훈련에 가장 적합하지 않을 수 있습니다. 과도한 배경 소음과 같이 ASR에 유용한 기능은 일반적으로 TTS에서는 바람직하지 않습니다. 그러나 최고 퀄리티의 다국어, 다중 화자 TTS 데이터세트를 찾는 것은 매우 어려울 수 있습니다.

데이터를 불러옵시다:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

20968개의 예시는 미세조정을 하기에 충분할 것입니다. SpeechT5는 오디오 데이터의 샘플링 속도가 16kHz일 것이라고 예상하므로, 데이터세트의 예시가 이 요구 사항을 만족하는지 확인하세요:

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## 데이터 전처리

사용할 모델 체크포인트를 정의하고 적절한 프로세서를 불러오는 것부터 시작합시다:

```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### SpeechT5 토큰화를 위한 텍스트 정리

텍스트 데이터를 정리하는 것부터 시작합니다. 텍스트를 처리하려면 프로세서의 토크나이저 부분이 필요할 것입니다:

```py
>>> tokenizer = processor.tokenizer
```

데이터세트 예시에는 `raw_text` 및 `normalized_text` 기능이 포함되어 있습니다. 텍스트 입력으로 어떤 기능을 사용할지 결정할 때, SpeechT5 토크나이저에 숫자에 대한 토큰이 없다는 점을 고려하세요. `normalized_text`에서는 숫자가 텍스트로 작성되어 있습니다. 따라서 `normalized_text`가 입력 텍스트로 더 적합하며, 이를 사용하는 것을 권장합니다.

SpeechT5는 영어로 훈련되었기 때문에 네덜란드어 데이터세트의 특정 문자를 인식하지 못할 수 있습니다. 그대로 두면 이 문자들은 `<unk>` 토큰으로 변환됩니다. 그러나 네덜란드어에서는 `à`와 같은 특정 문자가 음절을 강조하는 데 사용됩니다. 텍스트의 의미를 보존하기 위해 이 문자를 일반 `a`로 대체할 수 있습니다.

지원되지 않는 토큰을 식별하려면 문자를 토큰으로 사용하는 `SpeechT5Tokenizer`를 사용하여 데이터세트의 모든 고유 문자를 추출합니다. 이를 위해 모든 예시의 전사를 하나의 문자열로 연결하고 문자 집합으로 변환하는 `extract_all_chars` 매핑 함수를 작성하세요. 모든 전사를 한 번에 매핑 함수에서 사용할 수 있도록 `dataset.map()`에서 `batched=True` 및 `batch_size=-1`로 설정해야 합니다.

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

이제 두 가지 문자 집합이 있습니다: 하나는 데이터세트의 어휘, 다른 하나는 토크나이저의 어휘입니다. 데이터세트에서 지원되지 않는 문자를 식별하려면 이 두 집합의 차이를 취할 수 있습니다. 결과 집합에는 데이터세트에는 있지만 토크나이저에는 없는 문자들이 포함될 것입니다.

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

이전 단계에서 식별된 지원되지 않는 문자를 처리하려면 이러한 문자를 유효한 토큰에 매핑하는 함수를 정의합니다. 공백은 토크나이저에서 이미 `_`로 대체되므로 별도로 처리할 필요가 없습니다.

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

이제 텍스트의 특수 문자를 처리했으니, 오디오 데이터에 집중할 차례입니다.

### 화자

VoxPopuli 데이터tpxm에는 여러 화자의 음성이 포함되어 있는데, 데이터세트에는 몇 명의 화자가 나타나 있을까요? 이를 확인하기 위해 고유 화자의 수와 각 화자가 데이터세트 기여하는 예시의 수를 셀 수 있습니다. 데이터세트에 총 20,968개의 예시가 있으므로 이 정보는 데이터의 화자와 예시 분포에 대한 더 나은 이해를 제공할 것입니다.

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

히스토그램을 그리면 각 화자별 데이터 양을 파악할 수 있습니다.

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

히스토그램은 데이터세트의 약 3분의 1에 해당하는 화자가 100개 미만의 예시를 가지고 있고, 약 10명의 화자는 500개 이상의 예시를 가지고 있음을 보여줍니다. 훈련 효율성을 높이고 데이터세트의 균형을 맞추기 위해 100개에서 400개 사이의 예시를 가진 화자로 데이터를 제한할 수 있습니다.

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

남은 화자의 수를 확인해 보겠습니다.

```py
>>> len(set(dataset["speaker_id"]))
42
```

남은 예시의 수를 확인해 보겠습니다.

```py
>>> len(dataset)
9973
```

이제 약 40명의 고유 화자로부터 10,000개 미만의 예시가 남았으며, 이는 충분할 것입니다.

참고: 예시가 길면 예시가 적은 일부 화자도 더 많은 오디오를 사용할 수 있습니다. 그러나 각 화자별 총 오디오 양을 결정하려면 전체 데이터세트을 스캔해야 하며, 이는 각 오디오 파일을 로드하고 디코딩하는 시간이 오래 걸리는 과정입니다. 따라서 여기서는 이 단계를 건너뛰겠습니다.

### 화자 임베딩

TTS 모델이 여러 화자를 구분할 수 있도록 하려면 각 예시에 대한 화자 임베딩을 생성해야 합니다. 화자 임베딩은 특정 화자의 음성 특성을 포착하는 모델에 대한 추가 입력입니다. 이러한 화자 임베딩을 생성하려면 SpeechBrain의 사전 훈련된 [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) 모델을 사용합니다.

입력 오디오 파형을 가져와 해당 화자 임베딩을 포함하는 512요소 벡터를 출력하는 `create_speaker_embedding()` 함수를 생성합니다.

```py
>>> import os
>>> import torch
>>> from speechbrain.inference.classifiers import EncoderClassifier
>>> from accelerate.test_utils.testing import get_backend

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
>>> device, _, _ = get_backend() # 자동으로 기본 장치 유형(CUDA, CPU, XPU, MPS 등)을 감지합니다.
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

`speechbrain/spkrec-xvect-voxceleb` 모델은 VoxCeleb 데이터세트의 영어 음성으로 훈련되었지만, 이 가이드의 훈련 예시는 네덜란드어라는 점에 유의해야 합니다. 이 모델이 네덜란드어 데이터세트에 대해 합리적인 화자 임베딩을 생성할 것이라고 생각하지만, 이 가정이 항상 모든 경우에 적용되는 것은 아닙니다.

최적의 결과를 얻으려면 먼저 대상 음성에 대해 X-vector 모델을 훈련하는 것이 좋습니다. 이렇게 하면 모델이 네덜란드어에 존재하는 고유한 음성 특성을 더 잘 포착할 수 있습니다.

### 데이터셋 처리

마지막으로, 모델이 예상하는 형식으로 데이터를 처리해 보겠습니다. 단일 예시를 입력으로 받아 `SpeechT5Processor` 오브젝트를 사용하여 입력 텍스트를 토큰화하고 대상 오디오를 로그-멜 스펙트로그램으로 로드하는 `prepare_dataset` 함수를 만듭니다. 또한 화자 임베딩을 추가 입력으로 추가해야 합니다.

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

...     # SpeechBrain을 사용하여 x-vector를 얻습니다.
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

단일 예시를 살펴봄으로써 처리가 올바른지 확인합니다.

```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

화자 임베딩은 512요소 벡터여야 합니다.

```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

레이블은 80개의 멜 빈을 가진 로그-멜 스펙트로그램이어야 합니다.

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.imshow(processed_example["labels"].T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_1.png" alt="Log-mel spectrogram with 80 mel bins"/>
</div>

참고: 이 스펙트로그램이 혼란스럽다면, 낮은 주파수를 아래에, 높은 주파수를 위에 배치하는 일반적인 관습에 익숙하기 때문일 수 있습니다. 그러나 matplotlib 라이브러리를 사용하여 이미지를 스펙트로그램으로 플로팅할 때 y축이 뒤집혀 스펙트로그램이 거꾸로 나타납니다.

이제 처리 함수를 전체 데이터세트에 적용합니다. 이 과정은 5분에서 10분 정도 소요됩니다.

```py
>>> dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

데이터세트의 일부 예시가 모델이 처리할 수 있는 최대 입력 길이(600 토큰)보다 길다는 경고가 표시될 것입니다. 해당 예시들을 데이터세트에서 제거합니다. 여기서는 더 나아가 더 큰 배치 크기를 허용하기 위해 200 토큰 이상의 모든 것을 제거합니다.

```py
>>> def is_not_too_long(input_ids):
...     input_length = len(input_ids)
...     return input_length < 200


>>> dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
>>> len(dataset)
8259
```

다음으로, 기본적인 훈련/테스트 분할을 생성합니다.

```py
>>> dataset = dataset.train_test_split(test_size=0.1)
```

### 데이터 콜레이터

여러 예시를 배치로 결합하려면 사용자 지정 데이터 콜레이터를 정의해야 합니다. 이 콜레이터는 짧은 시퀀스를 패딩 토큰으로 채워 모든 예시가 동일한 길이를 갖도록 합니다. 스펙트로그램 레이블의 경우 패딩된 부분은 특수 값 `-100`으로 대체됩니다. 이 특수 값은 스펙트로그램 손실을 계산할 때 모델이 스펙트로그램의 해당 부분을 무시하도록 지시합니다.

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

...         # 입력과 대상을 배치로 정렬합니다.
...         batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

...         # 손실을 올바르게 무시하기 위해 패딩을 -100으로 대체합니다.
...         batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

...         # 미세조정 중에는 사용되지 않습니다.
...         del batch["decoder_attention_mask"]

...         # 감소 인자의 배수로 대상 길이를 반올림합니다.
...         if model.config.reduction_factor > 1:
...             target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
...             target_lengths = target_lengths.new(
...                 [length - length % model.config.reduction_factor for length in target_lengths]
...             )
...             max_length = max(target_lengths)
...             batch["labels"] = batch["labels"][:, :max_length]

...         # 화자 임베딩도 추가합니다.
...         batch["speaker_embeddings"] = torch.tensor(speaker_features)

...         return batch
```

SpeechT5에서 모델의 디코더 부분에 대한 입력은 인자 2만큼 줄어듭니다. 다시 말해, 대상 시퀀스에서 다른 모든 시간 단계를 버립니다. 그런 다음 디코더는 두 배 긴 시퀀스를 예측합니다. 원래 대상 시퀀스 길이가 홀수일 수 있으므로 데이터 콜레이터는 배치의 최대 길이가 2의 배수가 되도록 반올림합니다.

```py
>>> data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## 모델 훈련

프로세서를 로드하는 데 사용한 것과 동일한 체크포인트에서 사전 훈련된 모델을 로드합니다:

```py
>>> from transformers import SpeechT5ForTextToSpeech

>>> model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

`use_cache=True` 옵션은 그라디언트 체크포인팅과 호환되지 않습니다. 훈련을 위해 비활성화합니다.

```py
>>> model.config.use_cache = False
```

훈련 인자를 정의합니다. 여기서는 훈련 과정 중에 평가 지표를 계산하지 않습니다. 대신 손실만 확인합니다.

```python
>>> from transformers import Seq2SeqTrainingArguments

>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="speecht5_finetuned_voxpopuli_nl",  # 원하는 저장소 이름으로 변경
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

`Trainer` 오브젝트를 인스턴스화하고 모델, 데이터세트, 데이터 콜레이터를 전달합니다.

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

이제 훈련을 시작할 준비가 되었습니다! 훈련은 몇 시간이 걸릴 것입니다. GPU에 따라 훈련을 시작할 때 CUDA "메모리 부족" 오류가 발생할 수 있습니다. 이 경우 `per_device_train_batch_size`를 2의 인자만큼 점진적으로 줄이고 `gradient_accumulation_steps`를 2배로 늘려 보상할 수 있습니다.

```py
>>> trainer.train()
```

체크포인트를 파이프라인과 함께 사용하려면 프로세서를 체크포인트와 함께 저장해야 합니다:

```py
>>> processor.save_pretrained("YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

최종 모델을 🤗 Hub에 푸시합니다:

```py
>>> trainer.push_to_hub()
```

## 추론

### 파이프라인으로 추론

좋습니다. 이제 모델을 미세조정했으므로 추론에 사용할 수 있습니다!
먼저, 해당 파이프라인과 함께 사용하는 방법을 살펴보겠습니다. 체크포인트로 `"text-to-speech"` 파이프라인을 생성해 보겠습니다:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

네덜란드어로 내레이션하고 싶은 텍스트를 선택합니다. 예를 들어:

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

SpeechT5를 파이프라인과 함께 사용하려면 화자 임베딩이 필요합니다. 테스트 데이터세트의 예시에서 가져와 보겠습니다:

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

이제 텍스트와 화자 임베딩을 파이프라인에 전달하면 나머지는 파이프라인이 처리할 것입니다:

```py
>>> forward_params = {"speaker_embeddings": speaker_embeddings}
>>> output = pipe(text, forward_params=forward_params)
>>> output
{'audio': array([-6.82714235e-05, -4.26525949e-04,  1.06134125e-04, ...,
        -1.22392643e-03, -7.76011671e-04,  3.29112721e-04], dtype=float32),
 'sampling_rate': 16000}
```

결과를 들어볼 수 있습니다:

```py
>>> from IPython.display import Audio
>>> Audio(output['audio'], rate=output['sampling_rate'])
```

### 수동으로 추론

파이프라인을 사용하지 않고도 동일한 추론 결과를 얻을 수 있지만, 더 많은 단계가 필요할 것입니다,

🤗 Hub에서 모델을 로드합니다:

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

테스트 데이터셋에서 예시를 선택하여 화자 임베딩을 얻습니다.

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

입력 텍스트를 정의하고 토큰화합니다.

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
>>> inputs = processor(text=text, return_tensors="pt")
```

모델로 스펙트로그램을 생성합니다:

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

마지막으로, 보코더를 사용하여 스펙트로그램을 소리로 변환합니다.

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

저희 경험상 이 모델에서 만족스러운 결과를 얻는 것은 어려울 수 있습니다. 화자 임베딩의 퀄리티가 중요한 요소인 것으로 보입니다. SpeechT5는 영어 x-벡터로 사전 훈련되었기 때문에 영어 화자 임베딩을 사용할 때 가장 잘 작동합니다. 합성된 음성이 좋지 않다면 다른 화자 임베딩을 사용해 보세요.

훈련 기간을 늘리는 것도 결과의 품질을 향상시킬 가능성이 있습니다. 그럼에도 불구하고, 음성은 분명히 영어가 아닌 네덜란드어이며, 화자의 음성 특성을 포착합니다(예시의 원본 오디오와 비교했을 때).
실험할 또 다른 사항은 모델의 구성입니다. 예를 들어, `config.reduction_factor = 1`을 사용하여 결과가 향상되는지 확인해 보세요.

마지막으로 윤리적 고려 사항을 고려하는 것이 중요합니다. TTS 기술은 유용한 분야가 많지만, 타인의 음성을 허락이나 동의 없이 사용하는 것과 같이 좋진 않은 의도로 사용될 수 있습니다. TTS를 신중하고 책임감 있게 사용해 주시기 바랍니다.
