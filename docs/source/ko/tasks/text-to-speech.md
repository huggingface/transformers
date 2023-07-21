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

# Text to speech

[[open-in-colab]]

텍스트 음성화(Text-to-speech, TTS)는 텍스트로부터 자연스러운 음성을 생성하는 작업으로, 음성은 여러 언어와 여러 스피커에 대해 생성될 수 있습니다. 현재 🤗 Transformers에서 사용 가능한 유일한 텍스트 음성화 모델은 [SpeechT5](model_doc/speecht5)입니다. 하지만 앞으로 더 많은 모델이 추가될 예정입니다. SpeechT5는 텍스트와 음성 사이에서 공유되는 숨겨진 표현 공간을 학습하기 위해 음성-텍스트 및 텍스트-음성 데이터의 조합으로 사전 훈련되었습니다. 이는 동일한 사전 훈련 모델을 다른 작업에 대해 세밀하게 조정할 수 있음을 의미합니다. 또한, SpeechT5는 x-vector 스피커 임베딩을 통해 다중 스피커를 지원합니다.

이 가이드에서는 다음을 수행하는 방법을 설명합니다:

1. 원래 영어 음성으로 훈련된 [SpeechT5](model_doc/speecht5)를 네덜란드어 (`nl`) 언어 부분 집합의 [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 데이터셋에 대해 세밀 조정합니다.
2. 세밀 조정된 모델을 추론에 사용합니다.

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
pip install datasets soundfile speechbrain accelerate
```

SpeechT5의 일부 기능이 아직 공식 릴리스에 통합되지 않았으므로 🤗 Transformers를 소스에서 설치하세요:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

이 가이드를 따르기 위해 GPU가 필요합니다. 노트북에서 작업하고 있는 경우 다음 명령을 실행하여 GPU가 사용 가능한지 확인하세요.

```bash
!nvidia-smi
```

</Tip>

Hugging Face 계정에 로그인하여 모델을 업로드하고 공유하는 것을 권장합니다. 요청 시 토큰을 입력하여 로그인하세요.

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 데이터셋 로드

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)는 2009-2020년 유럽 의회 행사 녹음에서 추출된 대규모 다국어 음성 말뭉치입니다. 이 데이터셋은 15개 유럽 언어에 대한 레이블이 지정된 오디오-전사 데이터를 포함하고 있습니다. 이 가이드에서는 네덜란드어 언어 부분을 사용하며, 다른 부분을 선택해도 괜찮습니다.

VoxPopuli나 다른 자동 음성 인식(ASR) 데이터셋은 TTS 모델에 가장 적합하지 않을 수 있습니다. ASR에 유용한 기능(예: 과도한 배경 소음)은 일반적으로 TTS에는 바람직하지 않습니다. 그러나 고품질의 다국어 및 다중 스피커 TTS 데이터셋을 찾는 것은 꽤 어려울 수 있습니다.

데이터를 로드해 보겠습니다.

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

세부 사항은 생략하고 훈련에 충분한 20,968개의 예제를 사용하겠습니다. SpeechT5는 오디오 데이터의 샘플링 속도가 16 kHz여야 한다는 점을 유의하세요. 데이터셋의 예제가 이 요구 사항을 충족하는지 확인하세요.

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## 데이터 전처리

모델 체크포인트를 정의하고 적절한 프로세서를 로드하는 것부터 시작합니다:

```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### SpeechT5 토큰화를 위한 텍스트 정리

텍스트 데이터를 정리하는 것으로 시작합니다. 텍스트를 처리하기 위해 프로세서의 토크나이저 부분이 필요합니다:

```py
>>> tokenizer = processor.tokenizer
```

데이터셋 예제에는 `raw_text`와 `normalized_text` 특성이 있습니다. 텍스트 입력으로 사용할 특성을 결정할 때, SpeechT5 토크나이저에는 숫자를 나타내는 토큰이 없습니다. `normalized_text`에서는 숫자가 텍스트로 작성되어 있습니다. 따라서 `normalized_text`를 더 적합하며 입력 텍스트로 사용하기를 권장합니다.

SpeechT5는 영어로 훈련되었기 때문에 네덜란드어 데이터셋에서 일부 문자를 인식하지 못할 수 있습니다. 그대로 둔다면 이러한 문자는 `<unk>` 토큰으로 변환됩니다. 그러나 네덜란드어에서는 `à`와 같은 특정 문자가 음절을 강조하는 데 사용됩니다. 텍스트의 의미를 보존하기 위해 이 문자를 일반적인 `a`로 대체할 수 있습니다.

지원되지 않는 토큰을 식별하려면, `SpeechT5Tokenizer`를 사용하여 데이터셋에서 모든 고유한 문자를 추출합니다. 이를 위해 모든 예제의 전사를 하나의 문자열로 연결하고 문자 집합으로 변환하는 `extract_all_chars` 매핑 함수를 작성하세요. `dataset.map()`에서 `batched=True` 및 `batch_size=-1`를 설정하여 매핑 함수에서 모든 전사가 동시에 사용할 수 있도록 해야 합니다.

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

이제 두 개의 문자 집합이 있습니다. 하나는 데이터셋의 어휘이고 다른 하나는 토크나이저의 어휘입니다. 이 두 집합 사이의 차이를 취하면 데이터셋에 있는데 토크나이저에는 없는 문자가 포함된 집합이 됩니다.

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

이전 단계에서 식별한 지원되지 않는 문자를 처리하기 위해 이러한 문자를 유효한 토큰에 매핑하는 함수를 정의하세요. 토크나이저에서 이미 공백은 `▁`로 대체되었으므로 별도로 처리할 필요가 없습니다.

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

텍스트의 특수 문자를 처리했으므로 이제 오디오 데이터에 초점을 맞추겠습니다.

### 스피커

VoxPopuli 데이터셋에는 여러 스피커의 음성이 포함되어 있지만, 데이터셋에는 몇 명의 스피커가 포함되어 있는지 알아야 합니다. 이를 확인하기 위해 고유한 스피커 수와 각 스피커가 데이터셋에 기여하는 예제 수를 세어볼 수 있습니다. 데이터셋에는 총 20,968개의 예제가 있으므로 이 정보를 통해 데이터의 스피커 및 예제 분포에 대한 더 나은 이해를 얻을 수 있습니다.

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

히스토그램을 플롯하여 각 스피커에 대한 데이터의 양을 파악할 수 있습니다.

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

히스토그램에서 약 1/3 정도의 스피커가 100개 미만의 예제를 가지고 있고, 약 10명의 스피커가 500개 이상의 예제를 가지고 있음을 알 수 있습니다. 훈련 효율성을 높이고 데이터셋을 균형있게 조정하기 위해 100개에서 400개 사이의 예제를 가진 스피커로 데이터를 제한할 수 있습니다.

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

남은 스피커 수를 확인해 보겠습니다:

```py
>>> len(set(dataset["speaker_id"]))
42
```

남은 예제 수를 확인해 보겠습니다:

```py
>>> len(dataset)
9973
```

약 40명의 고유한 스피커로부터 약 10,000개의 예제가 남았습니다. 충분한 양입니다.

몇 개의 예제를 가진 일부 스피커는 예제가 길다면 실제로 더 많은 오디오를 사용할 수 있습니다. 그러나 각 스피커에 대한 총 오디오 양을 결정하려면 데이터셋 전체를 스캔해야 하므로 시간이 많이 소요되는 작업입니다. 따라서 이 단계를 생략하기로 결정했습니다.

### 스피커 임베딩

TTS 모델이 다중 스피커를 구분할 수 있도록 하려면 각 예제에 대해 스피커 임베딩을 생성해야 합니다. 스피커 임베딩은 해당 스피커의 음성 특성을 포착하는 모델의 추가 입력입니다. 이를 위해 SpeechBrain의 사전 훈련된 [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) 모델을 사용합니다.

입력 오디오 웨이브폼을 가져와 해당 스피커 임베딩을 포함하는 512개 요소 벡터를 출력하는 `create_speaker_embedding()` 함수를 생성하세요.

```py
>>> import os
>>> import torch
>>> from speechbrain.pretrained import EncoderClassifier

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
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

`speechbrain/spkrec-xvect-voxceleb` 모델은 VoxCeleb 데이터셋의 영어 음성에서 훈련되었습니다. 이 가이드의 훈련 예제는 네덜란드어입니다. 우리는 이 모델이 네덜란드어 데이터셋에 대해 합리적인 스피커 임베딩을 생성할 것으로 생각하지만, 모든 경우에 이 가정이 옳지 않을 수도 있습니다.

최적의 결과를 얻기 위해 목표 음성에서 X-vector 모델을 훈련하는 것을 권장합니다. 이렇게 하면 모델이 네덜란드어에서의 고유한 음성 특성을 더 잘 포착할 수 있습니다.

### 데이터셋 처리

마지막으로, 모델이 예상하는 형식으로 데이터를 처리하겠습니다. `SpeechT5Processor` 객체를 사용하여 입력 텍스트를 토큰화하고 대상 오디오를 로그 멜 스펙트로그램으로 로드하는 `prepare_dataset` 함수를 생성하세요. 스피커 임베딩을 추가 입력으로 추가해야 합니다.

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example = processor(
...         text=example["normalized_text"],
...         audio_target=audio["array"],
...         sampling_rate=audio["sampling_rate"],
...         return_attention_mask=False,
...     )

...     # strip off the batch dimension
...     example["labels"] = example["labels"][0]

...     # use SpeechBrain to obtain x-vector
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

단일 예제로 처리가 올바른지 확인해 보세요:

```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

스피커 임베딩은 512개 요소 벡터여야 합니다:

```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

레이블은 80개 멜 빈을 가진 로그 멜 스펙트로그램입니다.

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.imshow(processed_example["labels"].T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_1.png" alt="Log-mel spectrogram with 80 mel bins"/>
</div>

참고: 이 스펙트로그램이 혼란스러울 수 있다면, 플롯의 아래쪽에 저주파수를, 위쪽에 고주파수를 배치하는 관례에 익숙하지 않을 수 있습니다. 그러나 matplotlib 라이브러리를 사용하여 이미지로 스펙트로그램을 표시할 때는 y축이 뒤집혀서 스펙트로그램이 거꾸로 보입니다.

이제 처리 함수를 전체 데이터셋에 적용하세요. 이 작업에는 5분에서 10분이 소요됩니다.

```py
>>> dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

데이터셋에서 모델이 처리할 수 있는 최대 입력 길이(600개 토큰)를 초과하는 일부 예제가 있는 경고 메시지가 표시됩니다. 이러한 예제를 데이터셋에서 제거하세요. 여기서는 더 나아가 배치 크기를 크게하기 위해 200개 토큰을 초과하는 것을 제거합니다.

```py
>>> def is_not_too_long(input_ids):
...     input_length = len(input_ids)
...     return input_length < 200


>>> dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
>>> len(dataset)
8259
```

다음으로, 기본적인 훈련/테스트 분할을 생성하세요:

```py
>>> dataset = dataset.train_test_split(test_size=0.1)
```

### 데이터 콜레이터

여러 예제를 하나의 배치로 결합하기 위해 사용자 지정 데이터 콜레이터를 정의해야 합니다. 이 콜레이터는 더 짧은 시퀀스를 패딩 토큰으로 채워서 모든 예제가 동일한 길이를 가지도록 보장합니다. 스펙트로그램 레이블의 경우, 패딩된 부분은 특수 값 `-100`으로 대체됩니다. 이 특수 값은 스펙트로그램 손실을 계산할 때 해당 부분을 무시하도록 모델에 지시합니다.

```py
>>> from dataclasses import dataclass
>>> from typing import Any, Dict, List, Union


>>> @dataclass
... class TTSDataCollatorWithPadding:
...     processor: Any

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
...         label_features = [{"input_values": feature["labels"]} for feature in features]
...         speaker_features = [feature["speaker_embeddings"] for feature in features]

...         # collate the inputs and targets into a batch
...         batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

...         # not used during fine-tuning
...         del batch["decoder_attention_mask"]

...         # round down target lengths to multiple of reduction factor
...         if model.config.reduction_factor > 1:
...             target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
...             target_lengths = target_lengths.new(
...                 [length - length % model.config.reduction_factor for length in target_lengths]
...             )
...             max_length = max(target_lengths)
...             batch["labels"] = batch["labels"][:, :max_length]

...         # also add in the speaker embeddings
...         batch["speaker_embeddings"] = torch.tensor(speaker_features)

...         return batch
```

SpeechT5에서 모델의 디코더 부분으로의 입력은 2배로 줄어듭니다. 즉, 대상 시퀀스에서 매번 다른 타임스텝을 제거합니다. 디코더는 그런 다음 두 배 길이의 시퀀스를 예측합니다. 원래 대상 시퀀스 길이가 홀수일 수 있으므로, 데이터 콜레이터는 배치의 최대 길이를 반드시 2의 배수로 내림 처리하도록 합니다.

```py 
>>> data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## 모델 훈련

프로세서를 로드하는 데 사용한 체크포인트와 동일한 체크포인트에서 사전 훈련된 모델을 로드하세요:

```py
>>> from transformers import SpeechT5ForTextToSpeech

>>> model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

`use_cache=True` 옵션은 그래디언트 체크포인팅과 호환되지 않습니다. 훈련에는 이 옵션을 사용하지 않도록 설정하세요.

```py 
>>> model.config.use_cache = False
```

훈련 인수를 정의하세요. 여기서는 훈련 과정에서 평가 메트릭을 계산하지 않습니다. 대신 손실만 확인합니다:

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
...     evaluation_strategy="steps",
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

`Trainer` 객체를 인스턴스화하고 모델, 데이터셋 및 데이터 콜레이터를 전달하세요.

```py
>>> from transformers import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
...     args=training_args,
...     model=model,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     data_collator=data_collator,
...     tokenizer=processor,
... )
```

이제 훈련을 시작할 준비가 되었습니다! 훈련에는 몇 시간이 소요될 수 있습니다. GPU에 따라 훈련을 시작할 때 CUDA "out-of-memory" 오류가 발생할 수 있습니다. 이 경우, `per_device_train_batch_size`를 2의 배수로 반복적으로 줄이고 `gradient_accumulation_steps`를 2배로 증가시켜 보상하세요.

```py
>>> trainer.train()
```

최종 모델을 🤗 Hub에 업로드하세요:

```py
>>> trainer.push_to_hub()
```

## 추론

좋아요, 이제 세밀 조정된 모델을 추론에 사용할 수 있습니다! 🤗 Hub에서 모델을 로드하세요(다음 코드 스니펫에서 계정 이름을 사용하세요):

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

예제를 선택하세요. 여기서는 테스트 데이터셋에서 하나를 가져옵니다. 스피커 임베딩을 얻으세요.

```py 
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

일부 입력 텍스트를 정의하고 토큰화하세요.

```py 
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

입력 텍스트를 전처리하세요: 

```py
>>> inputs = processor(text=text, return_tensors="pt")
```

모델로 스펙트로그램을 생성하세요:

```py
>>> spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

스펙트로그램을 시각화하세요(원하는 경우):

```py
>>> plt.figure()
>>> plt.imshow(spectrogram.T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_2.png" alt="Generated log-mel spectrogram"/>
</div>

마지막으로, 보코더를 사용하여 스펙트로그램을 음성으로 변환하세요.

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

우리의 경험에 따르면, 이 모델로부터 만족할만한 결과를 얻는 것은 도전적일 수 있습니다. 스피커 임베딩의 품질이 중요한 요소인 것으로 보입니다. SpeechT5는 영어 x-vector로 사전 훈련되었으므로 영어 스피커 임베딩을 사용할 때 가장 좋은 성능을 발휘합니다. 합성된 음성이 좋지 않은 경우 다른 스피커 임베딩을 사용해 보세요.

훈련 기간을 늘리면 결과 품질이 향상될 가능성이 높습니다. 그럼에도 불구하고, 이 음성은 명확히 영어가 아닌 네덜란드어이며, 예제의 원본 오디오와 비교하면 화자의 음성 특성을 포착합니다.
모델의 구성도 실험해 볼 수 있는 요소입니다. 예를 들어, `config.reduction_factor = 1`을 사용하여 결과를 개선할 수 있는지 확인해 보세요.

마지막으로, 윤리적 고려 사항을 고려해야 합니다. TTS 기술은 다양한 유용한 응용 프로그램에 사용될 수 있지만, 누군가의 동의 없이 음성을 가장하는 악용 용도로 사용될 수도 있습니다. TTS를 신중하고 책임감 있게 사용하세요.