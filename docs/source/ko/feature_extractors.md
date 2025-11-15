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

# 특징 추출기[[feature-extractors]]

특징 추출기(feature extractor)는 오디오 데이터를 주어진 모델에 맞는 형식으로 전처리합니다. 원시 오디오 신호를 받아 모델에 입력할 수 있는 텐서로 변환합니다. 텐서의 형태는 모델에 따라 다르지만, 특징 추출기는 사용 중인 모델에 맞춰 오디오 데이터를 올바르게 전처리해 줍니다. 특징 추출기는 패딩, 잘라내기, 리샘플링을 위한 메소드도 포함합니다.

[`~AutoFeatureExtractor.from_pretrained`]를 호출하여 Hugging Face [Hub](https://hf.co/models) 또는 로컬 디렉터리에서 특징 추출기와 그 전처리기 설정을 가져오세요. 특징 추출기와 전처리기 설정은 [preprocessor_config.json](https://hf.co/openai/whisper-tiny/blob/main/preprocessor_config.json) 파일에 저장됩니다.

일반적으로 `array`에 저장된 오디오 신호를 특징 추출기에 전달하고, `sampling_rate` 매개변수를 사전 훈련된 오디오 모델의 샘플링 레이트로 설정하세요. 오디오 데이터의 샘플링 레이트가 사전 훈련된 오디오 모델이 학습한 데이터의 샘플링 레이트와 일치하는 것이 중요합니다.

```py
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
processed_sample = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=16000)
processed_sample
{'input_values': [array([ 9.4472744e-05,  3.0777880e-03, -2.8888427e-03, ...,
       -2.8888427e-03,  9.4472744e-05,  9.4472744e-05], dtype=float32)]}
```

특징 추출기는 모델이 바로 사용할 수 있는 입력값인 `input_values`를 반환합니다.

이 가이드에서는 특징 추출기 클래스와 오디오 데이터 전처리 방법에 대해 안내합니다.

## 특징 추출기 클래스[[feature-extractor-classes]]

Transformers 특징 추출기는 [`FeatureExtractionMixin`]을 기반으로 하는 [`SequenceFeatureExtractor`] 클래스를 상속받아 구현되어 있습니다.

- [`SequenceFeatureExtractor`]는 시퀀스 길이가 달라지는 것을 막기 위해 시퀀스를 특정 길이로 [`~SequenceFeatureExtractor.pad`]하는 메소드를 제공합니다.
[`FeatureExtractionMixin`]은 특징 추출기를 가져오고 저장하기 위한 [`~FeatureExtractionMixin.from_pretrained`]와 [`~FeatureExtractionMixin.save_pretrained`]를 제공합니다.

특징 추출기를 불러오는 방법은 두 가지가 있습니다. 하나는 [`AutoFeatureExtractor`]를 사용하는 것이고, 다른 하나는 모델에 특화된 특징 추출기 클래스를 사용하는 방법입니다.

<hfoptions id="feature-extractor-classes">
<hfoption id="AutoFeatureExtractor">

[AutoClass](./model_doc/auto) API는 주어진 모델에 맞는 특징 추출기를 자동으로 로드합니다.

[`~AutoFeatureExtractor.from_pretrained`]를 사용하여 특징 추출기를 로드하세요.

```py
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
<hfoption id="model-specific feature extractor">

모든 사전 훈련된 오디오 모델은 오디오 데이터를 올바르게 처리하기 위한 특정 특징 추출기를 가지고 있습니다. 특징 추출기를 가져오면 [preprocessor_config.json](https://hf.co/openai/whisper-tiny/blob/main/preprocessor_config.json)에서 특징 추출기의 설정(특성 크기, 청크 길이 등)을 가져옵니다.

특징 추출기는 모델별 클래스에서 직접 가져올 수 있습니다.

```py
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
</hfoptions>

## 전처리[[preprocess]]

특징 추출기는 특정 형태의 PyTorch 텐서를 입력으로 받습니다. 정확한 입력 형태는 사용 중인 특정 오디오 모델에 따라 다를 수 있습니다.

예를 들어, [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)는 `input_features`로 `(batch_size, feature_size, sequence_length)` 형태의 텐서를 입력받지만, [Wav2Vec2](https://hf.co/docs/transformers/model_doc/wav2vec2)는 `input_values`로 `(batch_size, sequence_length)` 형태의 텐서를 입력받습니다.

특징 추출기는 사용 중인 오디오 모델에 맞는 올바른 입력 형태를 생성합니다.

또한 특징 추출기는 오디오 파일의 샘플링 레이트(초당 추출되는 오디오 신호 값의 수)를 설정합니다. 오디오 데이터의 샘플링 레이트는 사전 훈련된 모델이 학습한 데이터 세트의 샘플링 레이트와 일치해야 합니다. 이 값은 일반적으로 모델 카드에 명시되어 있습니다.

[`~FeatureExtractionMixin.from_pretrained`]를 사용하여 데이터 세트와 특징 추출기를 가져옵니다.

```py
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

데이터 세트의 첫 번째 예제를 확인하고 원시 오디오 신호인 `array`가 포함된 `audio` 열에 접근해 보세요.

```py
dataset[0]["audio"]["array"]
array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
        0.        ,  0.        ])
```

특징 추출기는 `array`를 주어진 오디오 모델의 예상 입력 형식으로 전처리합니다. `sampling_rate` 매개변수를 사용하여 적절한 샘플링 레이트를 설정하세요.

```py
processed_dataset = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=16000)
processed_dataset
{'input_values': [array([ 9.4472744e-05,  3.0777880e-03, -2.8888427e-03, ...,
       -2.8888427e-03,  9.4472744e-05,  9.4472744e-05], dtype=float32)]}
```

### 패딩[[padding]]

오디오 시퀀스 길이가 서로 다르면 문제가 됩니다. Transformers는 모든 시퀀스의 길이가 같아야 배치 처리가 가능하기 때문입니다. 길이가 다른 시퀀스는 배치로 묶을 수 없습니다.

```py
dataset[0]["audio"]["array"].shape
(86699,)

dataset[1]["audio"]["array"].shape
(53248,)
```

패딩은 모든 시퀀스가 동일한 길이를 갖도록 특별한 *패딩 토큰*을 추가합니다. 특징 추출기는 `array`에 `0`(무음으로 해석됨)을 추가하여 패딩합니다. `padding=True`로 설정하여 배치 내 가장 긴 시퀀스 길이에 맞춰 패딩하세요.

```py
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])
processed_dataset["input_values"][0].shape
(86699,)

processed_dataset["input_values"][1].shape
(86699,)
```

### 잘라내기[[truncation]]

모델은 특정 길이까지의 시퀀스만 처리할 수 있으며, 그 이상은 오류를 발생시킬 수 있습니다.

잘라내기는 시퀀스가 최대 길이를 초과하지 않도록 초과하는 토큰을 제거하는 전략입니다. `truncation=True`로 설정하여 시퀀스를 `max_length` 매개변수에 지정된 길이로 잘라냅니다.

```py
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        max_length=50000,
        truncation=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])
processed_dataset["input_values"][0].shape
(50000,)

processed_dataset["input_values"][1].shape
(50000,)
```

### 리샘플링[[resampling]]

[Datasets](https://hf.co/docs/datasets/index) 라이브러리는 오디오 모델이 기대하는 샘플링 레이트에 맞게 오디오 데이터를 리샘플링할 수도 있습니다. 이 방법은 오디오 데이터를 가져올 때 실시간으로 리샘플링하므로, 전체 데이터 세트를 한 번에 리샘플링하는 것보다 빠를 수 있습니다.

지금까지 작업한 오디오 데이터 세트는 8kHz의 샘플링 레이트를 가지고 있지만, 사전 훈련된 모델은 16kHz를 기대합니다.

```py
dataset[0]["audio"]
{'path': '/root/.cache/huggingface/datasets/downloads/extracted/f507fdca7f475d961f5bb7093bcc9d544f16f8cab8608e772a2ed4fbeb4d6f50/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ]),
 'sampling_rate': 8000}
```

`audio` 열에 [`~datasets.Dataset.cast_column`]을 호출하여 샘플링 레이트를 16kHz로 업샘플링하세요.

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

이제 데이터 세트 샘플을 가져오면 16kHz로 리샘플링됩니다.

```py
dataset[0]["audio"]
{'path': '/root/.cache/huggingface/datasets/downloads/extracted/f507fdca7f475d961f5bb7093bcc9d544f16f8cab8608e772a2ed4fbeb4d6f50/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'array': array([ 1.70562416e-05,  2.18727451e-04,  2.28099874e-04, ...,
         3.43842403e-05, -5.96364771e-06, -1.76846661e-05]),
 'sampling_rate': 16000}
```