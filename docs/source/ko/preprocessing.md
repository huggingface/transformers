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

# 전처리[[preprocess]]

[[open-in-colab]]

모델을 훈련하려면 데이터 세트를 모델에 맞는 입력 형식으로 전처리해야 합니다. 텍스트, 이미지 또는 오디오인지 관계없이 데이터를 텐서 배치로 변환하고 조립할 필요가 있습니다. 🤗 Transformers는 모델에 대한 데이터를 준비하는 데 도움이 되는 일련의 전처리 클래스를 제공합니다. 이 튜토리얼에서는 다음 내용을 배울 수 있습니다:

* 텍스트는 [Tokenizer](./main_classes/tokenizer)를 사용하여 토큰 시퀀스로 변환하고 토큰의 숫자 표현을 만든 후 텐서로 조립합니다.
* 음성 및 오디오는 [Feature extractor](./main_classes/feature_extractor)를 사용하여 오디오 파형에서 시퀀스 특성을 파악하여 텐서로 변환합니다.
* 이미지 입력은 [ImageProcessor](./main_classes/image)을 사용하여 이미지를 텐서로 변환합니다.
* 멀티모달 입력은 [Processor](./main_classes/processors)을 사용하여 토크나이저와 특성 추출기 또는 이미지 프로세서를 결합합니다.

<Tip>

`AutoProcessor`는 **언제나** 작동하여 토크나이저, 이미지 프로세서, 특성 추출기 또는 프로세서 등 사용 중인 모델에 맞는 클래스를 자동으로 선택합니다.

</Tip>

시작하기 전에 🤗 Datasets를 설치하여 실험에 사용할 데이터를 불러올 수 있습니다:

```bash
pip install datasets
```

## 자연어처리[[natural-language-processing]]

<Youtube id="Yffk5aydLzg"/>

텍스트 데이터를 전처리하기 위한 기본 도구는 [tokenizer](main_classes/tokenizer)입니다. 토크나이저는 일련의 규칙에 따라 텍스트를 *토큰*으로 나눕니다. 토큰은 숫자로 변환되고 텐서는 모델 입력이 됩니다. 모델에 필요한 추가 입력은 토크나이저에 의해 추가됩니다.

<Tip>

사전훈련된 모델을 사용할 계획이라면 모델과 함께 사전훈련된 토크나이저를 사용하는 것이 중요합니다. 이렇게 하면 텍스트가 사전훈련 말뭉치와 동일한 방식으로 분할되고 사전훈련 중에 동일한 해당 토큰-인덱스 쌍(일반적으로 *vocab*이라고 함)을 사용합니다.

</Tip>

시작하려면 [`AutoTokenizer.from_pretrained`] 메소드를 사용하여 사전훈련된 토크나이저를 불러오세요. 모델과 함께 사전훈련된 *vocab*을 다운로드합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

그 다음으로 텍스트를 토크나이저에 넣어주세요:

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

토크나이저는 세 가지 중요한 항목을 포함한 딕셔너리를 반환합니다:

* [input_ids](glossary#input-ids)는 문장의 각 토큰에 해당하는 인덱스입니다.
* [attention_mask](glossary#attention-mask)는 토큰을 처리해야 하는지 여부를 나타냅니다.
* [token_type_ids](glossary#token-type-ids)는 두 개 이상의 시퀀스가 있을 때 토큰이 속한 시퀀스를 식별합니다.

`input_ids`를 디코딩하여 입력을 반환합니다:

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

토크나이저가 두 개의 특수한 토큰(분류 토큰 `CLS`와 분할 토큰 `SEP`)을 문장에 추가했습니다.
모든 모델에 특수한 토큰이 필요한 것은 아니지만, 필요하다면 토크나이저가 자동으로 추가합니다.

전처리할 문장이 여러 개 있는 경우에는 리스트로 토크나이저에 전달합니다:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### 패딩[[pad]]

모델 입력인 텐서는 모양이 균일해야 하지만, 문장의 길이가 항상 같지는 않기 때문에 문제가 될 수 있습니다. 패딩은 짧은 문장에 특수한 *패딩 토큰*을 추가하여 텐서를 직사각형 모양이 되도록 하는 전략입니다.

`padding` 매개변수를 `True`로 설정하여 배치 내의 짧은 시퀀스를 가장 긴 시퀀스에 맞춰 패딩합니다.

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

길이가 짧은 첫 문장과 세 번째 문장이 이제 `0`으로 채워졌습니다.

### 잘라내기[[truncation]]

한편, 때로는 시퀀스가 모델에서 처리하기에 너무 길 수도 있습니다. 이 경우, 시퀀스를 더 짧게 줄일 필요가 있습니다.

모델에서 허용하는 최대 길이로 시퀀스를 자르려면 `truncation` 매개변수를 `True`로 설정하세요:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

<Tip>

다양한 패딩과 잘라내기 인수에 대해 더 알아보려면 [패딩과 잘라내기](./pad_truncation) 개념 가이드를 확인해보세요.

</Tip>

### 텐서 만들기[[build-tensors]]

마지막으로, 토크나이저가 모델에 공급되는 실제 텐서를 반환하도록 합니다.

`return_tensors` 매개변수를 PyTorch의 경우 `pt`, TensorFlow의 경우 `tf`로 설정하세요:

<frameworkcontent>
<pt>

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt>
<tf>
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>,
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>,
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
```
</tf>
</frameworkcontent>

## 오디오[[audio]]

오디오 작업은 모델에 맞는 데이터 세트를 준비하기 위해 [특성 추출기](main_classes/feature_extractor)가 필요합니다. 특성 추출기는 원시 오디오 데이터에서 특성를 추출하고 이를 텐서로 변환하는 것이 목적입니다.

오디오 데이터 세트에 특성 추출기를 사용하는 방법을 보기 위해 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 데이터 세트를 가져오세요. (데이터 세트를 가져오는 방법은 🤗 [데이터 세트 튜토리얼](https://huggingface.co/docs/datasets/load_hub.html)에서 자세히 설명하고 있습니다.)

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

`audio` 열의 첫 번째 요소에 접근하여 입력을 살펴보세요. `audio` 열을 호출하면 오디오 파일을 자동으로 가져오고 리샘플링합니다.

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

이렇게 하면 세 가지 항목이 반환됩니다:

* `array`는 1D 배열로 가져와서 (필요한 경우) 리샘플링된 음성 신호입니다.
* `path`는 오디오 파일의 위치를 가리킵니다.
* `sampling_rate`는 음성 신호에서 초당 측정되는 데이터 포인트 수를 나타냅니다.

이 튜토리얼에서는 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) 모델을 사용합니다. 모델 카드를 보면 Wav2Vec2가 16kHz 샘플링된 음성 오디오를 기반으로 사전훈련된 것을 알 수 있습니다.
모델을 사전훈련하는 데 사용된 데이터 세트의 샘플링 레이트와 오디오 데이터의 샘플링 레이트가 일치해야 합니다. 데이터의 샘플링 레이트가 다르면 데이터를 리샘플링해야 합니다.

1. 🤗 Datasets의 [`~datasets.Dataset.cast_column`] 메소드를 사용하여 샘플링 레이트를 16kHz로 업샘플링하세요:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. 오디오 파일을 리샘플링하기 위해 `audio` 열을 다시 호출합니다:

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

다음으로, 입력을 정규화하고 패딩할 특성 추출기를 가져오세요. 텍스트 데이터의 경우, 더 짧은 시퀀스에 대해 `0`이 추가됩니다. 오디오 데이터에도 같은 개념이 적용됩니다.
특성 추출기는 배열에 `0`(묵음으로 해석)을 추가합니다.

[`AutoFeatureExtractor.from_pretrained`]를 사용하여 특성 추출기를 가져오세요:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

오디오 `array`를 특성 추출기에 전달하세요. 또한, 발생할 수 있는 조용한 오류(silent errors)를 더 잘 디버깅할 수 있도록 특성 추출기에 `sampling_rate` 인수를 추가하는 것을 권장합니다.

```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

토크나이저와 마찬가지로 배치 내에서 가변적인 시퀀스를 처리하기 위해 패딩 또는 잘라내기를 적용할 수 있습니다. 이 두 개의 오디오 샘플의 시퀀스 길이를 확인해보세요:

```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

오디오 샘플의 길이가 동일하도록 데이터 세트를 전처리하는 함수를 만드세요. 최대 샘플 길이를 지정하면 특성 추출기가 해당 길이에 맞춰 시퀀스를 패딩하거나 잘라냅니다:

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

`preprocess_function`을 데이터 세트의 처음 예시 몇 개에 적용해보세요:

```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

이제 샘플 길이가 모두 같고 지정된 최대 길이에 맞게 되었습니다. 드디어 전처리된 데이터 세트를 모델에 전달할 수 있습니다!

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

## 컴퓨터 비전[[computer-vision]]

컴퓨터 비전 작업의 경우, 모델에 대한 데이터 세트를 준비하기 위해 [이미지 프로세서](main_classes/image_processor)가 필요합니다.
이미지 전처리는 이미지를 모델이 예상하는 입력으로 변환하는 여러 단계로 이루어집니다.
이러한 단계에는 크기 조정, 정규화, 색상 채널 보정, 이미지의 텐서 변환 등이 포함됩니다.

<Tip>

이미지 전처리는 이미지 증강 기법을 몇 가지 적용한 뒤에 할 수도 있습니다.
이미지 전처리 및 이미지 증강은 모두 이미지 데이터를 변형하지만, 서로 다른 목적을 가지고 있습니다:

* 이미지 증강은 과적합(over-fitting)을 방지하고 모델의 견고함(resiliency)을 높이는 데 도움이 되는 방식으로 이미지를 수정합니다.
밝기와 색상 조정, 자르기, 회전, 크기 조정, 확대/축소 등 다양한 방법으로 데이터를 증강할 수 있습니다.
그러나 증강으로 이미지의 의미가 바뀌지 않도록 주의해야 합니다.
* 이미지 전처리는 이미지가 모델이 예상하는 입력 형식과 일치하도록 보장합니다.
컴퓨터 비전 모델을 미세 조정할 때 이미지는 모델이 초기에 훈련될 때와 정확히 같은 방식으로 전처리되어야 합니다.

이미지 증강에는 원하는 라이브러리를 무엇이든 사용할 수 있습니다. 이미지 전처리에는 모델과 연결된 `ImageProcessor`를 사용합니다.

</Tip>

[food101](https://huggingface.co/datasets/food101) 데이터 세트를 가져와서 컴퓨터 비전 데이터 세트에서 이미지 프로세서를 어떻게 사용하는지 알아보세요.
데이터 세트를 불러오는 방법은 🤗 [데이터 세트 튜토리얼](https://huggingface.co/docs/datasets/load_hub.html)을 참고하세요.

<Tip>

데이터 세트가 상당히 크기 때문에 🤗 Datasets의 `split` 매개변수를 사용하여 훈련 세트에서 작은 샘플만 가져오세요!

</Tip>

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

다음으로, 🤗 Datasets의 [`image`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=image#datasets.Image)로 이미지를 확인해보세요:

```py
>>> dataset[0]["image"]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/>
</div>

[`AutoImageProcessor.from_pretrained`]로 이미지 프로세서를 가져오세요:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

먼저 이미지 증강 단계를 추가해 봅시다. 아무 라이브러리나 사용해도 괜찮지만, 이번 튜토리얼에서는 torchvision의 [`transforms`](https://pytorch.org/vision/stable/transforms.html) 모듈을 사용하겠습니다.
다른 데이터 증강 라이브러리를 사용해보고 싶다면, [Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb) 또는 [Kornia notebooks](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)에서 어떻게 사용하는지 배울 수 있습니다.

1. [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html)로  [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html)와 [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) 등 변환을 몇 가지 연결하세요.
참고로 크기 조정에 필요한 이미지의 크기 요구사항은 `image_processor`에서 가져올 수 있습니다.
일부 모델은 정확한 높이와 너비를 요구하지만, 제일 짧은 변의 길이(`shortest_edge`)만 정의된 모델도 있습니다.

```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. 모델은 입력으로 [`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values)를 받습니다.
`ImageProcessor`는 이미지 정규화 및 적절한 텐서 생성을 처리할 수 있습니다.
배치 이미지에 대한 이미지 증강 및 이미지 전처리를 결합하고 `pixel_values`를 생성하는 함수를 만듭니다:

```py
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

위의 예에서는 이미지 증강 중에 이미지 크기를 조정했기 때문에 `do_resize=False`로 설정하고, 해당 `image_processor`에서 `size` 속성을 활용했습니다.
이미지 증강 중에 이미지 크기를 조정하지 않은 경우 이 매개변수를 생략하세요.
기본적으로는 `ImageProcessor`가 크기 조정을 처리합니다.

증강 변환 과정에서 이미지를 정규화하려면 `image_processor.image_mean` 및 `image_processor.image_std` 값을 사용하세요.

</Tip>

3. 🤗 Datasets의 [`set_transform`](https://huggingface.co/docs/datasets/process.html#format-transform)를 사용하여 실시간으로 변환을 적용합니다:

```py
>>> dataset.set_transform(transforms)
```

4. 이제 이미지에 접근하면 이미지 프로세서가 `pixel_values`를 추가한 것을 알 수 있습니다.
드디어 처리된 데이터 세트를 모델에 전달할 수 있습니다!

```py
>>> dataset[0].keys()
```

다음은 변형이 적용된 후의 이미지입니다. 이미지가 무작위로 잘려나갔고 색상 속성이 다릅니다.

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png"/>
</div>

<Tip>

`ImageProcessor`는 객체 감지, 시맨틱 세그멘테이션(semantic segmentation), 인스턴스 세그멘테이션(instance segmentation), 파놉틱 세그멘테이션(panoptic segmentation)과 같은 작업에 대한 후처리 방법을 제공합니다.
이러한 방법은 모델의 원시 출력을 경계 상자나 세그멘테이션 맵과 같은 의미 있는 예측으로 변환해줍니다.

</Tip>

### 패딩[[pad]]

예를 들어, [DETR](./model_doc/detr)와 같은 경우에는 모델이 훈련할 때 크기 조정 증강을 적용합니다.
이로 인해 배치 내 이미지 크기가 달라질 수 있습니다.
[`DetrImageProcessor`]의 [`DetrImageProcessor.pad`]를 사용하고 사용자 정의 `collate_fn`을 정의해서 배치 이미지를 처리할 수 있습니다.

```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## 멀티모달[[multimodal]]

멀티모달 입력이 필요한 작업의 경우, 모델에 데이터 세트를 준비하기 위한 [프로세서](main_classes/processors)가 필요합니다.
프로세서는 토크나이저와 특성 추출기와 같은 두 가지 처리 객체를 결합합니다.

[LJ Speech](https://huggingface.co/datasets/lj_speech) 데이터 세트를 가져와서 자동 음성 인식(ASR)을 위한 프로세서를 사용하는 방법을 확인하세요.
(데이터 세트를 가져오는 방법에 대한 자세한 내용은 🤗 [데이터 세트 튜토리얼](https://huggingface.co/docs/datasets/load_hub.html)에서 볼 수 있습니다.)

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

자동 음성 인식(ASR)에서는 `audio`와 `text`에만 집중하면 되므로, 다른 열들은 제거할 수 있습니다:

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

이제 `audio`와 `text`열을 살펴보세요:

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

기존에 사전훈련된 모델에서 사용된 데이터 세트와 새로운 오디오 데이터 세트의 샘플링 레이트를 일치시키기 위해 오디오 데이터 세트의 샘플링 레이트를 [리샘플링](preprocessing#audio)해야 합니다!

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

[`AutoProcessor.from_pretrained`]로 프로세서를 가져오세요:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. `array`에 들어 있는 오디오 데이터를 `input_values`로 변환하고 `text`를 토큰화하여 `labels`로 변환하는 함수를 만듭니다.
모델의 입력은 다음과 같습니다:

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. 샘플을 `prepare_dataset` 함수에 적용하세요:

```py
>>> prepare_dataset(lj_speech[0])
```

이제 프로세서가 `input_values`와 `labels`를 추가하고, 샘플링 레이트도 올바르게 16kHz로 다운샘플링했습니다.
드디어 처리된 데이터 세트를 모델에 전달할 수 있습니다!
