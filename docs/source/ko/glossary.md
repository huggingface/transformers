<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 용어집(Glossary)

이 용어집은 전반적인 머신러닝 및 🤗 Transformers 관련 용어를 정의하여 문서를 더 잘 이해하는 데 도움을 줍니다.

## A

### 어텐션 마스크 (attention mask)

어텐션 마스크(attention mask)는 여러 시퀀스를 배치(batch)로 처리할 때 사용되는 선택적 인자입니다.

<Youtube id="M6adb1j2jPI"/>

이 인자는 모델에게 어떤 토큰에 주의를 기울여야 하는지, 그리고 어떤 토큰은 무시해야 하는지를 알려줍니다.

예를 들어, 다음 두 개의 시퀀스가 있다고 가정해 봅시다:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

>>> sequence_a = "This is a short sequence."
>>> sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

>>> encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
>>> encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```

인코딩된 버전들의 길이가 다릅니다:

```python
>>> len(encoded_sequence_a), len(encoded_sequence_b)
(8, 19)
```

따라서 이 두 시퀀스를 그대로 하나의 텐서에 넣을 수는 없습니다. 첫 번째 시퀀스를 두 번째 길이에 맞춰 패딩 하거나, 반대로 두 번째 시퀀스를 첫 번째 길이에 맞춰 잘라내야 합니다.

첫 번째 경우에는 ID 목록이 패딩 인덱스로 확장됩니다. 이렇게 패딩을 적용하려면 토크나이저에 리스트를 전달하고 다음과 같이 요청할 수 있습니다:

```python
>>> padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
```

첫 번째 문장 오른쪽에 0이 추가되어 두 번째 문장과 길이가 같아진 것을 볼 수 있습니다:

```python
>>> padded_sequences["input_ids"]
[[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
```

이것은 PyTorch나 TensorFlow의 텐서로 변환될 수 있습니다. 어텐션 마스크는 모델이 패딩 된 인덱스를 참조하지 않도록 해당 위치를 나타내는 이진 텐서입니다. [`BertTokenizer`]의 경우, `1`은 어텐션이 필요한 값을 나타내고, `0`은 패딩 된 값을 나타냅니다. 이 어텐션 마스크는 토크나이저가 반환되는 딕셔너리의 "attention_mask" 키 아래에 포함되어 있습니다:

```python
>>> padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

### 오토인코딩 모델 (autoencoding models)

[인코더 모델](#encoder-models)과 [마스킹된 언어 모델링](#masked-language-modeling-mlm)을 참고하세요.

### 자기회귀 모델 (autoregressive models)

[인과적 언어 모델링](#causal-language-modeling)과 [디코더 모델](#decoder-models)을 참고하세요.

## B

### 백본 (backbone)

백본(backbone)은 원시(hidden) 은닉 상태(hidden state) 또는 특징(feature)을 출력하는 네트워크(임베딩과 레이어)입니다. 일반적으로 이 백본은 해당 특징을 입력으로 받아 예측을 수행하는 [헤드](#head)와 연결됩니다. 예를 들어, [`ViTModel`]은 특정 헤드가 없는 백본입니다. 다른 모델들도[`VitModel`]을 백본으로 사용할 수 있으며, [DPT](model_doc/dpt)등이 그 예시입니다.

## C

### 인과적 언어 모델링 (causal language modeling)

모델이 텍스트를 순서대로 읽으며 다음 단어를 예측해야 하는 사전 학습(pretraining) 작업입니다. 일반적으로 문장을 전체로 읽되, 모델 내부에서 특징 시점 이후의 토큰을 마스킹(masking)하여 다음 단어를 예측하게 됩니다.

### 채널 (channel)

컬러 이미지는 빨간색(R), 초록색(G), 파란색(B)의 세 채널 값을 조합하여 구성되며, 흑백 이미지는 단일 채널만을 가집니다. 🤗 Transformers에서는 이미지 텐서의 채널이 첫 번째 또는 마지막 차원에 위치할 수 있습니다:[`n_channels`, `height`, `width`] 또는 [`height`, `width`, `n_channels`]와 같은 형식입니다.

### 연결 시간분류(connectionist temporal classification, CTC) 

입력과 출력의 정렬 상태를 정확히 몰라도 모델이 학습할 수 있도록 돕는 알고리즘입니다. CTC는 주어진 입력에 대해 가능한 모든 출력의 확률 분포를 계산하고, 그중 가장 가능성이 높은 출력을 선택합니다. CTC는 말하는 속도의 차이 등 여러 이유로 음성과 텍스트가 항상 정확하게 일치하지 않기 때문에 음성 인식 작업에서 자주 사용됩니다.

### 컨볼루션 (convolution)

신경망에서 사용되는 레이어의 한 종류로, 입력 행렬에 대해 더 작은 행렬(커널 또는 필터)을 원소별로 곱한 뒤 그 값을 합산해 새로운 행렬을 만드는 연산입니다. 이 연산을 컨볼루션 연산이라고 하며, 입력 행렬 전체에 걸쳐 반복적으로 수행됩니다. 각 연산은 입력 행렬의 서로 다른 구간에 적용됩니다. 컨볼루션 신경망(CNN)은 컴퓨터 비전 분야에서 널리 사용됩니다.

## D

### 데이터 병렬화 (DataParallel)

여러 개의 GPU에서 훈련을 수행할 때 사용하는 병렬화 기법으로, 동일한 모델 구성이 여러 번 복제되며 각 인스턴스는 서로 다른 데이터 조각을 받습니다. 모든 인스턴스는 병렬로 처리를 수행하며, 각 훈련 단계가 끝난 후 결과를 동기화합니다.

DataParallel 방식에 대해 더 알아보려면 [여기](perf_train_gpu_many#dataparallel-vs-distributeddataparallel)를 참고하세요.

### 디코더 입력 ID (decoder input IDs)

이 입력은 인코더-디코더 모델에 특화된 것으로, 디코더에 전달될 input ID 들을 포함합니다. 이러한 입력은 번역이나 요약과 같은 시퀀스-투-시퀀스(sequence-to-sequence) 작업에 사용되며, 일반적으로 모델마다 고유한 방식으로 구성됩니다.

대부분의 인코더-디코더 모델(BART, T5 등)은 `labels`로부터 자동으로 `decoder_input_ids`를 생성합니다. 이러한 모델에서는 학습 시 `labels`를 전달하는 것이 일반적으로 권장됩니다.

시퀀스-투-시퀀스 학습에서 각 모델이 이러한 input ID를 어떻게 처리하는지는 모델 문서를 참고하시기를 바랍니다.

### 디코더 모델 (decoder models)

자기회귀 모델(Autoregressive models)이라고도 불리는 디코더 모델은 인과 언어 모델링(causal language modeling)이라 불리는 사전 학습 작업을 수행합니다. 이 작업에서는 모델이 텍스트를 순서대로 읽고 다음 단어를 예측해야 합니다. 일반적으로 문장의 전체를 읽되, 특정 시점 이후의 토큰은 마스크로 가려 예측하게 합니다.

<Youtube id="d_ixlCubqQw"/>

### 딥러닝 (deep learning)

여러 층의 신경망(neural network)을 사용하는 머신러닝 알고리즘입니다.

## E

### 인코더 모델 (encoder models)

자동 인코딩 모델(Autoencoding models)이라고도 불리는 인코더 모델은 텍스트나 이미지와 같은 입력을 받아 임베딩이라 불리는 압축된 수치 표현으로 반환합니다. 일반적으로 인코더 모델은 입력 시퀀스의 일부를 마스킹하고 더 의미 있는 표현을 생성하도록 학습하는 [masked language modeling](#masked-language-modeling-mlm)과 같은 기술을 사용하여 사전 학습됩니다.

<Youtube id="H39Z_720T5s"/>

## F

### 특징 추출 (feature extraction)

머신러닝 알고리즘이 더 효과적으로 학습할 수 있도록, 원시 데이터를 선택하고 변환하여 더 유용한 특징(feature) 집합으로 만드는 과정입니다. 예를 들어, 원시 텍스트를 워드 임베딩으로 변환하거나 이미지나 비디오 데이터에서 윤곽선이나 형태와 같은 중요한 특징을 추출하는 것이 있습니다.

### 피드 포워드 청킹 (feed forward chunking)

트랜스포머의 각 residual attention Block에서는 self-Attention Layer 다음에 보통 두 개의 Feed Forward Layer가 이어집니다. 이 Feed Forward Layers의 중간 임베딩 크기는 종종 모델의 히든 사이즈(hidden size)보다 큽니다(예:
`google-bert/bert-base-uncased` 모델의 경우).

입력 크기가 `[batch_size, sequence_length]`일 경우, 중간 Feed Forward 임베딩
`[batch_size, sequence_length, config.intermediate_size]`을 저장하는 데 필요한 메모리는 전체 메모리 사용량의 큰 부분을 차지할 수 있습니다.
[Reformer: The Efficient Transformer](https://huggingface.co/papers/2001.04451) 논문의 저자들은 이 연산이 `sequence_length` 차원에 대해 독립적이기 때문에,토큰마다 Feed Forward Layer의 출력 임베딩을 각 토큰별로 `[batch_size, config.hidden_size]`을 개별적으로 계산한 뒤, 이를 이어 붙여 `[batch_size, sequence_length, config.hidden_size]` 형태로 만들 수 있습니다.`n = sequence_length`. 이 방식은 계산 시간은 늘어나지만, 메모리 사용량은 줄어들게 됩니다.

[`apply_chunking_to_forward`] 함수를 사용하는 모델의 경우, `chunk_size`는 병렬로 계산되는 출력 임베딩의 개수를 정의하며, 이는 메모리 사용량과 계산 시간 간의 트레이드오프를 결정합니다.
`chunk_size`가 0으로 설정되면, 피드 포워드 청킹(Feed Forward Chunking)은 수행되지 않습니다.

### 파인튜닝 모델 (finetuned models)

파인튜닝(Finetuning)은 전이 학습(transfer learning)의 한 형태로, 사전 학습된 (pretrained) 모델을 사용하여 가중치를 고정(freeze)하고, 출력층을 새롭게 추가된 [모델 헤드](#head)로 교체한 뒤, 해당 모델 헤드를 목표 데이터셋에 맞게 학습시키는 방식입니다.

자세한 내용은 [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training) 튜토리얼을 참고하시고, 🤗 Transformers를 사용해 모델을 파인 튜닝하는 방법도 함께 확인해 보세요.

## H

### 헤드 (head)

모델 헤드(model head)란 신경망의 마지막 층을 의미하며, 이 층은 이전 층에서 나온 히든 상태(hidden states)를 받아 다른 차원으로 변환합니다. 각 작업(task)에 따라 서로 다른 모델 헤드가 사용됩니다. 예를 들어:

  * [`GPT2ForSequenceClassification`]은 기본 [`GPT2Model`] 위에 시퀀스 분류를 위한 선형계층(linear layer)을 추가한 모델 헤드입니다.
  * [`ViTForImageClassification`]은 이미지 분류를 위한 모델 헤드로, 기본 [`ViTModel`] 위에 `CLS` 토큰의 마지막 히든 상태에 선형 계층(linear layer)을 추가한 구조입니다.
  * [`Wav2Vec2ForCTC`]는 기본 [`Wav2Vec2Model`] 위에 [CTC](#connectionist-temporal-classification-ctc)를 적용한 언어 모델링 헤드입니다.

## I

### 이미지 패치 (image patch)

비전 기반 Transformer 모델은 이미지를 작은 패치로 분할한 후, 각 패치를 선형 임베딩하여 시퀀스로 모델에 입력합니다. 모델의 구성 파일에서 `patch_size`(또는 해상도)를 확인할 수 있습니다.

### 인퍼런스 (inference)

인퍼런스는 학습이 완료된 모델에 새로운 데이터를 입력하여 예측을 수행하는 과정입니다. 🤗 Transformer에서 인퍼런스를 수행하는 방법은 [Pipeline for inference](https://huggingface.co/docs/transformers/pipeline_tutorial) 튜토리얼을 참고하세요.

### 입력 ID (input IDs)

입력 ID는 종종 모델에 입력으로 전달해야 하는 유일한 필수 파라미터입니다. 이들은 토큰의 인덱스로, 모델이 입력으로 사용할 시퀀스를 구성하는 토큰들의 숫자 표현입니다.

<Youtube id="VFp38yj8h3A"/>

토크나이저마다 작동 방식은 다르지만, 기본 메커니즘은 동일합니다. 다음은 [WordPiece](https://huggingface.co/papers/1609.08144) 토크나이저인 BERT 토크나이저를 사용한 예시입니다:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

>>> sequence = "A Titan RTX has 24GB of VRAM"
```

토크나이저는 시퀀스를 토크나이저의 토큰 목록에 있는 항목으로 분리합니다.

```python
>>> tokenized_sequence = tokenizer.tokenize(sequence)
```

토큰은 단어이거나 서브 워드(subword)입니다. 예를 들어, "VRAM"은 모델의 어휘 사전에 없는 단어이기 때문에 "V", "RA", "M"으로 나뉘었습니다. 이 토큰들이 개별 단어가 아니라 같은 단어의 일부임을 나타내기 위해 "RA"와 "M" 앞에 더블 해시(`##`)가 추가 됩니다.

```python
>>> print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

이러한 토큰들은 모델이 이해할 수 있는 ID로 변환될 수 있습니다. 이 과정은 문장을 바로 토크나이저에 입력함으로써 수행되며, 성능 최적화를 위해 [🤗 Tokenizers](https://github.com/huggingface/tokenizers)의 Rust 구현을 활용합니다.

```python
>>> inputs = tokenizer(sequence)
```

토크나이저는 해당 모델이 올바르게 작동하는 데 필요한 모든 인자를 포함한 딕셔너리를 반환합니다. 토큰 인덱스는 `input_ids`라는 키에 저장됩니다.

```python
>>> encoded_sequence = inputs["input_ids"]
>>> print(encoded_sequence)
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
```

토크나이저는 (연결된 모델이 이를 사용하는 경우) 자동으로 "특수 토큰"을 추가합니다. 이들은 모델이 특정 상황에서 사용하는 특별한 ID입니다.

이전의 ID 시퀀스를 디코딩하면,

```python
>>> decoded_sequence = tokenizer.decode(encoded_sequence)
```

우리는 다음과 같은 결과를 보게 될 것입니다.

```python
>>> print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

이는 [`BertModel`]이 입력값을 기대하는 방식이기 때문입니다.

## L

### 레이블 (labels)

레이블은 모델이 손실(loss)을 직접 계산할 수 있도록 전달되는 선택적 인자입니다. 이 레이블은 모델이 예측해야 할 정답 값을 의미하며, 모델은 예측값과 이 정답(label) 사이의 차이를 표준 손실 함수를 이용해 계산하게 됩니다.

이 레이블(label)의 형태는 모델 헤드(model head)의 종류에 따라 달라집니다. 예를 들어:

- 시퀀스 분류 모델([`BertForSequenceClassification`] 등)의 경우, 모델은
  `(batch_size)` 차원의 텐서를 입력으로 받으며, 배치의 각 값은 전체 시퀀스에 대한 예상 레이블을 나타냅니다.
- 토큰 분류 모델([`BertForTokenClassification`] 등)의 경우, 모델은 `(batch_size, seq_length)` 차원의 텐서를 입력으로 받으며, 각 값은 개별 토큰에 대한 예상 레이블을 나타냅니다.
- 마스킹 언어 모델([`BertForMaskedLM`])의 경우, 모델은 `(batch_size,seq_length)` 차원의 텐서를 입력으로 받으며, 각 값은 개별 토큰에 대한 예상 레이블을 나타냅니다. 레이블은 마스킹 된 토큰의 토큰 ID이며, 나머지 토큰에 대해서는 무시할 값을 사용합니다(일반적으로 -100).
- 시퀀스 투 시퀀스 작업([`BartForConditionalGeneration`], [`MBartForConditionalGeneration`]등)의 경우, 모델은 `(batch_size, tgt_seq_length)` 차원의 텐서를 입력으로 받으며, 각 값은 입력 시퀀스에 대응하는 타겟 시퀀스를 나타냅니다. 학습 중에는 BART와 T5가 적절한 `decoder_input_ids`와 디코더 attention 마스크를 내부적으로 생성하므로, 일반적으로 따로 제공할 필요가 없습니다. 단, 이는 Encoder-Decoder 프레임워크를 직접 활용하는 모델에는 적용되지 않습니다.
- 이미지 분류 모델([`ViTForImageClassification`] 등)의 경우, 모델은 `(batch_size)` 차원의 텐서를 입력으로 받으며, 배치의 각 값은 개별 이미지에 대한 예상 레이블을 나타냅니다.
- 시멘틱 세그멘테이션 모델([`SegformerForSemanticSegmentation`] 등)의 경우, 모델은 `(batch_size, height, width)` 차원의 텐서를 입력으로 받으며, 배치의 각 값은 개별 픽셀에 대한 예상 레이블을 나타냅니다.
- 객체 탐지 모델([`DetrForObjectDetection`] 등)의 경우, 모델은 `class_labels`와 `boxes` 키를 포함하는 딕셔너리들의 리스트를 입력으로 받습니다. 배치의 각 값은 개별 이미지에 대한 예상 클래스 레이블과 바운딩 박스 정보를 나타냅니다.
- 자동 음성 인식 모델([`Wav2Vec2ForCTC`] 등)의 경우 모델은 `(batch_size,target_length)` 차원의 텐서를 입력으로 받으며, 각 값은 개별 토큰에 대한 예상 레이블을 나타냅니다.
  
<Tip>

모델마다 요구하는 레이블 형식이 다를 수 있으므로, 각 모델의 문서를 확인하여 해당 모델에 맞는 레이블 형식을 반드시 확인하세요!

</Tip>

기본 모델([`BertModel`] 등)은 레이블을 입력으로 받지 않습니다. 이러한 모델은 단순히 특징(feature)을 출력하는 기본 트랜스포머 모델이기 때문입니다.

### 대규모 언어 모델 (LLM)

대규모 데이터로 학습된 트랜스포머 언어 모델(GPT-3, BLOOM, OPT 등)을 지칭하는 일반적인 용어입니다. 이러한 모델은 학습할 수 있는 파라미터(parameter)의 수가 매우 많으며, 예를 들어 GPT-3는 약 1,750억 개의 파라미터를 가지고 있습니다.

## M

### 마스킹된 언어 모델링 (MLM)

사전 학습 단계 중 하나로, 모델은 일부 토큰이 무작위로 마스킹 된 손상된 문장을 입력받고, 원래의 문장을 예측해야 합니다.

### 멀티모달 (multimodal)

텍스트와 이미지와 같은 다른 형태의 입력을 함께 사용하는 작업입니다.

## N

### 자연어 생성 (NLG)

텍스트를 생성하는 모든 작업을 의미합니다. (예: [Write With Transformers](https://transformer.huggingface.co/), 번역 등).

### 자연어 처리 (NLP)

텍스트를 다루는 작업 전반을 지칭하는 일반적인 용어입니다.

### 자연어 이해 (NLU)

텍스트에 담긴 의미를 이해하는 모든 작업을 포함합니다. (예: 전체 문서 분류, 개별 단어 분류 등).

## P

### 파이프라인 (pipeline)

🤗 Transformers에서 파이프라인은 데이터를 전처리하고 변환한 후, 모델을 통해 예측값을 반환하는 일련의 단계를 순차적으로 수행하는 추상화된 개념입니다. 파이프라인에 포함될 수 있는 단계로는 데이터 전처리, 특징 추출(feature extraction), 정규화(normalization) 등이 있습니다.   

자세한 내용은 [Pipelines for inference](https://huggingface.co/docs/transformers/pipeline_tutorial) 문서를 참고하세요.

### 파이프라인 병렬화 (PP)

모델을 수직 방향(레이어 단위)으로 여러 GPU에 분할하여 병렬로 처리하는 병렬화 기법입니다. 각 GPU는 모델의 하나 또는 여러 개의 레이어만을 담당하며, 전체 파이프라인의 서로 다른 단계를 병렬로 처리하게 됩니다. 또한 각 GPU는 배치(batch)의 일부 작은 조각만 처리합니다. Pipeline Parallel 방식에 대해 더 알아보려면 [이 문서](perf_train_gpu_many#from-naive-model-parallelism-to-pipeline-parallelism)를 참고하세요.

### 픽셀 값 (pixel values)

이미지를 수치상으로 표현한 텐서로, 모델에 입력으로 전달됩니다. 이 텐서는 이미지 프로세서를 통해 생성되면, 값은 [`batch_size`, `num_channels`, `height`, `width`] 형태의 차원을 가집니다.

### 풀링 (pooling)

행렬의 특정 차원에서 최댓값이나 평균값을 취하여 더 작은 행렬로 줄이는 연산입니다. 풀링 계층은 주로 합성곱 계층 사이에 위치하여 특징 표현을 다운샘플링 하는 데 사용됩니다.

### 포지션 ID (position IDs)

RNN 모델과 달리 트랜스포머는 각 토큰의 위치 정보를 내부적으로 가지고 있지 않습니다. 따라서 모델은 `position_ids`를 사용하여 각 토큰이 시퀀스 내에서 어느 위치에 있는지를 인식합니다. 이 값은 선택적인 파라미터입니다. 모델에 `position_ids`를 전달하지 않으면, 절대 위치 임베딩 방식으로 자동 생성됩니다. 절대 위치 임베딩은 `[0, config.max_position_embeddings - 1]` 범위 내에서 선택됩니다. 일부 모델은 사인파 형태의 위치 임베딩(sinusoidal position embeddings) 또는 상대 위치 임베딩(relative position embeddings)과 같은 다른 유형의 위치 임베딩을 사용하기도 합니다.

### 전처리 (preprocessing)

머신러닝 모델이 쉽게 처리할 수 있도록 가공되지 않은 데이터를 정제하는 작업입니다. 예를 들어, 텍스트는 일반적으로 토큰화(tokenization) 과정을 거칩니다. 다른 입력 유형에 대한 전처리 방식이 궁금하다면 [Preprocess](https://huggingface.co/docs/transformers/preprocessing) 튜토리얼을 참고해 보세요.

### 사전 학습된 모델 (pretrained model)

일부 데이터(예: 위키피디아 전체)로 사전 학습(pretraining)된 모델입니다. 사전 학습은 자기 지도 학습(self-supervised learning)의 목표를 포함하며, 예를 들어 문장을 읽고 다음 단어를 예측하거나 ([causal language modeling](#causal-language-modeling)) 참고, 일부 단어를 마스킹하고 이를 예측하는 방식([masked language modeling](#masked-language-modeling-mlm))이 있습니다.

음성 및 비전 모델은 고유의 사전 학습 목표를 가지고 있습니다. 예를 들어, Wav2Vec2는 음성 표현 중 "진짜"를 "가짜" 중에서 구분하는 대조 학습(contrastive learning) 방식으로 사전 학습된 음성 모델입니다. 반면, BEiT는 이미지 패치 중 일부를 마스킹하고 이를 예측하는 마스킹 이미지 모델링 방식으로 사전 학습된 비전 모델입니다. 이는 마스킹 언어 모델링과 유사한 방식입니다.

## R

### 순환 신경망 (RNN)

텍스트와 같은 시퀀스 데이터를 처리하기 위해 레이어에 반복 구조(루프)를 사용하는 신경망 모델의 한 종류입니다.

### 표현학습 (representation learning)

머신러닝의 하위 분야로, 원시 데이터로부터 의미 있는 표현을 학습하는 데 중점을 둡니다. 대표적인 기법으로는 단어 임베딩, 오토인코더(autoencoder), 생성적 적대 신경망(GAN) 등이 있습니다.

## S

### 샘플링 속도 (sampling rate)

샘플링 속도는 1초에 추출하는 (오디오 신호) 샘플의 개수를 헤르츠(Hz) 단위로 나타낸 측정값입니다. 이는 음성처럼 연속적인 신호를 디지털화하여 이산적인 형태로 만드는 결과입니다.

### 셀프 어텐션 (self-attention)

입력의 각 요소가 다른 어떤 요소에 주목해야 하는지를 스스로 판단하는 메커니즘입니다. 이는 모델이 문장에서 특정 단어만을 보는 것이 아니라, 다른 단어들과의 관계를 고려하여 어떤 정보에 더 집중해야 할지를 학습하게 합니다.

### 자기지도 학습 (self-supervised learning) 

레이블이 없는 데이터로부터 모델이 스스로 학습 목표를 정의하여 학습하는 머신러닝 기법의 한 종류입니다. [비지도 학습](#unsupervised-learning)이나 [지도 학습](#supervised-learning)과 달리, 학습 과정 자체는 감독 방식 되지만, 라벨이 명시적으로 주어지는 것은 아닙니다.

예시로는 [마스크 언어 모델링](#masked-language-modeling-mlm)이 있으며, 이는 문장의 일부 토큰을 제거한 상태로 모델에 입력하고, 모델이 해당 토큰을 예측하도록 학습하는 방식입니다.

### 준지도 학습 (semi-supervised learning)

소량의 라벨이 달린 데이터와 대량의 라벨이 없는 데이터를 함께 사용하여 모델의 정확도를 높이는 머신러닝 훈련 기법의 넓은 범주입니다. 이는 [지도 학습](#supervised-learning)이나 [비지도 학습](#unsupervised-learning)과는 다른 방식입니다.

준지도 학습 기법의 예로는 "자기 학습(self-training)"이 있습니다. 이 방식은 먼저 라벨이 있는 데이터로 모델을 학습시키고, 그 모델을 사용해 라벨이 없는 데이터에 대한 예측을 수행합니다. 모델이 가장 높은 확신을 가지고 예측한 라벨이 없는 데이터 일부를 라벨이 있는 데이터로 추가하고, 이를 통해 모델을 다시 학습시킵니다.

### 시퀀스 투 시퀀스 (seq2seq)

입력으로부터 새로운 시퀀스를 생성하는 모델입니다. 예를 들어 번역 모델이나 요약 모델이 이에 해당하며, 대표적인 예로는 [Bart](model_doc/bart)나[T5](model_doc/t5) 모델이 있습니다.

### 분할 DDP (Sharded DDP)

[ZeRO](#zero-redundancy-optimizer-zero) 개념을 기반으로 다양한 구현에서 사용되는 다른 이름으로 불립니다.

### 스트라이드 (stride)

[convolution](#convolution) 또는 [pooling](#pooling)에서 스트라이드(stride)는 커널이 행렬 위를 이동하는 간격을 의미합니다. 스트라이드가 1이면 커널이 한 픽셀씩 이동하고, 2이면 두 픽셀씩 이동합니다.

### 지도학습 (supervised learning)

정답이 포함된 라벨링된 데이터를 직접 사용하여 모델의 성능을 개선하는 학습 방식입니다. 학습 중인 모델에 데이터를 입력하고, 예측 결과를 정답과 비교하여 오차를 계산합니다. 모델은 이 오차를 기반으로 가중치를 업데이트하며, 이러한 과정을 반복하여 성능을 최적화합니다.

## T

### 텐서 병렬화 (TP)

여러 GPU에서 훈련하기 위한 병렬화 기법으로, 각 텐서를 여러 덩어리(chunk)로 나눕니다. 따라서 전체 텐서가 단일 GPU에 상주하는 대신, 텐서의 각 조각(shard)이 지정된 GPU에 상주하게 됩니다. 이 조각들은 각각 다른 GPU에서 개별적으로 병렬 처리되며, 처리 단계가 끝날 때 결과가 동기화됩니다. 이러한 분할이 수평 방향으로 일어나기 때문에, 이는 때때로 수평적 병렬화라고 불립니다. Tensor Parallelism에 대해 더 알아보려면 [여기](perf_train_gpu_many#tensor-parallelism)를 참고하세요.

### 토큰 (token)

일반적인 단어 단위이지만, 때에 따라 서브 워드(자주 사용되지 않는 단어는 서브 워드로 분리됨)나 문장 부호도 포함될 수 있는 문장의 구성 요소입니다.

### 토큰 타입 ID (token type IDs)

일부 모델은 문장 쌍 분류나 질의 응답 작업을 수행하는 데 사용됩니다.

<Youtube id="0u3ioSwev3s"/>

이러한 작업에서는 두 개의 서로 다른 시퀀스를 하나의 "input_ids" 항목으로 결합해야 하며, 일반적으로 `[CLS]` 분류용 및 `[SEP]` 구분용과 같은 특수 토큰을 사용하여 처리합니다. 예를 들어, BERT 모델은 두 개의 시퀀스를 다음과 같은 방식으로 구성합니다:

```python
>>> # [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
```

두 개의 시퀀스를 `tokenizer`에 리스트가 아닌 개별 인자로 전달하면, 토크나이저가 자동으로 이러한 문장을 생성해 줍니다. 예시는 다음과 같습니다:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
>>> sequence_a = "HuggingFace is based in NYC"
>>> sequence_b = "Where is HuggingFace based?"

>>> encoded_dict = tokenizer(sequence_a, sequence_b)
>>> decoded = tokenizer.decode(encoded_dict["input_ids"])
```

결과는 아래와 같습니다:

```python
>>> print(decoded)
[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```

이 코드는 일부 모델이 두 개의 시퀀스를 어떻게 구분하는지 이해하는 데 충분합니다. 그러나 BERT와 같은 다른 모델은 토큰 타입 ID(또는 세그먼트 ID)를 추가로 사용합니다. 이 ID는 0과 1로 구성된 이진 마스크로, 두 시퀀스를 구분하는 역할을 합니다.

토크나이저는 이 마스크를 "token_type_id" 항목으로 반환합니다:

```python
>>> encoded_dict["token_type_ids"]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

질문에 사용되는 첫 번째 시퀀스인 "context"는 모든 토큰이 `0`으로 표시됩니다. 반면 두 번째 시퀀스인 "question"은 모든 토큰이 `1`로 표시됩니다.

일부 모델(예: [`XLNetModel`])은 `2`로 표시되는 추가 토큰을 사용하기도 합니다.

### 전이학습 (transfer learning)

사전 학습된(pretrained) 모델을 가져와 특정 작업에 맞는 데이터셋에 대해 추가 학습하는 기술입니다. 모델을 처음부터 학습시키는 대신, 기존 모델이 학습한 지식을 출발점으로 삼아 더욱 빠르게 학습할 수 있습니다. 이를 통해 학습 속도를 높이고 필요한 데이터양도 줄일 수 있습니다.

### 트랜스포머 (transformer)

셀프 어텐션 메커니즘을 기반으로 한 딥러닝 모델 아키텍처입니다.

## U

### 비지도 학습 (unsupervised learning)

정답(레이블)이 포함되지 않은 데이터를 이용해 모델을 학습시키는 방식입니다. 비지도 학습은 데이터 분포의 통계적 특성을 활용해 유용한 패턴을 찾아냅니다.

## Z

### Zero Redundancy Optimizer (ZeRO)

[TensorParallel](#tensor-parallelism-tp)과 유사하게 텐서를 샤딩(sharding)하는 병렬 처리 기법이지만, 순전파(forward)나 역전파(backward) 계산 시점에 전체 텐서를 다시 복원한다는 점에서 차이가 있습니다. 따라서 모델 자체를 수정할 필요가 없습니다. 이 방법은 GPU 메모리가 부족할 경우 이를 보완하기 위한 다양한 오프로딩 (offloading) 기법도 지원합니다. 
ZeRO에 대해 더 알아보려면 [이 문서](perf_train_gpu_many#zero-data-parallelism)를 참고하세요.
