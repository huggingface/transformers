<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BARTpho [[bartpho]]

## 개요 [[overview]]

BARTpho 모델은 Nguyen Luong Tran, Duong Minh Le, Dat Quoc Nguyen에 의해 [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese](https://arxiv.org/abs/2109.09701)에서 제안되었습니다.

이 논문의 초록은 다음과 같습니다:

*우리는 BARTpho_word와 BARTpho_syllable의 두 가지 버전으로 BARTpho를 제시합니다. 
이는 베트남어를 위해 사전훈련된 최초의 대규모 단일 언어 시퀀스-투-시퀀스 모델입니다. 
우리의 BARTpho는 시퀀스-투-시퀀스 디노이징 모델인 BART의 "large" 아키텍처와 사전훈련 방식을 사용하여, 생성형 NLP 작업에 특히 적합합니다. 
베트남어 텍스트 요약의 다운스트림 작업 실험에서, 
자동 및 인간 평가 모두에서 BARTpho가 강력한 기준인 mBART를 능가하고 최신 성능을 개선했음을 보여줍니다. 
우리는 향후 연구 및 베트남어 생성형 NLP 작업의 응용을 촉진하기 위해 BARTpho를 공개합니다.*

이 모델은 [dqnguyen](https://huggingface.co/dqnguyen)이 기여했습니다. 원본 코드는 [여기](https://github.com/VinAIResearch/BARTpho)에서 찾을 수 있습니다.

## 사용 예시 [[usage-example]]

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")

>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

>>> line = "Chúng tôi là những nghiên cứu viên."

>>> input_ids = tokenizer(line, return_tensors="pt")

>>> with torch.no_grad():
...     features = bartpho(**input_ids)  # 이제 모델 출력은 튜플입니다

>>> # With TensorFlow 2.0+:
>>> from transformers import TFAutoModel

>>> bartpho = TFAutoModel.from_pretrained("vinai/bartpho-syllable")
>>> input_ids = tokenizer(line, return_tensors="tf")
>>> features = bartpho(**input_ids)
```

## 사용 팁 [[usage-tips]]

- mBART를 따르며, BARTpho는 BART의 "large" 아키텍처에 인코더와 디코더의 상단에 추가적인 레이어 정규화 레이어를 사용합니다. 
따라서 [BART 문서](bart)에 있는 사용 예시를 BARTpho에 맞게 적용하려면 
BART 전용 클래스를 mBART 전용 클래스로 대체하여 조정해야 합니다. 
예를 들어:

```python
>>> from transformers import MBartForConditionalGeneration

>>> bartpho = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
>>> TXT = "Chúng tôi là <mask> nghiên cứu viên."
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = bartpho(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)
>>> print(tokenizer.decode(predictions).split())
```

- 이 구현은 토큰화만을 위한 것입니다: "monolingual_vocab_file"은 다국어
 XLM-RoBERTa에서 제공되는 사전훈련된 SentencePiece 모델 
 "vocab_file"에서 추출된 베트남어 전용 유형으로 구성됩니다.
  다른 언어들도 이 사전훈련된 다국어 SentencePiece 모델 "vocab_file"을 하위 단어 분할에 사용하면, 자신의 언어 전용 "monolingual_vocab_file"과 함께 BartphoTokenizer를 재사용할 수 있습니다.

## BartphoTokenizer [[bartphotokenizer]]

[[autodoc]] BartphoTokenizer
