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

# 일본어 BERT (BertJapanese) [[bertjapanese]]

## 개요 [[overview]]

일본어 문장에 학습된 BERT 모델 입니다.

각각 서로 다른 토큰화 방법을 사용하는 두 모델:

- MeCab와 WordPiece를 사용하여 토큰화합니다. 이를 위해 추가 의존성 [fugashi](https://github.com/polm/fugashi)이 필요합니다. (이는 [MeCab](https://taku910.github.io/mecab/)의 래퍼입니다.)
- 문자 단위로 토큰화합니다.

*MecabTokenizer*를 사용하려면, 의존성을 설치하기 위해 `pip install transformers["ja"]` (또는 소스에서 설치하는 경우 `pip install -e .["ja"]`) 명령을 실행해야 합니다.

자세한 내용은 [cl-tohoku 리포지토리](https://github.com/cl-tohoku/bert-japanese)에서 확인하세요.

MeCab과 WordPiece 토큰화를 사용하는 모델 예시:

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

문자 토큰화를 사용하는 모델 예시:

```python
>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

<Tip> 

이는 토큰화 방법을 제외하고는 BERT와 동일합니다. API 참조 정보는 [BERT 문서](https://huggingface.co/docs/transformers/main/en/model_doc/bert)를 참조하세요.
이 모델은 [cl-tohoku](https://huggingface.co/cl-tohoku)께서 기여하였습니다.

</Tip> 


## BertJapaneseTokenizer

[[autodoc]] BertJapaneseTokenizer
