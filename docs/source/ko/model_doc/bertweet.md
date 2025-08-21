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

# BERTweet [[bertweet]]

## 개요 [[overview]]

BERTweet 모델은 Dat Quoc Nguyen, Thanh Vu, Anh Tuan Nguyen에 의해 [BERTweet: A pre-trained language model for English Tweets](https://www.aclweb.org/anthology/2020.emnlp-demos.2.pdf) 에서 제안되었습니다.

해당 논문의 초록 :

*영어 트윗을 위한 최초의 공개 대규모 사전 학습된 언어 모델인 BERTweet을 소개합니다. 
BERTweet은 BERT-base(Devlin et al., 2019)와 동일한 아키텍처를 가지고 있으며, RoBERTa 사전 학습 절차(Liu et al., 2019)를 사용하여 학습되었습니다. 
실험 결과, BERTweet은 강력한 기준 모델인 RoBERTa-base 및 XLM-R-base(Conneau et al., 2020)의 성능을 능가하여 세 가지 트윗 NLP 작업(품사 태깅, 개체명 인식, 텍스트 분류)에서 이전 최신 모델보다 더 나은 성능을 보여주었습니다.*

이 모델은 [dqnguyen](https://huggingface.co/dqnguyen) 께서 기여하셨습니다. 원본 코드는 [여기](https://github.com/VinAIResearch/BERTweet).에서 확인할 수 있습니다.


## 사용 예시 [[usage-example]]

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

>>> # 트랜스포머 버전 4.x 이상 :
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

>>> # 트랜스포머 버전 3.x 이상:
>>> # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

>>> # 입력된 트윗은 이미 정규화되었습니다!
>>> line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = bertweet(input_ids)  # Models outputs are now tuples

>>> # With TensorFlow 2.0+:
>>> # from transformers import TFAutoModel
>>> # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
```

<Tip> 

이 구현은 토큰화 방법을 제외하고는 BERT와 동일합니다. API 참조 정보는 [BERT 문서](bert) 를 참조하세요.

</Tip>

## Bertweet 토큰화(BertweetTokenizer) [[transformers.BertweetTokenizer]]

[[autodoc]] BertweetTokenizer
