<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 다국어 모델 추론하기[[multilingual-models-for-inference]]

[[open-in-colab]]

🤗 Transformers에는 여러 종류의 다국어(multilingual) 모델이 있으며, 단일 언어(monolingual) 모델과 추론 시 사용법이 다릅니다.
그렇다고 해서 *모든* 다국어 모델의 사용법이 다른 것은 아닙니다.

[bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased)와 같은 몇몇 모델은 단일 언어 모델처럼 사용할 수 있습니다.
이번 가이드에서 다국어 모델의 추론 시 사용 방법을 알아볼 것입니다.

## XLM[[xlm]]

XLM에는 10가지 체크포인트(checkpoint)가 있는데, 이 중 하나만 단일 언어입니다. 
나머지 체크포인트 9개는 언어 임베딩을 사용하는 체크포인트와 그렇지 않은 체크포인트의 두 가지 범주로 나눌 수 있습니다.

### 언어 임베딩을 사용하는 XLM[[xlm-with-language-embeddings]]

다음 XLM 모델은 추론 시에 언어 임베딩을 사용합니다:

- `xlm-mlm-ende-1024` (마스킹된 언어 모델링, 영어-독일어)
- `xlm-mlm-enfr-1024` (마스킹된 언어 모델링, 영어-프랑스어)
- `xlm-mlm-enro-1024` (마스킹된 언어 모델링, 영어-루마니아어)
- `xlm-mlm-xnli15-1024` (마스킹된 언어 모델링, XNLI 데이터 세트에서 제공하는 15개 국어)
- `xlm-mlm-tlm-xnli15-1024` (마스킹된 언어 모델링 + 번역, XNLI 데이터 세트에서 제공하는 15개 국어)
- `xlm-clm-enfr-1024` (Causal language modeling, 영어-프랑스어)
- `xlm-clm-ende-1024` (Causal language modeling, 영어-독일어)

언어 임베딩은 모델에 전달된 `input_ids`와 동일한 shape의 텐서로 표현됩니다.
이러한 텐서의 값은 사용된 언어에 따라 다르며 토크나이저의 `lang2id` 및 `id2lang` 속성에 의해 식별됩니다.

다음 예제에서는 `xlm-clm-enfr-1024` 체크포인트(코잘 언어 모델링(causal language modeling), 영어-프랑스어)를 가져옵니다:

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
```

토크나이저의 `lang2id` 속성은 모델의 언어와 해당 ID를 표시합니다:

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

다음으로, 예제 입력을 만듭니다:

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # 배치 크기는 1입니다
```

언어 ID를 `"en"`으로 설정해 언어 임베딩을 정의합니다. 
언어 임베딩은 영어의 언어 ID인 `0`으로 채워진 텐서입니다.
이 텐서는 `input_ids`와 같은 크기여야 합니다. 

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # (batch_size, sequence_length) shape의 텐서가 되도록 만듭니다.
>>> langs = langs.view(1, -1)  # 이제 [1, sequence_length] shape이 되었습니다(배치 크기는 1입니다)
```

이제 `input_ids`와 언어 임베딩을 모델로 전달합니다:

```py
>>> outputs = model(input_ids, langs=langs)
```

[run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py) 스크립트로 `xlm-clm` 체크포인트를 사용해 텍스트와 언어 임베딩을 생성할 수 있습니다.

### 언어 임베딩을 사용하지 않는 XLM[[xlm-without-language-embeddings]]

다음 XLM 모델은 추론 시에 언어 임베딩이 필요하지 않습니다:

- `xlm-mlm-17-1280` (마스킹된 언어 모델링, 17개 국어)
- `xlm-mlm-100-1280` (마스킹된 언어 모델링, 100개 국어)

이전의 XLM 체크포인트와 달리 이 모델은 일반 문장 표현에 사용됩니다.

## BERT[[bert]]

다음 BERT 모델은 다국어 태스크에 사용할 수 있습니다:

- `bert-base-multilingual-uncased` (마스킹된 언어 모델링 + 다음 문장 예측, 102개 국어)
- `bert-base-multilingual-cased` (마스킹된 언어 모델링 + 다음 문장 예측, 104개 국어)

이러한 모델은 추론 시에 언어 임베딩이 필요하지 않습니다. 
문맥에서 언어를 식별하고, 식별된 언어로 추론합니다.

## XLM-RoBERTa[[xlmroberta]]

다음 XLM-RoBERTa 또한 다국어 다국어 태스크에 사용할 수 있습니다:

- `xlm-roberta-base` (마스킹된 언어 모델링, 100개 국어)
- `xlm-roberta-large` (마스킹된 언어 모델링, 100개 국어)

XLM-RoBERTa는 100개 국어에 대해 새로 생성되고 정제된 2.5TB 규모의 CommonCrawl 데이터로 학습되었습니다.
이전에 공개된 mBERT나 XLM과 같은 다국어 모델에 비해 분류, 시퀀스 라벨링, 질의 응답과 같은 다운스트림(downstream) 작업에서 이점이 있습니다.

## M2M100[[m2m100]]

다음 M2M100 모델 또한 다국어 다국어 태스크에 사용할 수 있습니다:

- `facebook/m2m100_418M` (번역)
- `facebook/m2m100_1.2B` (번역)

이 예제에서는 `facebook/m2m100_418M` 체크포인트를 가져와서 중국어를 영어로 번역합니다. 
토크나이저에서 번역 대상 언어(source language)를 설정할 수 있습니다:

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

문장을 토큰화합니다:

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

M2M100은 번역을 진행하기 위해 첫 번째로 생성되는 토큰은 번역할 언어(target language) ID로 강제 지정합니다.
영어로 번역하기 위해 `generate` 메소드에서 `forced_bos_token_id`를 `en`으로 설정합니다:

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart[[mbart]]

다음 MBart 모델 또한 다국어 태스크에 사용할 수 있습니다:

- `facebook/mbart-large-50-one-to-many-mmt` (일대다 다국어 번역, 50개 국어)
- `facebook/mbart-large-50-many-to-many-mmt` (다대다 다국어 번역, 50개 국어)
- `facebook/mbart-large-50-many-to-one-mmt` (다대일 다국어 번역, 50개 국어)
- `facebook/mbart-large-50` (다국어 번역, 50개 국어)
- `facebook/mbart-large-cc25`

이 예제에서는 핀란드어를 영어로 번역하기 위해 `facebook/mbart-large-50-many-to-many-mmt` 체크포인트를 가져옵니다. 
토크나이저에서 번역 대상 언어(source language)를 설정할 수 있습니다:

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

문장을 토큰화합니다:

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

MBart는 번역을 진행하기 위해 첫 번째로 생성되는 토큰은 번역할 언어(target language) ID로 강제 지정합니다.
영어로 번역하기 위해 `generate` 메소드에서 `forced_bos_token_id`를 `en`으로 설정합니다:

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id("en_XX"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

`facebook/mbart-large-50-many-to-one-mmt` 체크포인트를 사용하고 있다면, 첫 번째로 생성되는 토큰을 번역할 언어(target language) ID로 강제 지정할 필요는 없습니다.
