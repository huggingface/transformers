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

# MarianMT[[MarianMT]]

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=marian">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-marian-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/opus-mt-zh-en">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## 개요[[Overview]]

BART와 동일한 모델을 사용하는 번역 모델 프레임워크입니다. 번역 결과는 각 모델 카드의 테스트 세트와 유사하지만, 정확히 일치하지는 않을 수 있습니다. 이 모델은 [sshleifer](https://huggingface.co/sshleifer)가 제공했습니다.


## 구현 노트[[Implementation Notes]]

- 각 모델은 약 298 MB를 차지하며, 1,000개 이상의 모델이 제공됩니다.
- 지원되는 언어 쌍 목록은 [여기](https://huggingface.co/Helsinki-NLP)에서 확인할 수 있습니다.
- 모델들은 [Jörg Tiedemann](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann)에 의해 [Marian](https://marian-nmt.github.io/) C++ 라이브러리를 이용하여 학습되었습니다. 이 라이브러리는 빠른 학습과 번역을 지원합니다.
- 모든 모델은 6개 레이어로 이루어진 Transformer 기반의 인코더-디코더 구조입니다. 각 모델의 성능은 모델 카드에 기입되어 있습니다.
- BPE 전처리가 필요한 80개의 OPUS 모델은 지원되지 않습니다.
- 모델링 코드는 [`BartForConditionalGeneration`]을 기반으로 하며, 일부 수정사항이 반영되어 있습니다:

  - 정적 (사인 함수 기반) 위치 임베딩 사용 (`MarianConfig.static_position_embeddings=True`)
  - 임베딩 레이어 정규화 생략 (`MarianConfig.normalize_embedding=False`)
  - 모델은 생성 시 프리픽스로 `pad_token_id` (해당 토큰 임베딩 값은 0)를 사용하여 시작합니다 (Bart는
    `<s/>`를 사용),
- Marian 모델을 PyTorch로 대량 변환하는 코드는 `convert_marian_to_pytorch.py`에서 찾을 수 있습니다.


## 모델 이름 규칙[[Naming]]

- 모든 모델 이름은 `Helsinki-NLP/opus-mt-{src}-{tgt}` 형식을 따릅니다.
- 모델의 언어 코드 표기는 일관되지 않습니다. 두 자리 코드는 일반적으로 [여기](https://developers.google.com/admin-sdk/directory/v1/languages)에서 찾을 수 있으며, 세 자리 코드는 "언어 코드 {code}"로 구글 검색을 통해 찾습니다. 
- `es_AR`과 같은 형태의 코드는 `code_{region}` 형식을 의미합니다. 여기서의 예시는 아르헨티나의 스페인어를 의미합니다.
- 모델 변환은 두 단계로 이루어졌습니다. 처음 1,000개 모델은 ISO-639-2 코드를 사용하고, 두 번째 그룹은 ISO-639-5와 ISO-639-2 코드를 조합하여 언어를 식별합니다.


## 예시[[Examples]]

- Marian 모델은 라이브러리의 다른 번역 모델들보다 크기가 작아 파인튜닝 실험과 통합 테스트에 유용합니다.
- [GPU에서 파인튜닝하기](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/train_distil_marian_enro.sh)

## 다국어 모델 사용법[[Multilingual Models]]

- 모든 모델 이름은`Helsinki-NLP/opus-mt-{src}-{tgt}` 형식을 따릅니다.
- 다중 언어 출력을 지원하는 모델의 경우, 출력을 원하는 언어의 언어 코드를 `src_text`의 시작 부분에 추가하여 지정해야 합니다.
- 모델 카드에서 지원되는 언어 코드의 목록을 확인할 수 있습니다! 예를 들어 [opus-mt-en-roa](https://huggingface.co/Helsinki-NLP/opus-mt-en-roa)에서 확인할 수 있습니다.
- `Helsinki-NLP/opus-mt-roa-en`처럼 소스 측에서만 다국어를 지원하는 모델의 경우, 별도의 언어 코드 지정이 필요하지 않습니다.

[Tatoeba-Challenge 리포지토리](https://github.com/Helsinki-NLP/Tatoeba-Challenge)의 새로운 다국적 모델은 3자리 언어 코드를 사용합니다:


```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fra<< this is a sentence in english that we want to translate to french",
...     ">>por<< This should go to portuguese",
...     ">>esp<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-roa"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)
>>> print(tokenizer.supported_language_codes)
['>>zlm_Latn<<', '>>mfe<<', '>>hat<<', '>>pap<<', '>>ast<<', '>>cat<<', '>>ind<<', '>>glg<<', '>>wln<<', '>>spa<<', '>>fra<<', '>>ron<<', '>>por<<', '>>ita<<', '>>oci<<', '>>arg<<', '>>min<<']

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français",
 'Isto deve ir para o português.',
 'Y esto al español']
```

허브에 있는 모든 사전 학습된 모델을 확인하는 코드입니다:

```python
from huggingface_hub import list_models

model_list = list_models()
org = "Helsinki-NLP"
model_ids = [x.id for x in model_list if x.id.startswith(org)]
suffix = [x.split("/")[1] for x in model_ids]
old_style_multi_models = [f"{org}/{s}" for s in suffix if s != s.lower()]
```

## 구형 다국어 모델[[Old Style Multi-Lingual Models]]

이 모델들은 OPUS-MT-Train 리포지토리의 구형 다국어 모델들입니다. 각 언어 그룹에 포함된 언어들은 다음과 같습니다:

```python no-style
['Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU',
 'Helsinki-NLP/opus-mt-ROMANCE-en',
 'Helsinki-NLP/opus-mt-SCANDINAVIA-SCANDINAVIA',
 'Helsinki-NLP/opus-mt-de-ZH',
 'Helsinki-NLP/opus-mt-en-CELTIC',
 'Helsinki-NLP/opus-mt-en-ROMANCE',
 'Helsinki-NLP/opus-mt-es-NORWAY',
 'Helsinki-NLP/opus-mt-fi-NORWAY',
 'Helsinki-NLP/opus-mt-fi-ZH',
 'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI',
 'Helsinki-NLP/opus-mt-sv-NORWAY',
 'Helsinki-NLP/opus-mt-sv-ZH']
GROUP_MEMBERS = {
 'ZH': ['cmn', 'cn', 'yue', 'ze_zh', 'zh_cn', 'zh_CN', 'zh_HK', 'zh_tw', 'zh_TW', 'zh_yue', 'zhs', 'zht', 'zh'],
 'ROMANCE': ['fr', 'fr_BE', 'fr_CA', 'fr_FR', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'es_AR', 'es_CL', 'es_CO', 'es_CR', 'es_DO', 'es_EC', 'es_ES', 'es_GT', 'es_HN', 'es_MX', 'es_NI', 'es_PA', 'es_PE', 'es_PR', 'es_SV', 'es_UY', 'es_VE', 'pt', 'pt_br', 'pt_BR', 'pt_PT', 'gl', 'lad', 'an', 'mwl', 'it', 'it_IT', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la'],
 'NORTH_EU': ['de', 'nl', 'fy', 'af', 'da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SCANDINAVIA': ['da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SAMI': ['se', 'sma', 'smj', 'smn', 'sms'],
 'NORWAY': ['nb_NO', 'nb', 'nn_NO', 'nn', 'nog', 'no_nb', 'no'],
 'CELTIC': ['ga', 'cy', 'br', 'gd', 'kw', 'gv']
}
```

영어를 여러 로망스 언어로 번역하는 예제입니다. 여기서는 구형 2자리 언어 코드를 사용합니다:


```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fr<< this is a sentence in english that we want to translate to french",
...     ">>pt<< This should go to portuguese",
...     ">>es<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français", 
 'Isto deve ir para o português.',
 'Y esto al español']
```

## 자료[[Resources]]

- [번역 작업 가이드](../tasks/translation)
- [요약 작업 가이드](../tasks/summarization)
- [언어 모델링 작업 가이드](../tasks/language_modeling)

## MarianConfig

[[autodoc]] MarianConfig

## MarianTokenizer

[[autodoc]] MarianTokenizer
    - build_inputs_with_special_tokens

<frameworkcontent>
<pt>

## MarianModel

[[autodoc]] MarianModel
    - forward

## MarianMTModel

[[autodoc]] MarianMTModel
    - forward

## MarianForCausalLM

[[autodoc]] MarianForCausalLM
    - forward

</pt>
<tf>

## TFMarianModel

[[autodoc]] TFMarianModel
    - call

## TFMarianMTModel

[[autodoc]] TFMarianMTModel
    - call

</tf>
<jax>

## FlaxMarianModel

[[autodoc]] FlaxMarianModel
    - __call__

## FlaxMarianMTModel

[[autodoc]] FlaxMarianMTModel
    - __call__

</jax>
</frameworkcontent>
