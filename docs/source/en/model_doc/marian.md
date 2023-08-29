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

# MarianMT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=marian">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-marian-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/opus-mt-zh-en">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**Bugs:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title)
and assign @patrickvonplaten.

Translations should be similar, but not identical to output in the test set linked to in each model card.

Tips:

- A framework for translation models, using the same models as BART.

## Implementation Notes

- Each model is about 298 MB on disk, there are more than 1,000 models.
- The list of supported language pairs can be found [here](https://huggingface.co/Helsinki-NLP).
- Models were originally trained by [Jörg Tiedemann](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann) using the [Marian](https://marian-nmt.github.io/) C++ library, which supports fast training and translation.
- All models are transformer encoder-decoders with 6 layers in each component. Each model's performance is documented
  in a model card.
- The 80 opus models that require BPE preprocessing are not supported.
- The modeling code is the same as [`BartForConditionalGeneration`] with a few minor modifications:

  - static (sinusoid) positional embeddings (`MarianConfig.static_position_embeddings=True`)
  - no layernorm_embedding (`MarianConfig.normalize_embedding=False`)
  - the model starts generating with `pad_token_id` (which has 0 as a token_embedding) as the prefix (Bart uses
    `<s/>`),
- Code to bulk convert models can be found in `convert_marian_to_pytorch.py`.
- This model was contributed by [sshleifer](https://huggingface.co/sshleifer).

## Naming

- All model names use the following format: `Helsinki-NLP/opus-mt-{src}-{tgt}`
- The language codes used to name models are inconsistent. Two digit codes can usually be found [here](https://developers.google.com/admin-sdk/directory/v1/languages), three digit codes require googling "language
  code {code}".
- Codes formatted like `es_AR` are usually `code_{region}`. That one is Spanish from Argentina.
- The models were converted in two stages. The first 1000 models use ISO-639-2 codes to identify languages, the second
  group use a combination of ISO-639-5 codes and ISO-639-2 codes.


## Examples

- Since Marian models are smaller than many other translation models available in the library, they can be useful for
  fine-tuning experiments and integration tests.
- [Fine-tune on GPU](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/train_distil_marian_enro.sh)

## Multilingual Models

- All model names use the following format: `Helsinki-NLP/opus-mt-{src}-{tgt}`:
- If a model can output multiple languages, and you should specify a language code by prepending the desired output
  language to the `src_text`.
- You can see a models's supported language codes in its model card, under target constituents, like in [opus-mt-en-roa](https://huggingface.co/Helsinki-NLP/opus-mt-en-roa).
- Note that if a model is only multilingual on the source side, like `Helsinki-NLP/opus-mt-roa-en`, no language
  codes are required.

New multi-lingual models from the [Tatoeba-Challenge repo](https://github.com/Helsinki-NLP/Tatoeba-Challenge)
require 3 character language codes:

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

Here is the code to see all available pretrained models on the hub:

```python
from huggingface_hub import list_models

model_list = list_models()
org = "Helsinki-NLP"
model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
suffix = [x.split("/")[1] for x in model_ids]
old_style_multi_models = [f"{org}/{s}" for s in suffix if s != s.lower()]
```

## Old Style Multi-Lingual Models

These are the old style multi-lingual models ported from the OPUS-MT-Train repo: and the members of each language
group:

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

Example of translating english to many romance languages, using old-style 2 character language codes


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

## Documentation resources

- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)
- [Causal language modeling task guide](../tasks/language_modeling)

## MarianConfig

[[autodoc]] MarianConfig

## MarianTokenizer

[[autodoc]] MarianTokenizer
    - build_inputs_with_special_tokens

## MarianModel

[[autodoc]] MarianModel
    - forward

## MarianMTModel

[[autodoc]] MarianMTModel
    - forward

## MarianForCausalLM

[[autodoc]] MarianForCausalLM
    - forward

## TFMarianModel

[[autodoc]] TFMarianModel
    - call

## TFMarianMTModel

[[autodoc]] TFMarianMTModel
    - call

## FlaxMarianModel

[[autodoc]] FlaxMarianModel
    - __call__

## FlaxMarianMTModel

[[autodoc]] FlaxMarianMTModel
    - __call__
