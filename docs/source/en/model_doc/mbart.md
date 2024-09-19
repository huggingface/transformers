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

# MBart and MBart-50

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=mbart">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-mbart-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/mbart-large-50-one-to-many-mmt">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>


## Overview of MBart

The MBart model was presented in [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov Marjan
Ghazvininejad, Mike Lewis, Luke Zettlemoyer.

According to the abstract, MBART is a sequence-to-sequence denoising auto-encoder pretrained on large-scale monolingual
corpora in many languages using the BART objective. mBART is one of the first methods for pretraining a complete
sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only
on the encoder, decoder, or reconstructing parts of the text.

This model was contributed by [valhalla](https://huggingface.co/valhalla). The Authors' code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/mbart)

### Training of MBart

MBart is a multilingual encoder-decoder (sequence-to-sequence) model primarily intended for translation task. As the
model is multilingual it expects the sequences in a different format. A special language id token is added in both the
source and target text. The source text format is `X [eos, src_lang_code]` where `X` is the source text. The
target text format is `[tgt_lang_code] X [eos]`. `bos` is never used.

The regular [`~MBartTokenizer.__call__`] will encode source text format passed as first argument or with the `text`
keyword, and target text format passed with the `text_label` keyword argument.

- Supervised training

```python
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
>>> example_english_phrase = "UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"

>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")

>>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
>>> # forward pass
>>> model(**inputs)
```

- Generation

  While generating the target text set the `decoder_start_token_id` to the target language id. The following
  example shows how to translate English to Romanian using the *facebook/mbart-large-en-ro* model.

```python
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")
>>> article = "UN Chief Says There Is No Military Solution in Syria"
>>> inputs = tokenizer(article, return_tensors="pt")
>>> translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Şeful ONU declară că nu există o soluţie militară în Siria"
```

## Overview of MBart-50

MBart-50 was introduced in the [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401) paper by Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav
Chaudhary, Jiatao Gu, Angela Fan. MBart-50 is created using the original *mbart-large-cc25* checkpoint by extending
its embedding layers with randomly initialized vectors for an extra set of 25 language tokens and then pretrained on 50
languages.

According to the abstract

*Multilingual translation models can be created through multilingual finetuning. Instead of finetuning on one
direction, a pretrained model is finetuned on many directions at the same time. It demonstrates that pretrained models
can be extended to incorporate additional languages without loss of performance. Multilingual finetuning improves on
average 1 BLEU over the strongest baselines (being either multilingual from scratch or bilingual finetuning) while
improving 9.3 BLEU on average over bilingual baselines from scratch.*


### Training of MBart-50

The text format for MBart-50 is slightly different from mBART. For MBart-50 the language id token is used as a prefix
for both source and target text i.e the text format is `[lang_code] X [eos]`, where `lang_code` is source
language id for source text and target language id for target text, with `X` being the source or target text
respectively.


MBart-50 has its own tokenizer [`MBart50Tokenizer`].

-  Supervised training

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")

src_text = " UN Chief Says There Is No Military Solution in Syria"
tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")

model(**model_inputs)  # forward pass
```

- Generation

  To generate using the mBART-50 multilingual translation models, `eos_token_id` is used as the
  `decoder_start_token_id` and the target language id is forced as the first generated token. To force the
  target language id as the first generated token, pass the *forced_bos_token_id* parameter to the *generate* method.
  The following example shows how to translate between Hindi to French and Arabic to English using the
  *facebook/mbart-50-large-many-to-many* checkpoint.

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Hindi to French
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire en Syria."

# translate Arabic to English
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
```

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## MBartConfig

[[autodoc]] MBartConfig

## MBartTokenizer

[[autodoc]] MBartTokenizer
    - build_inputs_with_special_tokens

## MBartTokenizerFast

[[autodoc]] MBartTokenizerFast

## MBart50Tokenizer

[[autodoc]] MBart50Tokenizer

## MBart50TokenizerFast

[[autodoc]] MBart50TokenizerFast

<frameworkcontent>
<pt>

## MBartModel

[[autodoc]] MBartModel

## MBartForConditionalGeneration

[[autodoc]] MBartForConditionalGeneration

## MBartForQuestionAnswering

[[autodoc]] MBartForQuestionAnswering

## MBartForSequenceClassification

[[autodoc]] MBartForSequenceClassification

## MBartForCausalLM

[[autodoc]] MBartForCausalLM
    - forward

</pt>
<tf>

## TFMBartModel

[[autodoc]] TFMBartModel
    - call

## TFMBartForConditionalGeneration

[[autodoc]] TFMBartForConditionalGeneration
    - call

</tf>
<jax>

## FlaxMBartModel

[[autodoc]] FlaxMBartModel
    - __call__
    - encode
    - decode

## FlaxMBartForConditionalGeneration

[[autodoc]] FlaxMBartForConditionalGeneration
    - __call__
    - encode
    - decode

## FlaxMBartForSequenceClassification

[[autodoc]] FlaxMBartForSequenceClassification
    - __call__
    - encode
    - decode

## FlaxMBartForQuestionAnswering

[[autodoc]] FlaxMBartForQuestionAnswering
    - __call__
    - encode
    - decode

</jax>
</frameworkcontent>
