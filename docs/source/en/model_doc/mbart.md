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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# mBART

[mBART](https://huggingface.co/papers/2001.08210) is a multilingual machine translation model that pretrains the entire translation model (encoder-decoder) unlike previous methods that only focused on parts of the model. The model is trained on a denoising objective which reconstructs the corrupted text. This allows mBART to handle the source language and the target text to translate to.

[mBART-50](https://huggingface.co/paper/2008.00401) is pretrained on an additional 25 languages.

You can find all the original mBART checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=mbart) organization.

> [!TIP]
> Click on the mBart models in the right sidebar for more examples of applying mBart to different language tasks.

The example below demonstrates how to translate sentences with [`Pipeline`] or [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline(
    "translation",
    model="facebook/mbart-large-en-ro",
    src_lang="en_XX",
    tgt_lang="ro_XX"
)

# Translate to Romanian
translation = pipe("Hello! How are you?")
print(translation[0]['translation_text'])
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-en-ro")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-en-ro")

text = "Hello! How are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translated:", translated)
```

</hfoption>
</hfoptions>

## Notes

- **Training Text Format:** The source text format is `X [eos, src_lang_code]` where X is the source text. The target text format is `[tgt_lang_code] X [eos]`. `bos` is never used.
- **Applications:** Ideal for translation and other sequence-to-sequence tasks.
- **Further Reading:** See the [paper](https://arxiv.org/abs/2001.08210) for more in-depth details about the model.

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# mBart-50

**[facebook/mbart-large-50](https://huggingface.co/papers/2001.08210)** is a multilingual machine translation model that supports 50 languages. It extends the original [mBart](https://huggingface.co/facebook/mbart-large-cc25) checkpoint (`mbart-large-cc25`) by adding embedding layers. mBart-50 was one of the first models to show that you can scale multilingual translation to 50+ languages using a single model. It can translate between any pair of languages, not just English-centric pairs! The training uses language codes which are prefixed to both input and output, helping the model stay on track with which languages to focus on. Multilingual finetuning improves on average 1 BLEU over the strongest baselines (being either multilingual from scratch or bilingual finetuning) while improving 9.3 BLEU on average over bilingual baselines from scratch.

You can find all original "facebook/mbart-large-50" checkpoints under [facebook/mbart-large-50](https://huggingface.co/facebook/mbart-large-50)

> [!TIP]
> Click on the mBart-50 models in the right sidebar for more examples of applying mBart-50 to different translation tasks.

The example below demonstrates how to translate text with [`Pipeline`], [`AutoModel`], and from the command line with the [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) checkpoint. Usage is different for other mBART models like [facebook/mbart-large-50-one-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt). Refer to their Hub model cards for more details.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="translation",
    model="facebook/mbart-large-50-many-to-many-mmt",
    device=0,
    torch_dtype=torch.float16,
    src_lang="en_XX",
    tgt_lang="fr_XX",
)
pipeline("UN Chief Says There Is No Military Solution in Syria")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"

# translate Hindi to English.
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- You can check the full list of language codes via `tokenizer.lang_code_to_id.keys()`.
- **Use Case:** Optimized for many-to-many translation across 50 languages.
- **Further Reading:** Consult the [paper](https://arxiv.org/abs/2008.00401) for detailed insights into multilingual pretraining and fine-tuning strategies.

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## mBartConfig

[[autodoc]] MBartConfig

## mBartTokenizer

[[autodoc]] MBartTokenizer
    - build_inputs_with_special_tokens

## mBartTokenizerFast

[[autodoc]] MBartTokenizerFast

## mBart50Tokenizer

[[autodoc]] MBart50Tokenizer

## mBart50TokenizerFast

[[autodoc]] MBart50TokenizerFast

<frameworkcontent>
<pt>

## mBartModel

[[autodoc]] MBartModel

## mBartForConditionalGeneration

[[autodoc]] MBartForConditionalGeneration

## mBartForQuestionAnswering

[[autodoc]] MBartForQuestionAnswering

## mBartForSequenceClassification

[[autodoc]] MBartForSequenceClassification

## mBartForCausalLM

[[autodoc]] MBartForCausalLM
    - forward

</pt>
<tf>

## TFmBartModel

[[autodoc]] TFMBartModel
    - call

## TFmBartForConditionalGeneration

[[autodoc]] TFMBartForConditionalGeneration
    - call

</tf>
<jax>

## FlaxmBartModel

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
