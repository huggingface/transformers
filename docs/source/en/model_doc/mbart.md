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
*This model was released on 2020-01-22 and added to Hugging Face Transformers on 2020-11-16 and contributed by [valhalla](https://huggingface.co/valhalla).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MBart and MBart-50

[MBart](https://huggingface.co/papers/2001.08210) is a multilingual sequence-to-sequence denoising auto-encoder pretrained on large-scale monolingual corpora across multiple languages using the BART objective. It is notable for being one of the first models to pretrain a complete sequence-to-sequence model by denoising full texts in various languages. MBart is designed for translation tasks and requires special language ID tokens in both source and target texts. The model is trained using a supervised approach and generates translations by setting the `decoder_start_token_id` to the target language ID.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="translation_en_to_fr", model="facebook/mbart-large-50-many-to-many-mmt", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- Check the full list of language codes via `tokenizer.lang_code_to_id.keys()`.
- mBART requires a special language ID token in the source and target text during training. Source text format: `X [eos, src_lang_code]` where `X` is the source text. Target text format: `[tgt_lang_code] X [eos]`. The `bos` token is never used.
- [`~PreTrainedTokenizerBase._call_`] encodes the source text format passed as the first argument or with the `text` keyword. The target text format is passed with the `text_label` keyword.
- Set the `decoder_start_token_id` to the target language ID for mBART.
- mBART-50 has a different text format. The language ID token is used as the prefix for the source and target text. Text format: `[lang_code] X [eos]` where `lang_code` is the source language ID for source text and target language ID for target text. `X` is the source or target text respectively.
- Set the `eos_token_id` as the `decoder_start_token_id` for mBART-50. The target language ID is used as the first generated token by passing `forced_bos_token_id` to [`generate`].

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

