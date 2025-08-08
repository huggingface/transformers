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
> Click on the mBART models in the right sidebar for more examples of applying mBART to different language tasks.

> [!NOTE]
> The `head_mask` argument is ignored when using all attention implementation other than "eager". If you have a `head_mask` and want it to have effect, load the model with `XXXModel.from_pretrained(model_id, attn_implementation="eager")`

The example below demonstrates how to translate text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="translation",
    model="facebook/mbart-large-50-many-to-many-mmt",
    device=0,
    dtype=torch.float16,
    src_lang="en_XX",
    tgt_lang="fr_XX",
)
print(pipeline("UN Chief Says There Is No Military Solution in Syria"))
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

article_en = "UN Chief Says There Is No Military Solution in Syria"

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

tokenizer.src_lang = "en_XX"
encoded_hi = tokenizer(article_en, return_tensors="pt").to("cuda")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"], cache_implementation="static")
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- You can check the full list of language codes via `tokenizer.lang_code_to_id.keys()`.
- mBART requires a special language id token in the source and target text during training. The source text format is `X [eos, src_lang_code]` where `X` is the source text. The target text format is `[tgt_lang_code] X [eos]`. The `bos` token is never used. The [`~PreTrainedTokenizerBase._call_`] encodes the source text format passed as the first argument or with the `text` keyword. The target text format is passed with the `text_label` keyword.
- Set the `decoder_start_token_id` to the target language id for mBART.

    ```py
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-en-ro", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")

    article = "UN Chief Says There Is No Military Solution in Syria"
    inputs = tokenizer(article, return_tensors="pt")

    translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
    tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    ```

- mBART-50 has a different text format. The language id token is used as the prefix for the source and target text. The text format is `[lang_code] X [eos]` where `lang_code` is the source language id for the source text and target language id for the target text. `X` is the source or target text respectively.
- Set the `eos_token_id` as the `decoder_start_token_id` for mBART-50. The target language id is used as the first generated token by passing `forced_bos_token_id` to [`~GenerationMixin.generate`].

    ```py
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
    tokenizer.src_lang = "ar_AR"

    encoded_ar = tokenizer(article_ar, return_tensors="pt")
    generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    ```

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