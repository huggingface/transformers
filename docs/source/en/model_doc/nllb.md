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
*This model was released on 2022-07-11 and added to Hugging Face Transformers on 2022-07-18 and contributed by [lysandre](https://huggingface.co/lysandre).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# NLLB

[NLLB](https://huggingface.co/papers/2207.04672) addresses the challenge of translating low-resource languages by developing a conditional compute model based on Sparsely Gated Mixture of Experts. This model uses novel data mining techniques to train on thousands of tasks, improving overfitting resistance. Evaluated on over 40,000 translation directions with the Flores-200 benchmark and a toxicity benchmark, NLLB achieves a 44% BLEU improvement over the previous state-of-the-art, advancing towards a universal translation system.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="translation_en_to_fr", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("fra_Latn"))
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- The tokenizer was updated in April 2023. It now prefixes the source sequence with the source language instead of the target language. This prioritizes zero-shot performance at a minor cost to supervised performance.
- For non-English languages, specify the language's BCP-47 code with the `src_lang` keyword.

## NllbTokenizer

[[autodoc]] NllbTokenizer
    - build_inputs_with_special_tokens

## NllbTokenizerFast

[[autodoc]] NllbTokenizerFast
