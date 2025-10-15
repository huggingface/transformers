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
*This model was released on 2018-04-01 and added to Hugging Face Transformers on 2020-11-16 and contributed by [sshleifer](https://huggingface.co/sshleifer).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MarianMT

[MarianMT](https://huggingface.co/papers/1804.00344) is a neural machine translation framework built entirely in C++ with its own automatic differentiation engine based on dynamic computation graphs. It implements an encoder-decoder architecture designed for both research flexibility and performance. The framework emphasizes efficiency, offering fast training and translation speeds while remaining self-contained. This makes it a practical toolkit for developing and testing machine translation models without external dependencies.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- MarianMT models are ~298MB on disk. There are more than 1000 models available. Check the [supported language pairs list](https://huggingface.co/models?library=transformers&pipeline_tag=translation&sort=downloads) for available options.
- Language codes may be inconsistent. Two-digit codes are available in the [language codes list](https://huggingface.co/models?library=transformers&pipeline_tag=translation&sort=downloads). Three-digit codes may require further searching.
- Models that require BPE preprocessing aren't supported.
- All model names use this format: `Helsinki-NLP/opus-mt-{src}-{tgt}`. Language codes like `es_AR` refer to `code_{region}`. For example, `es_AR` refers to Spanish from Argentina.
- If a model outputs multiple languages, prepend the desired output language to `src_txt`. New multilingual models from the Tatoeba-Challenge require 3-character language codes. Older multilingual models use 2-character language codes.

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

