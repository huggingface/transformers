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
*This model was released on 2018-04-01 and added to Hugging Face Transformers on 2020-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MarianMT



[MarianMT](https://huggingface.co/papers/1804.00344) is a machine translation model trained with the Marian framework which is written in pure C++. The framework includes its own custom auto-differentiation engine and efficient meta-algorithms to train encoder-decoder models like BART.

All MarianMT models are transformer encoder-decoders with 6 layers in each component, use static sinusoidal positional embeddings, don't have a layernorm embedding, and the model starts generating with the prefix `pad_token_id` instead of `<s/>`.



You can find all the original MarianMT checkpoints under the [Language Technology Research Group at the University of Helsinki](https://huggingface.co/Helsinki-NLP/models?search=opus-mt) organization.


> [!TIP]
> This model was contributed by [sshleifer](https://huggingface.co/sshleifer).
>
> Click on the MarianMT models in the right sidebar for more examples of how to apply MarianMT to translation tasks.


The example below demonstrates how to translate text using [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python

import torch
from transformers import pipeline

pipeline = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de", dtype=torch.float16, device=0)
pipeline("Hello, how are you?")

```

</hfoption>

<hfoption id="AutoModel">

```python

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de", dtype=torch.float16, attn_implementation="sdpa", device_map="auto")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, cache_implementation="static")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

</hfoption>
</hfoptions>


Use the [AttentionMaskVisualizer](https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139) to better understand what tokens the model can and cannot attend to.

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("Helsinki-NLP/opus-mt-en-de")
visualizer("Hello, how are you?")
```
<div class="flex justify-center">
   <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/marianmt-attn-mask.png"/>
</div>

## Notes

- MarianMT models are ~298MB on disk and there are more than 1000 models. Check this [list](https://huggingface.co/Helsinki-NLP) for supported language pairs. The language codes may be inconsistent. Two digit codes can be found [here](https://developers.google.com/admin-sdk/directory/v1/languages) while three digit codes may require further searching.
- Models that require BPE preprocessing are not supported.
- All model names use the following format: `Helsinki-NLP/opus-mt-{src}-{tgt}`. Language codes formatted like `es_AR` usually refer to the `code_{region}`. For example, `es_AR` refers to Spanish from Argentina.
- If a model can output multiple languages, prepend the desired output language to `src_txt` as shown below. New multilingual models from the [Tatoeba-Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge) require 3 character language codes.

```python

from transformers import MarianMTModel, MarianTokenizer

# Model trained on multiple source languages → multiple target languages
# Example: multilingual to Arabic (arb)
model_name = "Helsinki-NLP/opus-mt-mul-mul"  # Tatoeba Challenge model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Prepend the desired output language code (3-letter ISO 639-3)
src_texts = ["arb>> Hello, how are you today?"]

# Tokenize and translate
inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs)

# Decode and print result
translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(translated_texts[0])

```

- Older multilingual models use 2 character language codes.

```python

from transformers import MarianMTModel, MarianTokenizer

# Example: older multilingual model (like en → many)
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # English → French, Spanish, Italian, etc.
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Prepend the 2-letter ISO 639-1 target language code (older format)
src_texts = [">>fr<< Hello, how are you today?"]

# Tokenize and translate
inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs)

# Decode and print result
translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(translated_texts[0])

```

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
