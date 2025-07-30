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
        <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
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

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
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
