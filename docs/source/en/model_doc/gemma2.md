
<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Gemma2

## Overview

The Gemma2 model was proposed in [Gemma2: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/google-gemma-2/) by Gemma2 Team, Google.
Two Gemma2 models are released, with parameters sizes of 9 billion (9B) and 27 billion (27B).

The abstract from the blog post is the following:

*Now we’re officially releasing Gemma 2 to researchers and developers globally. Available in both 9 billion (9B) and 27 billion (27B) parameter sizes, Gemma 2 is higher-performing and more efficient at inference than the first generation, with significant safety advancements built in. In fact, at 27B, it offers competitive alternatives to models more than twice its size, delivering the kind of performance that was only possible with proprietary models as recently as December.*

Tips:

- The original checkpoints can be converted using the conversion script `src/transformers/models/Gemma2/convert_Gemma2_weights_to_hf.py` 

<Tip warning={true}>

- Gemma2 uses sliding window attention every second layer, which makes it unsuitable for typical kv caching with [`~DynamicCache`] or tuples of tensors. To enable caching in Gemma2 forward call, you must initialize a [`~HybridCache`] instance and pass it as `past_key_values` to the forward call. Note, that you also have to prepare `cache_position` if the `past_key_values` already contains previous keys and values.

</Tip>

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ), [Pedro Cuenca](https://huggingface.co/pcuenq) and [Tom Arsen]().


## Gemma2Config

[[autodoc]] Gemma2Config

## Gemma2Model

[[autodoc]] Gemma2Model
    - forward

## Gemma2ForCausalLM

[[autodoc]] Gemma2ForCausalLM
    - forward

## Gemma2ForSequenceClassification

[[autodoc]] Gemma2ForSequenceClassification
    - forward

## Gemma2ForTokenClassification

[[autodoc]] Gemma2ForTokenClassification
    - forward
