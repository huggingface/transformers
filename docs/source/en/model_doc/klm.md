<!--Copyright 2026 The KLM authors and The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.
-->
*This model was contributed to Hugging Face Transformers on 2026-11-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# KLM

[KLM](https://huggingface.co/MinimaLabs/KLM-1.6) is a small causal language model that mixes two kinds of layers. Three out of four are Gated DeltaNet-2 linear attention layers, which read the past through a fixed-size recurrent state, and every fourth one is a full attention layer. Three more things set it apart from the usual decoder:

- **Block attention residuals.** There is no single residual stream. Every attention and MLP sublayer reads its own softmax mix of all the states computed so far: the token embeddings, the n-gram embeddings and the output of each previous layer.
- **N-gram embeddings.** The most frequent bigrams and trigrams have their own embeddings, found by exact lookup of the two and three last tokens.
- **Documents.** A sequence can hold several documents separated by the end of sequence token. Each one is read as if it were alone: the attention mask, the recurrent states, the convolutions and the n-grams all restart at its first token.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


pipe = pipeline(task="text-generation", model="MinimaLabs/KLM-1.6")
result = pipe("Photosynthesis is the process by which plants")
print(result)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("MinimaLabs/KLM-1.6")
model = AutoModelForCausalLM.from_pretrained("MinimaLabs/KLM-1.6", device_map="auto")
inputs = tokenizer("Photosynthesis is the process by which plants", return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- The model needs `input_ids`, to look up n-grams and to find where documents start. `inputs_embeds` is not supported.
- The first token after padding starts a document, so left padding is never read. Use left padding for batched generation. See [Padding side](../llm_tutorial#padding-side).
- The n-gram keys are computed from `vocab_size`. Resizing the token embeddings of a pretrained checkpoint changes `vocab_size`, and with it every key, so add tokens within the existing vocabulary instead.
- The recurrent states of the linear attention layers cannot be rolled back, so generation methods that crop the cache, such as assisted generation, are not available.
- The linear attention layers run in plain PyTorch. A sequence is solved one chunk of tokens at a time, and decoding with a cache updates the state one token at a time.

## KlmConfig

[[autodoc]] KlmConfig

## KlmModel

[[autodoc]] KlmModel
    - forward

## KlmForCausalLM

[[autodoc]] KlmForCausalLM
    - forward
