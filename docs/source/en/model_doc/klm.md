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

- The original Hub checkpoint's configuration and weight names are converted automatically by `from_pretrained`.
- The model needs `input_ids`, to look up n-grams and to find where documents start. `inputs_embeds` is not supported.
- The first token after padding starts a document, so left padding is never read. Use left padding for batched generation. See [Padding side](../llm_tutorial#padding-side).
- The n-gram keys are computed from `vocab_size`. Resizing the token embeddings of a pretrained checkpoint changes `vocab_size`, and with it every key, so add tokens within the existing vocabulary instead.
- The recurrent states of the linear attention layers cannot be rolled back, so generation methods that crop the cache, such as assisted generation, are not available.
- With compatible [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) installed, uncached CUDA forwards with float32 parameters use fused GDN-2, residual/prenorm and segmented convolution kernels. CPU/MPS, models with Accelerate placement/offload hooks, cached decoding, output capture, and compilation/export retain the portable operators.

## GPU fine-tuning

KLM uses the same optional kernel-dispatch mechanism as the other hybrid models. Its GDN-2 kernel has separate
channel-wise erase and write gates; the ordinary gated delta-rule kernel cannot replace it. Install an FLA
version providing `chunk_gdn2`, `fused_attnres`, and variable-length `causal_conv1d`. The implementation was
validated with revision `9c102739e348968753d86bb729c57dd6efa1c413`.

Keep the parameters in float32 and use autocast for projections. FlashAttention enables document-aware removal
of padding throughout the encoder; SDPA/eager attention also supports the fused operators while retaining the
original batch layout. Row boundaries, EOS resets, padding gaps and original rotary positions are preserved.
Padding is retained when a causal-LM loss supervises a padding-position logit, such as the prediction of the
first real token after left padding.
Unlabeled causal-LM calls also retain padding-position logits so external loss functions see the same values.
The initial fused path supports SiLU/Swish and linear head dimensions from 8 through 256; other configurations
use the reference operators.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MinimaLabs/KLM-1.6")
model = AutoModelForCausalLM.from_pretrained(
    "MinimaLabs/KLM-1.6", dtype=torch.float32, attn_implementation="flash_attention_2"
).cuda().train()
model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
batch = tokenizer(["The capital of France is Paris."], return_tensors="pt", padding=True).to("cuda")
labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(**batch, labels=labels, use_cache=False).loss
loss.backward()
```

FlashAttention is used for eligible uncached forwards under float16/bfloat16 autocast. Cached generation keeps
the reference recurrence and document-masked SDPA path; requesting attention weights uses eager attention.
First-use kernel compilation and autotuning can dominate short jobs. FLA's tuning cache can be configured before
importing Transformers; hardware-specific presets belong to that backend and are not stored in the model config.

## KlmConfig

[[autodoc]] KlmConfig

## KlmModel

[[autodoc]] KlmModel
    - forward

## KlmForCausalLM

[[autodoc]] KlmForCausalLM
    - forward
