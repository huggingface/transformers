<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-11.*

# Kolibri 1.1

Kolibri 1.1 is Aleph Alpha's successor to [Kolibri 1](./kolibri1).
Comparison to Kolibri 1:
- It keeps the mixture-of-experts layers and the sliding-window attention layers of Kolibri 1.
- The first two layers have a dense MLP instead of a mixture of experts.
- Every fifth layer attends to the whole context, but sparsely. These layers replace the full-attention layers of Kolibri 1. They are the `"indexed_attention"` entries of `layer_types`.
- Indexed-attention layers can attend to the keys and values of an earlier indexed-attention layer. Only the 5th, 20th, 25th and 30th layers compute their own. The other indexed-attention layers attend to those of the closest preceding of these, except the 50th layer, which attends to those of the 5th.

The sparse attention is designed as follows:

- Every query attends to the 8 first tokens, the 128 most recent tokens and the 2048 highest scoring other keys. Queries that see no more keys than that attend to all of them.
- Keys are scored without learned weights: by the ReLU of the logits of the first 12 query heads against key head 0, averaged over these heads.
- Indexer layers (the 5th, 15th, 20th and 30th) first pick a pool of 16384 candidate keys: all of them for short contexts, otherwise the 1024 blocks of 16 consecutive keys whose averaged key scores highest. The other indexed-attention layers rescore the pool of the closest preceding indexer.

The sparse attention keys and values, as well as the routing queries and keys, are rounded to NVFP4.

This is a reference implementation. Sparse layers apply their key selection as an explicit attention mask, so only the eager and SDPA attention implementations are supported, selecting keys costs time and memory quadratic in the prompt length, and every decoding step rescans the whole context. A prefill of about 50k tokens fits on two 180 GB GPUs; 128k tokens does not.


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<checkpoint>")
model = AutoModelForCausalLM.from_pretrained("<checkpoint>", device_map="auto", attn_implementation="sdpa")

messages = [{"role": "user", "content": "Erkläre kurz, was ein Mixture-of-Experts-Modell ist."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Kolibri1_1Config

[[autodoc]] Kolibri1_1Config

## Kolibri1_1Model

[[autodoc]] Kolibri1_1Model
    - forward

## Kolibri1_1ForCausalLM

[[autodoc]] Kolibri1_1ForCausalLM
    - forward
