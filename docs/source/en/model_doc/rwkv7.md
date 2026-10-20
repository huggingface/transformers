<!--Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-15.*

# RWKV-7

## Overview

RWKV-7 ("Goose") is an attention-free recurrent language model. Each layer replaces
self-attention with a *time-mix* block whose state is a fixed-size matrix per head,
updated by a generalised delta rule, and a *channel-mix* block that is a squared-ReLU
MLP over a one-token shift. Because the state does not grow with the sequence, there
is no KV cache: memory is constant in context length and each new token costs the
same as the first.

The model was released by the RWKV project; the reference implementation lives at
[BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM).

The backbone uses the standard names (`model.embed_tokens`, `model.layers.N`,
`model.norm`, `lm_head`). Inside a block the names follow the reference
implementation (`att.receptance`, the LoRA factors as raw `w1`/`w2` tensors, `ln_x`,
`ffn.key`, ...). Checkpoints stored with the reference backbone names
(`rwkv7.emb`, `rwkv7.blocks.N`, `rwkv7.ln_out`, `head`) are renamed when they are
loaded, and `save_pretrained` writes those names back.

## Checkpoints

The RWKV organisation publishes checkpoints that load into this implementation
directly, with no conversion and no `trust_remote_code`, for example:

- [`RWKV/RWKV7-G1k-1.5B-20260930`](https://huggingface.co/RWKV/RWKV7-G1k-1.5B-20260930),
  [`RWKV/RWKV7-G1k-2.9B-20260930`](https://huggingface.co/RWKV/RWKV7-G1k-2.9B-20260930),
  [`RWKV/RWKV7-G1k-7.2B-20260930`](https://huggingface.co/RWKV/RWKV7-G1k-7.2B-20260930),
  [`RWKV/RWKV7-G1k-13.3B-20260930`](https://huggingface.co/RWKV/RWKV7-G1k-13.3B-20260930)
- [`RWKV/RWKV7-1.5B-20260805`](https://huggingface.co/RWKV/RWKV7-1.5B-20260805) and the
  other `-20260805` sizes

Two other layouts exist and do need a step first:

- **native**: the reference `.pth` from [`BlinkDL/rwkv-7-world`](https://huggingface.co/BlinkDL/rwkv-7-world)
  and [`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1), a bare `torch.save`
  of a flat dict rather than a `transformers` directory.
- **fla**: the safetensors layout of the `flash-linear-attention` port, e.g.
  [`fla-hub/rwkv7-0.1B-g1`](https://huggingface.co/fla-hub/rwkv7-0.1B-g1), whose tensor
  names and LoRA layout differ.

Convert once, then use the output like any other checkpoint:

```bash
huggingface-cli download BlinkDL/rwkv-7-world RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth --local-dir .
python src/transformers/models/rwkv7/convert_rwkv7_checkpoint_to_hf.py \
    --checkpoint RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
    --flavour native --output_dir ./rwkv7-0.1b-hf
```

`--flavour fla` takes the fla safetensors layout instead. A converted `.pth` carries no
tokenizer; use the one from any of the official repos above.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("RWKV/RWKV7-1.5B-20260805")
model = AutoModelForCausalLM.from_pretrained("RWKV/RWKV7-1.5B-20260805", dtype=torch.bfloat16)

inputs = tokenizer("\nThe Eiffel Tower is located in the city of", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=20)[0]))
```

The recurrent state is returned in `past_key_values` as a [`DynamicCache`] whose layers
are linear-attention layers: per block, two width-1 conv states (the token shifts of the
time mix and of the channel mix) and one recurrent state (the WKV matrix, kept in
float32). Its size does not depend on the sequence length. Feed it back to continue:

```python
out = model(**inputs, use_cache=True)
next_token = out.logits[:, -1:].argmax(-1)
next_out = model(input_ids=next_token, past_key_values=out.past_key_values, use_cache=True)
```

A recurrent state cannot be rolled back to an earlier position, so assisted
generation is not available for this model.

### Padded batches

Pass `attention_mask` whenever a batch is left-padded (as `generate` does). Padding
positions are zeroed at the input of every mixer, so a left-padded row starts from a
zero state and decodes as if it had been run on its own. Right padding would run the
recurrence over the pads after the real tokens and is not supported.

### Kernels

The WKV recurrence uses `chunk_rwkv7` from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) when it is
installed. Without it, a per-token PyTorch reference implementation is used: it is
correct and runs anywhere, but a long prompt is processed one token at a time and is
much slower.

### Fine-tuning

The projections a PEFT adapter would normally attach to are `receptance`, `key`,
`value` and `output` in the time-mix, and `key` and `value` in the channel-mix.
The last two share their names with the time-mix pair, so a bare
`target_modules=["key", "value"]` attaches to both blocks:

```python
from peft import LoraConfig, get_peft_model

# To attach to the time-mix alone, qualify the names:
# ["att.receptance", "att.key", "att.value", "att.output"].
config = LoraConfig(r=8, lora_alpha=16, target_modules=["receptance", "key", "value", "output"])
model = get_peft_model(model, config)
```

## Rwkv7Config

[[autodoc]] Rwkv7Config

## Rwkv7Model

[[autodoc]] Rwkv7Model
    - forward

## Rwkv7ForCausalLM

[[autodoc]] Rwkv7ForCausalLM
    - forward
