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

This implementation keeps the reference parameter names (`blocks.N.att.receptance`,
the LoRA factors as raw `w1`/`w2` tensors, `emb`, `head`, …) rather than renaming
them, so converting a native `.pth` checkpoint needs no per-tensor mapping table.

## Checkpoints

The RWKV organisation publishes checkpoints that load into this implementation
directly, with no conversion and no `trust_remote_code`:

- [`RWKV/RWKV7-1.5B-20260805`](https://huggingface.co/RWKV/RWKV7-1.5B-20260805)
- [`RWKV/RWKV7-2.9B-20260805`](https://huggingface.co/RWKV/RWKV7-2.9B-20260805)
- [`RWKV/RWKV7-7.2B-20260805`](https://huggingface.co/RWKV/RWKV7-7.2B-20260805)
- [`RWKV/RWKV7-13.3B-20260805`](https://huggingface.co/RWKV/RWKV7-13.3B-20260805)

They are plain `safetensors` directories with a `tokenizer.json`, and their
`config.json` maps onto [`Rwkv7Config`] key for key.

Two other layouts exist and do need a step first:

- **native**: the reference `.pth` from [`BlinkDL/rwkv-7-world`](https://huggingface.co/BlinkDL/rwkv-7-world)
  and [`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1). Same parameter
  names as here, but a bare `torch.save` of a flat dict, not a `transformers` directory.
- **fla**: [`RWKV/RWKV7-Goose-World2.8-0.1B-HF`](https://huggingface.co/RWKV/RWKV7-Goose-World2.8-0.1B-HF),
  [`fla-hub/rwkv7-0.1B-g1`](https://huggingface.co/fla-hub/rwkv7-0.1B-g1) and friends.
  These are `transformers` directories, but they are `trust_remote_code=True` repos
  whose modelling code and tensor names come from `flash-linear-attention`, so they
  load their own implementation rather than this one.

Convert once, then use the output like any other checkpoint:

```bash
huggingface-cli download BlinkDL/rwkv-7-world RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth --local-dir .
python src/transformers/models/rwkv7/convert_rwkv7_checkpoint_to_hf.py \
    --checkpoint RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
    --flavour native --output_dir ./rwkv7-0.1b-hf
```

`--flavour fla` takes the safetensors layout instead, so a checkpoint that only exists
in fla form does not have to be re-trained to be used here. The tokenizer is the RWKV
world tokenizer either way; the fla repos carry it, and `--flavour native` leaves it to
you to point at one.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("RWKV/RWKV7-1.5B-20260805")
model = AutoModelForCausalLM.from_pretrained("RWKV/RWKV7-1.5B-20260805", dtype=torch.bfloat16)

inputs = tokenizer("\nThe Eiffel Tower is located in the city of", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=20)[0]))
# The Eiffel Tower is located in the city of Paris, France.
```

The World vocabulary these use is RWKV's own — greedy longest-match over raw bytes
with no pre-tokenisation — but the official repos ship it as a `tokenizer.json`, so
`AutoTokenizer` loads it as an ordinary fast tokenizer with no remote code. A `.pth`
you convert yourself carries no tokenizer, so point at one of the repos above for it:

```python
tokenizer = AutoTokenizer.from_pretrained("RWKV/RWKV7-1.5B-20260805")
model = AutoModelForCausalLM.from_pretrained("./rwkv7-0.1b-hf")   # converted above
```

The recurrent state is returned as `state` and can be fed back to continue a
sequence; it replaces `past_key_values` and is O(1) in the sequence length.

```python
out = model(**inputs, use_cache=True)
next_out = model(input_ids=next_token, state=out.state, use_cache=True)
```

### Padded batches

Pass `attention_mask` whenever a batch is padded. There is no attention to mask here, but
padding is neutralised inside the recurrence instead, by holding the state transition at
the identity for those positions, so a padded row decodes exactly as if it had been run
on its own. Without the mask the padding is fed through the recurrence like any other
token and moves the state before the real tokens arrive, which silently corrupts every
row shorter than the longest one. `generate` passes the mask through for you; an
all-ones mask costs nothing and changes nothing. A single decoded token is *not*
exempt: that shortcut was removed after a fully-masked one-token row was found moving
the state, so the mask is honoured at every length, and this paragraph said otherwise
for several commits after the code stopped doing it.

### Packed (varlen) batches

When the lengths in a batch vary a lot, padding is expensive twice over for a
recurrent model: the pads cost time as well as memory, because every one of them
is a step of the recurrence. Pack the sequences into a single row instead and pass
`cu_seq_lens_q`, the cumulative boundaries starting at 0 and ending at `seq_len`:

```python
packed = torch.cat([a, b, c], dim=1)                     # [1, len(a)+len(b)+len(c)]
cu_seq_lens_q = torch.tensor([0, a.shape[1], a.shape[1] + b.shape[1], packed.shape[1]])
out = model(input_ids=packed, cu_seq_lens_q=cu_seq_lens_q)
```

Each segment then decodes exactly as if it had been run on its own: the recurrent
state restarts at every boundary, and so does the token shift, which would
otherwise hand a segment's first token the previous sequence's last hidden state.
A malformed boundary list raises rather than quietly restarting the recurrence in
the wrong places.

### Fine-tuning

The projections a PEFT adapter would normally attach to are `receptance`, `key`,
`value` and `output` in the time-mix, and `key` and `value` in the channel-mix.
The last two share their names with the time-mix pair, so a bare
`target_modules=["key", "value"]` attaches to both blocks, not just the time-mix:

```python
from peft import LoraConfig, get_peft_model

# Six projections per layer. To attach to the time-mix alone, qualify the names:
# ["att.receptance", "att.key", "att.value", "att.output"].
config = LoraConfig(r=8, lora_alpha=16, target_modules=["receptance", "key", "value", "output"])
model = get_peft_model(model, config)
```

Trainers that chunk the loss over the output projection may read the state back
off the model output under an attention model's name. TRL's `SFTTrainer` defaults
to `loss_type="chunked_nll"`, whose chunked head reads `outputs.past_key_values`;
this model returns its recurrence in `state`, as `MambaOutput` returns its own in
`cache_params`, and neither name is found. Pass `loss_type="nll"` on TRL versions
where the chunked head still reads the field unconditionally; it skips that path
entirely.

### Getting good decode throughput

Single-stream decode is dominated by per-kernel launch overhead, so the compile mode
matters more than anything in the model itself: profiled on an RTX 3090 at 1.5B, the
GPU is busy for ~16ms of an ~83ms eager decode step -- the rest is the interpreter
feeding it. `mode="reduce-overhead"` removes that layer and captures the step into
CUDA graphs, for an order-of-magnitude decode speedup (measured 16x on that setup;
`"max-autotune"` came out *slower* than `"reduce-overhead"` here, its extra fusions
not paying for themselves on this chain of small kernels).

Two things have to line up, and neither is the default:

```python
model = Rwkv7ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
model = model.eval().cuda()
state = model.rwkv7.allocate_state(1)                 # BEFORE compiling, not state=None
compiled = torch.compile(model, mode="reduce-overhead")
```

1. **Compile the decode step, not the prefill.** Run the prompt through the *eager*
   model into the pre-allocated state, then loop the compiled model one token at a
   time. Handing the prefill to the compiled model builds a lazy, unpinned cache
   inside the traced region and trips the cudagraph output-buffer protection.
2. **Call `allocate_state` before compiling.** Anything first allocated *inside* the
   compiled region cannot have its address pinned (`mark_static_address` does not run
   during tracing), and inductor then declines CUDA graphs for a region that mutates
   its inputs. The recurrent state is in that category; on some torch builds the
   cudagraph pass does not merely skip but segfaults. Passing `state=None` and letting
   the model allocate is correct, just several times slower.

The compiled step hands back the same buffers every time, which is where the speed
comes from and also why a kept reference goes stale: a logits row read after the next
call has already been overwritten. That surfaces as an error rather than as quietly
wrong numbers, since torch checks it (`accessing tensor output of CUDAGraphs that has
been overwritten by a subsequent run`), but code that collects logits -- to score, or
to sample from once the loop is done -- has to `.clone()` each row. Consuming them on
the spot, as an inline `argmax` does, needs nothing.

For more than this, the kernels are a separate, optional package rather than part of
the model: `transformers` keeps the portable implementation that builds and runs
anywhere and that these are checked against.

### Performance notes

Prefill runs a chunk-parallel form of the recurrence rather than a per-token loop:
the running decay is factored out, which turns each chunk into one unit-lower-triangular
solve plus a few matmuls, leaving only the chunk-to-chunk carry serial. Decoding a
single token takes the plain sequential step, which is what it already is.

The model is `torch.compile`-friendly and benefits substantially from it, because the
time-mix is many small elementwise and low-rank operations whose per-op overhead
dominates at small batch.

### DeepEmbed

`config.use_deep_embed` enables the RWKV-8 "DeepEmbed" hook: a per-layer, per-token
vector that channelwise-modulates the channel-mix. The table is deliberately not a
model weight. The design keeps it in RAM/SSD and prefetches per token, which is what
makes it cheap on VRAM, so it is passed to the forward as `deep_embeds` with shape
`[num_layers, batch, seq_len, width]`, where a `width` of `hidden_size` reproduces the
reference "1x" variant and `intermediate_size` the "4x" variant. The forward picks
between them from the tensor it is handed rather than from the config. No
RWKV-7 checkpoint carries such a table; this is an extension point, off by default.

## Rwkv7Config

[[autodoc]] Rwkv7Config

## Rwkv7Cache

[[autodoc]] Rwkv7Cache

## Rwkv7Model

[[autodoc]] Rwkv7Model
    - forward

## Rwkv7ForCausalLM

[[autodoc]] Rwkv7ForCausalLM
    - forward
