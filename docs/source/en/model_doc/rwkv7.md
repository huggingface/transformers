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

No checkpoint on the Hub loads into this implementation as-is, and it is worth being
explicit about why rather than letting a `from_pretrained` line imply otherwise. The
RWKV-7 weights published today come in two layouts, neither of which is this one:

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
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Hakureirm/rwkv7-168m-pile-hf")
model = AutoModelForCausalLM.from_pretrained("Hakureirm/rwkv7-168m-pile-hf")

inputs = tokenizer("The capital of France is", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=20)[0]))
# The capital of France is the city of Paris.
```

That one is a Pile checkpoint, so its tokenizer is the GPT-NeoX-20B one and loads as an
ordinary fast tokenizer. The World checkpoints are the ones worth using — far more
training data, multilingual — but their vocabulary is RWKV's own, whose tokenizer is
greedy longest-match over raw bytes with no pre-tokenisation and ships as remote code,
so it has to be loaded from a repo that carries it:

```python
tokenizer = AutoTokenizer.from_pretrained("Hakureirm/rwkv7-0.1b-hf", trust_remote_code=True)
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
`cu_seq_lens`, the cumulative boundaries starting at 0 and ending at `seq_len`:

```python
packed = torch.cat([a, b, c], dim=1)                     # [1, len(a)+len(b)+len(c)]
cu_seq_lens = torch.tensor([0, a.shape[1], a.shape[1] + b.shape[1], packed.shape[1]])
out = model(input_ids=packed, cu_seq_lens=cu_seq_lens)
```

Each segment then decodes exactly as if it had been run on its own: the recurrent
state restarts at every boundary, and so does the token shift, which would
otherwise hand a segment's first token the previous sequence's last hidden state.
A malformed boundary list raises rather than quietly restarting the recurrence in
the wrong places.

### Swapping the WKV kernel

The recurrence is the one part worth replacing (everything around it is ordinary
linear algebra), so it is looked up by name instead of hard-coded:

```python
from transformers.models.rwkv7.modeling_rwkv7 import RWKV7_WKV_FUNCTIONS

RWKV7_WKV_FUNCTIONS["my_kernel"] = my_wkv
model = Rwkv7ForCausalLM.from_pretrained(checkpoint, wkv_implementation="my_kernel")
```

An entry receives `[batch, seq_len, num_heads, head_dim]` tensors for `r, w_log, k,
v, kk, a`, the `[batch, num_heads, head_dim, head_dim]` state, and `cu_seq_lens` as
a keyword, and returns `(output, new_state)`. The contract is to reproduce
`rwkv7_recurrent`, which is also what the test suite checks against, so a fused or
varlen kernel drops in without forking the model.

A worked example, against `vllm-rwkv`'s varlen WKV. Its calling convention is not
guessable from the signature and was read off that project's own call site, then
checked against `rwkv7_recurrent` before being relied on: `w` is the LoRA output
*before* the decay transform (the kernel applies that itself), the rank-one pair is
`(-kk, kk * a)`, activations are fp16 while the state is fp32, and `head_dim` must be
64. Single-token decode is a different entry point, `wkv_one`, whose state is fp16.

```python
import torch
from transformers.models.rwkv7.modeling_rwkv7 import RWKV7_WKV_FUNCTIONS

torch.ops.load_library("<vllm-rwkv>/vllm/rwkv7_ops.abi3.so")
_INV_SQRT_E = 0.6065306597126334


def vllm_varlen_wkv(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kwargs):
    batch, seq_len, heads, head_dim = r.shape
    channels = heads * head_dim
    # This model carries `w_log = -INV_SQRT_E * sigmoid(w_raw)`; the kernel wants
    # `w_raw` and applies the transform itself, so invert it here.
    sigmoid = (-w_log / _INV_SQRT_E).clamp(1e-6, 1 - 1e-6)
    w_raw = torch.log(sigmoid / (1 - sigmoid))
    flat = lambda t: t.reshape(batch * seq_len, channels).to(torch.float16).contiguous()

    # Without `cu_seq_lens` every row of the batch is its own segment.
    cu = cu_seq_lens if cu_seq_lens is not None else torch.arange(
        0, batch * seq_len + 1, seq_len, device=r.device)
    n_seq = cu.numel() - 1
    lengths = (cu[1:] - cu[:-1]).tolist()
    slot_state = torch.zeros(n_seq, heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
    y = torch.empty(batch * seq_len, channels, device=r.device, dtype=torch.float16)
    torch.ops.rwkv7_wkv_fp32_v2.forward_varlen(
        n_seq, batch * seq_len, max(lengths), channels, heads, cu.to(torch.int32),
        torch.arange(n_seq, device=r.device, dtype=torch.int32), slot_state,
        flat(r), flat(w_raw), flat(k), flat(v), flat(-kk), flat(kk * a), y)
    # Packed, the contract is the last segment's state, one row. Unpacked, every row
    # of the batch is its own sequence and every row's state has to come back: this
    # returned `slot_state[-1:]` in both cases, and since the caller copies into a
    # `[batch, ...]` cache slot, one row broadcast silently over all of them and rows
    # 0 and 1 continued from row 2's history with no error.
    new_state = slot_state[-1:] if cu_seq_lens is not None else slot_state
    return y.view(batch, seq_len, heads, head_dim).to(r.dtype), new_state.to(state.dtype)


RWKV7_WKV_FUNCTIONS["vllm_varlen"] = vllm_varlen_wkv
```

Checked against `rwkv7_recurrent` on three shapes, a packed row of uneven segments
(5 + 7), an unpacked `batch=1`, and an unpacked `batch=3`, at relative errors of
2.6e-04 to 5.2e-04, which is fp16 against an fp32 reference. That check compared the
returned activations on all three, and the returned *state* on none of them, which is
how the `batch>1` state truncation above survived it: an adapter can be right about
every value it emits and wrong about what it carries forward. A replacement kernel
should be checked on both. It speeds up prefill and leaves single-token decode
unchanged, because that step is bound by streaming the weights rather than by the
recurrence.

### Getting good decode throughput

Single-stream decode is dominated by per-kernel launch overhead, so the compile mode
matters more than anything in the model itself. Eager is by far the slowest, plain
`torch.compile()` helps, and `mode="max-autotune"` is the fastest here.

Two things have to line up, and neither is the default:

```python
model = Rwkv7ForCausalLM.from_pretrained(checkpoint, dtype=torch.float16)
model = model.eval().cuda()
state = model.rwkv7.allocate_state(1)                 # BEFORE compiling, not state=None
compiled = torch.compile(model, mode="max-autotune", dynamic=False)
```

1. **`mode="max-autotune"`.** Plain `torch.compile()` and `"reduce-overhead"` both
   help less; the decode is a long chain of small kernels, which is what autotuning
   has the most to work with.
2. **Call `allocate_state` before compiling.** Anything first allocated *inside* the
   compiled region cannot have its address pinned (`mark_static_address` does not run
   during tracing), and inductor then declines CUDA graphs for a region that mutates
   its inputs. The recurrent state is in that category; on some torch builds the
   cudagraph pass does not merely skip but segfaults. Passing `state=None` and letting
   the model allocate is correct, just several times slower.

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
