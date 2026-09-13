<!---
Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# HELIX measurements

Two scripts, one for each half of the claim HELIX makes: that it drops the quadratic cost of attention,
and that it does *not* drop attention's ability to recall an exact value from the distant past.

See [the model card](../../../docs/source/en/model_doc/helix.md) for the architecture, and
`tests/models/helix/test_modeling_helix.py` for the correctness properties (causality, decode/prefill
equivalence, padding invariance, cache shapes) that are asserted rather than measured.

## `scaling.py` — cost against context length

```bash
python scaling.py --lengths 4096 8192 16384 32768          # attention pairs + wall clock
python scaling.py --lengths 4096 8192 16384 32768 --cache  # what has to be kept to keep generating
```

Reports how many query/key products the forward pass actually forms — counted by instrumenting the
attention call, so it is exact and hardware-independent — alongside wall-clock time or the size of the
decode cache.

Measured on 4 CPU cores, 4 layers, `hidden_size=256`, 4 query heads over 2 key/value heads, HELIX with
`block_size=64`, `local_blocks=3`, `index_topk=8`, `index_layer_stride=2`:

| tokens | HELIX pairs/token | Llama pairs/token | HELIX s | Llama s | HELIX cache | Llama cache |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 096 | 8 192 | 32 776 | 0.4 | 0.2 | 8.7 MiB | 16.0 MiB |
| 8 192 | 8 192 | 65 544 | 0.9 | 0.6 | 16.8 MiB | 32.0 MiB |
| 16 384 | 8 192 | 131 080 | 1.7 | 1.9 | 32.9 MiB | 64.0 MiB |
| 32 768 | 8 192 | 262 152 | 4.3 | 6.7 | 65.2 MiB | 128.0 MiB |

Fitted exponents in `quantity ~ N^alpha`:

| | attention pairs | wall clock | decode cache |
| --- | ---: | ---: | ---: |
| HELIX | **1.00** | 1.11 | 0.97 |
| Llama | **2.00** | 1.70 | 1.00 |

Read the pair counts first: they are the asymptotics, exactly. HELIX forms the same 8 192 products per
token at every length — `(local_blocks + 1 + index_topk) * block_size` keys per query, per head, per
indexed layer — while full attention's grows with the context and is 32x larger by 32k tokens.

The wall clock lags the pair count because HELIX is plain PyTorch with no fused kernel for its gathered
attention, while `Llama` runs on a fused SDPA kernel. Even so it crosses over around 16k tokens and is
1.6x faster at 32k, and the gap widens from there.

Two honest caveats. A memory-efficient attention kernel already keeps full attention *linear* in activation
memory, so `--memory` does not separate the two — what is quadratic in a transformer is compute, not
working memory. And HELIX's cache is only ~2x smaller here rather than bounded: the layers that run the
index must keep the full key/value history, because reaching an arbitrary past token is the whole point.
Raising `index_layer_stride` trades recall depth for cache size directly.

## `mqar.py` — recall against number of bindings

```bash
python mqar.py --pairs 16 48 --steps 800
```

Trains three models from scratch on multi-query associative recall: each sequence writes a set of
key/value bindings, then queries a shuffled subset of them far enough away that no local window can still
see the write. Accuracy is measured only at the answer positions.

Three models, matched in width and depth:

| model | what it tests |
| --- | --- |
| `HELIX (L+R+I)` | the full braid |
| `HELIX no index (L+R)` | the same model with `index_layer_stride` pushed past the layer count, i.e. a local-window + delta-rule hybrid. This is the ablation: it isolates what the index strand contributes |
| `Llama (full attention)` | the ceiling — every token keeps every key |

Measured with 16 bindings, 128 possible keys, 16 possible values, 256-token sequences, 2 500 steps at
batch 24 — so chance is 6.2%, and the bindings are written ~200 tokens before they are queried, six times
further back than HELIX's widest local window of 32 tokens:

| model | recall |
| --- | ---: |
| HELIX (L+R+I) | **30.8%** |
| HELIX no index (L+R) | 6.0% |
| Llama (full attention) | 32.0% |
| *chance* | *6.2%* |

This is the result the architecture stands on. Strip the index strand out and the model is **at chance** —
its local window cannot see the bindings and its recurrent state cannot hold them. Put the index back and
the same model, at the same width and depth, recovers essentially all of what full attention gets, while
forming a fraction of the attention products.

None of the three is near saturation at this budget; the numbers are a comparison at equal training, not a
ceiling. What matters is the gap between the two HELIX rows, which is exactly the index strand's
contribution.
