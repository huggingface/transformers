<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Paged attention

This page documents the paged attention forward function used in [continuous batching](./continuous_batching). It wraps two versions of the flash attention kernel to handle different batch configurations efficiently.

## Varlen path

The `flash_attn_varlen_func` kernel handles variable length batches. This path is recommended for batches with a large number of requests in prefill.

### Cache behavior

This kernel has no mechanism to interact with the paged cache directly, so the cache is manually read and written using the [`~PagedAttentionCache.update`] method. This can become a bottleneck when sequence length grows large.

### Indexing mechanism

The kernel uses maximum sequence length (`max_seqlen_q`, `max_seqlen_k`) and cumulative sequence lengths (`cu_seq_lens_q`, `cu_seq_lens_k`) to compute attention for each sequence.

### Example

Consider a batch of 3 sequences with query lengths `[10, 3, 1]` and key lengths `[0, 1, 7]`:

```
cu_seq_lens_q = [0, 10, 13, 14]
cu_seq_lens_k = [0, 0, 1, 8]
max_seqlen_q = 10
max_seqlen_k = 7
```

Input shapes:

```
Q:      [1, 10+3+1, num_heads, head_dim] = [1, 14, num_heads, head_dim]
K or V: [1, 0+1+7, num_kv_heads, head_dim] = [1, 8, num_kv_heads, head_dim]
```

The kernel assigns each query and key/value token to a sequence using the cumulative sequence lengths:

```
Q request index: [r0, r0, r0, r0, r0, r0, r0, r0, r0, r0, r1, r1, r1, r2]
cu_seq_lens_q:   0____________________________________10__________13__14

K request index: [r1, r2, r2, r2, r2, r2, r2, r2]  (r0 has 0 K tokens)
cu_seq_lens_k: 0,0_1_______________________8
```

## Decode path

The `flash_attn_with_kvcache` kernel handles decode-only batches where each sequence has exactly one query token. This is more efficient than the varlen path but cannot handle batches with prefilling requests.

### Cache behavior

This kernel interacts with the paged cache using a `block_table` to index into the cache and update it in-place. The block table has shape `(batch_size, max_blocks_per_seq)`, where each row contains the physical locations of a request's cache blocks in the KV cache tensor.

### Indexing mechanism

The kernel uses `cache_seqlens` to retrieve the cache length for each sequence. It assumes each query token belongs to a different sequence (one token per sequence).

### Example

Consider a batch of 3 sequences with query lengths `[1, 1, 1]` and key lengths `[30, 32, 70]`. The cache block size is 32 and the maximum number of blocks per sequence is 4.

The cache sequence lengths are simply the key lengths:

```
cache_seqlens = [30, 32, 70]
```

The block table shape is `(3, 4)`. Using example addresses:

```
block_table = [[2, -1, -1, -1],
               [0,  1, -1, -1],
               [3,  5,  6, -1]]
```

Values of `-1` indicate unallocated blocks.

- **Sequence 0** (30 cached tokens): cache in `KV_cache[2]`. The new token fits (30 + 1 = 31 < 32).
- **Sequence 1** (32 cached tokens): cache in `KV_cache[0]` and `KV_cache[1]`. A second block is needed since 32 + 1 > 32.
- **Sequence 2** (70 cached tokens): cache in `KV_cache[3]`, `KV_cache[5]`, and `KV_cache[6]`. Note that blocks are not necessarily contiguous, which is the key advantage of paged cache. The new token fits in the third block.
