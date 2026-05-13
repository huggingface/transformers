<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Continuous batching architecture

Traditional batching processes a fixed group of requests together and waits for the slowest one to finish before starting the next group. This causes the GPU to idle between batches.

With continuous batching, at every generation step, the scheduler checks for finished requests and replaces them immediately with waiting ones. Short requests are kicked out as soon as they're done, and the GPU stays occupied the entire time. The result is significantly higher throughput and lower average latency.

## Request lifecycle

A request moves through four states from submission to completion.

```text
       WAIT IN QUEUE          LOAD PROMPT INTO KV       STREAM OUTPUT           DONE
    ┌────────────────┐    ┌─────────────────────┐    ┌─────────────────┐    ┌────────┐
    │    PENDING     │───▶│     PREFILLING      │───▶│    DECODING     │───▶│FINISHED│
    │  ○ ○ ○ · · ·   │    │ ████████░░░░░░░░░░  │    │  → → → → · ·    │    │   ✓    │
    └────────────────┘    └─────────────────────┘    └─────────────────┘    └────────┘
         others ahead            prompt chunks              +1 token/step
```

1. Pending — the request is queued and waiting to be scheduled.
2. Prefilling — prompt tokens are processed in a forward pass.
3. Decoding — output tokens are generated one at a time.
4. Finished — generation is complete and the result is available.

A request is moved from pending to prefilling when the scheduler finds enough token budget and cache space to admit the request. If the prompt is too long to fit in a single step, the scheduler splits it across multiple forward passes (chunk prefill).

## Chunked prefill

Processing a long prompt in a single step is expensive because it blocks other requests from generating. Chunked prefill solves this by splitting large prefills across multiple steps.

```text
Blocking (long prefill blocks the batch)
  Step:   1        2        3        4
  Req A  [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓]   ← one huge prefill
  Req B  ················ wait ················
  Req C  ················ wait ················

Chunked (same prompt split and interleave with decode)
  Step:   1          2          3          4
  Req A  [▓▓ pre]    [▓▓ pre]   [▓▓ pre]   [→ dec]
  Req B  [→ dec]     [→ dec]    [▓▓ pre]   [→ dec]
  Req C  [→ dec]     · idle ·   [→ dec]    · idle ·
```

When a new request's prompt exceeds the available token budget (set by `max_batch_tokens` in [`ContinuousBatchingConfig`]), the scheduler processes as many tokens as possible and holds the rest. On subsequent steps, it continues from where it left off, interleaving prefill work with ongoing decode steps. This keeps the batch productive and reduces time-to-first-token for other requests.

## Scheduler

The scheduler decides which requests join each forward pass based on two budgets.

- Token budget — the maximum number of query tokens processed in a single forward pass, set by `max_batch_tokens` in [`ContinuousBatchingConfig`].
- Cache budget — the total number of KV pages that can be read in a single pass, which is bounded by the total cache size.

### FIFO

[`FIFOScheduler`] is the default scheduler. It fills the batch in priority order. Active decode requests are processed first, then active prefill requests, and then new waiting requests in arrival order. This maximizes throughput by pushing existing requests through before pulling in new ones.

### PrefillFirst

[`PrefillFirstScheduler`] completes chunked prefill operations before resuming decode. This reduces fragmentation for workloads dominated by long prompts.

Set the scheduler in [`ContinuousBatchingConfig`].

```py
cb_config = ContinuousBatchingConfig(scheduler_type="prefill_first")
```

## Memory management

The KV cache is paged so requests of different lengths can share a fixed memory pool without fragmentation. The cache divides memory into fixed-size blocks, and each request holds a list of block IDs that the attention kernel reads from and writes to.

```text
Paged KV cache (num_blocks × block_size tokens per block)

    0   1   2   3   4   5   6   7   8   9   …
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ B │ · │ A │ B │ · │ A │ · │ A │ · │ · │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  A  request A: 2, 5, 7   B  request B: 0, 3   ·  free: 1, 4, 6, 8, 9, …
```

A page holds the key/value state for one token in one layer. A block is a span of `block_size` pages (default 256) and is the unit of allocation. Blocks are allocated per layer group. Layers within a group share a block ID, which keeps bookkeeping uniform for mixed-attention models.

### Cache sizing

The manager infers the number of blocks at startup from free GPU memory. The manager solves an equation that accounts for KV tensors, attention masks, activations, and bookkeeping indices, then sizes the pool to fit inside `max_memory_percent` (default 0.9) of the available memory.

You can pin the values explicitly in [`ContinuousBatchingConfig`].

```py
cb_config = ContinuousBatchingConfig(
    block_size=256,
    num_blocks=4096,
    max_batch_tokens=512,
    max_memory_percent=0.8,
)
```

### Admission

Before a request joins the batch, the scheduler checks that enough free blocks exist for every layer group. If any group would fall short, the request is rejected and nothing is allocated.

To avoid filling the cache until [offloading](#offloading) is the only option, the default FIFO scheduler enforces a safety margin. Once free blocks fall below 20% of the total, new prefills are held back and only active decodes continue.

### Prefix caching

When two requests share a prompt prefix, they can share the blocks that hold the KV for that prefix. Each completed block is content-hashed. A later request with a matching prefix reuses the block and skips the prefill for those tokens. Shared blocks are reference-counted and only return to the free pool once every request using them has finished.

Prefix caching is enabled by default and is active only when a model has exclusively full-attention layers. Set `allow_block_sharing=False` in [`ContinuousBatchingConfig`] for workloads with short prompts and long generations, where the bookkeeping outweighs the savings.

### Eviction

Freed blocks aren't wiped and they stay "initialized" with their content and hash intact so a later prefix match can reuse them. When the uninitialized pool runs low, the manager lazily demotes the most recent initialized block back to uninitialized, dropping their hash. The prefix cache is a best-effort layer that the allocator discards to keep serving new requests.

### Soft reset

If the KV cache fills completely during a long session, because requests are generating very long outputs, the manager triggers offloading rather than crashing. It selects the oldest active request (or the newest, if a previous soft reset already blocked new requests from joining), appends its generated tokens to its original prompt, frees its cache blocks, and adds it back to the queue.

When the cache has space again, the request resumes from where it left off with its generation history encoded in the prompt.

## CUDA graphs

Every generation step involves CPU overhead, from assembling the batch to dispatching GPU kernels and reading the results. CUDA graphs eliminate the overhead by recording the full GPU execution sequence once and replaying it for batches with matching shapes.

Because the batch shapes change every step, the continuous batching system handles this with padding and caching.

1. Query lengths are padded to the nearest multiple of `q_padding_interval_size`.
2. KV lengths are padded to the nearest multiple of `kv_padding_interval_size`.
3. Recorded graphs are stored in an LRU cache of up to `max_cached_graphs` entries.

When a batch's padded shape matches a cached graph, the graph replays without any CPU dispatch. New shapes trigger a new graph capture.

CUDA graphs require static control flow and are incompatible with attention masks. They're auto-detected by default and disabled when conditions aren't met.

## Async batching

Async batching uses two I/O buffer pairs and two CUDA streams to overlap CPU and GPU work.

```text
Sequential
  CPU  ── [prep N] ·········idle········· [prep N+1] ·········idle·········
  GPU  ·········idle········· [compute N] ·········idle········· [compute N+1]

Async
  CPU  ── [prep N] [prep N+1] [prep N+2] ──
  GPU  ── ········ [compute N] [compute N+1] ──
```

While the GPU computes batch N, the CPU prepares batch N+1. However, overlapping the two requires roughly double the VRAM.

Async batching requires CUDA graphs to be active, since graph replay provides the stable tensor addresses needed for stream overlap to be correct.

## Offloading

Requests that generate very long outputs can fill the KV cache during long sessions. The manager evicts one active request instead of crashing. It selects the oldest active request, or the newest if a previous eviction already blocked new requests from joining.

When `cpu_offload_space` is greater than `0.0`, the manager first tries to copy the evicted request's KV cache blocks to a pre-allocated pinned CPU buffer. The request moves back to the waiting queue. After GPU cache space becomes available, the manager copies the blocks back to the GPU and resumes the request without recomputing its prompt and generated tokens.

If CPU offloading is disabled or the CPU swap pool is full, the manager falls back to a soft reset. A soft reset appends the generated tokens to the original prompt, frees the request's cache blocks, and adds the request back to the queue. After the cache has space, the request resumes with its generation history encoded in the prompt.

## Next steps

- The [Continuous batching blog post](https://huggingface.co/blog/continuous_batching) covers KV caching, chunked prefill, and dynamic scheduling with performance benchmark numbers.
- For usage examples, see the [Continuous batching](./continuous_batching) doc.
