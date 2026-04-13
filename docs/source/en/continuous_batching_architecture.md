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

## Soft reset

If the KV cache fills completely during a long session, because requests are generating very long outputs, the manager triggers a soft reset rather than crashing. It selects the oldest active request (or the newest, if a previous soft reset already blocked new requests from joining), appends its generated tokens to its original prompt, frees its cache blocks, and adds it back to the queue.

When the cache has space again, the request resumes from where it left off with its generation history encoded in the prompt.

## Next steps

- The [Continuous batching blog post](https://huggingface.co/blog/continuous_batching) covers KV caching, chunked prefill, and dynamic scheduling with performance benchmark numbers.
- For usage examples, see the [Continuous batching](./continuous_batching) doc.
