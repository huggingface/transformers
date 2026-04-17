<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Continuous batching

Continuous batching maximizes GPU utilization by dynamically rescheduling the batch at every generation step. As requests finish, new ones join immediately instead of waiting for the whole batch to complete. The GPU stays full and throughput stays high.

> [!TIP]
> For production deployments, use [transformers serve](./serve-cli/serving_optims#continuous-batching). It builds on [`ContinuousBatchingManager`] and exposes an OpenAI-compatible HTTP endpoint.

## generate_batch

Continuous batching is supported through [`~ContinuousMixin.generate_batch`]. Pass a list of tokenized prompts and get back results for all of them when they're done. `generate_batch` handles scheduling internally and blocks until all requests are complete.

For serving and streaming use cases, use [ContinuousBatchingManager](#continuousbatchingmanager) directly to manage requests.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import ContinuousBatchingConfig, GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    attn_implementation="flash_attention_2",
    device_map="cuda",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

prompts = [
    "Whats up?",
    "Name a cat breed.",
    "Write a detailed history of quantum mechanics.",
]
inputs = [tokenizer.encode(p) for p in prompts]

generation_config = GenerationConfig(
    max_new_tokens=64,
    eos_token_id=tokenizer.eos_token_id,
)

outputs = model.generate_batch(inputs=inputs, generation_config=generation_config)

for request_id, output in outputs.items():
    text = tokenizer.decode(output.generated_tokens, skip_special_tokens=True)
    print(f"[{request_id}] {text}")
```


## ContinuousBatchingManager

[`ContinuousBatchingManager`] runs a background thread and lets you submit requests and retrieve results independently. Every generation step, it checks for finished requests and schedules new ones to join the batch. This is useful for streaming, real-time serving, or submitting requests as they arrive.

Use [`~ContinuousMixin.continuous_batching_context_manager`] to start and stop the manager safely. The example below contains variable length inputs. As soon as the shortest prompt is complete, it leaves the batch while the longer prompts continue generating. With static batching, you'd have to pad them all to the same length. Continuous batching frees up the completed prompt so you can start processing the next prompt immediately.

```py
with model.continuous_batching_context_manager(generation_config=generation_config) as manager:
    manager.add_request(
        input_ids=tokenizer.encode("Write a detailed history of quantum mechanics."),
        request_id="long",
        max_new_tokens=512,
    )
    manager.add_request(
        input_ids=tokenizer.encode("What's up?"),
        request_id="short_0",
        max_new_tokens=32,
    )
    manager.add_request(
        input_ids=tokenizer.encode("Name a cat breed."),
        request_id="short_1",
        max_new_tokens=32,
    )

    for result in manager:
        text = tokenizer.decode(result.generated_tokens, skip_special_tokens=True)
        print(f"[{result.request_id}] {text}")
```

You could also call [`~ContinuousMixin.init_continuous_batching`] to manage the lifecycle yourself.

```py
manager = model.init_continuous_batching(generation_config=generation_config)
manager.start()

# submit and retrieve requests...
```

Call [`ContinuousBatchingManager.stop`] to terminate the manager.

```py
manager.stop()
```

### Adding requests

[`~ContinuousBatchingManager.add_request`] submits a single request. Provide a `request_id` or let the manager generate one automatically.

```py
manager.add_request(input_ids=input_ids, request_id="my_request")
```

[`~ContinuousBatchingManager.add_requests`] submits a batch at once. It sorts inputs automatically to maximize prefix cache hits when block sharing is enabled.

```py
manager.add_requests(inputs=inputs)
```

Cancel a request with [`~ContinuousBatchingManager.cancel_request`].

```py
manager.cancel_request(request_id="my_request")
```

### Retrieving results

Iterate over the manager to receive results as they arrive.

```py
for result in manager:
    print(tokenizer.decode(result.generated_tokens, skip_special_tokens=True))
```

[`~ContinuousBatchingManager.get_result`] fetches the next result from the output queue. Pass a `request_id` to filter for a specific request. If the next result in the queue doesn't match, it's requeued and the method returns `None`.

```py
# next available result
result = manager.get_result()

# filter for a specific request
result = manager.get_result(request_id="my_request")
```

### Streaming

Set `streaming=True` on a request, then use [`~ContinuousBatchingManager.request_id_iter`] to iterate over partial outputs as tokens are generated.

```py
from transformers.generation.continuous_batching import RequestStatus

manager.add_request(input_ids=input_ids, request_id="streamed", streaming=True)

for chunk in manager.request_id_iter(request_id="streamed"):
    token = tokenizer.decode(chunk.generated_tokens[-1:], skip_special_tokens=True)
    print(token, end="", flush=True)
    if chunk.status == RequestStatus.FINISHED:
        break
```

## ContinuousBatchingConfig

[`ContinuousBatchingConfig`] controls the KV cache, scheduling, CUDA graphs, memory usage, and more. Pass it alongside [`GenerationConfig`] to customize continuous batching.

By default, `num_blocks` and `max_batch_tokens` are inferred automatically from available GPU memory. Use the table below to help you pick the appropriate features.

| Feature | Memory | Throughput | Latency |
|---|---|---|---|
| `max_memory_percent` / `block_size` | ✓ controls KV budget | | |
| `scheduler` | | ✓ scheduling policy | ✓ TTFT |
| CUDA graphs | ↑ graph storage | ✓ less dispatch overhead | ✓ |
| Async batching | ↑ ~2× I/O buffers | ✓ overlaps CPU/GPU | |
| Prefix caching | ↓ shared KV blocks | ✓ skips redundant prefill | ✓ TTFT |
| Paged attention | ↓ no fragmentation | ✓ dynamic batch membership | |
| Sliding window | ↓ bounded KV per layer | | |

```py
from transformers.generation import ContinuousBatchingConfig

cb_config = ContinuousBatchingConfig(
    max_memory_percent=0.8,  # fraction of free GPU memory to use for the KV cache
    block_size=256,          # KV cache block size in tokens
    scheduler_type="fifo",        # "fifo" or "prefill_first"
)

outputs = model.generate_batch(
    inputs=inputs,
    generation_config=generation_config,
    continuous_batching_config=cb_config,
)
```

### Log probabilities

[`ContinuousBatchingConfig`] returns each generated token's log probability when `return_logprobs=True`. This is useful for RL where logprobs are an input to some of the training loops.

```py
cb_config = ContinuousBatchingConfig(return_logprobs=True)

# generate_batch()

for request_id, output in outputs.items():
    for token_id, log_prob in zip(output.generated_tokens, output.logprobs):
        token = tokenizer.decode([token_id])
        print(f"{token} | logprob: {log_prob}")
```

### CUDA graphs

CUDA graphs eliminate CPU dispatch overhead by recording the GPU execution graph once and replaying it for batches with matching shapes. Enable them explicitly with `use_cuda_graph=True`.

```py
cb_config = ContinuousBatchingConfig(use_cuda_graph=True)
```

When active, the manager pads query and KV lengths to fixed intervals so shapes repeat and graphs reuse. Smaller values of `q_padding_interval_size` and `kv_padding_interval_size` reduce wasted compute on padding, but this means there are more unique shapes the graph has to record and store which costs more memory.

```py
cb_config = ContinuousBatchingConfig(
    use_cuda_graph=True,
    q_padding_interval_size=64,
    kv_padding_interval_size=16384,
    max_cached_graphs=32,
)
```

### Async batching

Async batching overlaps CPU scheduling of the next batch with GPU computation of the current one. It requires CUDA graphs and roughly doubles the VRAM used for input tensors.

```py
cb_config = ContinuousBatchingConfig(
    use_cuda_graph=True,
    use_async_batching=True,
)
```

### Prefix caching

When multiple requests share a common prefix, like a system prompt, the manager reuses their KV cache blocks instead of recomputing them. This is enabled by default and requires all model layers to use full attention (it's automatically disabled for sliding window models).

```py
cb_config = ContinuousBatchingConfig(
    allow_block_sharing=True,  # default
)
```

## Paged attention

Continuous batching requires a paged attention backend. Set `attn_implementation` when loading the model. If you load a model with a non-paged backend (`"flash_attention_2"`), the `"paged|"` prefix is added automatically when continuous batching starts.

| Backend | `attn_implementation` | Requirements |
|---|---|---|
| FlashAttention | <code>"paged&#124;flash_attention_2"</code> | `flash-attn` package |
| SDPA (PyTorch native) | <code>"paged&#124;sdpa"</code> | None |
| Eager | <code>"paged&#124;eager"</code> | None |

```py
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    attn_implementation="paged|flash_attention_2",
    device_map="cuda",
    dtype=torch.bfloat16,
)
```

## Sliding window attention

Models with sliding window attention (Mistral, Gemma 2) work with continuous batching. To manually configure a sliding window for fine-tuning or custom experiments, set it in the model config before loading.

```py
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("google/gemma-2-2b")
config.sliding_window = 4096

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    config=config,
    attn_implementation="paged|sdpa",
    device_map="cuda",
    dtype=torch.bfloat16,
)
```

Prefix caching is disabled automatically when sliding window attention is active.

## Next steps

- The [Continuous batching blog post](https://huggingface.co/blog/continuous_batching) covers KV caching, chunked prefill, and dynamic scheduling with performance benchmark numbers.
- For a deeper look at how the continuous batching system works, see the [Continuous batching architecture](./continuous_batching_architecture) doc.
