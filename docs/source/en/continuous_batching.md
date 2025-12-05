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

Continuous batching maximizes GPU utilization. It increases throughput and reduces latency by using dynamic scheduling to rearrange the batch at each step. The system removes completed requests and adds new requests immediately to prevent GPU idling. Chunked prefill prevents expensive prefill work from stalling the batch while still allowing new requests still join.

Continuous batching works with [transformers serve](./serving), a server for deploying local models, and [`~ContinuousMixin.generate_batch`].

## generate_batch

The [`~ContinuousMixin.generate_batch`] method works with all autoregressive text models. It accepts a list of tokenized inputs and a [`GenerationConfig`] to configure generation settings.

```py
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    attn_implementation="sdpa_paged",
    device_map="cuda",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", padding_side="left")

dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
dataset = dataset.select(range(args.samples))
tokenized_datasets = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

generation_config = GenerationConfig(
    max_new_tokens=32,
    use_cuda_graph=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
    max_batch_tokens=512,
)

batch_outputs = model.generate_batch(
    inputs=simple_batch_inputs,
    generation_config=generation_config,
)

for request_id, output in batch_outputs.items():
    generated_text = tokenizer.decode(output.generated_tokens, skip_special_tokens=True)
    print(f"Request {request_id} output: {generated_text}")
```

## ContinuousBatchingManager

The [`ContinuousBatchingManager`] orchestrates the background thread by pulling requests from the queue and filling the GPU to capacity. Every iteration checks for finished requests and schedules new ones to join the batch. Use this manager to customize request scheduling.

Call [`~ContinuousMixin.init_continuous_batching`] to initialize the manager with a [`GenerationConfig`] and [`~ContinuousBatchingManager.start`] the background thread.

```py
from transformers.generation.continuous_batching import RequestStatus

manager = model.init_continuous_batching(generation_config=generation_config)
manager.start()
```

Use [`~ContinuousBatchingManager.add_request`] to asynchronously submit individual requests. Provide a specific request id or the manager wgenerates one automatically.

```py
for i, input_ids in enumerate(simple_batch_inputs):
    request_id = manager.add_request(input_ids=input_ids, request_id=f"request_{i}")
```

Retrieve *all* results as they arrive with [`~ContinuousBatchingManager.get_result`].

```py
for request_id, request in manager.get_result():
    generated_text = tokenizer.decode(request.generated_tokens, skip_special_tokens=True)
    print(f"Request {request_id} output: {generated_text}")
```

Use the `request_id` of a specific request to get its results. This is a blocking operation that waits until the result is ready.

```py
result = manager.get_result(request_id="request_5")
```

Stream partial results for a specific request with [`~ContinuousBatchingManager.request_id_iter`].

```py
manager.add_request(
    input_ids=input_ids,
    request_id="streaming_request",
    stream=True,
)
for chunk in manager.request_id_iter(request_id="streaming_request"):
    generated_text = tokenizer.decode(chunk.generated_tokens, skip_special_tokens=True)
    print(generated_text)
    # FIXME: stop iteration in `request_id_iter` when finished instead of doing it externally
    if chunk.status == RequestStatus.FINISHED:
        break
```

Call [`~ContinuousBatchingManager.stop`] to terminate the manager.

```py
manager.stop()
```

## PagedAttention

PagedAttention breaks large key-value caches into smaller, non-contiguous fixed-size pages to avoid GPU memory fragmentation and support variable-length requests. Transformers automatically enables PagedAttention when using continuous batching.

You could explicitly enable PagedAttention when instantiating a model rather than waiting for [`~ContinuousMixin.generate_batch`] to dynamically enable it.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    attn_implementation="paged|flash_attention_2",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
```

## Sliding window attention

Sliding window attention limits the backward context of a token to save compute. Generation cost stays proportional to window size. This reduces compute per step and simplifies continuous batching.

Transformers models like Mistral and Gemma 2 natively support sliding window attention. Manually enable it in the model config if the architecture supports it. This helps with fine-tuning or running custom experiments.

```py
from transformers import AutoConfig

config = AutoConfig.from_pretrained("google/gemma-2-2b")
config.sliding_window = 4096

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    config=config,
    attn_implementation="paged|flash_attention_2",
    device_map="cuda",
    dtype=torch.bfloat16,
)
```

Usage remains the same with [`~ContinuousMixin.generate_batch`].

## How it works

The [`ContinuousMixin`] class serves as the main interface for continuous batching through [`~ContinuousMixin.generate_batch`]. This method internally creates a [`ContinuousBatchingManager`].

[`ContinuousBatchingManager`] manages requests by creating a background thread for the generation loop and adding requests to the queue. The manager is thread-safe, allowing asynchronous request additions while the model generates.

The [`Scheduler`] selects requests for processing at each step based on the token budget. [`FIFOScheduler`] is the default scheduler. It prioritizes decoding requests over prefilling requests and assigns them to specific memory blocks. [`PrefillFirstScheduler`] prioritizes prefill requests instead.

[`ContinuousBatchingManager`] runs the model forward pass for the scheduled requests. It then collects and returns the results.

## Resources

The [Continuous batching](https://huggingface.co/blog/continuous_batching) blog post explains KV caching, chunked prefill, and ragged batching with dynamic scheduling in more detail.