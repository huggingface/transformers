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

---
# Brainstorm

## Persona

Model developer that wants to evaluate his model implementation on a dataset or a model "trainer" that wants to run inference for his GRPO policy.
Pre reqs to understand the docs:
- knows what KV Cache is
- familiarity with transformers and infernece

## what we want do include in the doc

- [ ] CB usage examples
- [ ] CB API reference
- [x] light refresher on what is CB + links to blog post

- [ ] installation / setup instructions

- [x] open telemetry support

- [ ] subsection in Transformers > Inference

- [ ] supported & unsupported features

- [ ] performance considerations
  - [ ] note on benchmarks (CI + space)
  - [ ] cuda graphs
  - [ ] compile
  - [ ] attn impl

- [x] explicit intended use cases, the why of CB in transformers

- [x] integration with serving
---


# Continuous Batching

Continuous Batching (CB) is an advanced technique to optimize the inference of transformer models by dynamically grouping multiple requests into batches. This approach maximizes GPU utilization and throughput, specifically for workloads with many variable-length inputs.

We are particularly interested in having Continuous Batching in transformers for the following use cases:
- Evaluation of models on large datasets with variable-length inputs
- Generating outputs for multiple sequences for GRPO policies

CB is what makes inference engines like vLLM or SGLang efficient. That being said, transformers does not aim to be a production-ready inference engine, but a complete framework for model development. For this reason, CB is available in `transformers serve`.

If you are not familiar with some of the core concepts CB is built upon, we invite you to read the associated blog post: [Continuous Batching: Efficient Inference for Large Language Models](https://huggingface.co/blog/continuous-batching). _broken link for now_

## Installation

Nothing to do, it comes built-in with `transformers`! :nice:

## API Reference

## Usage Examples

The main way to use CB in transformers is via the `generate_batch` method.

Unlike `generate`, CB takes already tokenized inputs, known as input IDs. Each sequence of input IDs is represented as a list of integers, in python: `list[int]`. Since 

For a more detailed example, please refer to: [examples/continuous_batching](./path/to/example)

### `generate_batch` example

We have created a `ContinuousMixin` that is inherited by the `GenerationMixin` so that all auto regressive text models support CB.

This adds the `generate_batch` method to all models that inherit from `GenerationMixin`.

You can use it as follows:

```py
import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    attn_implementation="spda_paged",
    device_map="cuda",  # if you need cuda
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")

# prepare a batch of inputs
dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test")
dataset = dataset.select(range(args.samples))
tokenized_datasets = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
simple_batch_inputs = [item["input_ids"] for item in tokenized_datasets]

generation_config = GenerationConfig(
    max_new_tokens=32,
    use_cuda_graph=False,  # Not supported for simple version
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,
    max_batch_tokens=512,  # max number of tokens in a batch, this is just a default value you should tune based on your hardware
)

batch_outputs = model.generate_batch(
    inputs=simple_batch_inputs,
    generation_config=generation_config,
)

for request_id, output in batch_outputs.items():
    generated_text = tokenizer.decode(output.generated_tokens, skip_special_tokens=True)
    print(f"Request {request_id} output: {generated_text}")
```

### `ContinuousBatchingManager` example

If you want more control w.r.t. how you want to schedule requests using CB, you can use the `ContinuousBatchingManager` class directly.

This is what we use in `transformers serve` because requests arrive asynchronously and we can leverage the asynchronous nature of the CB process to make things more efficient.

Under the hood, the `ContinuousBatchingManager` creates a background thread that receives inputs from a python `queue.Queue` which it uses to get requests to batch in each forward pass.

```py
from transformers.generation.continuous_batching import ContinuousBatchingManager

# TODO:
```

## Supported & Unsupported Features

### Supported Features


### Unsupported Features


## Performance Considerations


## Integration with Serving

You can use CB in `transformers serve` by passing the `--continuous-batching` flag when starting the server.

## Monitoring

We have added `opentelemetry` support to Continuous Batching to help you monitor its performance in production. To enable it, you need to install the `opentelemetry` extra when installing `transformers`:

```sh
# this installs `opentelemetry-api`, `opentelemetry-sdk` and `opentelemetry-exporter-otlp`
pip install transformers[open-telemetry]
```

This will enable traces and metrics collection in CB. You will then have to setup the backend to collect and visualize the traces and metrics.


