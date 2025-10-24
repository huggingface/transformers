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

- CB usage examples
- CB API reference
- light refresher on what is CB + links to blog post

- installation / setup instructions

- open telemetry support

- subsection in Transformers > Inference

- supported & unsupported features

- performance considerations
  - note on benchmarks (CI + space)
  - cuda graphs
  - compile
  - attn impl

- explicit intended use cases, the why of CB in transformers

- integration with serving
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

### `ContinuousBatchingManager` example


## Supported & Unsupported Features

### Supported Features


### Unsupported Features

## Performance Considerations


## Integration with Serving


