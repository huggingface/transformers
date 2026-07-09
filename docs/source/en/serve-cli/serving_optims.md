<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Server optimizations

`transformers serve` includes optimizations to improve throughput and reduce memory usage.

## Continuous batching

[Continuous batching](../continuous_batching) dynamically groups and interleaves requests to share forward passes on the GPU. New requests join the batch as others progress through prefill. Completed requests drop out after decoding. This increases GPU utilization and throughput without compromising latency.

Add `--continuous-batching` to enable it.

```sh
transformers serve \
  --continuous-batching \
  --attn-implementation "sdpa"
```

## Quantization

[Quantization](../quantization/overview) reduces memory usage by mapping weights to a lower precision. `transformers serve` is compatible with all quantization methods in Transformers. It supports pre-quantized models and runtime quantization.

Pre-quantized models don't require any changes. They offer the best balance between performance and accuracy. Install the appropriate quantization library. Then pass the pre-quantized model from the Hub to the `model` argument.

```sh
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-GGUF",
    "stream": true,
    "input": "Tell me a three sentence bedtime story about a unicorn."
  }'
```

Use `--quantization` to quantize a model at runtime. This is useful for new checkpoints or finetunes without pre-quantized weights. Only [bitsandbytes](../quantization/bitsandbytes) 4-bit and 8-bit quantization are supported.

```sh
transformers serve \
  --quantization bnb-4bit
```

## Attention backend

An optimized [attention backend](../attention_interface) improves memory efficiency and speeds up inference.

```sh
transformers serve \
  --continuous-batching \
  --attn-implementation "flash_attention_2"
```

### Apple Silicon (Metal flash attention)

Install [kernels](https://github.com/huggingface/kernels) to make `transformers serve` default to [kernels-community/metal-flash-sdpa](https://huggingface.co/kernels-community/metal-flash-sdpa) on MPS. The Metal flash kernel runs 1.66x faster than SDPA with `generate_batch` on 100 samples of gsm8k, Qwen2.5-0.5B-Instruct and MPS fp16. It matches SDPA token-for-token under greedy decoding.

```sh
pip install kernels
transformers serve
```

A warning prints at startup confirming the auto-selection. Pass `--attn-implementation sdpa` to opt out.

```sh
transformers serve --attn-implementation sdpa
```

## Compile

[torch.compile](../perf_torch_compile) traces and compiles the decode loop for faster inference.

> [!NOTE]
> Compile is incompatible with continuous batching.

```sh
transformers serve \
  --compile
```

## Data type

The `"bfloat16"` or `"float16"` [data types](../models#model-data-type) save memory and increase throughput.

```sh
transformers serve \
  --continuous-batching \
  --dtype "bfloat16"
```
