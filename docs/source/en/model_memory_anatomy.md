<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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

# GPU memory usage

Training a 4B parameter model in mixed precision on a batch size of 16 requires roughly 85GB of GPU memory. Understanding what occupies GPU memory and the compute operations that occur during training help identify where you can reduce your memory usage.

```text
┌─────────────────────────── TENSORS ──────────────────────────────────┐
│                                                                      │
│  MODEL WEIGHTS       ████████████████████████  6 bytes/param         │
│  (mixed precision)   ├── fp16 copy (2B) — forward/backward pass      │
│                      └── fp32 copy (4B) — stable weight updates      │
│                                                                      │
│  OPTIMIZER STATES    ████████████████████████████████  8 bytes/param │
│  (Adam)              ├── fp32 momentum  (4B)                         │
│                      └── fp32 variance  (4B)                         │
│                      ↳ quantized Adam (bitsandbytes) → 2 bytes/param │
│                                                                      │
│  GRADIENTS           ████████████████  4 bytes/param  (fp32)         │
│                      └── computed in backward pass, per parameter    │
│                                                                      │
│  ACTIVATIONS         ████ varies — batch × seq_len × depth × hidden  │
│  (forward cache)     └── cached for backward; can OOM even if model  │
│                          fits (long seqs / large batches)            │
│                          bf16/fp16 if using mixed precision          │
│                                                                      │
│  TEMPORARY TENSORS   ▓ short-lived (softmax, matmul scratch)         │
│                      └── peak spikes can cause OOM                   │
│                                                                      │
│  OTHER OVERHEAD      ▒ beam search caches, large embedding tables    │
└──────────────────────────────────────────────────────────────────────┘
```

GPU memory holds two categories of things: stored tensors and the ops that process them.

## Tensors

Training requires and produces many tensor types that need to be stored.

- Model weights are stored on the GPU. In mixed precision training, two copies of the weights are required - one in fp16 for the forward/backward pass and one in fp32 as the "main copy" for stable weight updates. This equates to 6 bytes per parameter.

- Optimizer states such as Adam store two extra tensors per parameter, the momentum and variance, which are both in fp32. That's an additional 8 bytes per parameter.

    You could use a different optimizer like a quantized version of Adam from [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) to compress it to 2 bytes per parameter.

- Gradient tensors are computed for each parameter in the backward pass. This is kept in fp32 so that's 4 bytes per parameter.

- Forward activations are computed in the forward pass and cached for the backward pass to compute the gradients. These activations vary in size with batch size, sequence length, model depth, and hidden size. This is why batch size or sequence length can exhaust GPU memory even if the model itself fits.

- Temporary tensors are created by ops like softmax and matrix multiplications and released after each op. If the peak of a single op is very intensive, it can create a temporary spike that causes you to run out of memory.

- Some features add their own overhead. For example, beam search maintains multiple outputs, and embedding tables for large vocabularies can be very large.

## Ops

There are three main types of training ops.

- Matrix multiplications (matmuls) are the main op type, covering linear layers, QKV projections, attention output projections, and FFN layers. The main memory hogs are the attention score matrices which grow with the square of the sequence length. This is why long sequences are so expensive and different [attention backends](./attention_interface) exist to avoid materializing the full matrix in memory.

- Reduction ops, like softmax and layer norm, read a full tensor, compute a statistic across a dimension, and then read the tensor again to apply it. This requires accessing memory multiple times per operation.

- Element-wise ops, like activations and dropout, apply a function to each element independently and their memory usage is proportional to tensor size.

## Next steps

- See the [Gradient accumulation](./grad_accumulation) guide to learn how to simulate training on a larger effective batch size without running out of GPU memory.
- See the [Gradient checkpointing](./grad_checkpointing) guide to learn how to reduce memory usage by only storing some intermediate activations.
- See the [Mixed precision training](./mixed_precision_training) guide to learn how to use lower precision data types to reduce memory and speed up training.
- See the [Kernels](./kernels) guide to learn how to speed up training with custom fused kernels.
