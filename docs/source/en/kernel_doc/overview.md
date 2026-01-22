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

# Kernels

PyTorch operations are general-purpose. Hardware vendors and the community create specialized implementations that run faster on specific platforms. Installing these optimized kernels is a challenge because it requires matching compiler versions, CUDA toolkits, and platform-specific builds.

| platform | supported devices |
| :--- | :--- |
| NVIDIA GPUs (CUDA) | Modern architectures with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, Blackwell) |
| AMD GPUs (ROCm) | Compatible with ROCm-supported devices |
| Apple Silicon (Metal) | M-series chips (M1, M2, M3, M4 and newer) |
| Intel GPUs (XPU) | Intel Data Center GPU Max Series and compatible devices |

[Kernels](https://huggingface.co/docs/kernels/index) solves this by distributing precompiled binaries through the [Hub](https://huggingface.co/kernels-community). It detects your platform at runtime and loads the right binary automatically.

When `use_kernels=True`, Transformers identifies layers with available optimized kernel implementations. It downloads and [caches](../installation#cache-directory) kernels from the Hub only when needed to reduce startup time. Kernels accelerate compute-intensive operations such as attention, normalization, and fused operations.

Not all operations have kernel implementations. The library falls back to standard PyTorch when no kernel is available.

## Determinism

Some kernels produce slightly different results than PyTorch due to operation reordering or accumulation strategies. These differences are functionally equivalent but affect reproducibility.

For deterministic behavior, try the following.

- Check kernel repository documentation for determinism guarantees. For example, the SDPA kernel in [gpt-oss-metal-kernels](https://huggingface.co/kernels-community/gpt-oss-metal-kernels#4-scaled-dot-product-attention-sdpa) matches the PyTorch implementation 97% of the time.
- Disable specific kernels that affect your use case.
- Set random seeds and PyTorch deterministic flags.

## Resources

- [Loading kernels](./loading_kernels) guide to get started
- [Kernels](https://github.com/huggingface/kernels) GitHub repository
- [Enhance Your Models in 5 Minutes with the Hugging Face Kernel Hub](https://huggingface.co/blog/hello-hf-kernels) blog post
