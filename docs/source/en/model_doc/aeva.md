<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->


# Aeva

## Overview

The **Aeva** model architecture represents a significant advancement in open-source language models, designed for efficient long-context understanding and high-performance inference. Building upon the robust foundation of GLM-4.5, Aeva introduces several architectural innovations aimed at balancing performance with computational efficiency.

### Hierarchical Grouped Sparse Computation (HGSCMoE)

The **core innovation** of the Aeva architecture is the **Hierarchical Grouped Sparse Computation (HGSCMoE)**. Unlike traditional Mixture of Experts (MoE) models that route tokens uniformly to all experts, Aeva organizes experts into logical groups, each anchored by a shared "anchor expert".

- **Brain-Inspired Routing**: The routing mechanism utilizes a Leaky Integrate-and-Fire model, simulating neuronal membrane potentials. This event-driven routing ensures that only the most relevant experts are activated, reducing unnecessary computation.
- **Anchor Experts**: Anchor experts act as stabilizers within each group, ensuring consistent feature representations and improving the robustness of the sparse computations.
- **Efficiency**: By sharing capacity across groups and utilizing hierarchical selection, Aeva achieves superior load balancing compared to standard Top-K routing, reducing communication overhead and improving training stability.

### Hybrid Attention Mechanisms with ROSA

Aeva employs a sophisticated hybrid attention strategy to handle variable context lengths efficiently:

- **DeepEmbed Sliding Window Suffix Attention (DESWSA)**: Optimized for local context modeling, this mechanism operates on a sliding window to capture immediate dependencies.
- **DeepEmbed Linear Suffix Attention (DELSA)**: For global dependencies, Aeva switches to linear attention, reducing the complexity from quadratic to linear.
- **ROSA (Rapid Online Suffix Automaton)**: To accelerate autoregressive decoding, Aeva integrates ROSA. This component maintains a statistical fallback mechanism that significantly speeds up generation without sacrificing the quality of the attention mechanism.

### Deep Manifold Hyper-Streams

To enhance the richness of hidden representations, Aeva employs **Deep Manifold Hyper-Streams**:

- **Multi-Stream Processing**: The model processes information across multiple parallel streams rather than a single residual stream.
- **Manifold Constraints**: Using Sinkhorn-Knopp projections, the streams are constrained to remain on a manifold, which improves the geometric properties of the latent space.
- **DeepEmbed Scaling**: Token representations are dynamically scaled using DeepEmbedding techniques, further refining the signal before it passes through the transformer layers.

This architecture allows Aeva to maintain high performance on complex reasoning, coding, and long-context tasks while managing memory and compute requirements efficiently.

This model was contributed by [louzongzhi](https://huggingface.co/louzongzhi) .
The original code can be found [here](https://github.com/louzongzhi/Aeva) .
