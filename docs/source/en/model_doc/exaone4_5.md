<!--Copyright 2026 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-04-09 and added to Hugging Face Transformers on 2026-04-20.*

# EXAONE 4.5

## Overview

[EXAONE 4.5](https://github.com/LG-AI-EXAONE/EXAONE-4.5) model is the first open-weight vision language model developed by LG AI Research.
Integrating a dedicated visual encoder into the existing EXAONE 4.0 framework, we expand the model's capability toward multimodality.
EXAONE 4.5 features 33 billion parameters in total, including 1.2 billion parameters from the vision encoder. 
EXAONE 4.5 achieves competitive performance in general benchmark while outperforming SOTA models of similar size in document understanding and Korean contextual reasoning, inheriting powerful language capabilities from our previous language models.

For more details, please refer to the [technical report](http://arxiv.org/abs/2604.08644), [blog](https://www.lgresearch.ai/blog/view?seq=641) and [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.5).

All model weights including quantized version are available at [Huggingface Collections](https://huggingface.co/collections/LGAI-EXAONE/exaone-45).

## Model Details

### Model Configuration of EXAONE 4.5

- Model Type: Causal Language Model + Vision Encoder
- Number of Parameters (Language Model): 31.7B
- Number of Parameters (Vision Encoder): 1.29B
- Hidden Dimension: 5,120
- Intermediate size: 27,392
- Number of Layers: 64 Main layers + 1 MTP layers
  - Hybrid Attention Pattern: 16 x (3 Sliding window attention + 1 Global attention)
  - Reordered Norm: Apply normalization after Attention/MLP, and before residual connection
- Sliding Window Attention
  - Number of Attention Heads: 40 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - Sliding Window Size: 4,096
- Global Attention
  - Number of Attention Heads: 40 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - No Rotary Positional Embedding Used (NoPE)
- Vision Encoder
  - Grouped Query Attention (GQA) with 32 Q-heads and 8 KV-heads
  - 2D RoPE for vision embeddings
- Vocab Size: 153,600
- Context Length: 262,144 tokens
- Knowledge Cutoff: Dec 2024 (2024/12)

## Exaone4_5_Config

[[autodoc]] Exaone4_5_Config

## Exaone4_5_Model

[[autodoc]] Exaone4_5_Model
    - forward

## Exaone4_5_ForConditionalGeneration

[[autodoc]] Exaone4_5_ForConditionalGeneration
    - forward