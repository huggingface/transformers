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


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2024-07-29 and contributed to Hugging Face Transformers on 2026-09-01.*

# GTE

## Overview

GTE was proposed in [mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval](https://huggingface.co/papers/2407.19669) by Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li and Min Zhang.

The abstract from the paper is the following:

*We present systematic efforts in building long-context multilingual text representation model (TRM) and reranker from scratch for text retrieval. We first introduce a text encoder (base size) enhanced with RoPE and unpadding, pre-trained in a native 8192-token context (longer than 512 of previous multilingual encoders). Then we construct a hybrid TRM and a cross-encoder reranker by contrastive learning. Evaluations show that our text encoder outperforms the same-sized previous state-of-the-art XLM-R. Meanwhile, our TRM and reranker match the performance of large-sized state-of-the-art BGE-M3 models and achieve better results on long-context retrieval benchmarks. Further analysis demonstrate that our proposed models exhibit higher efficiency during both training and inference. We believe their efficiency and effectiveness could benefit various researches and industrial applications.*

GTE is a BERT-style bidirectional encoder that replaces absolute position embeddings with RoPE, uses a gated MLP, and applies layer normalization after each residual connection. The same architecture backs Alibaba's `gte-*-v1.5` and `gte-multilingual-*` checkpoints as well as Snowflake's `snowflake-arctic-embed-m-v2.0`.

This model was contributed by [Harshal Janjani](https://huggingface.co/harshaljanjani).
The original code can be found [here](https://huggingface.co/Alibaba-NLP/new-impl).

> [!TIP]
> Click on the GTE models in the right sidebar for more examples of how to apply GTE to different language tasks.

The example below demonstrates how to extract features (embeddings) with [`Pipeline`] and [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


# TODO: Remove revision
pipeline = pipeline(
    task="feature-extraction",
    model="Alibaba-NLP/gte-multilingual-base",
    revision="refs/pr/31",
    device=0
)
pipeline("Plants create oxygen through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoModel, AutoTokenizer


# TODO: Remove revision
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", revision="refs/pr/31")
model = AutoModel.from_pretrained(
    "Alibaba-NLP/gte-multilingual-base",
    revision="refs/pr/31",
    device_map="auto",
    attn_implementation="sdpa"
)
inputs = tokenizer("Plants create oxygen through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]

print(f"Embeddings shape: {embeddings.shape}")
```

</hfoption>
</hfoptions>

## Notes

- GTE uses RoPE, so for correct positional encoding either use right padding (the default), or use left padding and prepare `position_ids` accordingly.
- `type_vocab_size` differs across checkpoints. `Alibaba-NLP/gte-base-en-v1.5` sets it to `0`, in which case no token type embedding is created and `token_type_ids` are ignored.
- The `gte-*-v1.5` and `gte-multilingual-*` checkpoints apply static NTK scaling on top of RoPE. It is expressed as a `linear` [`~modeling_rope_utils.RopeParameters`] entry whose `rope_theta` is the base scaled by the NTK factor.

## GteConfig

[[autodoc]] GteConfig

## GteModel

[[autodoc]] GteModel
    - forward

## GteForMaskedLM

[[autodoc]] GteForMaskedLM
    - forward

## GteForSequenceClassification

[[autodoc]] GteForSequenceClassification
    - forward


## GteForTokenClassification

[[autodoc]] GteForTokenClassification
    - forward

