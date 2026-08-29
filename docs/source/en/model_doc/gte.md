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
*This model was published in HF papers on 2024-07-29 and contributed to Hugging Face Transformers on 2026-08-30.*

# GTE

## Overview

GTE was proposed in [mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval](https://huggingface.co/papers/2407.19669) by Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li and Min Zhang.

The abstract from the paper is the following:

*We present systematic efforts in building long-context multilingual text representation model (TRM) and reranker from scratch for text retrieval. We first introduce a text encoder (base size) enhanced with RoPE and unpadding, pre-trained in a native 8192-token context (longer than 512 of previous multilingual encoders). Then we construct a hybrid TRM and a cross-encoder reranker by contrastive learning. Evaluations show that our text encoder outperforms the same-sized previous state-of-the-art XLM-R. Meanwhile, our TRM and reranker match the performance of large-sized state-of-the-art BGE-M3 models and achieve better results on long-context retrieval benchmarks. Further analysis demonstrate that our proposed models exhibit higher efficiency during both training and inference. We believe their efficiency and effectiveness could benefit various researches and industrial applications.*

GTE is a BERT-style bidirectional encoder that replaces absolute position embeddings with RoPE, uses a gated MLP, and applies layer normalization after each residual connection. The same architecture backs Alibaba's `gte-*-v1.5` and `gte-multilingual-*` checkpoints as well as Snowflake's `snowflake-arctic-embed-m-v2.0`.

This model was contributed by [Harshal Janjani](https://huggingface.co/harshaljanjani).
The original code can be found [here](https://huggingface.co/Alibaba-NLP/new-impl).

## Usage examples

Embeddings are taken from the `[CLS]` token and normalized.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "Alibaba-NLP/gte-multilingual-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("what is the capital of China?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=-1)
```

Batched inference scores a query against several documents:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "Alibaba-NLP/gte-multilingual-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, device_map="auto")

texts = ["what is the capital of China?", "Beijing", "sorting algorithms"]
inputs = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=-1)
scores = embeddings[:1] @ embeddings[1:].T
```

The reranker checkpoints are cross-encoders and expose a single relevance logit per pair:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "Alibaba-NLP/gte-multilingual-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, device_map="auto")

pairs = [["what is the capital of China?", "Beijing"], ["what is the capital of China?", "sorting algorithms"]]
inputs = tokenizer(pairs, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(model.device)
with torch.no_grad():
    scores = model(**inputs).logits.view(-1)
```

Fine-tuning uses the standard forward and backward pass:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "Alibaba-NLP/gte-multilingual-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, device_map="auto")

inputs = tokenizer(["a positive review", "a negative review"], padding=True, return_tensors="pt").to(model.device)
labels = torch.tensor([1, 0], device=model.device)

loss = model(**inputs, labels=labels).loss
loss.backward()
```

The model is compatible with [`torch.compile`]:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "Alibaba-NLP/gte-multilingual-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, device_map="auto")
model = torch.compile(model)

inputs = tokenizer("what is the capital of China?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
```

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

## GteForMultipleChoice

[[autodoc]] GteForMultipleChoice
    - forward

## GteForTokenClassification

[[autodoc]] GteForTokenClassification
    - forward

## GteForQuestionAnswering

[[autodoc]] GteForQuestionAnswering
    - forward
