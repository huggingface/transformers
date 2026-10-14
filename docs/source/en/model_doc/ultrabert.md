<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
*This model was published in HF papers on 2026-10-10 and contributed to Hugging Face Transformers on 2026-10-14.*

# UltraBERT

## Overview

UltraBERT was introduced in [UltraBERT: Beyond the Limit of BERT Pre-Training](https://huggingface.co/papers/2610.15028) by Jungwoo Park, Taewhoo Lee, and Jaewoo Kang. It is a family of bidirectional text encoders trained with a combination of masking ratios, pruning, and distillation for classification and retrieval tasks.

The released encoders support sequences of up to 32,768 tokens. They use rotary position embeddings (RoPE), RMSNorm, SwiGLU feed-forward layers, and alternating local and global attention. Every third layer uses global attention. The other layers attend to the current token and up to 128 tokens on either side. The RoPE base is 10,000 for local layers and 2,560,000 for global layers.

| Checkpoint | Layers | Hidden size | Attention heads | Use |
|---|---|---|---|---|
| [UltraBERT-150M](https://huggingface.co/dmis-lab/UltraBERT-150M) | 24 | 768 | 6 | Masked language modeling and downstream fine-tuning |
| [UltraBERT-380M](https://huggingface.co/dmis-lab/UltraBERT-380M) | 24 | 1,024 | 8 | Masked language modeling and downstream fine-tuning |
| [UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) | 24 | 768 | 6 | Text embeddings |

The masked language modeling checkpoints contain the prediction head trained with the low masking ratio. Its output weights are untied from the input embeddings. The tokenizer uses the SmolLM2 vocabulary of 49,152 tokens.

## Usage examples

### Masked language modeling

Use [`AutoModelForMaskedLM`] to predict a masked token. The mask token is `<|mask|>`.

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "dmis-lab/UltraBERT-150M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint, attn_implementation="sdpa").eval()

inputs = tokenizer("The capital of France is <|mask|>.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

masked_logits = logits[inputs.input_ids == tokenizer.mask_token_id]
print(tokenizer.batch_decode(masked_logits.argmax(dim=-1)[:, None]))
```

### Text embeddings

UltraBERT-Embed-150M uses the first token's hidden state as the text embedding. For retrieval, prepend the task instruction to queries and encode documents without an instruction. Use right padding to preserve the first token's position. See the [model card](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) for instructions for other tasks.

The example below normalizes the embeddings so their dot product gives cosine similarity.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

checkpoint = "dmis-lab/UltraBERT-Embed-150M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="right")
model = AutoModel.from_pretrained(checkpoint, attn_implementation="sdpa").eval()

texts = [
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: What is the capital of France?",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
]
inputs = tokenizer(texts, padding=True, truncation=True, max_length=32768, return_tensors="pt")
with torch.no_grad():
    embeddings = F.normalize(model(**inputs).last_hidden_state[:, 0], p=2, dim=-1)

print(embeddings[:1] @ embeddings[1:].T)
```

## UltrabertConfig

[[autodoc]] UltrabertConfig

## UltrabertModel

[[autodoc]] UltrabertModel
    - forward

## UltrabertForMaskedLM

[[autodoc]] UltrabertForMaskedLM
    - forward

## UltrabertForSequenceClassification

[[autodoc]] UltrabertForSequenceClassification
    - forward

## UltrabertForTokenClassification

[[autodoc]] UltrabertForTokenClassification
    - forward

## UltrabertForQuestionAnswering

[[autodoc]] UltrabertForQuestionAnswering
    - forward
