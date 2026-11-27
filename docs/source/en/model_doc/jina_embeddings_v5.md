<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-27.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# JinaEmbeddingsV5

[Jina Embeddings v5](https://huggingface.co/collections/jinaai/jina-embeddings-v5-text) is a family of multilingual, multi-task text embedding models developed by [Jina AI](https://huggingface.co/jinaai). Under the `jina_embeddings_v5` model type, Jina AI releases two backbone variants:

- **`jina-embeddings-v5-text-small`** (`is_decoder=True`): causal Qwen3-0.6B backbone with QK-norm.
- **`jina-embeddings-v5-text-nano`** (`is_decoder=False`): bidirectional EuroBERT-210M backbone without QK-norm.

Both checkpoints ship task-specific LoRA adapters under `adapters/{retrieval,text-matching,clustering,classification}` and produce normalized embeddings via last-token pooling.

The example below demonstrates how to load a checkpoint, attach a task LoRA adapter, and compute last-token pooled embeddings with [`AutoModel`].

```python
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


model_id = "jinaai/jina-embeddings-v5-text-nano"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, device_map="auto")

task = "retrieval"
model.load_adapter(model_id, adapter_name=task, adapter_kwargs={"subfolder": f"adapters/{task}"})
model.set_adapter(task)

texts = [
    "Query: What is the capital of France?",
    "Document: Paris is the capital of France.",
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    attention_mask = inputs["attention_mask"]
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        pooled = outputs.last_hidden_state[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        pooled = outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0], device=model.device), seq_lens
        ]
    embeddings = F.normalize(pooled, p=2, dim=-1)
```

## JinaEmbeddingsV5Config

[[autodoc]] JinaEmbeddingsV5Config

## JinaEmbeddingsV5Model

[[autodoc]] JinaEmbeddingsV5Model
    - forward
