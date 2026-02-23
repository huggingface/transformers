<!--Copyright 2020 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️  Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

*This model was released on 2024-09-16 and added to Hugging Face Transformers on 2026-02-18.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>


# JinaEmbeddingsV3

The [Jina-Embeddings-v3](https://huggingface.co/papers/2409.10173) is a multilingual, multi-task text embedding model designed for a variety of NLP applications. Based on the XLM-RoBERTa architecture this model supports **Rotary Position Embeddings (RoPE)** replacing absolute position embeddings to support long input sequences up to 8192 tokens. Additionally, it features 5 built-in **Task-Specific LoRA Adapters:** that allow the model to generate task-specific embeddings (e.g., for retrieval vs. classification) without increasing inference latency significantly.


You can find the original Jina Embeddings v3 checkpoints under the [Jina AI](https://huggingface.co/jinaai) organization.


> [!TIP]
> Click on the Jina Embeddings v3 models in the right sidebar for more examples of how to apply the model to different language tasks.

The example below demonstrates how to predict the `[MASK]` token with [`Pipeline`], [`AutoModel`], and from the command line.


<hfoptions id="usage">

<hfoption id="Pipeline">


```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="jinaai/jina-embeddings-v3",
    dtype=torch.float16,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
pipeline("Plants create energy through a process known as [MASK].", top_k=5)
```


</hfoption>
<hfoption id="AutoModel">


```py

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
model = AutoModelForMaskedLM.from_pretrained(
    "jinaai/jina-embeddings-v3",
    dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto"
)

prompt = "Plants create energy through a process known as [MASK]."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    predictions = outputs.logits[0, mask_token_index]

top_k = torch.topk(predictions, k=5).indices.tolist()
for token_id in top_k[0]:
    print(f"Prediction: {tokenizer.decode([token_id])}")
```






## JinaEmbeddingsV3Config 

[[autodoc]] JinaEmbeddingsV3Config

## JinaEmbeddingsV3Model

[[autodoc]] JinaEmbeddingsV3Model
    - forward


## JinaEmbeddingsV3ForMaskedLM 

[[autodoc]] JinaEmbeddingsV3ForMaskedLM
    - forward

## JinaEmbeddingsV3ForSequenceClassification

[[autodoc]] JinaEmbeddingsV3ForSequenceClassification
    - forward

## JinaEmbeddingsV3ForTokenClassification

[[autodoc]] JinaEmbeddingsV3ForTokenClassification
    - forward

## JinaEmbeddingsV3ForQuestionAnswering

[[autodoc]] JinaEmbeddingsV3ForQuestionAnswering
    - forward

