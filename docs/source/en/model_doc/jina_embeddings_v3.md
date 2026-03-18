<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️  Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

*This model was released on 2024-09-16 and added to Hugging Face Transformers on 2026-03-18.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>


# JinaEmbeddingsV3

The [Jina-Embeddings-v3](https://huggingface.co/papers/2409.10173) is a multilingual, multi-task text embedding model designed for a variety of NLP applications. Based on the XLM-RoBERTa architecture, this model supports **Rotary Position Embeddings (RoPE)** replacing absolute position embeddings to support long input sequences up to 8192 tokens. Additionally, it features 5 built-in **Task-Specific LoRA Adapters:** that allow the model to generate task-specific embeddings (e.g., for retrieval vs. classification) without increasing inference latency significantly.


You can find the original Jina Embeddings v3 checkpoints under the [Jina AI](https://huggingface.co/jinaai) organization.


> [!TIP]
> Click on the Jina Embeddings v3 models in the right sidebar for more examples of how to apply the model to different language tasks.

The example below demonstrates how to extract features (embeddings) with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="feature-extraction",
    model="jinaai/jina-embeddings-v3-hf",
)
# Returns a list of lists containing the embeddings for each token
embeddings = pipeline("Jina Embeddings V3 is great for semantic search.")
```


</hfoption>
<hfoption id="AutoModel">


```py
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3-hf")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3-hf", device_map="auto")

prompt = "Jina Embeddings V3 is great for semantic search."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    # The base AutoModel returns the raw hidden states for all tokens
    last_hidden_states = outputs.last_hidden_state

print(f"Features shape: {last_hidden_states.shape}")
```

</hfoption>
</hfoptions>

## Task-Specific LoRA Adapters

A key feature of `JinaEmbeddingsV3` is it's LoRA adapters, which allow you to tailor the output embeddings to specific useful use cases without the overhead of loading entirely different models.

The following tasks are supported:

* **`retrieval.query`**: Used for query embeddings in asymmetric retrieval tasks (e.g., search queries).
* **`retrieval.passage`**: Used for passage embeddings in asymmetric retrieval tasks (e.g., the documents being searched).
* **`separation`**: Used for embeddings in clustering and re-ranking applications.
* **`classification`**: Used for embeddings in classification tasks.
* **`text-matching`**: Used for embeddings in tasks that quantify similarity between two texts, such as Semantic Textual Similarity (STS) or symmetric retrieval tasks.


To generate high-quality sentence or paragraph embeddings, you need to apply **mean pooling** to the model's token embeddings. Mean pooling takes all token embeddings from the model's output and averages them, masking out the padding tokens.

Here is how you can generate sentence embeddings tailored for a retrieval query task using the `AutoModel` API.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum the embeddings and divide by the number of non-padding tokens
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


sentences = [
    "How is the weather today?", 
    "What is the current weather like today?"
]

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3-hf")
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3-hf")

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)

# Set up the adapter mask for your specific task
task = 'retrieval_query'  # Can be any of (retrieval_passage, separation, classification, text_matching) depending on the use-case.

model.load_adapter("jinaai/jina-embeddings-v3-hf", adapter_name=task, adapter_kwargs={"subfolder": task})

model.set_adapter(task)

with torch.no_grad():
    model_output = model(**encoded_input)

embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
embeddings = F.normalize(embeddings, p=2, dim=1)

print(embeddings.shape)
# Output: torch.Size([2, 1024])
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
