<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Visual Document Retrieval

Documents are essentially multimodal data, rich in charts, tables, visuals as well as text. Retrieving information from such documents is challenging as text retrieval models miss out on infographics, and image retrieval models lack granularity and document processing capabilities. Visual document retrieval is a new paradigm to solve information retrieval from documents. This comes in handy in all document retrieval use cases, a major one is multimodal retrieval augmented generation (RAG). These models essentially take in documents (as images) and texts and calculate similarity between each.

This task guide aims to demonstrate how to index and retrieve documents using [ColPali](https://huggingface.co/vidore/colpali-v1.2-hf) with transformers.  

<Tip>

On a larger scale, you might want to index and retrieve your documents using a vector database. 

</Tip>

We will begin by installing transformers and datasets.

```bash
pip install -q datasets transformers
```

We will index a dataset of documents related to UFO sightings. We filter the examples where our column of interest is missing. It contains several columns, we are interested in the column `specific_detail_query` where it contains short summary of the document, and `image` column that contains our documents.

```python
from datasets import load_dataset

dataset = load_dataset("davanstrien/ufo-ColPali")
dataset = dataset["train"]
dataset = dataset.filter(lambda example: example["specific_detail_query"] is not None)
dataset
```
```
Dataset({
    features: ['image', 'raw_queries', 'broad_topical_query', 'broad_topical_explanation', 'specific_detail_query', 'specific_detail_explanation', 'visual_element_query', 'visual_element_explanation', 'parsed_into_json'],
    num_rows: 2172
})
```

Let's load the model and the tokenizer.

```python
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor

model_name = "vidore/colpali-v1.2-hf"

processor = ColPaliProcessor.from_pretrained(model_name)

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
).eval()
```

We can simply infer with the model with text query like below. We can get the indexed embeddings with `embeddings` in our output. For image-to-text search, you can swap text with `images` argument and pass in images instead. 

```python
inputs = processor(text="a document about Mars expedition").to("cuda")
with torch.no_grad():
  text_embeds = model(**inputs, return_tensors="pt").embeddings
```

We will do text-to-image retrieval. To do so, we need to index images offline, and during inference time, we get the text embeddings and simply get the closest image embedding to the query text embedding. Thus, we need to store image embeddings along with their images. A nice hack to index embeddings and write them on dataset is to use `dataset.map()` like below. We add an embeddings column that contains indexed embeddings. Since ColPali embeddings hold a lot of space, we need to remove them from GPU, so we detach them and write them as numpy vectors to store in CPU.

```python
ds_with_embeddings = dataset.map(lambda example: {'embeddings': model(**processor(images=example["image"]).to("cuda"), return_tensors="pt").embeddings.to(torch.float32).detach().cpu().numpy()})
```

For the online inference part, we need a function to search image embeddings in batches and retrieve k-most-relevant images. Below function takes in the indexed dataset and text embedding, number of top results and batch size to get the best results and returns the indices in the dataset and their scores.

```python
def find_top_k_indices_batched(dataset, text_embedding, processor, k=10, batch_size=4):
    scores_and_indices = []

    for start_idx in range(0, len(dataset), batch_size):

        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]        
        batch_embeddings = [torch.tensor(emb[0], dtype=torch.float32) for emb in batch["embeddings"]]
        scores = processor.score_retrieval(text_embedding.to("cpu").to(torch.float32), batch_embeddings)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()[0]

        for i, score in enumerate(scores):
            scores_and_indices.append((score, start_idx + i))

    sorted_results = sorted(scores_and_indices, key=lambda x: -x[0])

    topk = sorted_results[:k]
    indices = [idx for _, idx in topk]
    scores = [score for score, _ in topk]

    return indices, scores
```

Let's infer. We should first get text embeddings and call our helper function above.

```python
with torch.no_grad():
  text_embeds = model(**processor(text="a document about Mars expedition").to("cuda"), return_tensors="pt").embeddings
indices, scores = find_top_k_indices_batched(ds_with_embeddings, text_embeds, processor, k=3, batch_size=4)
print(indices, scores)
```

```
([440, 442, 443],
 [14.370786666870117,
  13.675487518310547,
  12.9899320602417])
```

Now displaying the images we can see the model has returned Mars related documents.

```python
for i in indices:
  display(dataset[i]["image"])
```

<div style="display: flex; align-items: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_1.png" 
         alt="Document 1" 
         style="height: 200px; object-fit: contain; margin-right: 10px;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_2.png" 
         alt="Document 2" 
         style="height: 200px; object-fit: contain;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_3.png" 
         alt="Document 3" 
         style="height: 200px; object-fit: contain;">
</div>