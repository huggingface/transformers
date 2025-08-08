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
# Visual document retrieval

Documents can contain multimodal data if they include charts, tables, and visuals in addition to text. Retrieving information from these documents is challenging because text retrieval models alone can't handle visual data and image retrieval models lack the granularity and document processing capabilities.

Visual document retrieval can help retrieve information from all types of documents, including multimodal retrieval augmented generation (RAG). These models accept documents (as images) and texts and calculates the similarity scores between them.

This guide demonstrates how to index and retrieve documents with [ColPali](../model_doc/colpali).  

> [!TIP]
> For large scale use cases, you may want to index and retrieve documents with a vector database.

Make sure Transformers and Datasets is installed.

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
    dtype=torch.bfloat16,
    device_map="cuda",
).eval()
```

Pass the text query to the processor and return the indexed text embeddings from the model. For image-to-text search, replace the `text` parameter in [`ColPaliProcessor`] with the `images` parameter to pass images.

```python
inputs = processor(text="a document about Mars expedition").to("cuda")
with torch.no_grad():
  text_embeds = model(**inputs, return_tensors="pt").embeddings
```

Index the images offline, and during inference, return the query text embeddings to get its closest image embeddings.

Store the image and image embeddings by writing them to the dataset with [`~datasets.Dataset.map`] as shown below. Add an `embeddings` column that contains the indexed embeddings. ColPali embeddings take up a lot of storage, so remove them from the GPU and store them in the CPU as NumPy vectors.

```python
ds_with_embeddings = dataset.map(lambda example: {'embeddings': model(**processor(images=example["image"]).to("cuda"), return_tensors="pt").embeddings.to(torch.float32).detach().cpu().numpy()})
```

For online inference, create a function to search the image embeddings in batches and retrieve the k-most relevant images. The function below returns the indices in the dataset and their scores for a given indexed dataset, text embeddings, number of top results, and the batch size.

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

Generate the text embeddings and pass them to the function above to return the dataset indices and scores.

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

Display the images to view the Mars related documents.

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