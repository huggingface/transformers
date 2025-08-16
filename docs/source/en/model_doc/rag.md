<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-05-22 and added to Hugging Face Transformers on 2020-11-16.*

# RAG

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
  </div>
</div>

[Retrieval-Augmented Generation (RAG)](https://huggingface.co/papers/2005.11401) mixes a **parametric generator** (seq2seq like BART/T5) with a **non-parametric memory** (a dense index queried by a neural retriever). At inference time, RAG fetches relevant passages and conditions its generation on them, effectively marginalizing over multiple documents. This often makes the answers **more factual** and lets you **update knowledge** by changing the index instead of retraining the whole model.

You can find official checkpoints under the RAG collection, for example: `facebook/rag-sequence-nq` and `facebook/rag-token-nq`.

> [!TIP]
> This model was contributed by [ola13](https://huggingface.co/ola13).
>
> Click on the RAG models in the right sidebar for more examples of how to apply RAG to different tasks.

The examples below show how to use RAG with a `pipeline` (when available) and with the explicit RAG classes.

<hfoptions id="usage">

<hfoption id="Pipeline">

```py
from transformers import pipeline

qa = pipeline("text2text-generation", model="facebook/rag-sequence-nq")
out = qa("Who wrote The Old Man and the Sea?")
print(out[0]["generated_text"])
```

</hfoption>

<hfoption id="AutoModel">

```py
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="exact",
    use_dummy_dataset=True,
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever,
)

inputs = tokenizer("Who discovered penicillin?", return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=32)

print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
```

</hfoption>

<hfoption id="transformers-cli">
<!-- No transformers-cli example for RAG, closing this block per the template. -->
</hfoption>

</hfoptions>

Quantization reduces memory by storing weights in lower precision. See the [Quantization](../quantization/overview) overview for supported backends.
**Note:** quantization here applies to the **generator weights**. The **retriever and vector index** keep their own formats/backends.

```py
import torch
from transformers import BitsAndBytesConfig, RagTokenizer, RagRetriever, RagSequenceForGeneration

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever,
    quantization_config=bnb,   # quantizes generator weights to 4bit
    device_map="auto",
)

inputs = tokenizer("When was CRISPR discovered?", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0], skip_special_tokens=True))
```

<!-- AttentionMaskVisualizer is not added for RAG -->

<div class="flex justify-center">
  <img src=""/>
</div>

## Notes

- **RAG-Sequence vs RAG-Token:** Sequence conditions on the same retrieved passages for the whole answer. Token can vary passages per token (more flexible, slightly slower).
- **Updating knowledge:** Rebuild or swap the index to refresh facts. No need to retrain the generator.
- **Latency:** Retrieval adds overhead. Tune top-k and batch retrieval for speed/quality trade-offs.

## Resources
- Paper: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
- Checkpoints: `facebook/rag-sequence-nq`, `facebook/rag-token-nq`
- Dense retriever used in RAG: DPR

## RagConfig

[[autodoc]] RagConfig

## RagTokenizer

[[autodoc]] RagTokenizer

## Rag specific outputs

[[autodoc]] models.rag.modeling_rag.RetrievAugLMMarginOutput

[[autodoc]] models.rag.modeling_rag.RetrievAugLMOutput

## RagRetriever

[[autodoc]] RagRetriever

<frameworkcontent>
<pt>

## RagModel

[[autodoc]] RagModel
    - forward

## RagSequenceForGeneration

[[autodoc]] RagSequenceForGeneration
    - forward
    - generate

## RagTokenForGeneration

[[autodoc]] RagTokenForGeneration
    - forward
    - generate

</pt>
<tf>

## TFRagModel

[[autodoc]] TFRagModel
    - call

## TFRagSequenceForGeneration

[[autodoc]] TFRagSequenceForGeneration
    - call
    - generate

## TFRagTokenForGeneration

[[autodoc]] TFRagTokenForGeneration
    - call
    - generate

</tf>
</frameworkcontent>
