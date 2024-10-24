<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ColPali

## Overview

The ColPali model was proposed in [ColPali: Efficient Document Retrieval with Vision Language Models](https://doi.org/10.48550/arXiv.2407.01449) by **Manuel Faysse***, **Hugues Sibille***, **Tony Wu***, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (* denotes equal contribution).

With our new model *ColPali*, we propose to leverage VLMs to construct efficient multi-vector embeddings in the visual space for document retrieval. By feeding the ViT output patches from PaliGemma-3B to a linear projection, we create a multi-vector representation of documents. We train the model to maximize the similarity between these document embeddings and the query embeddings, following the ColBERT method.

Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, ...) of a document. ColPali is also highly interpretable: similarity maps can be obtained between patches and query tokens. These maps highlight ColPali’s strong OCR capabilities and chart understanding.

The abstract from the paper is the following:

> Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents -often through lengthy and brittle processes-, they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG).
To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark *ViDoRe*, composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings.
The inherent complexity and performance shortcomings of modern systems motivate a new concept; doing document retrieval by directly embedding the images of the document pages. We release *ColPali*, a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, *ColPali* largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable.
We release models, data, code and benchmarks under open licenses at [https://huggingface.co/vidore](https://huggingface.co/vidore).

## Resources

- A blog post detailing ColPali, a vision retrieval model, can be found [here](https://huggingface.co/blog/manu/colpali). 📝
- The code for training ColPali and for the `colpali-engine` package can be found [here](https://github.com/illuin-tech/colpali). 🌎
- Cookbooks to fine-tune ColPali (with optional quantization) and generate similarity maps can be found [here](https://github.com/tonywu71/colpali-cookbooks). 📚

This model was contributed by [tonywu71](https://huggingface.co/tonywu71) and [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/illuin-tech/colpali). To be more precise, the Hf version of Colpali was adapter from [`colpali-engine==0.3.2`](https://github.com/illuin-tech/colpali/releases/tag/v0.3.2).

## ColPaliConfig

[[autodoc]] ColPaliConfig

## ColPaliProcessor

[[autodoc]] ColPaliProcessor

## ColPaliForRetrieval

[[autodoc]] ColPaliForRetrieval
    - forward
