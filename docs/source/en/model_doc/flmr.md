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

# FLMR

## Overview

The FLMR model was proposed in [Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering](https://openreview.net/forum?id=IWWWulAX7g) by Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. 

This work introduces Fine-grained Late-interaction Multi-modal Retrieval, FLMR, which achieved great performance on retrieving documents/passages for answering knowledge-intensive visually-grounded questions (such as [OK-VQA](https://okvqa.allenai.org/) and [InfoSeek](https://open-vision-language.github.io/infoseek/)). The model computes late-interaction scores between multi-modal query and context embeddings to achieve finer-grained embedding interaction.

The abstract from the paper is the following:

*Knowledge-based Visual Question Answering (KB-VQA) requires VQA systems to utilize knowledge from external knowledge bases to answer visually-grounded questions. Retrieval-Augmented Visual Question Answering (RA-VQA), a strong framework to tackle KB-VQA, first retrieves related documents with Dense Passage Retrieval (DPR) and then uses them to answer questions. This paper proposes Fine-grained Late-interaction Multi-modal Retrieval (FLMR) which significantly improves knowledge retrieval in RA-VQA. FLMR addresses two major limitations in RA-VQA's retriever: (1) the image representations obtained via image-to-text transforms can be incomplete and inaccurate and (2) similarity scores between queries and documents are computed with one-dimensional embeddings, which can be insensitive to finer-grained similarities. FLMR overcomes these limitations by obtaining image representations that complement those from the image-to-text transform using a vision model aligned with an existing text-based retriever through a simple alignment network. FLMR also encodes images and questions using multi-dimensional embeddings to capture finer-grained similarities between queries and documents. FLMR significantly improves the original RA-VQA retriever's PRRecall@5 by approximately 8%. Finally, we equipped RA-VQA with two state-of-the-art large multi-modal/language models to achieve 
62% VQA score in the OK-VQA dataset.*

Tips:

There are several versions of FLMR available:
- **FLMR**: available as `LinWeizheDragon/FLMR`. The original version of the proposed FLMR model, which uses 9 Regions-of-Interest to improve scene understanding.
- **PreFLMR**: available as `LinWeizheDragon/PreFLMR_ViT-G`/`LinWeizheDragon/PreFLMR_ViT-H`/`LinWeizheDragon/PreFLMR_ViT-L`/`LinWeizheDragon/PreFLMR_ViT-B`. The follow-up work of FLMR. PreFLMR was pretrained on a wide range of knowledge-intensive multi-modal retrieval tasks. PreFLMR was proposed in [PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers](https://arxiv.org/abs/2402.08327) by Weizhe Lin, Jingbiao Mei, Jinghong Chen, and Bill Byrne.

Applications:

This Huggingface implementation is to be used with the [ColBERT](https://github.com/stanford-futuredata/ColBERT) engine. Examples of how to use FLMR to index a corpus and perform retrieval can be found in `examples/research_projects/flmr-retrieval`.
Under this folder, the engine has been cleaned up and wrapped into a third-party package. Note that you should comply with the liscense of ColBERT if you use this provided third-party package.

This model was contributed by [Weizhe Lin](https://huggingface.co/LinWeizheDragon).
The original code can be found [here](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering).


## FLMRConfig

[[autodoc]] FLMRConfig

## FLMRContextEncoderTokenizer

[[autodoc]] FLMRContextEncoderTokenizer

## FLMRContextEncoderTokenizerFast

[[autodoc]] FLMRContextEncoderTokenizerFast

## FLMRQueryEncoderTokenizer

[[autodoc]] FLMRQueryEncoderTokenizer

## FLMRQueryEncoderTokenizerFast

[[autodoc]] FLMRQueryEncoderTokenizerFast

## FLMR specific outputs

[[autodoc]] models.flmr.modeling_flmr.FLMRContextEncoderOutput

[[autodoc]] models.flmr.modeling_flmr.FLMRQueryEncoderOutput

[[autodoc]] models.flmr.modeling_flmr.FLMRModelForRetrievalOutput

<frameworkcontent>
<pt>

## FLMRModelForRetrieval

[[autodoc]] FLMRModelForRetrieval
    - forward
    - query
    - doc

## FLMRTextModel

[[autodoc]] FLMRTextModel
    - forward

## FLMRVisionModel

[[autodoc]] FLMRVisionModel
    - forward

## FLMRPretrainedModelForRetrieval

[[autodoc]] FLMRPretrainedModelForRetrieval

## FLMRTextConfig

[[autodoc]] FLMRTextConfig

## FLMRVisionConfig

[[autodoc]] FLMRVisionConfig

</pt>
<tf>
