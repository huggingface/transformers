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
*This model was released on 2020-04-10 and added to Hugging Face Transformers on 2020-11-16.*

# DPR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Dense Passage Retrieval (DPR) is a set of tools and models for state-of-the-art open-domain Q&A research. It was
introduced in [Dense Passage Retrieval for Open-Domain Question Answering](https://huggingface.co/papers/2004.04906) by
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih.

The abstract from the paper is the following:

*Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional
sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can
be practically implemented using dense representations alone, where embeddings are learned from a small number of
questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets,
our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage
retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA
benchmarks.*

This model was contributed by [lhoestq](https://huggingface.co/lhoestq). The original code can be found [here](https://github.com/facebookresearch/DPR).

## Usage tips

- DPR consists in three models:

    * Question encoder: encode questions as vectors
    * Context encoder: encode contexts as vectors
    * Reader: extract the answer of the questions inside retrieved contexts, along with a relevance score (high if the inferred span actually answers the question).

## DPRConfig

[[autodoc]] DPRConfig

## DPRContextEncoderTokenizer

[[autodoc]] DPRContextEncoderTokenizer

## DPRContextEncoderTokenizerFast

[[autodoc]] DPRContextEncoderTokenizerFast

## DPRQuestionEncoderTokenizer

[[autodoc]] DPRQuestionEncoderTokenizer

## DPRQuestionEncoderTokenizerFast

[[autodoc]] DPRQuestionEncoderTokenizerFast

## DPRReaderTokenizer

[[autodoc]] DPRReaderTokenizer

## DPRReaderTokenizerFast

[[autodoc]] DPRReaderTokenizerFast

## DPR specific outputs

[[autodoc]] models.dpr.modeling_dpr.DPRContextEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRQuestionEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRReaderOutput

## DPRContextEncoder

[[autodoc]] DPRContextEncoder
    - forward

## DPRQuestionEncoder

[[autodoc]] DPRQuestionEncoder
    - forward

## DPRReader

[[autodoc]] DPRReader
    - forward
