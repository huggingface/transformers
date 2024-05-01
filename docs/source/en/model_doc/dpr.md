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

# DPR

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=dpr">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-dpr-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/dpr-question_encoder-bert-base-multilingual">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

Dense Passage Retrieval (DPR) is a set of tools and models for state-of-the-art open-domain Q&A research. It was
introduced in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) by
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

<frameworkcontent>
<pt>

## DPRContextEncoder

[[autodoc]] DPRContextEncoder
    - forward

## DPRQuestionEncoder

[[autodoc]] DPRQuestionEncoder
    - forward

## DPRReader

[[autodoc]] DPRReader
    - forward

</pt>
<tf>

## TFDPRContextEncoder

[[autodoc]] TFDPRContextEncoder
    - call

## TFDPRQuestionEncoder

[[autodoc]] TFDPRQuestionEncoder
    - call

## TFDPRReader

[[autodoc]] TFDPRReader
    - call

</tf>
</frameworkcontent>

