<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# REALM

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The REALM model was proposed in [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) by Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat and Ming-Wei Chang. It's a
retrieval-augmented language model that firstly retrieves documents from a textual knowledge corpus and then
utilizes retrieved documents to process question answering tasks.

The abstract from the paper is the following:

*Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks
such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network,
requiring ever-larger networks to cover more facts. To capture knowledge in a more modular and interpretable way, we
augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend
over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the
first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language
modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents. We
demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the
challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both
explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous
methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as
interpretability and modularity.*

This model was contributed by [qqaatw](https://huggingface.co/qqaatw). The original code can be found
[here](https://github.com/google-research/language/tree/master/language/realm).

## RealmConfig

[[autodoc]] RealmConfig

## RealmTokenizer

[[autodoc]] RealmTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary
    - batch_encode_candidates

## RealmTokenizerFast

[[autodoc]] RealmTokenizerFast
    - batch_encode_candidates

## RealmRetriever

[[autodoc]] RealmRetriever

## RealmEmbedder

[[autodoc]] RealmEmbedder
    - forward

## RealmScorer

[[autodoc]] RealmScorer
    - forward

## RealmKnowledgeAugEncoder

[[autodoc]] RealmKnowledgeAugEncoder
    - forward

## RealmReader

[[autodoc]] RealmReader
    - forward

## RealmForOpenQA

[[autodoc]] RealmForOpenQA
    - block_embedding_to
    - forward
