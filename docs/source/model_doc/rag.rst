.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

RAG
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieval-augmented generation ("RAG") models combine the powers of pretrained dense retrieval (DPR) and
sequence-to-sequence models. RAG models retrieve documents, pass them to a seq2seq model, then marginalize to generate
outputs. The retriever and seq2seq modules are initialized from pretrained models, and fine-tuned jointly, allowing
both retrieval and generation to adapt to downstream tasks.

It is based on the paper `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
<https://arxiv.org/abs/2005.11401>`__ by Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.

The abstract from the paper is the following:

*Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve
state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely
manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind
task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge
remain open research problems. Pre-trained models with a differentiable access mechanism to explicit nonparametric
memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a
general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained
parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a
pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a
pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages
across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our
models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks,
outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation
tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art
parametric-only seq2seq baseline.*



RagConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RagConfig
    :members:


RagTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RagTokenizer
    :members: prepare_seq2seq_batch


Rag specific outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput
    :members:

.. autoclass:: transformers.models.rag.modeling_rag.RetrievAugLMOutput
    :members:

RagRetriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RagRetriever
    :members:


RagModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RagModel
    :members: forward


RagSequenceForGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RagSequenceForGeneration
    :members: forward, generate


RagTokenForGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RagTokenForGeneration
    :members: forward, generate
