.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

OpenAI GPT
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenAI GPT model was proposed in `Improving Language Understanding by Generative Pre-Training
<https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf>`__
by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. It's a causal (unidirectional) transformer
pre-trained using language modeling on a large corpus will long range dependencies, the Toronto Book Corpus.

The abstract from the paper is the following:

*Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering,
semantic similarity assessment, and document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to
perform adequately. We demonstrate that large gains on these tasks can be realized by generative pretraining of a
language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. In
contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve
effective transfer while requiring minimal changes to the model architecture. We demonstrate the effectiveness of our
approach on a wide range of benchmarks for natural language understanding. Our general task-agnostic model outperforms
discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon
the state of the art in 9 out of the 12 tasks studied.*

Tips:

- GPT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- GPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as it can be
  observed in the `run_generation.py` example script.

`Write With Transformer <https://transformer.huggingface.co/doc/gpt>`__ is a webapp created and hosted by Hugging Face
showcasing the generative capabilities of several models. GPT is one of them.

This model was contributed by `thomwolf <https://huggingface.co/thomwolf>`__. The original code can be found `here
<https://github.com/openai/finetune-transformer-lm>`__.

Note:

If you want to reproduce the original tokenization process of the `OpenAI GPT` paper, you will need to install ``ftfy``
and ``SpaCy``:

.. code-block:: bash

    pip install spacy ftfy==4.4.3
    python -m spacy download en

If you don't install ``ftfy`` and ``SpaCy``, the :class:`~transformers.OpenAIGPTTokenizer` will default to tokenize
using BERT's :obj:`BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most usage, don't worry).

OpenAIGPTConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTConfig
    :members:


OpenAIGPTTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTTokenizer
    :members: save_vocabulary


OpenAIGPTTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTTokenizerFast
    :members:


OpenAI specific outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput
    :members:

.. autoclass:: transformers.models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput
    :members:


OpenAIGPTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTModel
    :members: forward


OpenAIGPTLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTLMHeadModel
    :members: forward


OpenAIGPTDoubleHeadsModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTDoubleHeadsModel
    :members: forward


OpenAIGPTForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTForSequenceClassification
    :members: forward


TFOpenAIGPTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFOpenAIGPTModel
    :members: call


TFOpenAIGPTLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFOpenAIGPTLMHeadModel
    :members: call


TFOpenAIGPTDoubleHeadsModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFOpenAIGPTDoubleHeadsModel
    :members: call

TFOpenAIGPTForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFOpenAIGPTForSequenceClassification
    :members: call
