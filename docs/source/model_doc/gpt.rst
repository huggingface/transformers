OpenAI GPT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

OpenAI GPT model was proposed in `Improving Language Understanding by Generative Pre-Training <https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf>`__
by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. It's a causal (unidirectional)
transformer pre-trained using language modeling on a large corpus will long range dependencies, the Toronto Book Corpus.

The abstract from the paper is the following:

*Natural language understanding comprises a wide range of diverse tasks such
as textual entailment, question answering, semantic similarity assessment, and
document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging for
discriminatively trained models to perform adequately. We demonstrate that large
gains on these tasks can be realized by generative pre-training of a language model
on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each
specific task. In contrast to previous approaches, we make use of task-aware input
transformations during fine-tuning to achieve effective transfer while requiring
minimal changes to the model architecture. We demonstrate the effectiveness of
our approach on a wide range of benchmarks for natural language understanding.
Our general task-agnostic model outperforms discriminatively trained models that
use architectures specifically crafted for each task, significantly improving upon the
state of the art in 9 out of the 12 tasks studied.*

Tips:

- GPT is a model with absolute position embeddings so it's usually advised to pad the inputs on
  the right rather than the left.
- GPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as
  it can be observed in the `run_generation.py` example script.

`Write With Transformer <https://transformer.huggingface.co/doc/gpt>`__ is a webapp created and hosted by
Hugging Face showcasing the generative capabilities of several models. GPT is one of them.

OpenAIGPTConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTConfig
    :members:


OpenAIGPTTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTTokenizer
    :members: save_vocabulary


OpenAIGPTModel
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTModel
    :members:


OpenAIGPTLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTLMHeadModel
    :members:


OpenAIGPTDoubleHeadsModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.OpenAIGPTDoubleHeadsModel
    :members:


TFOpenAIGPTModel
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFOpenAIGPTModel
    :members:


TFOpenAIGPTLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFOpenAIGPTLMHeadModel
    :members:


TFOpenAIGPTDoubleHeadsModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFOpenAIGPTDoubleHeadsModel
    :members:
