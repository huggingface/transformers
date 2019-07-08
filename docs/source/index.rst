Pytorch-Transformers
================================================================================================================================================


.. toctree::
    :maxdepth: 2
    :caption: Notes

    installation
    usage
    examples
    notebooks
    tpu
    cli
    torchscript


.. toctree::
    :maxdepth: 2
    :caption: Package Reference

    model_doc/overview
    model_doc/bert
    model_doc/gpt
    model_doc/transformerxl
    model_doc/gpt2
    model_doc/xlm
    model_doc/xlnet


.. image:: https://circleci.com/gh/huggingface/pytorch-pretrained-BERT.svg?style=svg
   :target: https://circleci.com/gh/huggingface/pytorch-pretrained-BERT
   :alt: CircleCI


This repository contains op-for-op PyTorch reimplementations, pre-trained models and fine-tuning examples for:


* `Google's BERT model <https://github.com/google-research/bert>`_\ ,
* `OpenAI's GPT model <https://github.com/openai/finetune-transformer-lm>`_\ ,
* `Google/CMU's Transformer-XL model <https://github.com/kimiyoung/transformer-xl>`_\ , and
* `OpenAI's GPT-2 model <https://blog.openai.com/better-language-models/>`_.

These implementations have been tested on several datasets (see the examples) and should match the performances of the associated TensorFlow implementations (e.g. ~91 F1 on SQuAD for BERT, ~88 F1 on RocStories for OpenAI GPT and ~18.3 perplexity on WikiText 103 for the Transformer-XL). You can find more details in the `Examples <#examples>`_ section below.

Here are some information on these models:

**BERT** was released together with the paper `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
This PyTorch implementation of BERT is provided with `Google's pre-trained models <https://github.com/google-research/bert>`_\ , examples, notebooks and a command-line interface to load any pre-trained TensorFlow checkpoint for BERT is also provided.

**OpenAI GPT** was released together with the paper `Improving Language Understanding by Generative Pre-Training <https://blog.openai.com/language-unsupervised/>`_ by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
This PyTorch implementation of OpenAI GPT is an adaptation of the `PyTorch implementation by HuggingFace <https://github.com/huggingface/pytorch-openai-transformer-lm>`_ and is provided with `OpenAI's pre-trained model <https://github.com/openai/finetune-transformer-lm>`_ and a command-line interface that was used to convert the pre-trained NumPy checkpoint in PyTorch.

**Google/CMU's Transformer-XL** was released together with the paper `Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context <http://arxiv.org/abs/1901.02860>`_ by Zihang Dai\ *, Zhilin Yang*\ , Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
This PyTorch implementation of Transformer-XL is an adaptation of the original `PyTorch implementation <https://github.com/kimiyoung/transformer-xl>`_ which has been slightly modified to match the performances of the TensorFlow implementation and allow to re-use the pretrained weights. A command-line interface is provided to convert TensorFlow checkpoints in PyTorch models.

**OpenAI GPT-2** was released together with the paper `Language Models are Unsupervised Multitask Learners <https://blog.openai.com/better-language-models/>`_ by Alec Radford\ *, Jeffrey Wu*\ , Rewon Child, David Luan, Dario Amodei\ ** and Ilya Sutskever**.
This PyTorch implementation of OpenAI GPT-2 is an adaptation of the `OpenAI's implementation <https://github.com/openai/gpt-2>`_ and is provided with `OpenAI's pre-trained model <https://github.com/openai/gpt-2>`_ and a command-line interface that was used to convert the TensorFlow checkpoint in PyTorch.

Content
-------

.. list-table::
   :header-rows: 1

   * - Section
     - Description
   * - `Installation <#installation>`_
     - How to install the package
   * - `Overview <#overview>`_
     - Overview of the package
   * - `Usage <#usage>`_
     - Quickstart examples
   * - `Doc <#doc>`_
     - Detailed documentation
   * - `Examples <#examples>`_
     - Detailed examples on how to fine-tune Bert
   * - `Notebooks <#notebooks>`_
     - Introduction on the provided Jupyter Notebooks
   * - `TPU <#tpu>`_
     - Notes on TPU support and pretraining scripts
   * - `Command-line interface <#Command-line-interface>`_
     - Convert a TensorFlow checkpoint in a PyTorch dump

Overview
--------

This package comprises the following classes that can be imported in Python and are detailed in the `Doc <#doc>`_ section of this readme:


*
  Eight **Bert** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py>`_ file):


  * `BertModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L639>`_ - raw BERT Transformer model (\ **fully pre-trained**\ ),
  * `BertForMaskedLM <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L793>`_ - BERT Transformer with the pre-trained masked language modeling head on top (\ **fully pre-trained**\ ),
  * `BertForNextSentencePrediction <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L854>`_ - BERT Transformer with the pre-trained next sentence prediction classifier on top  (\ **fully pre-trained**\ ),
  * `BertForPreTraining <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L722>`_ - BERT Transformer with masked language modeling head and next sentence prediction classifier on top (\ **fully pre-trained**\ ),
  * `BertForSequenceClassification <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L916>`_ - BERT Transformer with a sequence classification head on top (BERT Transformer is **pre-trained**\ , the sequence classification head **is only initialized and has to be trained**\ ),
  * `BertForMultipleChoice <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L982>`_ - BERT Transformer with a multiple choice head on top (used for task like Swag) (BERT Transformer is **pre-trained**\ , the multiple choice classification head **is only initialized and has to be trained**\ ),
  * `BertForTokenClassification <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L1051>`_ - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**\ , the token classification head **is only initialized and has to be trained**\ ),
  * `BertForQuestionAnswering <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L1124>`_ - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**\ , the token classification head **is only initialized and has to be trained**\ ).

*
  Three **OpenAI GPT** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_openai.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_openai.py>`_ file):


  * `OpenAIGPTModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_openai.py#L536>`_ - raw OpenAI GPT Transformer model (\ **fully pre-trained**\ ),
  * `OpenAIGPTLMHeadModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_openai.py#L643>`_ - OpenAI GPT Transformer with the tied language modeling head on top (\ **fully pre-trained**\ ),
  * `OpenAIGPTDoubleHeadsModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_openai.py#L722>`_ - OpenAI GPT Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT Transformer is **pre-trained**\ , the multiple choice classification head **is only initialized and has to be trained**\ ),

*
  Two **Transformer-XL** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_transfo_xl.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_transfo_xl.py>`_ file):


  * `TransfoXLModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_transfo_xl.py#L983>`_ - Transformer-XL model which outputs the last hidden state and memory cells (\ **fully pre-trained**\ ),
  * `TransfoXLLMHeadModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_transfo_xl.py#L1260>`_ - Transformer-XL with the tied adaptive softmax head on top for language modeling which outputs the logits/loss and memory cells (\ **fully pre-trained**\ ),

*
  Three **OpenAI GPT-2** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_gpt2.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_gpt2.py>`_ file):


  * `GPT2Model <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_gpt2.py#L479>`_ - raw OpenAI GPT-2 Transformer model (\ **fully pre-trained**\ ),
  * `GPT2LMHeadModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_gpt2.py#L559>`_ - OpenAI GPT-2 Transformer with the tied language modeling head on top (\ **fully pre-trained**\ ),
  * `GPT2DoubleHeadsModel <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_gpt2.py#L624>`_ - OpenAI GPT-2 Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT-2 Transformer is **pre-trained**\ , the multiple choice classification head **is only initialized and has to be trained**\ ),

*
  Tokenizers for **BERT** (using word-piece) (in the `tokenization.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py>`_ file):


  * ``BasicTokenizer`` - basic tokenization (punctuation splitting, lower casing, etc.),
  * ``WordpieceTokenizer`` - WordPiece tokenization,
  * ``BertTokenizer`` - perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization.

*
  Tokenizer for **OpenAI GPT** (using Byte-Pair-Encoding) (in the `tokenization_openai.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization_openai.py>`_ file):


  * ``OpenAIGPTTokenizer`` - perform Byte-Pair-Encoding (BPE) tokenization.

*
  Tokenizer for **Transformer-XL** (word tokens ordered by frequency for adaptive softmax) (in the `tokenization_transfo_xl.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization_transfo_xl.py>`_ file):


  * ``OpenAIGPTTokenizer`` - perform word tokenization and can order words by frequency in a corpus for use in an adaptive softmax.

*
  Tokenizer for **OpenAI GPT-2** (using byte-level Byte-Pair-Encoding) (in the `tokenization_gpt2.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization_gpt2.py>`_ file):


  * ``GPT2Tokenizer`` - perform byte-level Byte-Pair-Encoding (BPE) tokenization.

*
  Optimizer for **BERT** (in the `optimization.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/pytorch_pretrained_bert/optimization.py>`_ file):


  * ``BertAdam`` - Bert version of Adam algorithm with weight decay fix, warmup and linear decay of the learning rate.

*
  Optimizer for **OpenAI GPT** (in the `optimization_openai.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/optimization_openai.py>`_ file):


  * ``OpenAIAdam`` - OpenAI GPT version of Adam algorithm with weight decay fix, warmup and linear decay of the learning rate.

*
  Configuration classes for BERT, OpenAI GPT and Transformer-XL (in the respective `modeling.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py>`_\ , `modeling_openai.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_openai.py>`_\ , `modeling_transfo_xl.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_transfo_xl.py>`_ files):


  * ``BertConfig`` - Configuration class to store the configuration of a ``BertModel`` with utilities to read and write from JSON configuration files.
  * ``OpenAIGPTConfig`` - Configuration class to store the configuration of a ``OpenAIGPTModel`` with utilities to read and write from JSON configuration files.
  * ``GPT2Config`` - Configuration class to store the configuration of a ``GPT2Model`` with utilities to read and write from JSON configuration files.
  * ``TransfoXLConfig`` - Configuration class to store the configuration of a ``TransfoXLModel`` with utilities to read and write from JSON configuration files.

The repository further comprises:


*
  Five examples on how to use **BERT** (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`_\ ):


  * `run_bert_extract_features.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_bert_extract_features.py>`_ - Show how to extract hidden states from an instance of ``BertModel``\ ,
  * `run_bert_classifier.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_bert_classifier.py>`_ - Show how to fine-tune an instance of ``BertForSequenceClassification`` on GLUE's MRPC task,
  * `run_bert_squad.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_bert_squad.py>`_ - Show how to fine-tune an instance of ``BertForQuestionAnswering`` on SQuAD v1.0 and SQuAD v2.0 tasks.
  * `run_swag.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_swag.py>`_ - Show how to fine-tune an instance of ``BertForMultipleChoice`` on Swag task.
  * `simple_lm_finetuning.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/simple_lm_finetuning.py>`_ - Show how to fine-tune an instance of ``BertForPretraining`` on a target text corpus.

*
  One example on how to use **OpenAI GPT** (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`_\ ):


  * `run_openai_gpt.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_openai_gpt.py>`_ - Show how to fine-tune an instance of ``OpenGPTDoubleHeadsModel`` on the RocStories task.

*
  One example on how to use **Transformer-XL** (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`_\ ):


  * `run_transfo_xl.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_transfo_xl.py>`_ - Show how to load and evaluate a pre-trained model of ``TransfoXLLMHeadModel`` on WikiText 103.

*
  One example on how to use **OpenAI GPT-2** in the unconditional and interactive mode (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`_\ ):


  * `run_gpt2.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py>`_ - Show how to use OpenAI GPT-2 an instance of ``GPT2LMHeadModel`` to generate text (same as the original OpenAI GPT-2 examples).

  These examples are detailed in the `Examples <#examples>`_ section of this readme.

*
  Three notebooks that were used to check that the TensorFlow and PyTorch models behave identically (in the `notebooks folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks>`_\ ):


  * `Comparing-TF-and-PT-models.ipynb <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing-TF-and-PT-models.ipynb>`_ - Compare the hidden states predicted by ``BertModel``\ ,
  * `Comparing-TF-and-PT-models-SQuAD.ipynb <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing-TF-and-PT-models-SQuAD.ipynb>`_ - Compare the spans predicted by  ``BertForQuestionAnswering`` instances,
  * `Comparing-TF-and-PT-models-MLM-NSP.ipynb <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing-TF-and-PT-models-MLM-NSP.ipynb>`_ - Compare the predictions of the ``BertForPretraining`` instances.

  These notebooks are detailed in the `Notebooks <#notebooks>`_ section of this readme.


*
  A command-line interface to convert TensorFlow checkpoints (BERT, Transformer-XL) or NumPy checkpoint (OpenAI) in a PyTorch save of the associated PyTorch model:

  This CLI is detailed in the `Command-line interface <#Command-line-interface>`_ section of this readme.
