Pytorch-Transformers
================================================================================================================================================


.. toctree::
    :maxdepth: 2
    :caption: Notes

    installation
    philosophy
    usage
    examples
    notebooks
    converting_tensorflow_models
    migration
    bertology
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


* `Google's BERT model <https://github.com/google-research/bert>`__\ ,
* `OpenAI's GPT model <https://github.com/openai/finetune-transformer-lm>`__\ ,
* `Google/CMU's Transformer-XL model <https://github.com/kimiyoung/transformer-xl>`__\ , and
* `OpenAI's GPT-2 model <https://blog.openai.com/better-language-models/>`__.

These implementations have been tested on several datasets (see the examples) and should match the performances of the associated TensorFlow implementations (e.g. ~91 F1 on SQuAD for BERT, ~88 F1 on RocStories for OpenAI GPT and ~18.3 perplexity on WikiText 103 for the Transformer-XL). You can find more details in the `Examples <./examples.html>`__ section.

Here are some information on these models:

**BERT** was released together with the paper `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
This PyTorch implementation of BERT is provided with `Google's pre-trained models <https://github.com/google-research/bert>`__\ , examples, notebooks and a command-line interface to load any pre-trained TensorFlow checkpoint for BERT is also provided.

**OpenAI GPT** was released together with the paper `Improving Language Understanding by Generative Pre-Training <https://blog.openai.com/language-unsupervised/>`__ by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
This PyTorch implementation of OpenAI GPT is an adaptation of the `PyTorch implementation by HuggingFace <https://github.com/huggingface/pytorch-openai-transformer-lm>`__ and is provided with `OpenAI's pre-trained model <https://github.com/openai/finetune-transformer-lm>`__ and a command-line interface that was used to convert the pre-trained NumPy checkpoint in PyTorch.

**Google/CMU's Transformer-XL** was released together with the paper `Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context <http://arxiv.org/abs/1901.02860>`__ by Zihang Dai\*, Zhilin Yang\* , Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
This PyTorch implementation of Transformer-XL is an adaptation of the original `PyTorch implementation <https://github.com/kimiyoung/transformer-xl>`__ which has been slightly modified to match the performances of the TensorFlow implementation and allow to re-use the pretrained weights. A command-line interface is provided to convert TensorFlow checkpoints in PyTorch models.

**OpenAI GPT-2** was released together with the paper `Language Models are Unsupervised Multitask Learners <https://blog.openai.com/better-language-models/>`__ by Alec Radford\*, Jeffrey Wu\* , Rewon Child, David Luan, Dario Amodei\*\* and Ilya Sutskever\*\*.
This PyTorch implementation of OpenAI GPT-2 is an adaptation of the `OpenAI's implementation <https://github.com/openai/gpt-2>`__ and is provided with `OpenAI's pre-trained model <https://github.com/openai/gpt-2>`__ and a command-line interface that was used to convert the TensorFlow checkpoint in PyTorch.

**Facebook Research's XLM** was released together with the paper `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`__ by Guillaume Lample and Alexis Conneau.
This PyTorch implementation of XLM is an adaptation of the original `PyTorch implementation <https://github.com/facebookresearch/XLM>`__.

**Google's XLNet** was released together with the paper `XLNet: Generalized Autoregressive Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`__ by Zhilin Yang\*, Zihang Dai\*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov and Quoc V. Le.
This PyTorch implementation of XLM is an adaptation of the `Tensorflow implementation <https://github.com/zihangdai/xlnet>`__.


Content
-------

.. list-table::
   :header-rows: 1

   * - Section
     - Description
   * - `Installation <./installation.html>`__
     - How to install the package
   * - `Philosphy <./philosophy.html>`__
     - The philosophy behind this package
   * - `Usage <./usage.html>`__
     - Quickstart examples
   * - `Examples <./examples.html>`__
     - Detailed examples on how to fine-tune Bert
   * - `Notebooks <./notebooks.html>`__
     - Introduction on the provided Jupyter Notebooks
   * - `TPU <./tpu.html>`__
     - Notes on TPU support and pretraining scripts
   * - `Command-line interface <./cli.html>`__
     - Convert a TensorFlow checkpoint in a PyTorch dump
   * - `Migration <./migration.html>`__
     - Migrating from ``pytorch_pretrained_BERT`` (v0.6) to ``pytorch_transformers`` (v1.0)
   * - `Bertology <./bertology.html>`__
     - Exploring the internals of the pretrained models.
   * - `TorchScript <./torchscript.html>`__
     - Convert a model to TorchScript for use in other programming languages

.. list-table::
   :header-rows: 1

   * - Section
     - Description
   * - `Overview <./model_doc/overview.html>`__
     - Overview of the package
   * - `BERT <./model_doc/bert.html>`__
     - BERT Models, Tokenizers and optimizers
   * - `OpenAI GPT <./model_doc/gpt.html>`__
     - GPT Models, Tokenizers and optimizers
   * - `TransformerXL <./model_doc/transformerxl.html>`__
     - TransformerXL Models, Tokenizers and optimizers
   * - `OpenAI GPT2 <./model_doc/gpt2.html>`__
     - GPT2 Models, Tokenizers and optimizers
   * - `XLM <./model_doc/xlm.html>`__
     - XLM Models, Tokenizers and optimizers
   * - `XLNet <./model_doc/xlnet.html>`__
     - XLNet Models, Tokenizers and optimizers

Overview
--------

This package comprises the following classes that can be imported in Python and are detailed in the `documentation <./model_doc/overview.html>`__ section of this package:


*
  Eight **Bert** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_bert.py <./_modules/pytorch_transformers/modeling_bert.html>`__ file):


  * `BertModel <./model_doc/bert.html#pytorch_transformers.BertModel>`__ - raw BERT Transformer model (\ **fully pre-trained**\ ),
  * `BertForMaskedLM <./model_doc/bert.html#pytorch_transformers.BertForMaskedLM>`__ - BERT Transformer with the pre-trained masked language modeling head on top (\ **fully pre-trained**\ ),
  * `BertForNextSentencePrediction <./model_doc/bert.html#pytorch_transformers.BertForNextSentencePrediction>`__ - BERT Transformer with the pre-trained next sentence prediction classifier on top  (\ **fully pre-trained**\ ),
  * `BertForPreTraining <./model_doc/bert.html#pytorch_transformers.BertForPreTraining>`__ - BERT Transformer with masked language modeling head and next sentence prediction classifier on top (\ **fully pre-trained**\ ),
  * `BertForSequenceClassification <./model_doc/bert.html#pytorch_transformers.BertForSequenceClassification>`__ - BERT Transformer with a sequence classification head on top (BERT Transformer is **pre-trained**\ , the sequence classification head **is only initialized and has to be trained**\ ),
  * `BertForMultipleChoice <./model_doc/bert.html#pytorch_transformers.BertForMultipleChoice>`__ - BERT Transformer with a multiple choice head on top (used for task like Swag) (BERT Transformer is **pre-trained**\ , the multiple choice classification head **is only initialized and has to be trained**\ ),
  * `BertForTokenClassification <./model_doc/bert.html#pytorch_transformers.BertForTokenClassification>`__ - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**\ , the token classification head **is only initialized and has to be trained**\ ),
  * `BertForQuestionAnswering <./model_doc/bert.html#pytorch_transformers.BertForQuestionAnswering>`__ - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**\ , the token classification head **is only initialized and has to be trained**\ ).

*
  Three **OpenAI GPT** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_openai.py <./_modules/pytorch_transformers/modeling_openai.html>`__ file):


  * `OpenAIGPTModel <./model_doc/gpt.html#pytorch_transformers.OpenAIGPTModel>`__ - raw OpenAI GPT Transformer model (\ **fully pre-trained**\ ),
  * `OpenAIGPTLMHeadModel <./model_doc/gpt.html#pytorch_transformers.OpenAIGPTLMHeadModel>`__ - OpenAI GPT Transformer with the tied language modeling head on top (\ **fully pre-trained**\ ),
  * `OpenAIGPTDoubleHeadsModel <./model_doc/gpt.html#pytorch_transformers.OpenAIGPTDoubleHeadsModel>`__ - OpenAI GPT Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT Transformer is **pre-trained**\ , the multiple choice classification head **is only initialized and has to be trained**\ ),

*
  Two **Transformer-XL** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_transfo_xl.py <./_modules/pytorch_transformers/modeling_transfo_xl.html>`__ file):


  * `TransfoXLModel <./model_doc/transformerxl.html#pytorch_transformers.TransfoXLModel>`__ - Transformer-XL model which outputs the last hidden state and memory cells (\ **fully pre-trained**\ ),
  * `TransfoXLLMHeadModel <./model_doc/transformerxl.html#pytorch_transformers.TransfoXLLMHeadModel>`__ - Transformer-XL with the tied adaptive softmax head on top for language modeling which outputs the logits/loss and memory cells (\ **fully pre-trained**\ ),

*
  Three **OpenAI GPT-2** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_gpt2.py <./_modules/pytorch_transformers/modeling_gpt2.html>`__ file):


  * `GPT2Model <./model_doc/gpt2.html#pytorch_transformers.GPT2Model>`__ - raw OpenAI GPT-2 Transformer model (\ **fully pre-trained**\ ),
  * `GPT2LMHeadModel <./model_doc/gpt2.html#pytorch_transformers.GPT2LMHeadModel>`__ - OpenAI GPT-2 Transformer with the tied language modeling head on top (\ **fully pre-trained**\ ),
  * `GPT2DoubleHeadsModel <./model_doc/gpt2.html#pytorch_transformers.GPT2DoubleHeadsModel>`__ - OpenAI GPT-2 Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT-2 Transformer is **pre-trained**\ , the multiple choice classification head **is only initialized and has to be trained**\ ),

*
  Four **XLM** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_xlm.py <./_modules/pytorch_transformers/modeling_xlm.html>`__ file):


  * `XLMModel <./model_doc/xlm.html#pytorch_transformers.XLMModel>`__ - raw XLM Transformer model (\ **fully pre-trained**\ ),
  * `XLMWithLMHeadModel <./model_doc/xlm.html#pytorch_transformers.XLMWithLMHeadModel>`__ - XLM Transformer with the tied language modeling head on top (\ **fully pre-trained**\ ),
  * `XLMForSequenceClassification <./model_doc/xlm.html#pytorch_transformers.XLMForSequenceClassification>`__ - XLM Transformer with a sequence classification head on top (XLM Transformer is **pre-trained**\ , the sequence classification head **is only initialized and has to be trained**\ ),
  * `XLMForQuestionAnswering <./model_doc/xlm.html#pytorch_transformers.XLMForQuestionAnswering>`__ - XLM Transformer with a token classification head on top (XLM Transformer is **pre-trained**\ , the token classification head **is only initialized and has to be trained**\ )

*
  Four **XLNet** PyTorch models (\ ``torch.nn.Module``\ ) with pre-trained weights (in the `modeling_xlnet.py <./_modules/pytorch_transformers/modeling_xlnet.html>`__ file):


  * `XLNetModel <./model_doc/xlnet.html#pytorch_transformers.XLNetModel>`__ - raw XLNet Transformer model (\ **fully pre-trained**\ ),
  * `XLNetLMHeadModel <./model_doc/xlnet.html#pytorch_transformers.XLNetLMHeadModel>`__ - XLNet Transformer with the tied language modeling head on top (\ **fully pre-trained**\ ),
  * `XLNetForSequenceClassification <./model_doc/xlnet.html#pytorch_transformers.XLNetForSequenceClassification>`__ - XLNet Transformer with a sequence classification head on top (XLM Transformer is **pre-trained**\ , the sequence classification head **is only initialized and has to be trained**\ ),
  * `XLNetForQuestionAnswering <./model_doc/xlnet.html#pytorch_transformers.XLNetForQuestionAnswering>`__ - XLNet Transformer with a token classification head on top (XLNet Transformer is **pre-trained**\ , the token classification head **is only initialized and has to be trained**\ )


TODO Lysandre filled: I filled in XLM and XLNet. I didn't do the Tokenizers because I don't know the current philosophy behind them.

*
  Tokenizers for **BERT** (using word-piece) (in the `tokenization_bert.py <./_modules/pytorch_transformers/tokenization_bert.html>`__ file):

  * ``BasicTokenizer`` - basic tokenization (punctuation splitting, lower casing, etc.),
  * ``WordpieceTokenizer`` - WordPiece tokenization,
  * ``BertTokenizer`` - perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization.


*
  Tokenizer for **OpenAI GPT** (using Byte-Pair-Encoding) (in the `tokenization_openai.py <./_modules/pytorch_transformers/tokenization_openai.html>`__ file):

  * ``OpenAIGPTTokenizer`` - perform Byte-Pair-Encoding (BPE) tokenization.


*
  Tokenizer for **OpenAI GPT-2** (using byte-level Byte-Pair-Encoding) (in the `tokenization_gpt2.py <./_modules/pytorch_transformers/tokenization_gpt2.html>`__ file):

  * ``GPT2Tokenizer`` - perform byte-level Byte-Pair-Encoding (BPE) tokenization.


*
  Tokenizer for **Transformer-XL** (word tokens ordered by frequency for adaptive softmax) (in the `tokenization_transfo_xl.py <./_modules/pytorch_transformers/tokenization_transfo_xl.html>`__ file):

  * ``OpenAIGPTTokenizer`` - perform word tokenization and can order words by frequency in a corpus for use in an adaptive softmax.


*
  Tokenizer for **XLNet** (SentencePiece based tokenizer) (in the `tokenization_xlnet.py <./_modules/pytorch_transformers/tokenization_xlnet.html>`__ file):

  * ``XLNetTokenizer`` - perform SentencePiece tokenization.


*
  Tokenizer for **XLM** (using Byte-Pair-Encoding) (in the `tokenization_xlm.py <./_modules/pytorch_transformers/tokenization_xlm.html>`__ file):

  * ``GPT2Tokenizer`` - perform Byte-Pair-Encoding (BPE) tokenization.


*
  Optimizer (in the `optimization.py <./_modules/pytorch_transformers/optimization.html>`__ file):


  * ``AdamW`` - Version of Adam algorithm with weight decay fix, warmup and linear decay of the learning rate.


*
  Configuration classes for BERT, OpenAI GPT, Transformer-XL, XLM and XLNet (in the respective \
  `modeling_bert.py <./_modules/pytorch_transformers/modeling_bert.html>`__\ , \
  `modeling_openai.py <./_modules/pytorch_transformers/modeling_openai.html>`__\ , \
  `modeling_transfo_xl.py <./_modules/pytorch_transformers/modeling_transfo_xl.html>`__, \
  `modeling_xlm.py <./_modules/pytorch_transformers/modeling_xlm.html>`__, \
  `modeling_xlnet.py <./_modules/pytorch_transformers/modeling_xlnet.html>`__ \
  files):


  * ``BertConfig`` - Configuration class to store the configuration of a ``BertModel`` with utilities to read and write from JSON configuration files.
  * ``OpenAIGPTConfig`` - Configuration class to store the configuration of a ``OpenAIGPTModel`` with utilities to read and write from JSON configuration files.
  * ``GPT2Config`` - Configuration class to store the configuration of a ``GPT2Model`` with utilities to read and write from JSON configuration files.
  * ``TransfoXLConfig`` - Configuration class to store the configuration of a ``TransfoXLModel`` with utilities to read and write from JSON configuration files.
  * ``XLMConfig`` - Configuration class to store the configuration of a ``XLMModel`` with utilities to read and write from JSON configuration files.
  * ``XLNetConfig`` - Configuration class to store the configuration of a ``XLNetModel`` with utilities to read and write from JSON configuration files.

The repository further comprises:


*
  Five examples on how to use **BERT** (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`__\ ):


  * `run_bert_extract_features.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_bert_extract_features.py>`__ - Show how to extract hidden states from an instance of ``BertModel``\ ,
  * `run_bert_classifier.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_bert_classifier.py>`__ - Show how to fine-tune an instance of ``BertForSequenceClassification`` on GLUE's MRPC task,
  * `run_bert_squad.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_bert_squad.py>`__ - Show how to fine-tune an instance of ``BertForQuestionAnswering`` on SQuAD v1.0 and SQuAD v2.0 tasks.
  * `run_swag.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_swag.py>`__ - Show how to fine-tune an instance of ``BertForMultipleChoice`` on Swag task.
  * `simple_lm_finetuning.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/simple_lm_finetuning.py>`__ - Show how to fine-tune an instance of ``BertForPretraining`` on a target text corpus.

*
  One example on how to use **OpenAI GPT** (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`__\ ):


  * `run_openai_gpt.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_openai_gpt.py>`__ - Show how to fine-tune an instance of ``OpenGPTDoubleHeadsModel`` on the RocStories task.

*
  One example on how to use **Transformer-XL** (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`__\ ):


  * `run_transfo_xl.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_transfo_xl.py>`__ - Show how to load and evaluate a pre-trained model of ``TransfoXLLMHeadModel`` on WikiText 103.

*
  One example on how to use **OpenAI GPT-2** in the unconditional and interactive mode (in the `examples folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples>`__\ ):


  * `run_gpt2.py <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py>`__ - Show how to use OpenAI GPT-2 an instance of ``GPT2LMHeadModel`` to generate text (same as the original OpenAI GPT-2 examples).

  These examples are detailed in the `Examples <#examples>`__ section of this readme.

*
  Three notebooks that were used to check that the TensorFlow and PyTorch models behave identically (in the `notebooks folder <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks>`__\ ):


  * `Comparing-TF-and-PT-models.ipynb <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing-TF-and-PT-models.ipynb>`__ - Compare the hidden states predicted by ``BertModel``\ ,
  * `Comparing-TF-and-PT-models-SQuAD.ipynb <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing-TF-and-PT-models-SQuAD.ipynb>`__ - Compare the spans predicted by  ``BertForQuestionAnswering`` instances,
  * `Comparing-TF-and-PT-models-MLM-NSP.ipynb <https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/notebooks/Comparing-TF-and-PT-models-MLM-NSP.ipynb>`__ - Compare the predictions of the ``BertForPretraining`` instances.

  These notebooks are detailed in the `Notebooks <#notebooks>`__ section of this readme.


*
  A command-line interface to convert TensorFlow checkpoints (BERT, Transformer-XL) or NumPy checkpoint (OpenAI) in a PyTorch save of the associated PyTorch model:

  This CLI is detailed in the `Command-line interface <#Command-line-interface>`__ section of this readme.
