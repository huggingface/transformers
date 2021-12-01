.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Converting Tensorflow Checkpoints
=======================================================================================================================

A command-line interface is provided to convert original Bert/GPT/GPT-2/Transformer-XL/XLNet/XLM checkpoints to models
that can be loaded using the ``from_pretrained`` methods of the library.

.. note::
    Since 2.3.0 the conversion script is now part of the transformers CLI (**transformers-cli**) available in any
    transformers >= 2.3.0 installation.

    The documentation below reflects the **transformers-cli convert** command format.

BERT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can convert any TensorFlow checkpoint for BERT (in particular `the pre-trained models released by Google
<https://github.com/google-research/bert#pre-trained-models>`_) in a PyTorch save file by using the
:prefix_link:`convert_bert_original_tf_checkpoint_to_pytorch.py
<src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py>` script.

This CLI takes as input a TensorFlow checkpoint (three files starting with ``bert_model.ckpt``) and the associated
configuration file (``bert_config.json``), and creates a PyTorch model for this configuration, loads the weights from
the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can
be imported using ``from_pretrained()`` (see example in :doc:`quicktour` , :prefix_link:`run_glue.py
<examples/pytorch/text-classification/run_glue.py>` ).

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow
checkpoint (the three files starting with ``bert_model.ckpt``) but be sure to keep the configuration file (\
``bert_config.json``) and the vocabulary file (``vocab.txt``) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (``pip install
tensorflow``). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained ``BERT-Base Uncased`` model:

.. code-block:: shell

    export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

    transformers-cli convert --model_type bert \
      --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
      --config $BERT_BASE_DIR/bert_config.json \
      --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin

You can download Google's pre-trained models for the conversion `here
<https://github.com/google-research/bert#pre-trained-models>`__.

ALBERT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert TensorFlow model checkpoints of ALBERT to PyTorch using the
:prefix_link:`convert_albert_original_tf_checkpoint_to_pytorch.py
<src/transformers/models/albert/convert_albert_original_tf_checkpoint_to_pytorch.py>` script.

The CLI takes as input a TensorFlow checkpoint (three files starting with ``model.ckpt-best``) and the accompanying
configuration file (``albert_config.json``), then creates and saves a PyTorch model. To run this conversion you will
need to have TensorFlow and PyTorch installed.

Here is an example of the conversion process for the pre-trained ``ALBERT Base`` model:

.. code-block:: shell

    export ALBERT_BASE_DIR=/path/to/albert/albert_base

    transformers-cli convert --model_type albert \
      --tf_checkpoint $ALBERT_BASE_DIR/model.ckpt-best \
      --config $ALBERT_BASE_DIR/albert_config.json \
      --pytorch_dump_output $ALBERT_BASE_DIR/pytorch_model.bin

You can download Google's pre-trained models for the conversion `here
<https://github.com/google-research/albert#pre-trained-models>`__.

OpenAI GPT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of the conversion process for a pre-trained OpenAI GPT model, assuming that your NumPy checkpoint
save as the same format than OpenAI pretrained model (see `here <https://github.com/openai/finetune-transformer-lm>`__\
)

.. code-block:: shell

    export OPENAI_GPT_CHECKPOINT_FOLDER_PATH=/path/to/openai/pretrained/numpy/weights

    transformers-cli convert --model_type gpt \
      --tf_checkpoint $OPENAI_GPT_CHECKPOINT_FOLDER_PATH \
      --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
      [--config OPENAI_GPT_CONFIG] \
      [--finetuning_task_name OPENAI_GPT_FINETUNED_TASK] \


OpenAI GPT-2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of the conversion process for a pre-trained OpenAI GPT-2 model (see `here
<https://github.com/openai/gpt-2>`__)

.. code-block:: shell

    export OPENAI_GPT2_CHECKPOINT_PATH=/path/to/gpt2/pretrained/weights

    transformers-cli convert --model_type gpt2 \
      --tf_checkpoint $OPENAI_GPT2_CHECKPOINT_PATH \
      --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
      [--config OPENAI_GPT2_CONFIG] \
      [--finetuning_task_name OPENAI_GPT2_FINETUNED_TASK]

Transformer-XL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of the conversion process for a pre-trained Transformer-XL model (see `here
<https://github.com/kimiyoung/transformer-xl/tree/master/tf#obtain-and-evaluate-pretrained-sota-models>`__)

.. code-block:: shell

    export TRANSFO_XL_CHECKPOINT_FOLDER_PATH=/path/to/transfo/xl/checkpoint

    transformers-cli convert --model_type transfo_xl \
      --tf_checkpoint $TRANSFO_XL_CHECKPOINT_FOLDER_PATH \
      --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
      [--config TRANSFO_XL_CONFIG] \
      [--finetuning_task_name TRANSFO_XL_FINETUNED_TASK]


XLNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of the conversion process for a pre-trained XLNet model:

.. code-block:: shell

    export TRANSFO_XL_CHECKPOINT_PATH=/path/to/xlnet/checkpoint
    export TRANSFO_XL_CONFIG_PATH=/path/to/xlnet/config

    transformers-cli convert --model_type xlnet \
      --tf_checkpoint $TRANSFO_XL_CHECKPOINT_PATH \
      --config $TRANSFO_XL_CONFIG_PATH \
      --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
      [--finetuning_task_name XLNET_FINETUNED_TASK] \


XLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of the conversion process for a pre-trained XLM model:

.. code-block:: shell

    export XLM_CHECKPOINT_PATH=/path/to/xlm/checkpoint

    transformers-cli convert --model_type xlm \
      --tf_checkpoint $XLM_CHECKPOINT_PATH \
      --pytorch_dump_output $PYTORCH_DUMP_OUTPUT
     [--config XML_CONFIG] \
     [--finetuning_task_name XML_FINETUNED_TASK]


T5
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example of the conversion process for a pre-trained T5 model:

.. code-block:: shell

    export T5=/path/to/t5/uncased_L-12_H-768_A-12

    transformers-cli convert --model_type t5 \
      --tf_checkpoint $T5/t5_model.ckpt \
      --config $T5/t5_config.json \
      --pytorch_dump_output $T5/pytorch_model.bin
