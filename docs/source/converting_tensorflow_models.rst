Converting Tensorflow Checkpoints
================================================

A command-line interface is provided to convert a TensorFlow checkpoint in a PyTorch dump of the ``BertForPreTraining`` class  (for BERT) or NumPy checkpoint in a PyTorch dump of the ``OpenAIGPTModel`` class  (for OpenAI GPT).

BERT
^^^^

You can convert any TensorFlow checkpoint for BERT (in particular `the pre-trained models released by Google <https://github.com/google-research/bert#pre-trained-models>`_\ ) in a PyTorch save file by using the `convert_tf_checkpoint_to_pytorch.py <https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/convert_tf_checkpoint_to_pytorch.py>`_ script.

This CLI takes as input a TensorFlow checkpoint (three files starting with ``bert_model.ckpt``\ ) and the associated converter configuration file (\ ``converter_config.json``\ ), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using ``torch.load()`` (see examples in `run_bert_extract_features.py <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/run_bert_extract_features.py>`_\ , `run_bert_classifier.py <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/run_bert_classifier.py>`_ and `run_bert_squad.py <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/run_bert_squad.py>`_\ ).


You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with ``bert_model.ckpt``\ ) but be sure to keep the configuration file (\ ``bert_config.json``\ ) and the vocabulary file (\ ``vocab.txt``\ ) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (\ ``pip install tensorflow``\ ). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained ``BERT-Base Uncased`` model:

.. code-block:: shell

   export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

   pytorch_transformers bert \
     $BERT_BASE_DIR/bert_model.ckpt \
     converter_config.json \
     $BERT_BASE_DIR/pytorch_model.bin

Examples of ``converter_config.json``:

Converter config fields:
1. (**Required**) ``bert_config_path`` - path to original BERT configuration file 
2. (**Required**) ``bert_model_class`` - name of the PyTorch class which the TF model will be converted to, i.e. subclass of ``BertPreTrainedModel``
3. `tf_torch_mappings` - key-value formatted mappings from original TF model layer names to the corresponding PyTorch ones (`bert_model_class` specific)
4. `args`- key-value formatted arguments passed to the ``bert_model_class``'s constructor. The names should match exactly with constructor's (e.g. `num_labels` for ``BertForTokenClassification``\ )

Converter config for ``QuestionAnswering`` pre-trained on SQuAD:
.. code-block:: json

{
   "bert_config_path":"$BERT_BASE_DIR/bert_config.json",
   "bert_model_class":"BertForQuestionAnswering",
   "tf_torch_mappings":{
      "tf_output_layer_name":"qa_output"
   }
}
```

Converter config for `TokenClassification` with two output labels:
.. code-block:: json

{
   "bert_config_path":"$BERT_BASE_DIR/bert_config.json",
   "bert_model_class":"BertForTokenClassification",
   "tf_torch_mappings":{
      "tf_output_layer_name":"classifier"
   },
   "args":{
      "num_labels":2,
   }
}


You can download Google's pre-trained models for the conversion `here <https://github.com/google-research/bert#pre-trained-models>`__.

OpenAI GPT
^^^^^^^^^^

Here is an example of the conversion process for a pre-trained OpenAI GPT model, assuming that your NumPy checkpoint save as the same format than OpenAI pretrained model (see `here <https://github.com/openai/finetune-transformer-lm>`__\ )

.. code-block:: shell

   export OPENAI_GPT_CHECKPOINT_FOLDER_PATH=/path/to/openai/pretrained/numpy/weights

   pytorch_transformers gpt \
     $OPENAI_GPT_CHECKPOINT_FOLDER_PATH \
     $PYTORCH_DUMP_OUTPUT \
     [OPENAI_GPT_CONFIG]

Transformer-XL
^^^^^^^^^^^^^^

Here is an example of the conversion process for a pre-trained Transformer-XL model (see `here <https://github.com/kimiyoung/transformer-xl/tree/master/tf#obtain-and-evaluate-pretrained-sota-models>`__\ )

.. code-block:: shell

   export TRANSFO_XL_CHECKPOINT_FOLDER_PATH=/path/to/transfo/xl/checkpoint

   pytorch_transformers transfo_xl \
     $TRANSFO_XL_CHECKPOINT_FOLDER_PATH \
     $PYTORCH_DUMP_OUTPUT \
     [TRANSFO_XL_CONFIG]

GPT-2
^^^^^

Here is an example of the conversion process for a pre-trained OpenAI's GPT-2 model.

.. code-block:: shell

   export GPT2_DIR=/path/to/gpt2/checkpoint

   pytorch_transformers gpt2 \
     $GPT2_DIR/model.ckpt \
     $PYTORCH_DUMP_OUTPUT \
     [GPT2_CONFIG]

XLNet
^^^^^

Here is an example of the conversion process for a pre-trained XLNet model, fine-tuned on STS-B using the TensorFlow script:

.. code-block:: shell

   export TRANSFO_XL_CHECKPOINT_PATH=/path/to/xlnet/checkpoint
   export TRANSFO_XL_CONFIG_PATH=/path/to/xlnet/config

   pytorch_transformers xlnet \
     $TRANSFO_XL_CHECKPOINT_PATH \
     $TRANSFO_XL_CONFIG_PATH \
     $PYTORCH_DUMP_OUTPUT \
     STS-B \
