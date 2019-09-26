Notebooks
================================================

We include `three Jupyter Notebooks <https://github.com/huggingface/transformers/tree/master/notebooks>`_ that can be used to check that the predictions of the PyTorch model are identical to the predictions of the original TensorFlow model.


*
  The first NoteBook (\ `Comparing-TF-and-PT-models.ipynb <https://github.com/huggingface/transformers/blob/master/notebooks/Comparing-TF-and-PT-models.ipynb>`_\ ) extracts the hidden states of a full sequence on each layers of the TensorFlow and the PyTorch models and computes the standard deviation between them. In the given example, we get a standard deviation of 1.5e-7 to 9e-7 on the various hidden state of the models.

*
  The second NoteBook (\ `Comparing-TF-and-PT-models-SQuAD.ipynb <https://github.com/huggingface/transformers/blob/master/notebooks/Comparing-TF-and-PT-models-SQuAD.ipynb>`_\ ) compares the loss computed by the TensorFlow and the PyTorch models for identical initialization of the fine-tuning layer of the ``BertForQuestionAnswering`` and computes the standard deviation between them. In the given example, we get a standard deviation of 2.5e-7 between the models.

*
  The third NoteBook (\ `Comparing-TF-and-PT-models-MLM-NSP.ipynb <https://github.com/huggingface/transformers/blob/master/notebooks/Comparing-TF-and-PT-models-MLM-NSP.ipynb>`_\ ) compares the predictions computed by the TensorFlow and the PyTorch models for masked token language modeling using the pre-trained masked language modeling model.

Please follow the instructions given in the notebooks to run and modify them.
