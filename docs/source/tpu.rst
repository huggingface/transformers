TPU
================================================

TPU support and pretraining scripts
------------------------------------------------

TPU are not supported by the current stable release of PyTorch (0.4.1). However, the next version of PyTorch (v1.0) should support training on TPU and is expected to be released soon (see the recent `official announcement <https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud>`_\ ).

We will add TPU support when this next release is published.

The original TensorFlow code further comprises two scripts for pre-training BERT: `create_pretraining_data.py <https://github.com/google-research/bert/blob/master/create_pretraining_data.py>`_ and `run_pretraining.py <https://github.com/google-research/bert/blob/master/run_pretraining.py>`_.

Since, pre-training BERT is a particularly expensive operation that basically requires one or several TPUs to be completed in a reasonable amout of time (see details `here <https://github.com/google-research/bert#pre-training-with-bert>`_\ ) we have decided to wait for the inclusion of TPU support in PyTorch to convert these pre-training scripts.
