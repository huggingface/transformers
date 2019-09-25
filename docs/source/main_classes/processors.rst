Processors
----------------------------------------------------

This library includes processors for several traditional tasks. These processors can be used to process a dataset into
examples that can be fed to a model.

Processors
~~~~~~~~~~~~~~~~~~~~~

All processors follow the same architecture which is that of the
:class:`~pytorch_transformers.data.processors.utils.DataProcessor`. The processor returns a list
of :class:`~pytorch_transformers.data.processors.utils.InputExample`. These
:class:`~pytorch_transformers.data.processors.utils.InputExample` can be converted to
:class:`~pytorch_transformers.data.processors.utils.InputFeatures` in order to be fed to the model.

.. autoclass:: pytorch_transformers.data.processors.utils.DataProcessor
    :members:


.. autoclass:: pytorch_transformers.data.processors.utils.InputExample
    :members:


.. autoclass:: pytorch_transformers.data.processors.utils.InputFeatures
    :members:


GLUE
~~~~~~~~~~~~~~~~~~~~~

`General Language Understanding Evaluation (GLUE) <https://gluebenchmark.com/>`__ is a benchmark that evaluates
the performance of models across a diverse set of existing NLU tasks. It was released together with the paper
`GLUE: A multi-task benchmark and analysis platform for natural language understanding <https://openreview.net/pdf?id=rJ4km2R5t7>`__

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched),
CoLA, SST2, STSB, QQP, QNLI, RTE and WNLI.

Those processors are:
    - :class:`~pytorch_transformers.data.processors.utils.MrpcProcessor`
    - :class:`~pytorch_transformers.data.processors.utils.MnliProcessor`
    - :class:`~pytorch_transformers.data.processors.utils.MnliMismatchedProcessor`
    - :class:`~pytorch_transformers.data.processors.utils.Sst2Processor`
    - :class:`~pytorch_transformers.data.processors.utils.StsbProcessor`
    - :class:`~pytorch_transformers.data.processors.utils.QqpProcessor`
    - :class:`~pytorch_transformers.data.processors.utils.QnliProcessor`
    - :class:`~pytorch_transformers.data.processors.utils.RteProcessor`
    - :class:`~pytorch_transformers.data.processors.utils.WnliProcessor`

Additionally, the following method  can be used to load values from a data file and convert them to a list of
:class:`~pytorch_transformers.data.processors.utils.InputExample`.

.. automethod:: pytorch_transformers.data.processors.glue.glue_convert_examples_to_features

Example usage
^^^^^^^^^^^^^^^^^^^^^^^^^

An example using these processors is given in the
`run_glue.py <https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py>`__ script.