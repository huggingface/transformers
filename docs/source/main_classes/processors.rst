Processors
----------------------------------------------------

This library includes processors for several traditional tasks. These processors can be used to process a dataset into
examples that can be fed to a model.

``GLUE``
~~~~~~~~~~~~~~~~~~~~~

`General Language Understanding Evaluation (GLUE)<https://gluebenchmark.com/>`__ is a benchmark that evaluates
the performance of models across a diverse set of existing NLU tasks. It was released together with the paper
`GLUE: A multi-task benchmark and analysis platform for natural language understanding<https://openreview.net/pdf?id=rJ4km2R5t7>`__

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched),
CoLA, SST2, STSB, QQP, QNLI, RTE and WNLI.

.. autoclass:: pytorch_transformers.data.processors.glue.MrpcProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.MnliProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.MnliMismatchedProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.ColaProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.Sst2Processor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.StsbProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.QqpProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.QnliProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.RteProcessor
    :members:

.. autoclass:: pytorch_transformers.data.processors.glue.WnliProcessor
    :members:
