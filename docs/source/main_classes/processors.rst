Processors
----------------------------------------------------

This library includes processors for several traditional tasks. These processors can be used to process a dataset into
examples that can be fed to a model.

Processors
~~~~~~~~~~~~~~~~~~~~~

All processors follow the same architecture which is that of the
:class:`~transformers.data.processors.utils.DataProcessor`. The processor returns a list
of :class:`~transformers.data.processors.utils.InputExample`. These
:class:`~transformers.data.processors.utils.InputExample` can be converted to
:class:`~transformers.data.processors.utils.InputFeatures` in order to be fed to the model.

.. autoclass:: transformers.data.processors.utils.DataProcessor
    :members:


.. autoclass:: transformers.data.processors.utils.InputExample
    :members:


.. autoclass:: transformers.data.processors.utils.InputFeatures
    :members:


GLUE
~~~~~~~~~~~~~~~~~~~~~

`General Language Understanding Evaluation (GLUE) <https://gluebenchmark.com/>`__ is a benchmark that evaluates
the performance of models across a diverse set of existing NLU tasks. It was released together with the paper
`GLUE: A multi-task benchmark and analysis platform for natural language understanding <https://openreview.net/pdf?id=rJ4km2R5t7>`__

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched),
CoLA, SST2, STSB, QQP, QNLI, RTE and WNLI.

Those processors are:
    - :class:`~transformers.data.processors.utils.MrpcProcessor`
    - :class:`~transformers.data.processors.utils.MnliProcessor`
    - :class:`~transformers.data.processors.utils.MnliMismatchedProcessor`
    - :class:`~transformers.data.processors.utils.Sst2Processor`
    - :class:`~transformers.data.processors.utils.StsbProcessor`
    - :class:`~transformers.data.processors.utils.QqpProcessor`
    - :class:`~transformers.data.processors.utils.QnliProcessor`
    - :class:`~transformers.data.processors.utils.RteProcessor`
    - :class:`~transformers.data.processors.utils.WnliProcessor`

Additionally, the following method  can be used to load values from a data file and convert them to a list of
:class:`~transformers.data.processors.utils.InputExample`.

.. automethod:: transformers.data.processors.glue.glue_convert_examples_to_features

Example usage
^^^^^^^^^^^^^^^^^^^^^^^^^

An example using these processors is given in the
`run_glue.py <https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_glue.py>`__ script.


XNLI
~~~~~~~~~~~~~~~~~~~~~

`The Cross-Lingual NLI Corpus (XNLI) <https://www.nyu.edu/projects/bowman/xnli/>`__ is a benchmark that evaluates
the quality of cross-lingual text representations. 
XNLI is crowd-sourced dataset based on `MultiNLI <http://www.nyu.edu/projects/bowman/multinli/>`: pairs of text are labeled with textual entailment 
annotations for 15 different languages (including both high-ressource language such as English and low-ressource languages such as Swahili).

It was released together with the paper
`XNLI: Evaluating Cross-lingual Sentence Representations <https://arxiv.org/abs/1809.05053>`__

This library hosts the processor to load the XNLI data:
    - :class:`~transformers.data.processors.utils.XnliProcessor`

Please note that since the gold labels are available on the test set, evaluation is performed on the test set.

Example usage
^^^^^^^^^^^^^^^^^^^^^^^^^

An example using these processors is given in the
`run_xnli.py <https://github.com/huggingface/pytorch-transformers/blob/master/examples/run_xnli.py>`__ script.