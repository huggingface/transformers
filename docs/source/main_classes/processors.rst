.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Processors
-----------------------------------------------------------------------------------------------------------------------

This library includes processors for several traditional tasks. These processors can be used to process a dataset into
examples that can be fed to a model.

Processors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All processors follow the same architecture which is that of the
:class:`~transformers.data.processors.utils.DataProcessor`. The processor returns a list of
:class:`~transformers.data.processors.utils.InputExample`. These
:class:`~transformers.data.processors.utils.InputExample` can be converted to
:class:`~transformers.data.processors.utils.InputFeatures` in order to be fed to the model.

.. autoclass:: transformers.data.processors.utils.DataProcessor
    :members:


.. autoclass:: transformers.data.processors.utils.InputExample
    :members:


.. autoclass:: transformers.data.processors.utils.InputFeatures
    :members:


GLUE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`General Language Understanding Evaluation (GLUE) <https://gluebenchmark.com/>`__ is a benchmark that evaluates the
performance of models across a diverse set of existing NLU tasks. It was released together with the paper `GLUE: A
multi-task benchmark and analysis platform for natural language understanding
<https://openreview.net/pdf?id=rJ4km2R5t7>`__

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched), CoLA, SST2, STSB,
QQP, QNLI, RTE and WNLI.

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

Additionally, the following method can be used to load values from a data file and convert them to a list of
:class:`~transformers.data.processors.utils.InputExample`.

.. automethod:: transformers.data.processors.glue.glue_convert_examples_to_features

Example usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example using these processors is given in the `run_glue.py
<https://github.com/huggingface/pytorch-transformers/blob/master/examples/text-classification/run_glue.py>`__ script.


XNLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`The Cross-Lingual NLI Corpus (XNLI) <https://www.nyu.edu/projects/bowman/xnli/>`__ is a benchmark that evaluates the
quality of cross-lingual text representations. XNLI is crowd-sourced dataset based on `MultiNLI
<http://www.nyu.edu/projects/bowman/multinli/>`: pairs of text are labeled with textual entailment annotations for 15
different languages (including both high-resource language such as English and low-resource languages such as Swahili).

It was released together with the paper `XNLI: Evaluating Cross-lingual Sentence Representations
<https://arxiv.org/abs/1809.05053>`__

This library hosts the processor to load the XNLI data:

    - :class:`~transformers.data.processors.utils.XnliProcessor`

Please note that since the gold labels are available on the test set, evaluation is performed on the test set.

An example using these processors is given in the `run_xnli.py
<https://github.com/huggingface/pytorch-transformers/blob/master/examples/text-classification/run_xnli.py>`__ script.


SQuAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`The Stanford Question Answering Dataset (SQuAD) <https://rajpurkar.github.io/SQuAD-explorer//>`__ is a benchmark that
evaluates the performance of models on question answering. Two versions are available, v1.1 and v2.0. The first version
(v1.1) was released together with the paper `SQuAD: 100,000+ Questions for Machine Comprehension of Text
<https://arxiv.org/abs/1606.05250>`__. The second version (v2.0) was released alongside the paper `Know What You Don't
Know: Unanswerable Questions for SQuAD <https://arxiv.org/abs/1806.03822>`__.

This library hosts a processor for each of the two versions:

Processors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Those processors are:

    - :class:`~transformers.data.processors.utils.SquadV1Processor`
    - :class:`~transformers.data.processors.utils.SquadV2Processor`

They both inherit from the abstract class :class:`~transformers.data.processors.utils.SquadProcessor`

.. autoclass:: transformers.data.processors.squad.SquadProcessor
    :members:

Additionally, the following method can be used to convert SQuAD examples into
:class:`~transformers.data.processors.utils.SquadFeatures` that can be used as model inputs.

.. automethod:: transformers.data.processors.squad.squad_convert_examples_to_features

These processors as well as the aforementionned method can be used with files containing the data as well as with the
`tensorflow_datasets` package. Examples are given below.


Example usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example using the processors as well as the conversion method using data files:

.. code-block::

    # Loading a V2 processor
    processor = SquadV2Processor()
    examples = processor.get_dev_examples(squad_v2_data_dir)

    # Loading a V1 processor
    processor = SquadV1Processor()
    examples = processor.get_dev_examples(squad_v1_data_dir)

    features = squad_convert_examples_to_features( 
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
    )

Using `tensorflow_datasets` is as easy as using a data file:

.. code-block::

    # tensorflow_datasets only handle Squad V1.
    tfds_examples = tfds.load("squad")
    examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

    features = squad_convert_examples_to_features( 
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=max_query_length,
        is_training=not evaluate,
    )


Another example using these processors is given in the :prefix_link:`run_squad.py
<examples/question-answering/run_squad.py>` script.
