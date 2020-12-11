.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Benchmarks
=======================================================================================================================

Let's take a look at how 🤗 Transformer models can be benchmarked, best practices, and already available benchmarks.

A notebook explaining in more detail how to benchmark 🤗 Transformer models can be found `here
<https://github.com/huggingface/transformers/blob/master/notebooks/05-benchmark.ipynb>`__.

How to benchmark 🤗 Transformer models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classes :class:`~transformers.PyTorchBenchmark` and :class:`~transformers.TensorFlowBenchmark` allow to flexibly
benchmark 🤗 Transformer models. The benchmark classes allow us to measure the `peak memory usage` and `required time`
for both `inference` and `training`.

.. note::

  Hereby, `inference` is defined by a single forward pass, and `training` is defined by a single forward pass and
  backward pass.

The benchmark classes :class:`~transformers.PyTorchBenchmark` and :class:`~transformers.TensorFlowBenchmark` expect an
object of type :class:`~transformers.PyTorchBenchmarkArguments` and
:class:`~transformers.TensorFlowBenchmarkArguments`, respectively, for instantiation.
:class:`~transformers.PyTorchBenchmarkArguments` and :class:`~transformers.TensorFlowBenchmarkArguments` are data
classes and contain all relevant configurations for their corresponding benchmark class. In the following example, it
is shown how a BERT model of type `bert-base-cased` can be benchmarked.

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

    >>> args = PyTorchBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
    >>> benchmark = PyTorchBenchmark(args)

    >>> ## TENSORFLOW CODE
    >>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

    >>> args = TensorFlowBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
    >>> benchmark = TensorFlowBenchmark(args)


Here, three arguments are given to the benchmark argument data classes, namely ``models``, ``batch_sizes``, and
``sequence_lengths``. The argument ``models`` is required and expects a :obj:`list` of model identifiers from the
`model hub <https://huggingface.co/models>`__ The :obj:`list` arguments ``batch_sizes`` and ``sequence_lengths`` define
the size of the ``input_ids`` on which the model is benchmarked. There are many more parameters that can be configured
via the benchmark argument data classes. For more detail on these one can either directly consult the files
``src/transformers/benchmark/benchmark_args_utils.py``, ``src/transformers/benchmark/benchmark_args.py`` (for PyTorch)
and ``src/transformers/benchmark/benchmark_args_tf.py`` (for Tensorflow). Alternatively, running the following shell
commands from root will print out a descriptive list of all configurable parameters for PyTorch and Tensorflow
respectively.

.. code-block:: bash

    ## PYTORCH CODE
    python examples/benchmarking/run_benchmark.py --help

    ## TENSORFLOW CODE
    python examples/benchmarking/run_benchmark_tf.py --help


An instantiated benchmark object can then simply be run by calling ``benchmark.run()``.

.. code-block::

    >>> ## PYTORCH CODE
    >>> results = benchmark.run()
    >>> print(results)
    ====================       INFERENCE - SPEED - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length     Time in s                  
    --------------------------------------------------------------------------------
    bert-base-uncased          8               8             0.006     
    bert-base-uncased          8               32            0.006     
    bert-base-uncased          8              128            0.018     
    bert-base-uncased          8              512            0.088     
    --------------------------------------------------------------------------------

    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
    bert-base-uncased          8               8             1227
    bert-base-uncased          8               32            1281
    bert-base-uncased          8              128            1307
    bert-base-uncased          8              512            1539
    --------------------------------------------------------------------------------

    ====================        ENVIRONMENT INFORMATION         ====================
    - transformers_version: 2.11.0
    - framework: PyTorch
    - use_torchscript: False
    - framework_version: 1.4.0
    - python_version: 3.6.10
    - system: Linux
    - cpu: x86_64
    - architecture: 64bit
    - date: 2020-06-29
    - time: 08:58:43.371351
    - fp16: False
    - use_multiprocessing: True
    - only_pretrain_model: False
    - cpu_ram_mb: 32088
    - use_gpu: True
    - num_gpus: 1
    - gpu: TITAN RTX
    - gpu_ram_mb: 24217
    - gpu_power_watts: 280.0
    - gpu_performance_state: 2
    - use_tpu: False

    >>> ## TENSORFLOW CODE
    >>> results = benchmark.run()
    >>> print(results)
    ====================       INFERENCE - SPEED - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length     Time in s                  
    --------------------------------------------------------------------------------
    bert-base-uncased          8               8             0.005
    bert-base-uncased          8               32            0.008
    bert-base-uncased          8              128            0.022
    bert-base-uncased          8              512            0.105
    --------------------------------------------------------------------------------

    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length    Memory in MB 
    --------------------------------------------------------------------------------
    bert-base-uncased          8               8             1330
    bert-base-uncased          8               32            1330
    bert-base-uncased          8              128            1330
    bert-base-uncased          8              512            1770
    --------------------------------------------------------------------------------

    ====================        ENVIRONMENT INFORMATION         ====================
    - transformers_version: 2.11.0
    - framework: Tensorflow
    - use_xla: False
    - framework_version: 2.2.0
    - python_version: 3.6.10
    - system: Linux
    - cpu: x86_64
    - architecture: 64bit
    - date: 2020-06-29
    - time: 09:26:35.617317
    - fp16: False
    - use_multiprocessing: True
    - only_pretrain_model: False
    - cpu_ram_mb: 32088
    - use_gpu: True
    - num_gpus: 1
    - gpu: TITAN RTX
    - gpu_ram_mb: 24217
    - gpu_power_watts: 280.0
    - gpu_performance_state: 2
    - use_tpu: False

By default, the `time` and the `required memory` for `inference` are benchmarked. In the example output above the first
two sections show the result corresponding to `inference time` and `inference memory`. In addition, all relevant
information about the computing environment, `e.g.` the GPU type, the system, the library versions, etc... are printed
out in the third section under `ENVIRONMENT INFORMATION`. This information can optionally be saved in a `.csv` file
when adding the argument :obj:`save_to_csv=True` to :class:`~transformers.PyTorchBenchmarkArguments` and
:class:`~transformers.TensorFlowBenchmarkArguments` respectively. In this case, every section is saved in a separate
`.csv` file. The path to each `.csv` file can optionally be defined via the argument data classes.

Instead of benchmarking pre-trained models via their model identifier, `e.g.` `bert-base-uncased`, the user can
alternatively benchmark an arbitrary configuration of any available model class. In this case, a :obj:`list` of
configurations must be inserted with the benchmark args as follows.

.. code-block::

    >>> ## PYTORCH CODE
    >>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

    >>> args = PyTorchBenchmarkArguments(models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
    >>> config_base = BertConfig()
    >>> config_384_hid = BertConfig(hidden_size=384)
    >>> config_6_lay = BertConfig(num_hidden_layers=6)

    >>> benchmark = PyTorchBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
    >>> benchmark.run()
    ====================       INFERENCE - SPEED - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length       Time in s                  
    --------------------------------------------------------------------------------
    bert-base                  8              128            0.006
    bert-base                  8              512            0.006
    bert-base                  8              128            0.018     
    bert-base                  8              512            0.088     
    bert-384-hid              8               8             0.006     
    bert-384-hid              8               32            0.006     
    bert-384-hid              8              128            0.011     
    bert-384-hid              8              512            0.054     
    bert-6-lay                 8               8             0.003     
    bert-6-lay                 8               32            0.004     
    bert-6-lay                 8              128            0.009     
    bert-6-lay                 8              512            0.044
    --------------------------------------------------------------------------------

    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length      Memory in MB 
    --------------------------------------------------------------------------------
    bert-base                  8               8             1277
    bert-base                  8               32            1281
    bert-base                  8              128            1307     
    bert-base                  8              512            1539     
    bert-384-hid              8               8             1005     
    bert-384-hid              8               32            1027     
    bert-384-hid              8              128            1035     
    bert-384-hid              8              512            1255     
    bert-6-lay                 8               8             1097     
    bert-6-lay                 8               32            1101     
    bert-6-lay                 8              128            1127     
    bert-6-lay                 8              512            1359
    --------------------------------------------------------------------------------

    ====================        ENVIRONMENT INFORMATION         ====================
    - transformers_version: 2.11.0
    - framework: PyTorch
    - use_torchscript: False
    - framework_version: 1.4.0
    - python_version: 3.6.10
    - system: Linux
    - cpu: x86_64
    - architecture: 64bit
    - date: 2020-06-29
    - time: 09:35:25.143267
    - fp16: False
    - use_multiprocessing: True
    - only_pretrain_model: False
    - cpu_ram_mb: 32088
    - use_gpu: True
    - num_gpus: 1
    - gpu: TITAN RTX
    - gpu_ram_mb: 24217
    - gpu_power_watts: 280.0
    - gpu_performance_state: 2
    - use_tpu: False

    >>> ## TENSORFLOW CODE
    >>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

    >>> args = TensorFlowBenchmarkArguments(models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
    >>> config_base = BertConfig()
    >>> config_384_hid = BertConfig(hidden_size=384)
    >>> config_6_lay = BertConfig(num_hidden_layers=6)

    >>> benchmark = TensorFlowBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
    >>> benchmark.run()
    ====================       INFERENCE - SPEED - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length       Time in s                  
    --------------------------------------------------------------------------------
    bert-base                  8               8             0.005
    bert-base                  8               32            0.008
    bert-base                  8              128            0.022
    bert-base                  8              512            0.106
    bert-384-hid              8               8             0.005
    bert-384-hid              8               32            0.007
    bert-384-hid              8              128            0.018
    bert-384-hid              8              512            0.064
    bert-6-lay                 8               8             0.002
    bert-6-lay                 8               32            0.003
    bert-6-lay                 8              128            0.0011
    bert-6-lay                 8              512            0.074
    --------------------------------------------------------------------------------

    ====================      INFERENCE - MEMORY - RESULT       ====================
    --------------------------------------------------------------------------------
    Model Name             Batch Size     Seq Length      Memory in MB 
    --------------------------------------------------------------------------------
    bert-base                  8               8             1330
    bert-base                  8               32            1330
    bert-base                  8              128            1330
    bert-base                  8              512            1770
    bert-384-hid              8               8             1330
    bert-384-hid              8               32            1330
    bert-384-hid              8              128            1330
    bert-384-hid              8              512            1540
    bert-6-lay                 8               8             1330
    bert-6-lay                 8               32            1330
    bert-6-lay                 8              128            1330
    bert-6-lay                 8              512            1540
    --------------------------------------------------------------------------------

    ====================        ENVIRONMENT INFORMATION         ====================
    - transformers_version: 2.11.0
    - framework: Tensorflow
    - use_xla: False
    - framework_version: 2.2.0
    - python_version: 3.6.10
    - system: Linux
    - cpu: x86_64
    - architecture: 64bit
    - date: 2020-06-29
    - time: 09:38:15.487125
    - fp16: False
    - use_multiprocessing: True
    - only_pretrain_model: False
    - cpu_ram_mb: 32088
    - use_gpu: True
    - num_gpus: 1
    - gpu: TITAN RTX
    - gpu_ram_mb: 24217
    - gpu_power_watts: 280.0
    - gpu_performance_state: 2
    - use_tpu: False


Again, `inference time` and `required memory` for `inference` are measured, but this time for customized configurations
of the :obj:`BertModel` class. This feature can especially be helpful when deciding for which configuration the model
should be trained.


Benchmark best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section lists a couple of best practices one should be aware of when benchmarking a model.

- Currently, only single device benchmarking is supported. When benchmarking on GPU, it is recommended that the user
  specifies on which device the code should be run by setting the ``CUDA_VISIBLE_DEVICES`` environment variable in the
  shell, `e.g.` ``export CUDA_VISIBLE_DEVICES=0`` before running the code.
- The option :obj:`no_multi_processing` should only be set to :obj:`True` for testing and debugging. To ensure accurate
  memory measurement it is recommended to run each memory benchmark in a separate process by making sure
  :obj:`no_multi_processing` is set to :obj:`True`.
- One should always state the environment information when sharing the results of a model benchmark. Results can vary
  heavily between different GPU devices, library versions, etc., so that benchmark results on their own are not very
  useful for the community.


Sharing your benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously all available core models (10 at the time) have been benchmarked for `inference time`, across many different
settings: using PyTorch, with and without TorchScript, using TensorFlow, with and without XLA. All of those tests were
done across CPUs (except for TensorFlow XLA) and GPUs.

The approach is detailed in the `following blogpost
<https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2>`__ and the results are
available `here
<https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing>`__.

With the new `benchmark` tools, it is easier than ever to share your benchmark results with the community `here
<https://github.com/huggingface/transformers/blob/master/examples/benchmarking/README.md>`__.
