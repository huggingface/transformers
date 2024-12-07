<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 基准测试

<Tip warning={true}>

小提示：Hugging Face的基准测试工具已经不再更新，建议使用外部基准测试库来衡量Transformer模
型的速度和内存复杂度。

</Tip>

[[open-in-colab]]

让我们来看看如何对🤗 Transformers模型进行基准测试，以及进行测试的推荐策略和已有的基准测试结果。

如果你需要更详细的回答，可以在这里找到关于如何对🤗 Transformers模型进行基准测试的[说明](https://github.com/huggingface/notebooks/tree/main/examples/benchmark.ipynb)。

<!-- Let's take a look at how 🤗 Transformers models can be benchmarked, best practices, and already available benchmarks.

A notebook explaining in more detail how to benchmark 🤗 Transformers models can be found [here](https://github.com/huggingface/notebooks/tree/main/examples/benchmark.ipynb). -->

<!-- ## How to benchmark 🤗 Transformers models -->
## 如何对🤗 Transformers模型进行基准测试

<!-- The classes [`PyTorchBenchmark`] and [`TensorFlowBenchmark`] allow to flexibly benchmark 🤗 Transformers models. The benchmark classes allow us to measure the _peak memory usage_ and _required time_ for both _inference_ and _training_. -->
使用[`PyTorchBenchmark`]和[`TensorFlowBenchmark`]类可以灵活地对🤗 Transformers模型进行基准测试。这些基准测试类可以衡量模型在**推理**和**训练**过程中所需的**峰值内存**和**时间**。

<Tip>

<!-- Here, _inference_ is defined by a single forward pass, and _training_ is defined by a single forward pass and
backward pass. -->
这里的**推理**指的是一次前向传播(forward pass)，训练则指的是一次前向传播和反向传播(backward pass)。

</Tip>

<!-- The benchmark classes [`PyTorchBenchmark`] and [`TensorFlowBenchmark`] expect an object of type [`PyTorchBenchmarkArguments`] and
[`TensorFlowBenchmarkArguments`], respectively, for instantiation. [`PyTorchBenchmarkArguments`] and [`TensorFlowBenchmarkArguments`] are data classes and contain all relevant configurations for their corresponding benchmark class. In the following example, it is shown how a BERT model of type _bert-base-cased_ can be benchmarked. -->

基准测试类 [`PyTorchBenchmark`] 和 [`TensorFlowBenchmark`] 需要分别传入 [`PyTorchBenchmarkArguments`] 和 [`TensorFlowBenchmarkArguments`] 类型的对象来进行实例化。这些类是数据类型，包含了所有相关的配置参数，用于其对应的基准测试类。

在下面的示例中，我们展示了如何对类型为 **bert-base-cased** 的BERT模型进行基准测试：

<frameworkcontent>
<pt>
```py
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

>>> args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
>>> benchmark = PyTorchBenchmark(args)
```
</pt>
<tf>
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

>>> args = TensorFlowBenchmarkArguments(
...     models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> benchmark = TensorFlowBenchmark(args)
```
</tf>
</frameworkcontent>

在这里，基准测试的参数数据类接受了三个主要的参数，即 `models`、`batch_sizes` 和`sequence_lengths`。其中，`models` 是必需的参数，它期望一个来自[模型库](https://huggingface.co/models)的模型标识符列表。`batch_sizes` 和 `sequence_lengths` 是列表类型的参数，定义了进行基准测试时 `input_ids` 的批量大小和序列长度。

这些是基准测试数据类中可以配置的一些主要参数。除此之外，基准测试数据类中还可以配置很多其他参数。如需要查看更详细的配置参数，可以直接查看以下文件：

* `src/transformers/benchmark/benchmark_args_utils.py`
* `src/transformers/benchmark/benchmark_args.py`（针对 PyTorch）
* `src/transformers/benchmark/benchmark_args_tf.py`（针对 TensorFlow）
  
另外，您还可以通过在根目录下运行以下命令，查看针对 PyTorch 和 TensorFlow 的所有可配置参数的描述列表：
``` bash python examples/pytorch/benchmarking/run_benchmark.py --help ```
这些命令将列出所有可以配置的参数。有了这些工具的帮助，您可以更加灵活地进行基准测试。


<!-- Here, three arguments are given to the benchmark argument data classes, namely `models`, `batch_sizes`, and
`sequence_lengths`. The argument `models` is required and expects a `list` of model identifiers from the
[model hub](https://huggingface.co/models) The `list` arguments `batch_sizes` and `sequence_lengths` define
the size of the `input_ids` on which the model is benchmarked. There are many more parameters that can be configured
via the benchmark argument data classes. For more detail on these one can either directly consult the files
`src/transformers/benchmark/benchmark_args_utils.py`, `src/transformers/benchmark/benchmark_args.py` (for PyTorch)
and `src/transformers/benchmark/benchmark_args_tf.py` (for Tensorflow). Alternatively, running the following shell
commands from root will print out a descriptive list of all configurable parameters for PyTorch and Tensorflow
respectively. -->

<frameworkcontent>
<pt>
<!-- ```bash
python examples/pytorch/benchmarking/run_benchmark.py --help
``` -->

<!-- An instantiated benchmark object can then simply be run by calling `benchmark.run()`. -->

以下代码通过`PyTorchBenchmarkArguments`设置模型批处理大小和序列长度，然后调用`benchmark.run()`执行基准测试。

```py
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.006     
google-bert/bert-base-uncased          8               32            0.006     
google-bert/bert-base-uncased          8              128            0.018     
google-bert/bert-base-uncased          8              512            0.088     
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1227
google-bert/bert-base-uncased          8               32            1281
google-bert/bert-base-uncased          8              128            1307
google-bert/bert-base-uncased          8              512            1539
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
```
</pt>
<tf>
```bash
python examples/tensorflow/benchmarking/run_benchmark_tf.py --help
```

<!-- An instantiated benchmark object can then simply be run by calling `benchmark.run()`. -->
接下来，只需要调用 `benchmark.run()` 就能轻松运行已经实例化的基准测试对象。

```py
>>> results = benchmark.run()
>>> print(results)
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.005
google-bert/bert-base-uncased          8               32            0.008
google-bert/bert-base-uncased          8              128            0.022
google-bert/bert-base-uncased          8              512            0.105
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1330
google-bert/bert-base-uncased          8               32            1330
google-bert/bert-base-uncased          8              128            1330
google-bert/bert-base-uncased          8              512            1770
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
```
</tf>
</frameworkcontent>

<!-- By default, the _time_ and the _required memory_ for _inference_ are benchmarked. In the example output above the first
two sections show the result corresponding to _inference time_ and _inference memory_. In addition, all relevant
information about the computing environment, _e.g._ the GPU type, the system, the library versions, etc... are printed
out in the third section under _ENVIRONMENT INFORMATION_. This information can optionally be saved in a _.csv_ file
when adding the argument `save_to_csv=True` to [`PyTorchBenchmarkArguments`] and
[`TensorFlowBenchmarkArguments`] respectively. In this case, every section is saved in a separate
_.csv_ file. The path to each _.csv_ file can optionally be defined via the argument data classes. -->


在一般情况下，基准测试会测量推理（inference）的**时间**和**所需内存**。在上面的示例输出中，前两部分显示了与**推理时间**和**推理内存**对应的结果。与此同时，关于计算环境的所有相关信息（例如 GPU 类型、系统、库版本等）会在第三部分的**环境信息**中打印出来。你可以通过在 [`PyTorchBenchmarkArguments`] 和 [`TensorFlowBenchmarkArguments`] 中添加 `save_to_csv=True`参数，将这些信息保存到一个 .csv 文件中。在这种情况下，每一部分的信息会分别保存在不同的 .csv 文件中。每个 .csv 文件的路径也可以通过参数数据类进行定义。

<!-- Instead of benchmarking pre-trained models via their model identifier, _e.g._ `google-bert/bert-base-uncased`, the user can
alternatively benchmark an arbitrary configuration of any available model class. In this case, a `list` of
configurations must be inserted with the benchmark args as follows. -->

您可以选择不通过预训练模型的模型标识符（如 `google-bert/bert-base-uncased`）进行基准测试，而是对任何可用模型类的任意配置进行基准测试。在这种情况下，我们必须将一系列配置与基准测试参数一起传入，方法如下：

<frameworkcontent>
<pt>
```py
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

>>> args = PyTorchBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
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
```
</pt>
<tf>
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

>>> args = TensorFlowBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
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
```
</tf>
</frameworkcontent>

<!-- Again, _inference time_ and _required memory_ for _inference_ are measured, but this time for customized configurations
of the `BertModel` class. This feature can especially be helpful when deciding for which configuration the model
should be trained. -->

重新测量的是 **推理时间** 和 **推理所需内存**，不过这次是针对 `BertModel` 类的自定义配置进行基准测试。这个功能在决定模型应该使用哪种配置进行训练时尤其有用。


## 基准测试的推荐策略
本节列出了一些在对模型进行基准测试时比较推荐的策略：

* 目前，该模块只支持单设备基准测试。在进行 GPU 基准测试时，建议用户通过设置 `CUDA_VISIBLE_DEVICES` 环境变量来指定代码应在哪个设备上运行，例如在运行代码前执行 `export CUDA_VISIBLE_DEVICES=0`。
* `no_multi_processing` 选项仅应在测试和调试时设置为 `True`。为了确保内存测量的准确性，建议将每个内存基准测试单独运行在一个进程中，并确保 `no_multi_processing` 设置为 `True`。
* 当您分享模型基准测试结果时，应始终提供环境信息。由于 GPU 设备、库版本等之间可能存在较大差异，单独的基准测试结果对社区的帮助有限。

<!-- ## Benchmark best practices

This section lists a couple of best practices one should be aware of when benchmarking a model.

- Currently, only single device benchmarking is supported. When benchmarking on GPU, it is recommended that the user
  specifies on which device the code should be run by setting the `CUDA_VISIBLE_DEVICES` environment variable in the
  shell, _e.g._ `export CUDA_VISIBLE_DEVICES=0` before running the code.
- The option `no_multi_processing` should only be set to `True` for testing and debugging. To ensure accurate
  memory measurement it is recommended to run each memory benchmark in a separate process by making sure
  `no_multi_processing` is set to `True`.
- One should always state the environment information when sharing the results of a model benchmark. Results can vary
  heavily between different GPU devices, library versions, etc., as a consequence, benchmark results on their own are not very
  useful for the community. -->

## 分享您的基准测试结果

以前，所有可用的核心模型（当时有10个）都已针对 **推理时间** 进行基准测试，涵盖了多种不同的设置：使用 PyTorch（包不包含 TorchScript），使用 TensorFlow（包不包含 XLA）。所有这些测试都在 CPU（除了 TensorFlow XLA）和 GPU 上进行。

这种方法的详细信息可以在 [这篇博客](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2) 中找到，测试结果可以在 [这里](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing) 查看。


您可以借助新的 **基准测试** 工具比以往任何时候都更容易地分享您的基准测试结果！

- [PyTorch 基准测试结果](https://github.com/huggingface/transformers/tree/main/examples/pytorch/benchmarking/README.md)
- [TensorFlow 基准测试结果](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/benchmarking/README.md)


<!-- ## Sharing your benchmark

Previously all available core models (10 at the time) have been benchmarked for _inference time_, across many different
settings: using PyTorch, with and without TorchScript, using TensorFlow, with and without XLA. All of those tests were
done across CPUs (except for TensorFlow XLA) and GPUs.

The approach is detailed in the [following blogpost](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2) and the results are
available [here](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing).

With the new _benchmark_ tools, it is easier than ever to share your benchmark results with the community

- [PyTorch Benchmarking Results](https://github.com/huggingface/transformers/tree/main/examples/pytorch/benchmarking/README.md).
- [TensorFlow Benchmarking Results](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/benchmarking/README.md). -->
