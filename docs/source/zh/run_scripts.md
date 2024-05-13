<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 使用脚本进行训练

除了 🤗 Transformers [notebooks](./notebooks)，还有示例脚本演示了如何使用[PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch)、[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow)或[JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax)训练模型以解决特定任务。

您还可以在这些示例中找到我们在[研究项目](https://github.com/huggingface/transformers/tree/main/examples/research_projects)和[遗留示例](https://github.com/huggingface/transformers/tree/main/examples/legacy)中使用过的脚本，这些脚本主要是由社区贡献的。这些脚本已不再被积极维护，需要使用特定版本的🤗 Transformers， 可能与库的最新版本不兼容。

示例脚本可能无法在初始配置下直接解决每个问题，您可能需要根据要解决的问题调整脚本。为了帮助您，大多数脚本都完全暴露了数据预处理的方式，允许您根据需要对其进行编辑。

如果您想在示例脚本中实现任何功能，请在[论坛](https://discuss.huggingface.co/)或[issue](https://github.com/huggingface/transformers/issues)上讨论，然后再提交Pull Request。虽然我们欢迎修复错误，但不太可能合并添加更多功能的Pull Request，因为这会降低可读性。

本指南将向您展示如何在[PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)和[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization)中运行示例摘要训练脚本。除非另有说明，否则所有示例都可以在两个框架中工作。

## 设置

要成功运行示例脚本的最新版本，您必须在新虚拟环境中**从源代码安装 🤗 Transformers**：

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

对于旧版本的示例脚本，请点击下面的切换按钮：

<details>
  <summary>老版本🤗 Transformers示例 </summary>
	<ul>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

然后切换您clone的 🤗 Transformers 仓到特定的版本，例如v3.5.1：

```bash
git checkout tags/v3.5.1
```

在安装了正确的库版本后，进入您选择的版本的`example`文件夹并安装例子要求的环境：

```bash
pip install -r requirements.txt
```

## 运行脚本

<frameworkcontent>
<pt>

示例脚本从🤗 [Datasets](https://huggingface.co/docs/datasets/)库下载并预处理数据集。然后，脚本通过[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)使用支持摘要任务的架构对数据集进行微调。以下示例展示了如何在[CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)数据集上微调[T5-small](https://huggingface.co/google-t5/t5-small)。由于T5模型的训练方式，它需要一个额外的`source_prefix`参数。这个提示让T5知道这是一个摘要任务。

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>

示例脚本从  🤗 [Datasets](https://huggingface.co/docs/datasets/) 库下载并预处理数据集。然后，脚本使用 Keras 在支持摘要的架构上微调数据集。以下示例展示了如何在 [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) 数据集上微调 [T5-small](https://huggingface.co/google-t5/t5-small)。T5 模型由于训练方式需要额外的 `source_prefix` 参数。这个提示让 T5 知道这是一个摘要任务。

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## 分布式训练和混合精度

[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) 支持分布式训练和混合精度，这意味着你也可以在脚本中使用它。要启用这两个功能，可以做如下设置：

- 添加 `fp16` 参数以启用混合精度。
- 使用 `nproc_per_node` 参数设置使用的GPU数量。


```bash
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

TensorFlow脚本使用[`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)进行分布式训练，您无需在训练脚本中添加任何其他参数。如果可用，TensorFlow脚本将默认使用多个GPU。

## 在TPU上运行脚本

<frameworkcontent>
<pt>

张量处理单元（TPUs）是专门设计用于加速性能的。PyTorch使用[XLA](https://www.tensorflow.org/xla)深度学习编译器支持TPU（更多细节请参见[这里](https://github.com/pytorch/xla/blob/master/README.md)）。要使用TPU，请启动`xla_spawn.py`脚本并使用`num_cores`参数设置要使用的TPU核心数量。

```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>

张量处理单元（TPUs）是专门设计用于加速性能的。TensorFlow脚本使用[`TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy)在TPU上进行训练。要使用TPU，请将TPU资源的名称传递给`tpu`参数。

```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## 基于🤗 Accelerate运行脚本

🤗 [Accelerate](https://huggingface.co/docs/accelerate) 是一个仅支持 PyTorch 的库，它提供了一种统一的方法来在不同类型的设置（仅 CPU、多个 GPU、多个TPU）上训练模型，同时保持对 PyTorch 训练循环的完全可见性。如果你还没有安装 🤗 Accelerate，请确保你已经安装了它：

> 注意：由于 Accelerate 正在快速发展，因此必须安装 git 版本的 accelerate 来运行脚本。

```bash
pip install git+https://github.com/huggingface/accelerate
```

你需要使用`run_summarization_no_trainer.py`脚本，而不是`run_summarization.py`脚本。🤗 Accelerate支持的脚本需要在文件夹中有一个`task_no_trainer.py`文件。首先运行以下命令以创建并保存配置文件：

```bash
accelerate config
```
检测您的设置以确保配置正确：

```bash
accelerate test
```

现在您可以开始训练模型了：

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

## 使用自定义数据集

摘要脚本支持自定义数据集，只要它们是CSV或JSON Line文件。当你使用自己的数据集时，需要指定一些额外的参数：
- `train_file` 和 `validation_file` 分别指定您的训练和验证文件的路径。
- `text_column` 是输入要进行摘要的文本。
- `summary_column` 是目标输出的文本。

使用自定义数据集的摘要脚本看起来是这样的：


```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

## 测试脚本

通常，在提交整个数据集之前，最好先在较少的数据集示例上运行脚本，以确保一切按预期工作,因为完整数据集的处理可能需要花费几个小时的时间。使用以下参数将数据集截断为最大样本数：

- `max_train_samples`
- `max_eval_samples`
- `max_predict_samples`


```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

并非所有示例脚本都支持`max_predict_samples`参数。如果您不确定您的脚本是否支持此参数，请添加`-h`参数进行检查：

```bash
examples/pytorch/summarization/run_summarization.py -h
```

## 从checkpoint恢复训练

另一个有用的选项是从之前的checkpoint恢复训练。这将确保在训练中断时，您可以从之前停止的地方继续进行，而无需重新开始。有两种方法可以从checkpoint恢复训练。

第一种方法使用`output_dir previous_output_dir`参数从存储在`output_dir`中的最新的checkpoint恢复训练。在这种情况下，您应该删除`overwrite_output_dir`：

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir previous_output_dir \
    --predict_with_generate
```

第二种方法使用`resume_from_checkpoint path_to_specific_checkpoint`参数从特定的checkpoint文件夹恢复训练。


```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate
```

## 分享模型

所有脚本都可以将您的最终模型上传到[Model Hub](https://huggingface.co/models)。在开始之前，请确保您已登录Hugging Face：

```bash
huggingface-cli login
```

然后，在脚本中添加`push_to_hub`参数。这个参数会创建一个带有您Hugging Face用户名和`output_dir`中指定的文件夹名称的仓库。

为了给您的仓库指定一个特定的名称，使用`push_to_hub_model_id`参数来添加它。该仓库将自动列出在您的命名空间下。

以下示例展示了如何上传具有特定仓库名称的模型：


```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```