<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的担保或条件，无论是明示还是暗示。有关许可证的详细信息，请参阅特定语言下的权限和限制。许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档构建器的特定语法（类似 MDX），可能无法在您的 Markdown 查看器中正确呈现。
-->

# 使用脚本进行训练

除了🤗 Transformers [notebooks](./noteboks/README) 之外，还有演示如何使用 [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch)、[TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow) 或 [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax) 训练模型的示例脚本。

您还将在我们的 [研究项目](https://github.com/huggingface/transformers/tree/main/examples/research_projects) 和 [旧示例](https://github.com/huggingface/transformers/tree/main/examples/legacy) 中找到我们使用的脚本，这些脚本主要由社区贡献。这些脚本没有被积极维护，并且需要与最新版本的库不兼容的特定版本的🤗 Transformers。

不希望示例脚本能够立即在每个问题上正常工作，并且您可能需要根据您要解决的问题调整脚本。为了帮助您，大多数脚本完全暴露了数据的预处理方式，允许您根据需要进行编辑以适应您的用例。

如果您想在示例脚本中实现任何功能，请在提交 Pull Request 之前在 [论坛](https://discuss.huggingface.co/) 或 [问题](https://github.com/huggingface/transformers/issues) 中进行讨论。虽然我们欢迎修复错误，但我们不太可能合并添加更多功能但可读性降低的 Pull Request。

本指南将演示如何在 [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) 和 [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization) 中运行一个示例摘要训练脚本。除非另有说明，所有示例都预计可与这两个框架一起工作。

## 设置

要成功运行最新版本的示例脚本，您需要在新的虚拟环境中 **从源代码安装🤗 Transformers**：
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

对于旧版本的示例脚本，请单击下面的切换：
<details>  <summary> 🤗 Transformers 旧版本示例 </summary>	<ul>		<li> <a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples"> v4.5.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples"> v4.4.2 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples"> v4.3.3 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples"> v4.2.2 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples"> v4.1.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples"> v4.0.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples"> v3.5.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples"> v3.4.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples"> v3.3.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples"> v3.2.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples"> v3.1.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples"> v3.0.2 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples"> v2.11.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples"> v2.10.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples"> v2.9.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples"> v2.8.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples"> v2.7.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples"> v2.6.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples"> v2.5.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples"> v2.4.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples"> v2.3.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples"> v2.2.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples"> v2.1.1 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples"> v2.0.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples"> v1.2.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples"> v1.1.0 </a> </li>		<li> <a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples"> v1.0.0 </a> </li>	</ul> </details>
然后将您当前的🤗 Transformers 克隆切换到特定版本，例如 v3.5.1：
```bash
git checkout tags/v3.5.1
```

设置正确的库版本后，转到您选择的示例文件夹并安装示例特定要求：
```bash
pip install -r requirements.txt
```

## 运行脚本

<frameworkcontent> 
<pt> 
 示例脚本从🤗 [数据集](https://huggingface.co/docs/datasets/) 库下载和预处理数据集。然后，脚本使用 [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) 在支持摘要的架构上对数据集进行微调。以下示例展示了如何在 [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) 数据集上对 [T5-small](https://huggingface.co/t5-small) 进行微调。由于 T5 是如何训练的，T5 模型需要额外的 `source_prefix` 参数。此提示让 T5 知道这是一个摘要任务。
```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
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

示例脚本从🤗 [数据集](https://huggingface.co/docs/datasets/) 库下载和预处理数据集。然后，脚本在支持摘要的架构上使用 Keras 对数据集进行微调。以下示例展示了如何在 [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) 数据集上对 [T5-small](https://huggingface.co/t5-small) 进行微调。由于 T5 是如何训练的，T5 模型需要额外的 `source_prefix` 参数。此提示让 T5 知道这是一个摘要任务。

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path t5-small \
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

[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) 支持分布式训练和混合精度，这意味着您也可以在脚本中使用它。

要启用这两个功能：
- 添加 `fp16` 参数以启用混合精度。- 使用 `nproc_per_node` 参数设置要使用的 GPU 数量。

```bash
python -m torch.distributed.launch \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path t5-small \
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

TensorFlow 脚本使用 [`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) 进行分布式训练，您不需要向训练脚本添加任何额外的参数。如果可用，TensorFlow 脚本将默认使用多个 GPU。

## 在 TPU 上运行脚本

<frameworkcontent> 
<pt> 
 Tensor Processing Units (TPUs)专为提高性能而设计。PyTorch 使用 [XLA](https://www.tensorflow.org/xla) 深度学习编译器来支持 TPU（更多细节请参见 [此处](https://github.com/pytorch/xla/blob/master/README.md)）。要使用 TPU，请启动 `xla_spawn.py` 脚本，并使用 `num_cores` 参数设置要使用的 TPU 核心数量。
```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path t5-small \
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

Tensor Processing Units (TPUs)专为提高性能而设计。TensorFlow 脚本使用 [`TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy) 在 TPU 上进行训练。

要使用 TPU，请将 TPU 资源的名称传递给 `tpu` 参数。

```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path t5-small \
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


## 使用🤗 Accelerate 运行脚本

🤗 [Accelerate](https://huggingface.co/docs/accelerate) 是一个仅适用于 PyTorch 的库，它提供了一种在多种类型的设置（仅 CPU、多个 GPU、TPU）上训练模型的统一方法，同时完全可见 PyTorch 训练循环。

如果尚未安装🤗 Accelerate，请确保安装了它：
> 注意：由于 Accelerate 正在快速开发中，必须安装加速版本的 git 才能运行脚本。

```bash
pip install git+https://github.com/huggingface/accelerate
```

您需要使用 `run_summarization_no_trainer.py` 脚本而不是 `run_summarization.py` 脚本。支持🤗 Accelerate 的脚本将在文件夹中有一个 `task_no_trainer.py` 文件。首先运行以下命令以创建并保存配置文件：
```bash
accelerate config
```

使用以下参数测试您的设置以确保配置正确：
```bash
accelerate test
```

现在，您可以启动训练了：
```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

## 使用自定义数据集
摘要脚本支持自定义数据集，只要它们是 CSV 或 JSON Line 文件即可。当使用自己的数据集时，您需要指定几个额外的参数：
- `train_file` 和 `validation_file` 指定训练和验证文件的路径。- `text_column` 是要摘要的输入文本。- `summary_column` 是要输出的目标文本。
使用自定义数据集的摘要脚本如下所示：
```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
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

在提交整个数据集之前，将脚本应用于较少数量的数据集示例通常是一个好主意，以确保一切正常运行。使用以下参数将数据集截断为最大样本数：

- `max_train_samples`- `max_eval_samples`- `max_predict_samples`
```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
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

并非所有示例脚本都支持 `max_predict_samples` 参数。如果不确定脚本是否支持此参数，请添加 `-h` 参数进行检查：
```bash
examples/pytorch/summarization/run_summarization.py -h
```

## 从检查点恢复训练

还有一个有用的选项是启用从先前检查点恢复训练的功能。这将确保您可以在中断训练后继续之前的进度，而无需重新开始。有两种方法可以从检查点恢复训练。

第一种方法使用 `output_dir previous_output_dir` 参数从 `output_dir` 中存储的最新检查点恢复训练

。在这种情况下，您应该删除 `overwrite_output_dir`：
```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
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

第二种方法使用 `resume_from_checkpoint path_to_specific_checkpoint` 参数从特定检查点文件夹恢复训练。
```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
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

## 共享您的模型

所有脚本都可以将最终模型上传到 [Model Hub](https://huggingface.co/models)。确保您已登录 Hugging Face：
```bash
huggingface-cli login
```

然后将 `push_to_hub` 参数添加到脚本中。此参数将使用您的 Hugging Face 用户名和 `output_dir` 中指定的文件夹名称创建一个存储库。

要为存储库指定特定名称，请使用 `push_to_hub_model_id` 参数进行添加。存储库将自动列在您的命名空间下。

以下示例显示了如何上传具有特定存储库名称的模型：

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path t5-small \
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