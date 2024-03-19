<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer
在Transformers 库中，[`Trainer`] 提供了一个用于 PyTorch 模型的训练和评估完整训练器。您只需传递必要的训练部件（模型、tokenizer、数据集、评价函数、训练超参数等），[`Trainer`] 类会处理剩下的事情。这使得模型训练变得更加简易，而无需每次手动编写自己的训练循环。与此同时，[`Trainer`] 是非常灵活的，并提供了大量的训练选项，因此您可以根据自己的训练需求进行自定义。

<Tip>

除了 [`Trainer`] 类之外，Transformers 还提供了 [`Seq2SeqTrainer`] 类，用于序列到序列的任务，比如翻译或摘要。还有来自[TRL](https://hf.co/docs/trl) 库的 [`~trl.SFTTrainer`] 类，它封装了 [`Trainer`] 类，并进行了优化，用于训练 Llama-2 和 Mistral 等自回归技术的语言模型。[`~trl.SFTTrainer`] 还支持其他特性，如序列打包、LoRA、量化和 DeepSpeed，可有效扩展到任何模型大小。

<br>

随时查看其他 [`Trainer`] 类型的[API reference](./main_classes/trainer)参考，以了解何时使用哪种类型。总体而言，[`Trainer`] 是最通用的选择，并适用于广泛的任务。[`Seq2SeqTrainer`] 适用于序列到序列的任务，[`~trl.SFTTrainer`] 适用于训练语言模型。

</Tip>

在开始之前，请确保已安装 [Accelerate](https://hf.co/docs/accelerate) - 一个用于在分布式环境中启用和运行 PyTorch 训练的库。

```bash
pip install accelerate

# upgrade
pip install accelerate --upgrade
```

本指南提供了 [`Trainer`] 类的概述。

## 基本用法
[`Trainer`] 包括您在基本训练循环中的所有代码：

1. 执行训练步骤以计算损失
2. 使用 [`~accelerate.Accelerator.backward`] 方法计算梯度
3. 基于梯度更新权重
4. 重复此过程，直到达到预定的轮数

[`Trainer`] 类将所有这些代码抽象出来，因此您无需每次手动编写训练循环，或者如果您刚开始使用 PyTorch 进行训练并不知道该怎么做。您只需提供用于训练所需的基本组件（如模型和数据集），[`Trainer`] 类会处理其他一切。

如果您想指定任何训练选项或超参数，可以在 [`TrainingArguments`] 类中找到它们。例如，让我们定义将模型保存在`output_dir`中，并在训练后将模型推送到Hub，使用`push_to_hub=True`。

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
```

将 `training_args`与模型、数据集、用于预处理数据集的内容（根据您的数据类型，可能是tokenizer、feature extractor或image processor）、数据收集器，以及用于计算训练期间想要跟踪的指标的函数一起传递给 [`Trainer`]。

最后，调用 [`~Trainer.train`] 开始训练！

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### checkpoints

[`Trainer`]类将您的模型checkpoints保存到[`TrainingArguments`]中指定的`output_dir`参数所指定的目录中。您会发现checkpoints保存在一个`checkpoint-000`子文件夹中，其中末尾的数字对应训练步骤。保存checkpoints对于稍后恢复训练非常有用。

```py
# 从最新的checkpoint恢复训练
trainer.train(resume_from_checkpoint=True)

# 从输出目录中保存的特定checkpoint恢复
trainer.train(resume_from_checkpoint="your-model/checkpoint-1000")
```

您可以通过在[`TrainingArguments`]中设置`push_to_hub=True`将您的checkpoints（默认情况下不保存优化器状态）保存到Hub以进行提交和推送。可以通过设置[`hub_strategy`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_strategy)参数决定如何保存checkpoints：

* `hub_strategy="checkpoint"` 将最新的checkpoints推送到名为 "last-checkpoint" 的子文件夹中，从中您可以恢复训练
* `hub_strategy="all_checkpoints"` 将所有checkpoints推送到在`output_dir`中定义的目录（您将在模型存储库中的每个文件夹中看到一个checkpoints）

当您从checkpoints恢复训练时，[`Trainer`]尝试保持 Python、NumPy 和 PyTorch 的 RNG 状态与保存checkpoints时相同。但由于 PyTorch 具有各种非确定性的默认设置，RNG 状态不能保证是相同的。如果您想实现完全确定性，可以查看[Controlling sources of randomness](https://pytorch.org/docs/stable/notes/randomness#controlling-sources-of-randomness)指南，了解哪些设置能够使您的训练完全确定性。但请注意，通过使某些设置确定性化，训练可能会变慢。

## 自定义训练器
虽然[`Trainer`]类旨在易于访问和使用，但它也为更有冒险精神的用户提供了大量的可定制性。许多[`Trainer`]的方法可以被子类化和重写，以支持您想要的功能，而无需从头开始重写整个训练循环以适应它。这些方法包括：

* [`~Trainer.get_train_dataloader`] 创建一个训练数据加载器
* [`~Trainer.get_eval_dataloader`] 创建一个评估数据加载器
* [`~Trainer.get_test_dataloader`] 创建一个测试数据加载器
* [`~Trainer.log`] 记录观察训练的各种对象的信息
* [`~Trainer.create_optimizer_and_scheduler`] 如果没有在 `__init__`中定义，则创建一个优化器和学习率调度器；也可以分别指定为[`~Trainer.create_optimizer`]和[`~Trainer.create_scheduler`]
* [`~Trainer.compute_loss`] 计算一批训练输入的损失
* [`~Trainer.training_step`] 执行训练步骤
* [`~Trainer.prediction_step`] 执行预测和测试步骤
* [`~Trainer.evaluate`] 评估模型并返回评估指标
* [`~Trainer.predict`] 在测试集上进行预测（如果提供了标签，则会输出指标）

例如，如果您想自定义[' ~Trainer.compute_loss ']方法来使用加权损失。

```py
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

### 回调函数



另一个自定义[`Trainer`]的选项是使用[回调函数](callbacks)。回调函数不会改变训练循环中的任何内容。它们会检查训练循环状态，然后根据状态执行某些操作（提前停止、记录结果等）。换句话说，回调函数不能用于实现诸如自定义损失函数之类的东西，您需要对[`~Trainer.compute_loss`]方法进行子类化和重写。

例如，如果您想在10步后向训练循环添加一个早停回调。

```py
from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}
```
然后将其传递给[`Trainer`]的 `callback` 参数。

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callback=[EarlyStoppingCallback()],
)
```

## 日志记录
<Tip>

查看[`logging`](./main_classes/logging) API 参考以获取有关不同日志记录级别的更多信息。

</Tip>

[`Trainer`] 默认设置为`logging.INFO`，报告错误、警告和其他基本信息。在分布式环境中，[`Trainer`] 副本的设置为`logging.WARNING`，仅报告错误和警告。您可以使用[`TrainingArguments`]中的[`log_level`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level) 和 [`log_level_replica`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level_replica)参数来更改日志记录级别。

使用[`log_on_each_node`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.log_on_each_node)参数为每个节点配置日志记录级别设置，以确定是否在每个节点上使用日志记录级别或仅在主节点上使用。

<Tip>

[`Trainer`] 在 [`Trainer.__init__`] 方法中为每个节点单独设置日志记录级别，因此如果在创建[`Trainer`]对象之前使用其他 Transformers 功能，您可能希望更早设置。

</Tip>

例如，将主代码和模块设置为根据每个节点使用相同的日志级别:

```py
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

使用不同的`log_level` 和 `log_level_replica`组合来配置每个节点的日志记录内容。


<hfoptions id="logging">
<hfoption id="single node">

```bash
my_app.py ... --log_level warning --log_level_replica error
```

</hfoption>
<hfoption id="multi-node">
在多节点环境中添加参数`log_on_each_node 0`。

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0

# 设置仅报告错误
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

</hfoption>
</hfoptions>

## NEFTune
[`NEFTune`](https://hf.co/papers/2310.05914)是一种在训练过程中通过向嵌入向量添加噪声来提高性能的技术。要在[`Trainer`]中启用它，请在 [`TrainingArguments`]中设置 `neftune_noise_alpha`参数以控制添加了多少噪声。

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(..., neftune_noise_alpha=0.1)
trainer = Trainer(..., args=training_args)
```
在训练完成后，NEFTune 将被禁用，从而恢复原始嵌入层，以避免任何意外行为。

## Accelerate 和 Trainer

[`Trainer`] 类由 [Accelerate](https://hf.co/docs/accelerate) 提供支持，它是一个在分布式环境中轻松训练 PyTorch 模型的库，支持集成 [FullyShardedDataParallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) 和 [DeepSpeed](https://www.deepspeed.ai/) 等功能。

<Tip>

了解更多关于 FSDP 分片策略、CPU 卸载等内容，可以查看[Fully Sharded Data Parallel](fsdp)指南。

</Tip>

要在 [`Trainer`] 中使用 Accelerate，请运行[`accelerate.config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) 命令，为您的训练环境设置训练。该命令会创建一个 `config_file.yaml` ，在您启动训练脚本时将被使用。

例如，一些示例配置。

<hfoptions id="config">
<hfoption id="DistributedDataParallel">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
distributed_type: MULTI_GPU                                                                                                    
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0 #根据节点更改排名
main_process_ip: 192.168.20.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="FSDP">

```yml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed">

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/user/configs/ds_zero3_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed with Accelerate plugin">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
deepspeed_config:                                                                                                              
  gradient_accumulation_steps: 1
  gradient_clipping: 0.7
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
</hfoptions>

[accelerate_launch](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) 命令是在分布式系统上启用 Accelerate 和 [`Trainer`] 的训练脚本的推荐方式，您可以指定 `config_file.yaml` 中的参数。这个文件被保存到 Accelerate 缓存文件夹中，并在运行`accelerate_launch`时自动加载。

例如，要使用 FSDP 配置运行 [run_glue.py](https://github.com/huggingface/transformers/blob/f4db565b695582891e43a5e042e5d318e28f20b8/examples/pytorch/text-classification/run_glue.py#L4)训练脚本：


```bash
accelerate launch \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```
您也可以直接在命令行中指定来自`config_file.yaml`文件的参数：

```bash
accelerate launch --num_processes=2 \
    --use_fsdp \
    --mixed_precision=bf16 \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
    --fsdp_transformer_layer_cls_to_wrap="BertLayer" \
    --fsdp_sharding_strategy=1 \
    --fsdp_state_dict_type=FULL_STATE_DICT \
    ./examples/pytorch/text-classification/run_glue.py
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

查看启动您的 [Launching your Accelerate scripts](https://huggingface.co/docs/accelerate/basic_tutorials/launch)教程，了解有关`accelerate_launch`和自定义配置的更多信息。
