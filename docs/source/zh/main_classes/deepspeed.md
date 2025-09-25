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

# DeepSpeed集成

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)实现了[ZeRO论文](https://huggingface.co/papers/1910.02054)中描述的所有内容。目前，它提供对以下功能的全面支持：

1. 优化器状态分区（ZeRO stage 1）
2. 梯度分区（ZeRO stage 2）
3. 参数分区（ZeRO stage 3）
4. 自定义混合精度训练处理
5. 一系列基于CUDA扩展的快速优化器
6. ZeRO-Offload 到 CPU 和 NVMe

ZeRO-Offload有其自己的专门论文：[ZeRO-Offload: Democratizing Billion-Scale Model Training](https://huggingface.co/papers/2101.06840)。而NVMe支持在论文[ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://huggingface.co/papers/2104.07857)中进行了描述。

DeepSpeed ZeRO-2主要用于训练，因为它的特性对推理没有用处。

DeepSpeed ZeRO-3也可以用于推理，因为它允许将单个GPU无法加载的大模型加载到多个GPU上。

🤗 Transformers通过以下两种方式集成了[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)：

1. 通过[`Trainer`]集成核心的DeepSpeed功能。这是一种“为您完成一切”式的集成 - 您只需提供自定义配置文件或使用我们的模板配置文件。本文档的大部分内容都集中在这个功能上。
2. 如果您不使用[`Trainer`]并希望在自己的Trainer中集成DeepSpeed，那么像`from_pretrained`和`from_config`这样的核心功能函数将包括ZeRO stage 3及以上的DeepSpeed的基础部分，如`zero.Init`。要利用此功能，请阅读有关[非Trainer DeepSpeed集成](#nontrainer-deepspeed-integration)的文档。

集成的内容：

训练：

1. DeepSpeed ZeRO训练支持完整的ZeRO stages 1、2和3，以及ZeRO-Infinity（CPU和NVMe offload）。

推理：

1. DeepSpeed ZeRO推理支持ZeRO stage 3和ZeRO-Infinity。它使用与训练相同的ZeRO协议，但不使用优化器和学习率调度器，只有stage 3与推理相关。更多详细信息请参阅：[zero-inference](#zero-inference)。

此外还有DeepSpeed推理 - 这是一种完全不同的技术，它使用张量并行而不是ZeRO（即将推出）。


<a id='deepspeed-trainer-integration'></a>


## Trainer DeepSpeed 集成


<a id='deepspeed-installation'></a>

### 安装

通过pypi安装库：


```bash
pip install deepspeed
```

或通过 `transformers` 的 `extras`安装：

```bash
pip install transformers[deepspeed]
```

或在 [DeepSpeed 的 GitHub 页面](https://github.com/deepspeedai/DeepSpeed#installation) 和
[高级安装](https://www.deepspeed.ai/tutorials/advanced-install/) 中查找更多详细信息。

如果构建过程中仍然遇到问题，请首先确保阅读 [CUDA 扩展安装注意事项](trainer#cuda-extension-installation-notes)。

如果您没有预先构建扩展而是在运行时构建它们，而且您尝试了以上所有解决方案都无效，下一步可以尝试在安装之前预先构建扩展。

进行 DeepSpeed 的本地构建：


```bash
git clone https://github.com/deepspeedai/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

如果您打算使用 NVMe offload，您还需要在上述说明中添加 `DS_BUILD_AIO=1`（并且还需要在系统范围内安装 *libaio-dev*）。

编辑 `TORCH_CUDA_ARCH_LIST` 以插入您打算使用的 GPU 卡的架构代码。假设您的所有卡都是相同的，您可以通过以下方式获取架构：

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

因此，如果您得到 `8, 6`，则使用 `TORCH_CUDA_ARCH_LIST="8.6"`。如果您有多个不同的卡，您可以像这样列出所有卡 `TORCH_CUDA_ARCH_LIST="6.1;8.6"`。

如果您需要在多台机器上使用相同的设置，请创建一个二进制 wheel：


```bash
git clone https://github.com/deepspeedai/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

它将生成类似于 `dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl` 的文件，现在您可以在本地或任何其他机器上安装它，如 `pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`。

再次提醒确保调整 `TORCH_CUDA_ARCH_LIST` 以匹配目标架构。

您可以在[这里](https://developer.nvidia.com/cuda-gpus)找到完整的 NVIDIA GPU 列表及其对应的 **计算能力**（与此上下文中的架构相同）。

您可以使用以下命令检查 PyTorch 构建时使用的架构：


```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

以下是如何查找已安装 GPU 中的一张卡的架构。例如，对于 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))"
```

如果输出结果如下：

```bash
_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)
```

然后您就知道这张卡的架构是 `8.6`。

您也可以完全省略 `TORCH_CUDA_ARCH_LIST`，然后构建程序将自动查询构建所在的 GPU 的架构。这可能与目标机器上的 GPU 不匹配，因此最好明确指定所需的架构。

如果尝试了所有建议的方法仍然遇到构建问题，请继续在 [Deepspeed](https://github.com/deepspeedai/DeepSpeed/issues)的 GitHub Issue 上提交问题。


<a id='deepspeed-multi-gpu'></a>

### 多GPU启用

为了启用DeepSpeed 集成，调整 [`Trainer`] 的命令行参数，添加一个新的参数 `--deepspeed ds_config.json`，其中 `ds_config.json` 是 DeepSpeed 配置文件，如文档 [这里](https://www.deepspeed.ai/docs/config-json/) 所述。文件命名由您决定。
建议使用 DeepSpeed 的 `add_config_arguments` 程序将必要的命令行参数添加到您的代码中。
有关更多信息，请参阅 [DeepSpeed 的参数解析](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) 文档。

在这里，您可以使用您喜欢的启动器。您可以继续使用 PyTorch 启动器：


```bash
torch.distributed.run --nproc_per_node=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

或使用由 `deepspeed` 提供的启动器：


```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json
```


正如您所见，这两个启动器的参数不同，但对于大多数需求，任何一个都可以满足工作需求。有关如何配置各个节点和 GPU 的完整详细信息，请查看 [此处](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。

当您使用 `deepspeed` 启动器并且希望使用所有可用的 GPU 时，您可以简单地省略 `--num_gpus` 标志。

以下是在 DeepSpeed 中启用使用所有可用 GPU情况下， 运行 `run_translation.py` 的示例：


```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

请注意，在 DeepSpeed 文档中，您可能会看到 `--deepspeed --deepspeed_config ds_config.json` - 即两个与 DeepSpeed 相关的参数，但为简单起见，并且因为已经有很多参数要处理，我们将两者合并为一个单一参数。

有关一些实际使用示例，请参阅 [此帖](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400)。



<a id='deepspeed-one-gpu'></a>

### 单GPU启用

要使用一张 GPU 启用 DeepSpeed，调整 [`Trainer`] 的命令行参数如下：


```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

这与多 GPU 的情况几乎相同，但在这里我们通过 `--num_gpus=1` 明确告诉 DeepSpeed 仅使用一张 GPU。默认情况下，DeepSpeed 启用给定节点上可以看到的所有 GPU。如果您一开始只有一张 GPU，那么您不需要这个参数。以下 [文档](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) 讨论了启动器的选项。

为什么要在仅使用一张 GPU 的情况下使用 DeepSpeed 呢？

1. 它具有 ZeRO-offload 功能，可以将一些计算和内存委托给主机的 CPU 和 内存，从而为模型的需求保留更多 GPU 资源 - 例如更大的批处理大小，或启用正常情况下无法容纳的非常大模型。
2. 它提供了智能的 GPU 内存管理系统，最小化内存碎片，这再次允许您容纳更大的模型和数据批次。

虽然接下来我们将详细讨论配置，但在单个 GPU 上通过 DeepSpeed 实现巨大性能提升的关键是在配置文件中至少有以下配置：


```json
{
  "zero_optimization": {
     "stage": 2,
     "offload_optimizer": {
         "device": "cpu",
         "pin_memory": true
     },
     "allgather_partitions": true,
     "allgather_bucket_size": 2e8,
     "reduce_scatter": true,
     "reduce_bucket_size": 2e8,
     "overlap_comm": true,
     "contiguous_gradients": true
  }
}
```

这会启用`optimizer offload`和一些其他重要功能。您可以尝试不同的buffer大小，有关详细信息，请参见下面的讨论。

关于这种启用类型的实际使用示例，请参阅 [此帖](https://github.com/huggingface/transformers/issues/8771#issuecomment-759176685)。

您还可以尝试使用本文后面进一步解释的支持`CPU 和 NVMe offload`功能的ZeRO-3 。


<!--- TODO: Benchmark whether we can get better performance out of ZeRO-3 vs. ZeRO-2 on a single GPU, and then
recommend ZeRO-3 config as starting one. -->

注意：

- 如果您需要在特定的 GPU 上运行，而不是 GPU 0，则无法使用 `CUDA_VISIBLE_DEVICES` 来限制可用 GPU 的可见范围。相反，您必须使用以下语法：

  ```bash
  deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...
  ```

  在这个例子中，我们告诉 DeepSpeed 使用 GPU 1（第二个 GPU）。



<a id='deepspeed-multi-node'></a>

### 多节点启用

这一部分的信息不仅适用于 DeepSpeed 集成，也适用于任何多节点程序。但 DeepSpeed 提供了一个比其他启动器更易于使用的 `deepspeed` 启动器，除非您在 SLURM 环境中。

在本节，让我们假设您有两个节点，每个节点有 8 张 GPU。您可以通过 `ssh hostname1` 访问第一个节点，通过 `ssh hostname2` 访问第二个节点，两者必须能够在本地通过 ssh 无密码方式相互访问。当然，您需要将这些主机（节点）名称重命名为您实际使用的主机名称。


#### torch.distributed.run启动器


例如，要使用 `torch.distributed.run`，您可以执行以下操作：

```bash
python -m torch.distributed.run --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

您必须 ssh 到每个节点，并在每个节点上运行相同的命令！不用担心，启动器会等待两个节点同步完成。

有关更多信息，请参阅 [torchrun](https://pytorch.org/docs/stable/elastic/run.html)。顺便说一下，这也是替代了几个 PyTorch 版本前的 `torch.distributed.launch` 的启动器。


#### deepspeed启动器

要改用 `deepspeed` 启动器，首先需要创建一个 `hostfile` 文件：

```
hostname1 slots=8
hostname2 slots=8
```
然后，您可以这样启动：

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

与 `torch.distributed.run` 启动器不同，`deepspeed` 将自动在两个节点上启动此命令！

更多信息，请参阅[资源配置（多节点）](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。


#### 在 SLURM 环境中启动

在 SLURM 环境中，可以采用以下方法。以下是一个 SLURM 脚本 `launch.slurm`，您需要根据您的具体 SLURM 环境进行调整。

```bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

剩下的就是运行它：

```bash
sbatch launch.slurm
```

`srun` 将负责在所有节点上同时启动程序。


#### 使用非共享文件系统

默认情况下，DeepSpeed 假定多节点环境使用共享存储。如果不是这种情况，每个节点只能看到本地文件系统，你需要调整配置文件，包含一个 [`checkpoint` 部分](https://www.deepspeed.ai/docs/config-json/#checkpoint-options)并设置如下选项：

```json
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

或者，你还可以使用 [`Trainer`] 的 `--save_on_each_node` 参数，上述配置将自动添加。


<a id='deepspeed-notebook'></a>

### 在Notebooks启用

在将`notebook cells`作为脚本运行的情况下，问题在于没有正常的 `deepspeed` 启动器可依赖，因此在某些设置下，我们必须仿真运行它。

如果您只使用一个 GPU，以下是如何调整notebook中的训练代码以使用 DeepSpeed。

```python
# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Now proceed as normal, plus pass the deepspeed config file
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

注意：`...` 代表您传递给函数的正常参数。

如果要使用多于一个 GPU，您必须在 DeepSpeed 中使用多进程环境。也就是说，您必须使用专门的启动器来实现这一目的，而不能通过仿真本节开头呈现的分布式环境来完成。

如果想要在notebook中动态创建配置文件并保存在当前目录，您可以在一个专用的cell中使用：

```python no-style
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

如果训练脚本在一个普通文件中而不是在notebook cells中，您可以通过笔记本中的 shell 正常启动 `deepspeed`。例如，要使用 `run_translation.py`，您可以这样启动：

```python no-style
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

或者使用 `%%bash` 魔术命令，您可以编写多行代码，用于运行 shell 程序：

```python no-style
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

在这种情况下，您不需要本节开头呈现的任何代码。

注意：虽然 `%%bash` 魔术命令很方便，但目前它会缓冲输出，因此在进程完成之前您看不到日志。


<a id='deepspeed-config'></a>

### 配置

有关可以在 DeepSpeed 配置文件中使用的完整配置选项的详细指南，请参阅[以下文档](https://www.deepspeed.ai/docs/config-json/)。

您可以在 [DeepSpeedExamples 仓库](https://github.com/deepspeedai/DeepSpeedExamples)中找到解决各种实际需求的数十个 DeepSpeed 配置示例。

```bash
git clone https://github.com/deepspeedai/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
```

延续上面的代码，假设您要配置 Lamb 优化器。那么您可以通过以下方式在示例的 `.json` 文件中进行搜索：

```bash
grep -i Lamb $(find . -name '*json')
```

还可以在[主仓](https://github.com/deepspeedai/DeepSpeed)中找到更多示例。

在使用 DeepSpeed 时，您总是需要提供一个 DeepSpeed 配置文件，但是一些配置参数必须通过命令行进行配置。您将在本指南的剩余章节找到这些细微差别。

为了了解 DeepSpeed 配置文件，这里有一个激活 ZeRO stage 2 功能的示例，包括优化器状态的 CPU offload，使用 `AdamW` 优化器和 `WarmupLR`  调度器，并且如果传递了 `--fp16` 参数将启用混合精度训练：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}
```

当您执行程序时，DeepSpeed 将把它从 [`Trainer`] 收到的配置日志输出到console，因此您可以看到传递给它的最终配置。



<a id='deepspeed-config-passing'></a>

### 传递配置

正如本文档讨论的那样，通常将 DeepSpeed 配置作为指向 JSON 文件的路径传递，但如果您没有使用命令行界面配置训练，而是通过 [`TrainingArguments`] 实例化 [`Trainer`]，那么对于 `deepspeed` 参数，你可以传递一个嵌套的 `dict`。这使您能够即时创建配置，而无需在将其传递给 [`TrainingArguments`] 之前将其写入文件系统。

总结起来，您可以这样做：

```python
TrainingArguments(..., deepspeed="/path/to/ds_config.json")
```

或者:

```python
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
TrainingArguments(..., deepspeed=ds_config_dict)
```

<a id='deepspeed-config-shared'></a>

### 共享配置


> [!WARNING]
> 这一部分是必读的。

一些配置值对于 [`Trainer`] 和 DeepSpeed 正常运行都是必需的，因此，为了防止定义冲突及导致的难以检测的错误，我们选择通过 [`Trainer`] 命令行参数配置这些值。

此外，一些配置值是基于模型的配置自动派生的，因此，与其记住手动调整多个值，最好让 [`Trainer`] 为您做大部分配置。

因此，在本指南的其余部分，您将找到一个特殊的配置值：`auto`，当设置时将自动将参数替换为正确或最有效的值。请随意选择忽略此建议或显式设置该值，在这种情况下，请务必确保 [`Trainer`] 参数和 DeepSpeed 配置保持一致。例如，您是否使用相同的学习率、批量大小或梯度累积设置？如果这些不匹配，训练可能以非常难以检测的方式失败。请重视该警告。

还有一些参数是仅适用于 DeepSpeed 的，并且这些参数必须手动设置以适应您的需求。

在您自己的程序中，如果您想要作为主动修改 DeepSpeed 配置并以此配置 [`TrainingArguments`]，您还可以使用以下方法。步骤如下：

1. 创建或加载要用作主配置的 DeepSpeed 配置
2. 根据这些参数值创建 [`TrainingArguments`] 对象

请注意，一些值，比如 `scheduler.params.total_num_steps`，是在 [`Trainer`] 的 `train` 过程中计算的，但当然您也可以自己计算这些值。


<a id='deepspeed-zero'></a>

### ZeRO

[Zero Redundancy Optimizer (ZeRO)](https://www.deepspeed.ai/tutorials/zero/) 是 DeepSpeed 的工作核心。它支持3个不同级别（stages）的优化。Stage 1 对于扩展性来说不是很有趣，因此本文档重点关注Stage 2和Stage 3。Stage 3通过最新的 ZeRO-Infinity 进一步改进。你可以在 DeepSpeed 文档中找到更详细的信息。

配置文件的 `zero_optimization` 部分是最重要的部分（[文档](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)），因为在这里您定义了要启用哪些 ZeRO stages 以及如何配置它们。您可以在 DeepSpeed 文档中找到每个参数的解释。

这一部分必须通过 DeepSpeed 配置文件单独配置 - [`Trainer`] 不提供相应的命令行参数。

注意：目前 DeepSpeed 不验证参数名称，因此如果您拼错了任何参数，它将使用拼写错误的参数的默认设置。您可以观察 DeepSpeed 引擎启动日志消息，看看它将使用哪些值。

<a id='deepspeed-zero2-config'></a>

#### ZeRO-2 配置

以下是 ZeRO stage 2 的配置示例：

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    }
}
```

**性能调优：**

- 启用 `offload_optimizer` 应该减少 GPU 内存使用（需要 `"stage": 2`）。
- `"overlap_comm": true` 通过增加 GPU 内存使用来降低all-reduce 的延迟。 `overlap_comm` 使用了 `allgather_bucket_size` 和 `reduce_bucket_size` 值的4.5倍。因此，如果它们设置为 `5e8`，这将需要一个9GB的内存占用（`5e8 x 2Bytes x 2 x 4.5`）。因此，如果您的 GPU 内存为8GB或更小，为了避免出现OOM错误，您需要将这些参数减小到约 `2e8`，这将需要3.6GB。如果您的 GPU 容量更大，当您开始遇到OOM时，你可能也需要这样做。
- 当减小这些buffers时，您以更慢的通信速度来换取更多的 GPU 内存。buffers大小越小，通信速度越慢，GPU 可用于其他任务的内存就越多。因此，如果更大的批处理大小很重要，那么稍微减慢训练时间可能是一个很好的权衡。

此外，`deepspeed==0.4.4` 添加了一个新选项 `round_robin_gradients`，您可以通过以下方式启用：

```json
{
    "zero_optimization": {
        "round_robin_gradients": true
    }
}
```
这是一个用于 CPU offloading 的stage 2优化，通过细粒度梯度分区在 ranks 之间并行复制到 CPU 内存，从而实现了性能的提升。性能优势随着梯度累积步骤（在优化器步骤之间进行更多复制）或 GPU 数量（增加并行性）增加而增加。

<a id='deepspeed-zero3-config'></a>

#### ZeRO-3 配置

以下是 ZeRO stage 3的配置示例：

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

如果您因为你的模型或激活值超过 GPU 内存而遇到OOM问题，并且您有未使用的 CPU 内存，可以通股票使用 `"device": "cpu"` 将优化器状态和参数卸载到 CPU 内存中，来解决这个限制。如果您不想卸载到 CPU 内存，可以在 `device` 条目中使用 `none` 代替 `cpu`。将优化器状态卸载到 NVMe 上会在后面进一步讨论。

通过将 `pin_memory` 设置为 `true` 启用固定内存。此功能会以减少可用于其他进程的内存为代价来提高吞吐量。固定内存被分配给特定请求它的进程，通常比普通 CPU 内存访问速度更快。

**性能调优：**

- `stage3_max_live_parameters`: `1e9`
- `stage3_max_reuse_distance`: `1e9`

如果遇到OOM问题，请减小 `stage3_max_live_parameters` 和 `stage3_max_reuse_distance`。它们对性能的影响应该很小，除非您正在进行激活值checkpointing。`1e9` 大约会消耗 ~2GB。内存由 `stage3_max_live_parameters` 和 `stage3_max_reuse_distance` 共享，所以它不是叠加的，而是总共2GB。

`stage3_max_live_parameters` 是在任何给定时间要在 GPU 上保留多少个完整参数的上限。"reuse distance" 是我们用来确定参数在将来何时会再次使用的度量标准，我们使用 `stage3_max_reuse_distance` 来决定是丢弃参数还是保留参数。如果一个参数在不久的将来（小于 `stage3_max_reuse_distance`）将被再次使用，那么我们将其保留以减少通信开销。这在启用激活值checkpoing时非常有用，其中我们以单层粒度进行前向重计算和反向传播，并希望在反向传播期间保留前向重计算中的参数。

以下配置值取决于模型的隐藏大小：

- `reduce_bucket_size`: `hidden_size*hidden_size`
- `stage3_prefetch_bucket_size`: `0.9 * hidden_size * hidden_size`
- `stage3_param_persistence_threshold`: `10 * hidden_size`

因此，将这些值设置为 `auto`，[`Trainer`] 将自动分配推荐的参数值。当然，如果您愿意，也可以显式设置这些值。

`stage3_gather_16bit_weights_on_model_save` 在模型保存时启用模型的 fp16 权重整合。对于大模型和多个 GPU，无论是在内存还是速度方面，这都是一项昂贵的操作。目前如果计划恢复训练，这是必需的。请注意未来的更新可能会删除此限制并让使用更加灵活。

如果您从 ZeRO-2 配置迁移，请注意 `allgather_partitions`、`allgather_bucket_size` 和 `reduce_scatter` 配置参数在 ZeRO-3 中不被使用。如果保留这些配置文件，它们将被忽略。

- `sub_group_size`: `1e9`

`sub_group_size` 控制在优化器步骤期间更新参数的粒度。参数被分组到大小为 `sub_group_size` 的桶中，每个桶逐个更新。在 ZeRO-Infinity 中与 NVMe offload一起使用时，`sub_group_size` 控制了在优化器步骤期间在 NVMe 和 CPU 内存之间移动模型状态的粒度。这可以防止非常大的模型耗尽 CPU 内存。

当不使用 NVMe offload时，可以将 `sub_group_size` 保留为其默认值 *1e9*。在以下情况下，您可能需要更改其默认值：

1. 在优化器步骤中遇到OOM：减小 `sub_group_size` 以减少临时buffers的内存利用
2. 优化器步骤花费很长时间：增加 `sub_group_size` 以提高由于增加的数据buffers而导致的带宽利用率。


#### ZeRO-0 配置

请注意，我们将 Stage 0 和 1 放在最后，因为它们很少使用。

Stage 0 禁用了所有类型的分片，只是将 DeepSpeed 作为 DDP 使用。您可以通过以下方式启用：

```json
{
    "zero_optimization": {
        "stage": 0
    }
}
```

这将实质上禁用 ZeRO，而无需更改其他任何内容。


#### ZeRO-1 配置


Stage 1 等同于 Stage 2 减去梯度分片。您可以尝试使用以下配置，仅对优化器状态进行分片，以稍微加速：


```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```



<a id='deepspeed-nvme'></a>

### NVMe 支持

ZeRO-Infinity 通过使用 NVMe 内存扩展 GPU 和 CPU 内存，从而允许训练非常大的模型。由于智能分区和平铺算法，在offload期间每个 GPU 需要发送和接收非常小量的数据，因此 NVMe 被证明适用于训练过程中提供更大的总内存池。ZeRO-Infinity 需要启用 ZeRO-3。

以下配置示例启用 NVMe 来offload优化器状态和参数：

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
}
```

您可以选择将优化器状态和参数都卸载到 NVMe，也可以只选择其中一个，或者都不选择。例如，如果您有大量的 CPU 内存可用，只卸载到 CPU 内存训练速度会更快（提示："device": "cpu"）。

这是有关卸载 [优化器状态](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) 和 [参数](https://www.deepspeed.ai/docs/config-json/#parameter-offloading) 的完整文档。

确保您的 `nvme_path` 实际上是一个 NVMe，因为它与普通硬盘或 SSD 一起工作，但速度会慢得多。快速可扩展的训练是根据现代 NVMe 传输速度设计的（截至本文撰写时，可以达到 ~3.5GB/s 读取，~3GB/s 写入的峰值速度）。

为了找出最佳的 `aio` 配置块，您必须在目标设置上运行一个基准测试，具体操作请参见[说明](https://github.com/deepspeedai/DeepSpeed/issues/998)。



<a id='deepspeed-zero2-zero3-performance'></a>

#### ZeRO-2 和 ZeRO-3 性能对比

如果其他一切都配置相同，ZeRO-3 可能比 ZeRO-2 慢，因为前者除了 ZeRO-2 的操作外，还必须收集模型权重。如果 ZeRO-2 满足您的需求，而且您不需要扩展到几个 GPU 以上，那么您可以选择继续使用它。重要的是要理解，ZeRO-3 以速度为代价实现了更高的可扩展性。

可以调整 ZeRO-3 配置使其性能接近 ZeRO-2：

- 将 `stage3_param_persistence_threshold` 设置为一个非常大的数字 - 大于最大的参数，例如 `6 * hidden_size * hidden_size`。这将保留参数在 GPU 上。
- 关闭 `offload_params`，因为 ZeRO-2 没有这个选项。

即使不更改 `stage3_param_persistence_threshold`，仅将 `offload_params` 关闭，性能可能会显著提高。当然，这些更改将影响您可以训练的模型的大小。因此，这些更改可根据需求帮助您在可扩展性和速度之间进行权衡。



<a id='deepspeed-zero2-example'></a>

#### ZeRO-2 示例

这是一个完整的 ZeRO-2 自动配置文件 `ds_config_zero2.json`：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

这是一个完整的手动设置的启用所有功能的 ZeRO-2 配置文件。主要是为了让您看到典型的参数值是什么样的，但我们强烈建议使用其中包含多个 `auto` 设置的配置文件。

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

<a id='deepspeed-zero3-example'></a>

#### ZeRO-3 示例

这是一个完整的 ZeRO-3 自动配置文件 `ds_config_zero3.json`：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

这是一个完整的 手动设置的启用所有功能的ZeRO-3 配置文件。主要是为了让您看到典型的参数值是什么样的，但我们强烈建议使用其中包含多个 `auto` 设置的配置文件。

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 0.94e6,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

#### 如何选择最佳性能的ZeRO Stage和 offloads

了解了这些不同stages后，现在您需要决定使用哪个stage。本节将尝试回答这个问题。

通常，以下规则适用：

- 速度方面（左边比右边快）

  stage 0（DDP） > stage 1 > stage 2 > stage 2 + offload  > stage 3 > stage3 + offload

- GPU内存使用方面（右边比左边更节省GPU内存）

  stage 0（DDP） < stage 1 < stage 2 < stage 2 + offload < stage 3 < stage 3 + offload

所以，当您希望在尽量使用较少数量的GPU的同时获得最快的执行速度时，可以按照以下步骤进行。我们从最快的方法开始，如果遇到GPU内存溢出，然后切换到下一个速度较慢但使用的GPU内存更少的方法。以此类推。

首先，将批量大小设置为1（您始终可以使用梯度累积来获得任何所需的有效批量大小）。


1. 启用 `--gradient_checkpointing 1`（HF Trainer）或直接 `model.gradient_checkpointing_enable()` - 如果发生OOM（Out of Memory），则执行以下步骤。
2. 首先尝试 ZeRO stage 2。如果发生OOM，则执行以下步骤。
3. 尝试 ZeRO stage 2 + `offload_optimizer` - 如果发生OOM，则执行以下步骤。
4. 切换到 ZeRO stage 3 - 如果发生OOM，则执行以下步骤。
5. 启用 `offload_param` 到 `cpu` - 如果发生OOM，则执行以下步骤。
6. 启用 `offload_optimizer` 到 `cpu` - 如果发生OOM，则执行以下步骤。
7. 如果仍然无法适应批量大小为1，请首先检查各种默认值并尽可能降低它们。例如，如果使用 `generate` 并且不使用宽搜索束，将其缩小，因为它会占用大量内存。
8. 绝对要使用混合半精度而非fp32 - 在Ampere及更高的GPU上使用bf16，在旧的GPU体系结构上使用fp16。
9. 如果仍然发生OOM，可以添加更多硬件或启用ZeRO-Infinity - 即切换 `offload_param` 和 `offload_optimizer` 到 `nvme`。您需要确保它是非常快的NVMe。作为趣闻，我曾经能够在一个小型GPU上使用BLOOM-176B进行推理，使用了ZeRO-Infinity，尽管速度非常慢。但它奏效了！

当然，您也可以按相反的顺序进行这些步骤，从最节省GPU内存的配置开始，然后逐步反向进行，或者尝试进行二分法。

一旦您的批量大小为1不会导致OOM，就测量您的有效吞吐量。

接下来尝试将批量大小增加到尽可能大，因为批量大小越大，GPU的效率越高，特别是在它们乘法运算的矩阵很大时。

现在性能优化游戏开始了。您可以关闭一些offload特性，或者降低ZeRO stage，并增加/减少批量大小，再次测量有效吞吐量。反复尝试，直到满意为止。

不要花费太多时间，但如果您即将开始一个为期3个月的训练 - 请花几天时间找到吞吐量方面最有效的设置。这样您的训练成本将最低，而且您会更快地完成训练。在当前快节奏的机器学习世界中，如果您花费一个额外的月份来训练某样东西，你很可能会错过一个黄金机会。当然，这只是我分享的一种观察，我并不是在催促你。在开始训练BLOOM-176B之前，我花了2天时间进行这个过程，成功将吞吐量从90 TFLOPs提高到150 TFLOPs！这一努力为我们节省了一个多月的训练时间。

这些注释主要是为训练模式编写的，但它们在推理中也应该大部分适用。例如，在推理中，Gradient Checkpointing 是无用的，因为它只在训练过程中有用。此外，我们发现，如果你正在进行多GPU推理并且不使用 [DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/)，[Accelerate](https://huggingface.co/blog/bloom-inference-pytorch-scripts) 应该提供更优越的性能。

其他与性能相关的快速注释：
- 如果您从头开始训练某个模型，请尽量确保张量的形状可以被16整除（例如隐藏层大小）。对于批量大小，至少尝试可被2整除。如果您想从GPU中挤取更高性能，还有一些硬件特定的[wave和tile量化](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/)的可整除性。



### Activation Checkpointing 或 Gradient Checkpointing

Activation Checkpointing和Gradient Checkpointing是指相同方法的两个不同术语。这确实让人感到困惑，但事实就是这样。

Gradient Checkpointing允许通过牺牲速度来换取GPU内存，这要么使您能够克服GPU内存溢出，要么增加批量大小来获得更好的性能。

HF Transformers 模型对DeepSpeed的Activation Checkpointing一无所知，因此如果尝试在DeepSpeed配置文件中启用该功能，什么都不会发生。

因此，您有两种方法可以利用这个非常有益的功能：

1. 如果您想使用 HF Transformers 模型，你可以使用 `model.gradient_checkpointing_enable()` 或在 HF Trainer 中使用 `--gradient_checkpointing`，它会自动为您启用这个功能。在这里使用了 `torch.utils.checkpoint`。
2. 如果您编写自己的模型并希望使用DeepSpeed的Activation Checkpointing，可以使用[规定的API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html)。您还可以使用 HF Transformers 的模型代码，将 `torch.utils.checkpoint` 替换为 DeepSpeed 的API。后者更灵活，因为它允许您将前向激活值卸载到CPU内存，而不是重新计算它们。


### Optimizer 和 Scheduler

只要你不启用 `offload_optimizer`，您可以混合使用DeepSpeed和HuggingFace的调度器和优化器，但有一个例外，即不要使用HuggingFace调度器和DeepSpeed优化器的组合：


| Combos       | HF Scheduler | DS Scheduler |
|:-------------|:-------------|:-------------|
| HF Optimizer | Yes          | Yes          |
| DS Optimizer | No           | Yes          |

在启用 `offload_optimizer` 的情况下，可以使用非DeepSpeed优化器，只要该优化器具有CPU和GPU的实现（除了LAMB）。

<a id='deepspeed-optimizer'></a>

#### Optimizer

DeepSpeed的主要优化器包括Adam、AdamW、OneBitAdam和Lamb。这些优化器已经与ZeRO进行了彻底的测试，因此建议使用它们。然而，也可以导入`torch`中的其他优化器。完整的文档在[这里](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)。

如果在配置文件中不配置`optimizer`条目，[`Trainer`] 将自动将其设置为 `AdamW`，并使用提供的值或以下命令行参数的默认值：`--learning_rate`、`--adam_beta1`、`--adam_beta2`、`--adam_epsilon` 和 `--weight_decay`。

以下是`AdamW` 的自动配置示例：

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

请注意，命令行参数将设置配置文件中的值。这是为了有一个明确的值来源，并避免在不同地方设置学习率等值时难以找到的错误。命令行参数配置高于其他。被覆盖的值包括：

- `lr` 的值为 `--learning_rate`
- `betas` 的值为 `--adam_beta1 --adam_beta2`
- `eps` 的值为 `--adam_epsilon`
- `weight_decay` 的值为 `--weight_decay`

因此，请记住在命令行上调整共享的超参数。

您也可以显式地设置这些值：

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": 0.001,
         "betas": [0.8, 0.999],
         "eps": 1e-8,
         "weight_decay": 3e-7
       }
   }
}
```

但在这种情况下，您需要自己同步[`Trainer`]命令行参数和DeepSpeed配置。

如果您想使用上面未列出的其他优化器，您将不得不将其添加到顶层配置中。

```json
{
   "zero_allow_untested_optimizer": true
}
```

类似于 `AdamW`，您可以配置其他官方支持的优化器。只是记住这些可能有不同的配置值。例如，对于Adam，您可能需要将 `weight_decay` 设置在 `0.01` 左右。

此外，当与DeepSpeed的CPU Adam优化器一起使用时，offload的效果最好。如果您想在offload时使用不同的优化器，自 `deepspeed==0.8.3` 起，您还需要添加：


```json
{
   "zero_force_ds_cpu_optimizer": false
}
```
到顶层配置中。



<a id='deepspeed-scheduler'></a>

#### Scheduler

DeepSpeed支持`LRRangeTest`、`OneCycle`、`WarmupLR`和`WarmupDecayLR`学习率调度器。完整文档在[这里](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters)。

以下是🤗 Transformers 和 DeepSpeed 之间的调度器重叠部分：

- 通过 `--lr_scheduler_type constant_with_warmup` 实现 `WarmupLR`
- 通过 `--lr_scheduler_type linear` 实现 `WarmupDecayLR`。这也是 `--lr_scheduler_type` 的默认值，因此，如果不配置调度器，这将是默认配置的调度器。

如果在配置文件中不配置 `scheduler` 条目，[`Trainer`] 将使用 `--lr_scheduler_type`、`--learning_rate` 和 `--warmup_steps` 或 `--warmup_ratio` 的值来配置其🤗 Transformers 版本。

以下是 `WarmupLR` 的自动配置示例：

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

由于使用了 *"auto"*，[`Trainer`] 的参数将在配置文件中设置正确的值。这是为了有一个明确的值来源，并避免在不同地方设置学习率等值时难以找到的错误。命令行配置高于其他。被设置的值包括：

- `warmup_min_lr` 的值为 `0`。
- `warmup_max_lr` 的值为 `--learning_rate`。
- `warmup_num_steps` 的值为 `--warmup_steps`（如果提供）。否则，将使用 `--warmup_ratio` 乘以训练步骤的数量，并四舍五入。
- `total_num_steps` 的值为 `--max_steps` 或者如果没有提供，将在运行时根据环境、数据集的大小和其他命令行参数（对于 `WarmupDecayLR` 来说需要）自动推导。

当然，您可以接管任何或所有的配置值，并自行设置这些值：

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": 0,
             "warmup_max_lr": 0.001,
             "warmup_num_steps": 1000
         }
     }
}
```

但在这种情况下，您需要自己同步[`Trainer`]命令行参数和DeepSpeed配置。

例如，对于 `WarmupDecayLR`，您可以使用以下条目：

```json
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "last_batch_iteration": -1,
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

然后，`total_num_steps`、`warmup_max_lr`、`warmup_num_steps` 和 `total_num_steps` 将在加载时设置。


<a id='deepspeed-fp32'></a>

### fp32精度

DeepSpeed支持完整的fp32和fp16混合精度。

由于fp16混合精度具有更小的内存需求和更快的速度，唯一不使用它的时候是当您使用的模型在这种训练模式下表现不佳时。通常，当模型没有在fp16混合精度下进行预训练时（例如，bf16预训练模型经常出现这种情况），会出现这种情况。这样的模型可能会发生溢出或下溢，导致 `NaN` 损失。如果是这种情况，那么您将希望使用完整的fp32模式，通过显式禁用默认启用的fp16混合精度模式：

```json
{
    "fp16": {
        "enabled": false,
    }
}
```

如果您使用基于Ampere架构的GPU，PyTorch版本1.7及更高版本将自动切换到使用更高效的tf32格式进行一些操作，但结果仍将以fp32格式呈现。有关详细信息和基准测试，请参见[TensorFloat-32(TF32) on Ampere devices](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)。如果出于某种原因您不希望使用它，该文档包括有关如何禁用此自动转换的说明。

在🤗 Trainer中，你可以使用 `--tf32` 来启用它，或使用 `--tf32 0` 或 `--no_tf32` 来禁用它。默认情况下，使用PyTorch的默认设置。



<a id='deepspeed-amp'></a>

### 自动混合精度

您可以使用自动混合精度，可以选择使用类似 PyTorch AMP 的方式，也可以选择使用类似 Apex 的方式：

### fp16

要配置PyTorch AMP-like 的 fp16（float16） 模式，请设置：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

并且，[`Trainer`]将根据`args.fp16_backend`的值自动启用或禁用它。其余的配置值由您决定。

当传递`--fp16 --fp16_backend amp`或`--fp16_full_eval`命令行参数时，此模式将被启用。

您也可以显式地启用/禁用此模式：

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

但是之后您需要自己同步[`Trainer`]命令行参数和DeepSpeed配置。

以下是[相关文档](https://www.deepspeed.ai/docs/config-json/#fp16-training-options)


### bf16

如果需要使用bfloat16而不是fp16，那么可以使用以下配置部分：

```json
{
    "bf16": {
        "enabled": "auto"
    }
}
```

bf16具有与fp32相同的动态范围，因此不需要损失缩放。

当传递`--bf16`或`--bf16_full_eval`命令行参数时，启用此模式。

您还可以显式地启用/禁用此模式：

```json
{
    "bf16": {
        "enabled": true
    }
}
```

> [!TIP]
> 在`deepspeed==0.6.0`版本中，bf16支持是新的实验性功能。
>
> 如果您启用了bf16来进行[梯度累积](#gradient-accumulation)，您需要意识到它会以bf16累积梯度，这可能不是您想要的，因为这种格式的低精度可能会导致lossy accumulation。
>
> 修复这个问题的工作正在努力进行，同时提供了使用更高精度的`dtype`（fp16或fp32）的选项。


### NCCL集合

在训练过程中，有两种数据类型：`dtype`和用于通信收集操作的`dtype`，如各种归约和收集/分散操作。

所有的gather/scatter操作都是在数据相同的`dtype`中执行的，所以如果您正在使用bf16的训练模式，那么它将在bf16中进行gather操作 - gather操作是非损失性的。

各种reduce操作可能会是非常损失性的，例如当梯度在多个gpu上平均时，如果通信是在fp16或bf16中进行的，那么结果可能是有损失性的 - 因为当在一个低精度中添加多个数字时，结果可能不是精确的。更糟糕的是，bf16比fp16具有更低的精度。通常，当平均梯度时，损失最小，这些梯度通常非常小。因此，对于半精度训练，默认情况下，fp16被用作reduction操作的默认值。但是，您可以完全控制这个功能，如果你选择的话，您可以添加一个小的开销，并确保reductions将使用fp32作为累积数据类型，只有当结果准备好时，它才会降级到您在训练中使用的半精度`dtype`。

要覆盖默认设置，您只需添加一个新的配置条目：

```json
{
    "communication_data_type": "fp32"
}
```

根据这个信息，有效的值包括"fp16"、"bfp16"和"fp32"。

注意：在stage zero 3中，bf16通信数据类型存在一个bug，该问题已在`deepspeed==0.8.1`版本中得到修复。


### apex

配置apex AMP-like模式：

```json
"amp": {
    "enabled": "auto",
    "opt_level": "auto"
}
```

并且，[`Trainer`]将根据`args.fp16_backend`和`args.fp16_opt_level`的值自动配置它。

当传递`--fp16 --fp16_backend apex --fp16_opt_level 01`命令行参数时，此模式将被启用。

您还可以显式配置此模式：

```json
{
    "amp": {
        "enabled": true,
        "opt_level": "O1"
    }
}
```

但是，您需要自己同步[`Trainer`]命令行参数和DeepSpeed配置。

这里是[文档](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options)


<a id='deepspeed-bs'></a>

### Batch Size

配置batch size可以使用如下参数:

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

并且，[`Trainer`]将自动将`train_micro_batch_size_per_gpu`设置为`args.per_device_train_batch_size`的值，并将`train_batch_size`设置为`args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps`。

您也可以显式设置这些值：

```json
{
    "train_batch_size": 12,
    "train_micro_batch_size_per_gpu": 4
}
```

但是，您需要自己同步[`Trainer`]命令行参数和DeepSpeed配置。


<a id='deepspeed-grad-acc'></a>

### Gradient Accumulation

配置gradient accumulation设置如下:

```json
{
    "gradient_accumulation_steps": "auto"
}
```

并且，[`Trainer`]将自动将其设置为`args.gradient_accumulation_steps`的值。

您也可以显式设置这个值：

```json
{
    "gradient_accumulation_steps": 3
}
```

但是，您需要自己同步[`Trainer`]命令行参数和DeepSpeed配置。


<a id='deepspeed-grad-clip'></a>

### Gradient Clipping

配置gradient clipping如下:

```json
{
    "gradient_clipping": "auto"
}
```

并且，[`Trainer`]将自动将其设置为`args.max_grad_norm`的值。

您也可以显式设置这个值：

```json
{
    "gradient_clipping": 1.0
}
```

但是，您需要自己同步[`Trainer`]命令行参数和DeepSpeed配置。



<a id='deepspeed-weight-extraction'></a>

### 获取模型权重

只要您继续使用DeepSpeed进行训练和恢复，您就不需要担心任何事情。DeepSpeed在其自定义检查点优化器文件中存储fp32主权重，这些文件是`global_step*/*optim_states.pt`（这是glob模式），并保存在正常的checkpoint下。

**FP16权重：**

当模型保存在ZeRO-2下时，您最终会得到一个包含模型权重的普通`pytorch_model.bin`文件，但它们只是权重的fp16版本。

在ZeRO-3下，事情要复杂得多，因为模型权重分布在多个GPU上，因此需要`"stage3_gather_16bit_weights_on_model_save": true`才能让`Trainer`保存fp16版本的权重。如果这个设置是`False`，`pytorch_model.bin`将不会被创建。这是因为默认情况下，DeepSpeed的`state_dict`包含一个占位符而不是实际的权重。如果我们保存这个`state_dict`，就无法再加载它了。


```json
{
    "zero_optimization": {
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

**FP32权重：**

虽然fp16权重适合恢复训练，但如果您完成了模型的微调并希望将其上传到[models hub](https://huggingface.co/models)或传递给其他人，您很可能想要获取fp32权重。这最好不要在训练期间完成，因为这需要大量内存，因此最好在训练完成后离线进行。但是，如果需要并且有充足的空闲CPU内存，可以在相同的训练脚本中完成。以下部分将讨论这两种方法。

**实时FP32权重恢复：**

如果您的模型很大，并且在训练结束时几乎没有剩余的空闲CPU内存，这种方法可能不起作用。

如果您至少保存了一个检查点，并且想要使用最新的一个，可以按照以下步骤操作：

```python
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

如果您在使用`--load_best_model_at_end`类：*~transformers.TrainingArguments*参数（用于跟踪最佳
检查点），那么你可以首先显式地保存最终模型，然后再执行相同的操作：

```python
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
trainer.deepspeed.save_checkpoint(checkpoint_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

> [!TIP]
> 注意，一旦运行了`load_state_dict_from_zero_checkpoint`，该模型将不再可以在相同的应用程序的DeepSpeed上下文中使用。也就是说，您需要重新初始化deepspeed引擎，因为`model.load_state_dict(state_dict)`会从其中移除所有的DeepSpeed相关点。所以您只能训练结束时这样做。

当然，您不必使用类：*~transformers.Trainer*，您可以根据你的需求调整上面的示例。

如果您出于某种原因想要更多的优化，您也可以提取权重的fp32 `state_dict`并按照以下示例进行操作：

```python
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)  # already on cpu
model = model.cpu()
model.load_state_dict(state_dict)
```

**离线FP32权重恢复：**

DeepSpeed会创建一个特殊的转换脚本`zero_to_fp32.py`，并将其放置在checkpoint文件夹的顶层。使用此脚本，您可以在任何时候提取权重。该脚本是独立的，您不再需要配置文件或`Trainer`来执行提取操作。

假设您的checkpoint文件夹如下所示：

```bash
$ ls -l output_dir/checkpoint-1/
-rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
-rw-rw-r-- 1 stas stas   12 Mar 27 13:16 latest
-rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
-rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
-rw-rw-r-- 1 stas stas  623 Mar 27 20:42 scheduler.pt
-rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
-rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
-rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
-rw-rw-r-- 1 stas stas  339 Mar 27 20:42 trainer_state.json
-rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.bin
-rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*
```

在这个例子中，只有一个DeepSpeed检查点子文件夹*global_step1*。因此，要重构fp32权重，只需运行：

```bash
python zero_to_fp32.py . pytorch_model.bin
```

这就是它。`pytorch_model.bin`现在将包含从多个GPUs合并的完整的fp32模型权重。

该脚本将自动能够处理ZeRO-2或ZeRO-3 checkpoint。

`python zero_to_fp32.py -h`将为您提供使用细节。

该脚本将通过文件`latest`的内容自动发现deepspeed子文件夹，在当前示例中，它将包含`global_step1`。

注意：目前该脚本需要2倍于最终fp32模型权重的通用内存。


### ZeRO-3 和 Infinity Nuances

ZeRO-3与ZeRO-2有很大的不同，主要是因为它的参数分片功能。

ZeRO-Infinity进一步扩展了ZeRO-3，以支持NVMe内存和其他速度和可扩展性改进。

尽管所有努力都是为了在不需要对模型进行任何特殊更改的情况下就能正常运行，但在某些情况下，您可能需要以下信息。


#### 构建大模型

DeepSpeed/ZeRO-3可以处理参数量达到数万亿的模型，这些模型可能无法适应现有的内存。在这种情况下，如果您还是希望初始化更快地发生，可以使用*deepspeed.zero.Init()*上下文管理器（也是一个函数装饰器）来初始化模型，如下所示：

```python
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(config)
```

如您所见，这会为您随机初始化一个模型。

如果您想使用预训练模型，`model_class.from_pretrained`将在`is_deepspeed_zero3_enabled()`返回`True`的情况下激活此功能，目前这是通过传递的DeepSpeed配置文件中的ZeRO-3配置部分设置的。因此，在调用`from_pretrained`之前，您必须创建**TrainingArguments**对象。以下是可能的顺序示例：

```python
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

如果您使用的是官方示例脚本，并且命令行参数中包含`--deepspeed ds_config.json`且启用了ZeRO-3配置，那么一切都已经为您准备好了，因为这是示例脚本的编写方式。

注意：如果模型的fp16权重无法适应单个GPU的内存，则必须使用此功能。

有关此方法和其他相关功能的完整详细信息，请参阅[构建大模型](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models)。

此外，在加载fp16预训练模型时，您希望`from_pretrained`使用`dtype=torch.float16`。详情请参见[from_pretrained-torch-dtype](#from_pretrained-torch-dtype)。


#### 参数收集

在多个GPU上使用ZeRO-3时，没有一个GPU拥有所有参数，除非它是当前执行层的参数。因此，如果您需要一次访问所有层的所有参数，有一个特定的方法可以实现。
您可能不需要它，但如果您需要，请参考[参数收集](https://deepspeed.readthedocs.io/en/latest/zero3.html#manual-parameter-coordination)。

然而，我们在多个地方确实使用了它，其中一个例子是在`from_pretrained`中加载预训练模型权重。我们一次加载一层，然后立即将其分区到所有参与的GPU上，因为对于非常大的模型，无法在一个GPU上一次性加载并将其分布到多个GPU上，因为内存限制。

此外，在ZeRO-3下，如果您编写自己的代码并遇到看起来像这样的模型参数权重：

```python
tensor([1.0], device="cuda:0", dtype=torch.float16, requires_grad=True)
```

强调`tensor([1.])`，或者如果您遇到一个错误，它说参数的大小是`1`，而不是某个更大的多维形状，这意味着参数被划分了，你看到的是一个ZeRO-3占位符。



<a id='deepspeed-zero-inference'></a>


### ZeRO 推理

"ZeRO 推断" 使用与 "ZeRO-3 训练" 相同的配置。您只需要去掉优化器和调度器部分。实际上，如果您希望与训练共享相同的配置文件，您可以将它们保留在配置文件中，它们只会被忽略。

您只需要传递通常的[`TrainingArguments`]参数。例如：

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json
```

唯一的重要事情是您需要使用ZeRO-3配置，因为ZeRO-2对于推理没有任何优势，因为只有ZeRO-3才对参数进行分片，而ZeRO-1则对梯度和优化器状态进行分片。

以下是在DeepSpeed下运行`run_translation.py`启用所有可用GPU的示例：

```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --output_dir output_dir \
--do_eval --max_eval_samples 50 --warmup_steps 50  \
--max_source_length 128 --val_max_target_length 128 \
--overwrite_output_dir --per_device_eval_batch_size 4 \
--predict_with_generate --dataset_config "ro-en" --fp16 \
--source_lang en --target_lang ro --dataset_name wmt16 \
--source_prefix "translate English to Romanian: "
```

由于在推理阶段，优化器状态和梯度不需要额外的大量内存，您应该能够将更大的批次和/或序列长度放到相同的硬件上。

此外，DeepSpeed目前正在开发一个名为Deepspeed-Inference的相关产品，它与ZeRO技术无关，而是使用张量并行来扩展无法适应单个GPU的模型。这是一个正在进行的工作，一旦该产品完成，我们将提供集成。


### 内存要求

由于 DeepSpeed ZeRO 可以将内存卸载到 CPU（和 NVMe），该框架提供了一些工具，允许根据使用的 GPU 数量告知将需要多少 CPU 和 GPU 内存。

让我们估计在单个GPU上微调"bigscience/T0_3B"所需的内存：

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

因此，您可以将模型拟合在单个80GB的GPU上，不进行CPU offload，或者使用微小的8GB GPU，但需要约60GB的CPU内存。（请注意，这仅是参数、优化器状态和梯度所需的内存 - 您还需要为CUDA内核、激活值和临时变量分配更多的内存。）

然后，这是成本与速度的权衡。购买/租用较小的 GPU（或较少的 GPU，因为您可以使用多个 GPU 进行 Deepspeed ZeRO）。但这样会更慢，因此即使您不关心完成某项任务的速度，减速也直接影响 GPU 使用的持续时间，从而导致更大的成本。因此，请进行实验并比较哪种方法效果最好。

如果您有足够的GPU内存，请确保禁用CPU/NVMe卸载，因为这会使所有操作更快。

例如，让我们重复相同的操作，使用2个GPU：

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 2 GPUs per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.74GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=1
   31.11GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=0

```

所以，您需要2个32GB或更高的GPU，且不进行CPU卸载。

如需了解更多信息，请参阅[内存估算器](https://deepspeed.readthedocs.io/en/latest/memory.html)。



### 归档Issues

请按照以下步骤提交问题，以便我们能够迅速找到问题并帮助您解除工作阻塞。

在您的报告中，请始终包括以下内容：

1. 完整的Deepspeed配置文件
2. 如果使用了[`Trainer`]，则包括命令行参数；如果自己编写了Trainer设置，则包括[`TrainingArguments`]参数。请不要导出[`TrainingArguments`]，因为它有几十个与问题无关的条目。
3. 输出：

    ```bash
    python -c 'import torch; print(f"torch: {torch.__version__}")'
    python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
    python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
    ```

4. 如果可能，请包含一个Google Colab notebook链接，我们可以使用它来重现问题。您可以使用这个[notebook](https://github.com/stas00/porting/blob/master/transformers/deepspeed/DeepSpeed_on_colab_CLI.ipynb)作为起点。
5. 除非不可能，否则请始终使用标准数据集，而不是自定义数据集。
6. 如果可能，尝试使用现有[示例](https://github.com/huggingface/transformers/tree/main/examples/pytorch)之一来重现问题。

需要考虑的因素：

- Deepspeed通常不是问题的原因。

  一些已提交的问题被证明与Deepspeed无关。也就是说，一旦将Deepspeed从设置中移除，问题仍然存在。

  因此，如果问题明显与DeepSpeed相关，例如您可以看到有一个异常并且可以看到DeepSpeed模块涉及其中，请先重新测试没有DeepSpeed的设置。只有当问题仍然存在时，才向Deepspeed提供所有必需的细节。

- 如果您明确问题是在Deepspeed核心中而不是集成部分，请直接向[Deepspeed](https://github.com/deepspeedai/DeepSpeed/)提交问题。如果您不确定，请不要担心，无论使用哪个issue跟踪问题都可以，一旦您发布问题，我们会弄清楚并将其重定向到另一个issue跟踪（如果需要的话）。



### Troubleshooting

#### 启动时`deepspeed`进程被终止，没有回溯

如果启动时`deepspeed`进程被终止，没有回溯，这通常意味着程序尝试分配的CPU内存超过了系统的限制或进程被允许分配的内存，操作系统内核杀死了该进程。这是因为您的配置文件很可能将`offload_optimizer`或`offload_param`或两者都配置为卸载到`cpu`。如果您有NVMe，可以尝试在ZeRO-3下卸载到NVMe。这里是如何[估计特定模型所需的内存](https://deepspeed.readthedocs.io/en/latest/memory.html)。

#### 训练和/或评估/预测loss为`NaN`

这种情况通常发生在使用bf16混合精度模式预训练的模型试图在fp16（带或不带混合精度）下使用时。大多数在TPU上训练的模型以及由谷歌发布的模型都属于这个类别（例如，几乎所有基于t5的模型）。在这种情况下，解决方案是要么使用fp32，要么在支持的情况下使用bf16（如TPU、Ampere GPU或更新的版本）。

另一个问题可能与使用fp16有关。当您配置此部分时：

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

并且您在日志中看到Deepspeed报告`OVERFLOW`如下

```
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

这意味着Deepspeed损失缩放器无法找到一个克服损失溢出的缩放系数。

在这种情况下，通常需要提高`initial_scale_power`的值。将其设置为`"initial_scale_power": 32`通常会解决问题。



### 注意事项

- 尽管 DeepSpeed 有一个可安装的 PyPI 包，但强烈建议从源代码安装它，以最好地匹配您的硬件，如果您需要启用某些功能，如 1-bit Adam，这些功能在 pypi 发行版中不可用。
- 您不必使用🤗  Transformers的 [`Trainer`] 来使用 DeepSpeed   - 您可以使用任何模型与自己的训练器，您还需要根据 [DeepSpeed 集成说明](https://www.deepspeed.ai/getting-started/#writing-deepspeed-models) 调整后者。



## Non-Trainer Deepspeed集成

当`Trainer`没有被使用时，`~integrations.HfDeepSpeedConfig`被用来将Deepspeed集成到huggingface的Transformers核心功能中。它唯一做的事情就是在`from_pretrained`调用期间处理Deepspeed ZeRO-3参数收集和将模型自动分割到多个GPU上。除此之外，您需要自己完成其他所有工作。

当使用`Trainer`时，所有事情都自动得到了处理。

当不使用`Trainer`时，为了高效地部署Deepspeed ZeRO-3，您必须在实例化模型之前实例化`~integrations.HfDeepSpeedConfig`对象并保持该对象活跃。

如果您正在使用Deepspeed ZeRO-1或ZeRO-2，您根本不需要使用`HfDeepSpeedConfig`。

以预训练模型为例:

```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

ds_config = {...}  # deepspeed config object or path to the file
# must run before instantiating the model to detect zero 3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
model = AutoModel.from_pretrained("openai-community/gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

或者以非预训练模型为例：

```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel, AutoConfig
import deepspeed

ds_config = {...}  # deepspeed config object or path to the file
# must run before instantiating the model to detect zero 3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
config = AutoConfig.from_pretrained("openai-community/gpt2")
model = AutoModel.from_config(config)
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

请注意，如果您没有使用[`Trainer`]集成，您完全需要自己动手。基本上遵循[Deepspeed](https://www.deepspeed.ai/)网站上的文档。同时，您必须显式配置配置文件 - 不能使用`"auto"`值，而必须放入实际值。


## HfDeepSpeedConfig

[[autodoc]] integrations.HfDeepSpeedConfig
    - all

### 自定义DeepSpeed ZeRO推理

以下是一个示例，演示了在无法将模型放入单个 GPU 时如果不使用[Trainer]进行 DeepSpeed ZeRO 推理 。该解决方案包括使用额外的 GPU 或/和将 GPU 内存卸载到 CPU 内存。

这里要理解的重要细微差别是，ZeRO的设计方式可以让您在不同的GPU上并行处理不同的输入。

这个例子有很多注释，并且是自文档化的。

请确保：

1. 如果您有足够的GPU内存（因为这会减慢速度），禁用CPU offload。
2. 如果您拥有Ampere架构或更新的GPU，启用bf16以加快速度。如果您没有这种硬件，只要不使用任何在bf16混合精度下预训练的模型（如大多数t5模型），就可以启用fp16。否则这些模型通常在fp16中溢出，您会看到输出无效结果。

```python
#!/usr/bin/env python

# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py


from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

# ds_config notes
#
# - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
# faster.
#
# - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
# all official t5 models are bf16-pretrained
#
# - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
# - want CPU offload
#
# - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
# - which params should remain on gpus - the larger the value the smaller the offload size
#
# For in-depth info on Deepspeed config see
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# keeping the same format as json for consistency, except it uses lower case for true/false
# fmt: off
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# Deepspeed ZeRO can process unrelated inputs on each GPU. So for 2 gpus you process 2 inputs at once.
# If you use more GPUs adjust for more.
# And of course if you have just one input to process you then need to pass the same string to both gpus
# If you use only one GPU, then you will have only rank 0.
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
elif rank == 1:
    text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")
```

让我们保存它为 `t0.py`并运行：
```bash
$ deepspeed --num_gpus 2 t0.py
rank0:
   in=Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy
  out=Positive
rank1:
   in=Is this review positive or negative? Review: this is the worst restaurant ever
  out=negative
```

这是一个非常基本的例子，您需要根据自己的需求进行修改。

### `generate` 的差异

在使用ZeRO stage 3的多GPU时，需要通过调用`generate(..., synced_gpus=True)`来同步GPU。如果一个GPU在其它GPU之前完成生成，整个系统将挂起，因为其他GPU无法从停止生成的GPU接收权重分片。

从`transformers>=4.28`开始，如果没有明确指定`synced_gpus`，检测到这些条件后它将自动设置为`True`。但如果您需要覆盖`synced_gpus`的值，仍然可以这样做。



## 测试 DeepSpeed 集成

如果您提交了一个涉及DeepSpeed集成的PR，请注意我们的CircleCI PR CI设置没有GPU，因此我们只在另一个CI夜间运行需要GPU的测试。因此，如果您在PR中获得绿色的CI报告，并不意味着DeepSpeed测试通过。

要运行DeepSpeed测试，请至少运行以下命令：

```bash
RUN_SLOW=1 pytest tests/deepspeed/test_deepspeed.py
```

如果你更改了任何模型或PyTorch示例代码，请同时运行多模型测试。以下将运行所有DeepSpeed测试：

```bash
RUN_SLOW=1 pytest tests/deepspeed
```

## 主要的DeepSpeed资源

- [项目GitHub](https://github.com/deepspeedai/DeepSpeed)
- [使用文档](https://www.deepspeed.ai/getting-started/)
- [API文档](https://deepspeed.readthedocs.io/en/latest/index.html)
- [博客文章](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

论文:

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://huggingface.co/papers/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://huggingface.co/papers/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://huggingface.co/papers/2104.07857)

最后，请记住，HuggingFace [`Trainer`]仅集成了DeepSpeed，因此如果您在使用DeepSpeed时遇到任何问题或疑问，请在[DeepSpeed GitHub](https://github.com/deepspeedai/DeepSpeed/issues)上提交一个issue。
