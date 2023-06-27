<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，按“原样”分发的软件在许可证下分发；没有任何形式的担保或条件，无论是明示的还是隐含的。请参阅许可证以获取特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 实例化一个大模型

当您想要使用一个非常大的预训练模型时，一个挑战是最小化 RAM 的使用。常规的 PyTorch 工作流程是：
1. 使用随机权重创建您的模型。
2. 加载预训练权重。
3. 将这些预训练权重放入您的随机模型中。

步骤 1 和 2 都需要完整版本的模型存储在内存中，这在大多数情况下不是问题，但如果您的模型开始重达几个千兆字节，这两个副本可能导致内存不足。更糟糕的是，如果您使用 `torch.distributed` 来启动分布式训练，每个进程都会加载预训练模型并将这两个副本存储在内存中。


<Tip>

请注意，随机创建的模型使用“空”张量进行初始化，这些张量占用内存空间但不填充内容（因此随机值实际上是在内存的某个时间点上存在的内容）。适用于所实例化的模型/参数类型的适当分布的随机初始化（例如正态分布）只在步骤 3 上对未初始化的权重执行，以尽可能快地完成!
</Tip>

在本指南中，我们将探讨 Transformers 提供的解决方案来解决此问题。请注意，这是一个活跃开发领域，因此这里解释的 API 可能会在将来略有变化。

## 分片检查点 Sharded checkpoints

从版本 4.18.0 开始，占用超过 10GB 空间的模型检查点会自动分成较小的片段。在执行 `model.save_pretrained(save_dir)` 时，您将获得多个部分检查点（每个部分检查点的大小都小于 10GB）和一个将参数名称映射到存储文件的索引。

您可以使用 `max_shard_size` 参数来控制分片之前的最大大小，因此为了举例，我们将使用一个具有小分片大小的普通大小模型：传统的 BERT 模型。
```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

如果使用 [`~PreTrainedModel.save_pretrained`] 进行保存，您将得到一个新文件夹，其中包含模型的配置和权重两个文件：
```py
>>> import os
>>> import tempfile

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir)
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model.bin']
```

现在让我们使用最大分片大小为 200MB：
```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']
```

除了模型的配置，我们看到三个不同的权重文件和一个 `index.json` 文件，这是我们的索引。

可以使用 [`~PreTrainedModel.from_pretrained`] 方法完全重新加载此类检查点：

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

对于大型模型，这样做的主要优点是在上述工作流程的第 2 步中，检查点的每个分片都在前一个分片之后加载，将 RAM 中的内存使用量限制为模型大小加上最大分片的大小。

在幕后，索引文件用于确定检查点中的哪些键以及相应的权重存储在哪里。我们可以像任何 json 一样加载该索引并获得一个字典：
```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     with open(os.path.join(tmp_dir, "pytorch_model.bin.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

目前，元数据仅包含模型的总大小。我们计划在将来添加其他信息：
```py
>>> index["metadata"]
{'total_size': 433245184}
```

权重映射是此索引的主要部分，它将每个参数名称（通常在 PyTorch 模型 `state_dict` 中找到）映射到其存储的文件：
```py
>>> index["weight_map"]
{'embeddings.LayerNorm.bias': 'pytorch_model-00001-of-00003.bin',
 'embeddings.LayerNorm.weight': 'pytorch_model-00001-of-00003.bin',
 ...
```

如果要在不使用 [`~PreTrainedModel.from_pretrained`] 的情况下直接加载此类分片检查点（就像对完整检查点执行 `model.load_state_dict()` 一样），应使用 [`~modeling_utils.load_sharded_checkpoint`]：
```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     load_sharded_checkpoint(model, tmp_dir)
```

## 低内存加载

分片检查点减少了上述工作流程的第 2 步中的内存使用量，但为了在低内存环境中使用该模型，我们建议利用基于 Accelerate 库的工具。
请阅读以下指南以获取更多信息：[使用 Accelerate 进行大模型加载](./main_classes/model#large-model-loading)