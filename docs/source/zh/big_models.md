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

# 实例化大型模型

当你想使用一个非常大的预训练模型时，一个挑战是尽量减少对内存的使用。通常从PyTorch开始的工作流程如下：

1. 用随机权重创建你的模型。
2. 加载你的预训练权重。
3. 将这些预训练权重放入你的随机模型中。

步骤1和2都需要完整版本的模型在内存中，这在大多数情况下不是问题，但如果你的模型开始达到几个GB的大小，这两个副本可能会让你超出内存的限制。更糟糕的是，如果你使用`torch.distributed`来启动分布式训练，每个进程都会加载预训练模型并将这两个副本存储在内存中。

<Tip>

请注意，随机创建的模型使用“空”张量进行初始化，这些张量占用内存空间但不填充它（因此随机值是给定时间内该内存块中的任何内容）。在第3步之后，对未初始化的权重执行适合模型/参数种类的随机初始化（例如正态分布），以尽可能提高速度！

</Tip>

在本指南中，我们将探讨 Transformers 提供的解决方案来处理这个问题。请注意，这是一个积极开发的领域，因此这里解释的API在将来可能会略有变化。

## 分片checkpoints

自4.18.0版本起，占用空间超过10GB的模型检查点将自动分成较小的片段。在使用`model.save_pretrained(save_dir)`时，您最终会得到几个部分`checkpoints`（每个的大小都小于10GB）以及一个索引，该索引将参数名称映射到存储它们的文件。

您可以使用`max_shard_size`参数来控制分片之前的最大大小。为了示例的目的，我们将使用具有较小分片大小的普通大小的模型：让我们以传统的BERT模型为例。


```py
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

如果您使用 [`PreTrainedModel.save_pretrained`](模型预训练保存) 进行保存，您将得到一个新的文件夹，其中包含两个文件：模型的配置和权重：

```py
>>> import os
>>> import tempfile

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir)
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model.bin']
```

现在让我们使用最大分片大小为200MB：

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']
```

在模型配置文件最上方，我们可以看到三个不同的权重文件，以及一个`index.json`索引文件。这样的`checkpoint`可以使用[`~PreTrainedModel.from_pretrained`]方法完全重新加载：

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

对于大型模型来说，这样做的主要优点是在上述工作流程的步骤2中，每个`checkpoint`的分片在前一个分片之后加载，从而将内存中的内存使用限制在模型大小加上最大分片的大小。

在后台，索引文件用于确定`checkpoint`中包含哪些键以及相应的权重存储在哪里。我们可以像加载任何json一样加载该索引，并获得一个字典：

```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     with open(os.path.join(tmp_dir, "pytorch_model.bin.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

目前元数据仅包括模型的总大小。我们计划在将来添加其他信息：
```py
>>> index["metadata"]
{'total_size': 433245184}
```

权重映射是该索引的主要部分，它将每个参数的名称（通常在PyTorch模型的`state_dict`中找到）映射到存储该参数的文件：

```py
>>> index["weight_map"]
{'embeddings.LayerNorm.bias': 'pytorch_model-00001-of-00003.bin',
 'embeddings.LayerNorm.weight': 'pytorch_model-00001-of-00003.bin',
 ...
```

如果您想直接在模型内部加载这样的分片`checkpoint`，而不使用 [`PreTrainedModel.from_pretrained`](就像您会为完整`checkpoint`执行 `model.load_state_dict()` 一样)，您应该使用 [`modeling_utils.load_sharded_checkpoint`]：


```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     load_sharded_checkpoint(model, tmp_dir)
```

## 低内存加载

分片`checkpoints`在上述工作流的第2步中降低了内存使用，但为了在低内存环境中使用该模型，我们建议使用基于 Accelerate 库的工具。

请阅读以下指南以获取更多信息：[使用 Accelerate 进行大模型加载](./main_classes/model#large-model-loading)
