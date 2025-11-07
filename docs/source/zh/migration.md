<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 从旧版本迁移

## 从 Transformers `v3.x` 迁移到 `v4.x`

从版本 3 升级到版本 4 时引入了一些更改。以下是预期更改的摘要：

#### 1. AutoTokenizer 和 pipeline 现在默认使用快速分词器（Rust）。

Python 和 Rust 分词器的 API 大致相同，但 Rust 分词器具有更全面的功能集。

这引入了两个主要更改：

- Python 和 Rust 分词器处理词元溢出的方式不同。

- Rust 分词器在编码方法中不接受整数。

##### 如何在 v4.x 中实现与 v3.x 相同的行为

- Pipeline 现在包含一些开箱即用的附加功能。请参阅带有 `grouped_entities` 标志的词元分类 pipeline。

- Auto-tokenizer 现在返回 Rust 分词器。要使用 Python 分词器，用户必须将 `use_fast` 标志设置为 `False`：

在 `v3.x` 版本中：

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

在 `v4.x` 版本中实现相同功能：

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased", use_fast=False)
```

#### 2. SentencePiece 已从必需依赖项中移除

`setup.py` 文件中已移除对 SentencePiece 的依赖。这样做是为了在不依赖 `conda-forge` 的情况下，提供一个基于 Anaconda 云的通道。这意味着，依赖于 SentencePiece 库的分词器将无法在标准的 `transformers` 安装中使用。

这包括以下速度较慢的版本：

- `XLNetTokenizer`

- `AlbertTokenizer`

- `CamembertTokenizer`

- `MBartTokenizer`

- `PegasusTokenizer`

- `T5Tokenizer`

- `ReformerTokenizer`

- `XLMRobertaTokenizer`

##### 如何在 v4.x 版本中获得与 v3.x 相同的行为

要获得与 `v3.x` 版本相同的行为，您还必须安装 `sentencepiece`：

在 `v3.x` 版本中：

```bash
pip install tr​​ansformers
```

要在 `v4.x` 版本中获得相同的行为：

```bash
pip install tr​​ansformers[sentencepiece]
```

或
```bash
pip install tr​​ansformers sentencepiece
```

#### 3. 仓库架构已更新，每个模型都有自己的文件夹

随着新模型的添加，仓库中的文件数量也会增加。 `src/transformers` 文件夹持续增长，导致其难以浏览和理解。我们决定将每个模型及其相关文件分别放置在各自的子文件夹中。

这是一项重大变更，因为使用模型模块直接导入中间层需要通过不同的路径。

##### 如何在 v4.x 中实现与 v3.x 相同的行为

要实现与 `v3.x` 版本相同的行为，您必须更新用于访问图层的路径。

在 `v3.x` 版本中：

```bash
from transformers.modeling_bert import BertLayer
```

要在 `v4.x` 版本中实现相同的功能：

```bash
from transformers.models.bert.modeling_bert import BertLayer
```

#### 4. 将 `return_dict` 参数默认设置为 `True`

`return_dict` 参数（`main_classes/output`）允许返回包含模型输出的类似字典的 Python 对象，而不是标准元组。该对象具有自文档性，因为键可用于检索值（其行为也类似于元组），并且用户可以检索对象以进行索引或切片。

这是一个重大更改，因为元组无法解包：`value0, value1 = outputs` 将无法工作。

##### 如何在 v4.x 中实现与 v3.x 相同的行为

要实现与 `v3.x` 版本相同的行为，请在模型配置和下一步中都将 `return_dict` 参数指定为 `False`。

在 `v3.x` 版本中：

```bash
model = BertModel.from_pretrained("google-bert/bert-base-cased")
outputs = model(**inputs)
```

要在 `v4.x` 版本中实现相同的功能：

```bash
model = BertModel.from_pretrained("google-bert/bert-base-cased")
outputs = model(**inputs, return_dict=False)
```

或

```bash
model = BertModel.from_pretrained("google-bert/bert-base-cased", return_dict=False)
outputs = model(**inputs)
```

#### 5. 移除某些已弃用的属性

如果属性已弃用至少一个月，则将其移除。已弃用属性的完整列表可在 [#8604](https://github.com/huggingface/transformers/pull/8604) 中找到。

以下是这些属性/方法/参数的列表及其替代方案：

在多个模型中,标签变得与其他模型一致：

- `masked_lm_labels` 在 `AlbertForMaskedLM` 和 `AlbertForPreTraining` 中变为 `labels`。
- `masked_lm_labels` 在 `BertForMaskedLM` 和 `BertForPreTraining` 中变为 `labels`。
- `masked_lm_labels` 在 `DistilBertForMaskedLM` 中变为 `labels`。
- `masked_lm_labels` 在 `ElectraForMaskedLM` 中变为 `labels`。
- `masked_lm_labels` 在 `LongformerForMaskedLM` 中变为 `labels`。
- `masked_lm_labels` 在 `MobileBertForMaskedLM` 中变为 `labels`。
- `masked_lm_labels` 在 `RobertaForMaskedLM` 中变为 `labels`。
- `lm_labels` 在 `BartForConditionalGeneration` 中变为 `labels`。
- `lm_labels` 在 `GPT2DoubleHeadsModel` 中变为 `labels`。
- `lm_labels` 在 `OpenAIGPTDoubleHeadsModel` 中变为 `labels`。
- `lm_labels` 在 `T5ForConditionalGeneration` 中变为 `labels`。

在多个模型中，缓存机制变得与其他模型一致：

- `decoder_cached_states` 在所有类 BART、FSMT 和 T5 模型中变为 `past_key_values`。
- `decoder_past_key_values` 在所有类 BART、FSMT 和 T5 模型中变为 `past_key_values`。
- `past` 在所有 CTRL 模型中变为 `past_key_values`。
- `past` 在所有 GPT-2 模型中变为 `past_key_values`。

关于分词器类：

- 分词器属性 `max_len` 变为 `model_max_length`。
- 分词器属性 `return_lengths` 变为 `return_length`。
- 分词器编码参数 `is_pretokenized` 变为 `is_split_into_words`。

关于 `Trainer` 类：

- `Trainer` 的 `tb_writer` 参数已移除,改用回调函数 `TensorBoardCallback(tb_writer=...)`。
- `Trainer` 的 `prediction_loss_only` 参数已移除,改用类参数 `args.prediction_loss_only`。
- `Trainer` 的 `data_collator` 属性将是可调用的。
- `Trainer` 的 `_log` 方法已弃用,改用 `log`。
- `Trainer` 的 `_training_step` 方法已弃用,改用 `training_step`。
- `Trainer` 的 `_prediction_loop` 方法已弃用,改用 `prediction_loop`。
- `Trainer` 的 `is_local_master` 方法已弃用,改用 `is_local_process_zero`。
- `Trainer` 的 `is_world_master` 方法已弃用,改用 `is_world_process_zero`。

关于 `TrainingArguments` 类：

- `TrainingArguments` 的 `evaluate_during_training` 参数已弃用,改用 `eval_strategy`。

关于 Transfo-XL 模型：

- Transfo-XL 的配置属性 `tie_weight` 变为 `tie_words_embeddings`。
- Transfo-XL 的建模方法 `reset_length` 变为 `reset_memory_length`。

关于 pipeline：

- `FillMaskPipeline` 的 `topk` 参数变为 `top_k`。

## 从 pytorch-transformers 迁移到 🤗 Transformers

以下是从 `pytorch-transformers` 迁移到 🤗 Transformers 时需要注意的事项的简要摘要。

### 某些模型关键字输入的位置顺序（`attention_mask`、`token_type_ids`...）已更改

为了使用 Torchscript（参见 #1010、#1204 和 #1195），某些模型的**关键字输入**（`attention_mask`、`token_type_ids`...）的特定顺序已被修改。

如果您使用关键字参数初始化模型，例如 `model(inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)`，这不应该引起任何变化。

如果您使用位置参数初始化模型，例如 `model(inputs_ids, attention_mask, token_type_ids)`，您可能需要仔细检查输入参数的确切顺序。

## 从 pytorch-pretrained-bert 迁移

以下是从 `pytorch-pretrained-bert` 迁移到 🤗 Transformers 时需要注意的事项的简要摘要。

### 模型始终返回 `tuple`

从 `pytorch-pretrained-bert` 迁移到 🤗 Transformers 的主要重大变化是，模型的预测方法始终返回一个 `tuple`，其中包含各种元素，具体取决于模型和配置参数。

每个模型的元组的确切内容详细显示在模型的文档字符串和[文档](https://huggingface.co/transformers/)中。

在几乎所有情况下，只需获取输出的第一个元素即可获得您以前在 `pytorch-pretrained-bert` 中使用的内容。

以下是从 `pytorch-pretrained-bert` 转换为 🤗 Transformers 的 `BertForSequenceClassification` 分类模型的示例：

```python
# 加载我们的模型
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

# 如果您在 pytorch-pretrained-bert 中使用此行：
loss = model(input_ids, labels=labels)

# 现在在 🤗 Transformers 中使用此行从输出元组中提取损失：
outputs = model(input_ids, labels=labels)
loss = outputs[0]

# 在 🤗 Transformers 中，您还可以访问 logits：
loss, logits = outputs[:2]

# 如果您将模型配置为返回注意力权重，您还可以访问它们（以及其他输出，请参阅文档字符串和文档）
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
outputs = model(input_ids, labels=labels)
loss, logits, attentions = outputs
```

### 序列化

`from_pretrained()` 方法的重大变化：

1. 当使用 `from_pretrained()` 方法时，模型现在默认设置为评估模式。要训练它们，不要忘记将它们恢复到训练模式（`model.train()`）以激活 dropout 模块。

2. 提供给 `from_pretrained()` 方法的额外参数 `*inputs` 和 `**kwargs` 以前直接传递给底层模型类的 `__init__()` 方法。现在它们首先用于更新模型的配置属性，这可能不适用于基于先前 `BertForSequenceClassification` 示例构建的派生模型类。更准确地说，提供给 `from_pretrained()` 的位置参数 `*inputs` 会直接转发到模型的 `__init__()` 方法，而关键字参数 `**kwargs` (i) 与配置类的属性匹配的，用于更新这些属性 (ii) 与配置类的任何属性都不匹配的，则转发到 `__init__()` 方法。

此外，虽然这不是重大变化，但序列化方法已经标准化，如果您以前使用任何其他序列化方法，您可能应该切换到新的 `save_pretrained(save_directory)` 方法。

以下是一个示例：

```python
### 加载模型和分词器
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

### 让我们对模型和分词器做一些事情
# 例如：向我们模型的词汇表和嵌入中添加新标记
tokenizer.add_tokens(["[SPECIAL_TOKEN_1]", "[SPECIAL_TOKEN_2]"])
model.resize_token_embeddings(len(tokenizer))

# 训练我们的模型
train(model)

### 现在将我们的模型和分词器保存到文件夹
model.save_pretrained("./my_saved_model_directory/")
tokenizer.save_pretrained("./my_saved_model_directory/")

### 重新加载模型和分词器
model = BertForSequenceClassification.from_pretrained("./my_saved_model_directory/")

tokenizer = BertTokenizer.from_pretrained("./my_saved_model_directory/")
```

### 优化器：BertAdam 和 OpenAIAdam 现在是 AdamW，调度器是标准 PyTorch

以前包含的两个优化器 `BertAdam` 和 `OpenAIAdam` 已被单个 `AdamW` 替换，它具有一些差异：

- 它仅实现权重衰减校正，
- 调度器现在是外部的（见下文），
- 梯度裁剪现在也是外部的（见下文）。

新的优化器 `AdamW` 与 PyTorch 的 `Adam` API 相匹配，并允许您使用 PyTorch 或 apex 方法进行调度和裁剪。

调度器现在是标准的 [PyTorch 学习率调度器](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)，不再是优化器的一部分。

以下是使用 `BertAdam` 和 `AdamW` 进行线性预热和衰减的示例：

```python
# 参数：
lr = 1e-3
max_grad_norm = 1.0
num_training_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

### 以前 BertAdam 优化器是这样实例化的：
optimizer = BertAdam(
   model.parameters(),
   lr=lr,
   schedule="warmup_linear",
   warmup=warmup_proportion,
   num_training_steps=num_training_steps,
)

### 并且这样使用：
for batch in train_data:
   loss = model(batch)
   loss.backward()
   optimizer.step()

### 在 🤗 Transformers 中，优化器和调度器是分开的，并且这样使用：
optimizer = AdamW(
   model.parameters(), lr=lr, correct_bias=False
)  # 要复制 BertAdam 的特定行为，请设置 correct_bias=False

scheduler = get_linear_schedule_with_warmup(
   optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)  # PyTorch 调度器

### 应该这样使用：
for batch in train_data:
   loss = model(batch)
   loss.backward()
   torch.nn.utils.clip_grad_norm_(
      model.parameters(), max_grad_norm
   )  # 梯度裁剪不再在 AdamW 中（因此您可以毫无问题地使用 amp）
   optimizer.step()
   scheduler.step()
```