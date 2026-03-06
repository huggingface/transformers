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

# 问答

[[open-in-colab]]

<Youtube id="ajPx5LwJD-I"/>

问答任务根据给定的问题返回答案。相信您肯定在日常生活中接触过问答模型, 比如您可能使用过 豆包、Siri 等虚拟助手询问天气情况。问答任务通常分为两种类型：

- 抽取式：从给定的上下文中提取答案。
- 生成式：根据上下文生成能够正确回答问题的答案。

本指南将向您展示如何：

1. 在 [SQuAD](https://huggingface.co/datasets/squad) 数据集上微调 [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)，用于抽取式问答。
2. 使用微调后的模型进行推断。

<Tip>

如果您想查看所有与本任务兼容的架构和检查点，最好查看[任务页](https://huggingface.co/tasks/question-answering)。

</Tip>

在开始之前，请确保您已安装所有必要的库：

```bash
pip install transformers datasets evaluate
```

建议您登录 Hugging Face 账户，以便将模型上传并分享给社区。在提示时，输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 SQuAD 数据集

首先从 🤗 Datasets 库中加载 SQuAD 数据集的一个较小子集。这样您可以先进行实验，确保一切正常，再花更多时间在完整数据集上进行训练。

```py
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

使用 [`~datasets.Dataset.train_test_split`] 方法将数据集的 `train` 划分为训练集和测试集：

```py
>>> squad = squad.train_test_split(test_size=0.2)
```

然后查看一个示例：

```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

这里有几个重要字段：

- `answers`：答案词元的起始位置及答案文本。
- `context`：模型需要从中提取答案的背景信息。
- `question`：模型应该回答的问题。

## 预处理

<Youtube id="qgaM0weJHpA"/>

下一步是加载 DistilBERT 分词器，对 `question` 和 `context` 字段进行处理：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

问答任务有一些特别的预处理步骤需要注意：

1. 数据集中的某些示例可能具有非常长的 `context`，超过了模型的最大输入长度。为处理较长的序列，仅截断 `context` 部分，设置 `truncation="only_second"`。
2. 接下来，通过设置 `return_offset_mapping=True`，将答案的起始和结束位置映射回原始的 `context`。
3. 有了映射后，即可找到答案的起始和结束词元。使用 [`~tokenizers.Encoding.sequence_ids`] 方法找出偏移量的哪部分对应 `question`，哪部分对应 `context`。

下面是创建函数以截断并将 `answer` 的起止词元映射到 `context` 的方法：

```py
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding="max_length",
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         # 找到上下文的起始和结束位置
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # 如果答案不完全在上下文内，标记为 (0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # 否则为答案的起止词元位置
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

使用 🤗 Datasets 的 [`~datasets.Dataset.map`] 函数将预处理函数应用于整个数据集。通过设置 `batched=True` 一次处理数据集的多个元素，可以加速 `map` 函数。删除不需要的列：

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

现在使用 [`DefaultDataCollator`] 创建一批样本。与 🤗 Transformers 中的其他数据整理器不同，[`DefaultDataCollator`] 不会应用任何额外的预处理（如填充）。

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## 训练

<Tip>

如果您不熟悉使用 [`Trainer`] 微调模型，请查看[这里](../training#train-with-pytorch-trainer)的基础教程！

</Tip>

现在可以开始训练模型了！使用 [`AutoModelForQuestionAnswering`] 加载 DistilBERT：

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

此时，只剩三个步骤：

1. 在 [`TrainingArguments`] 中定义训练超参数。唯一必需的参数是 `output_dir`，它指定保存模型的位置。通过设置 `push_to_hub=True`，将模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。
2. 将训练参数传递给 [`Trainer`]，同时传入模型、数据集、分词器和数据整理器。
3. 调用 [`~Trainer.train`] 微调您的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将模型分享到 Hub，让所有人都能使用您的模型：

```py
>>> trainer.push_to_hub()
```

<Tip>

如需了解如何微调问答模型的更深入示例，请参阅相应的
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)。

</Tip>

## 评估

问答任务的评估需要大量后处理工作。为了不占用您太多时间，本指南跳过了评估步骤。[`Trainer`] 在训练过程中仍然会计算评估损失，因此您对模型性能并非完全一无所知。

如果您有更多时间，并且对如何评估问答模型感兴趣，可以查看 🤗 Hugging Face 课程中的[问答](https://huggingface.co/course/chapter7/7?fw=pt#post-processing)章节！

## 推断

很好，现在您已经微调了模型，可以用它进行推断了！

准备一个问题和一些您希望模型作出预测的上下文：

```py
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

使用微调后的模型进行推断最简单的方式是在 [`pipeline`] 中使用它。用您的模型实例化一个问答 `pipeline`，并将文本传递给它：

```py
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401,
 'start': 10,
 'end': 95,
 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
```

如果您愿意，也可以手动复现 `pipeline` 的结果：

对文本进行分词并返回 PyTorch 张量：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

将输入传递给模型并返回输出：

```py
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

从模型输出中获取起始和结束位置的最高概率：

```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

解码预测的词元以获取答案：

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
