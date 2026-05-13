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

# 摘要

[[open-in-colab]]

<Youtube id="yHnr5Dk2zCI"/>

摘要任务生成文档或文章的简短版本，同时保留所有重要信息。与翻译类似，它是另一个可以表述为序列到序列任务的例子。摘要可以分为：

- 抽取式：从文档中提取最相关的信息。
- 生成式：生成能够捕获最重要信息的新文本。

本指南将向您展示如何：

1. 在 [BillSum](https://huggingface.co/datasets/billsum) 数据集的加利福尼亚州法案子集上微调 [T5](https://huggingface.co/google-t5/t5-small)，用于生成式摘要。
2. 使用微调后的模型进行推断。

<Tip>

如果您想查看所有与本任务兼容的架构和检查点，最好查看[任务页](https://huggingface.co/tasks/summarization)。

</Tip>

在开始之前，请确保您已安装所有必要的库：

```bash
pip install transformers datasets evaluate rouge_score
```

建议您登录 Hugging Face 账户，以便将模型上传并分享给社区。在提示时，输入您的令牌进行登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 BillSum 数据集

首先从 🤗 Datasets 库中加载 BillSum 数据集中较小的加利福尼亚州法案子集：

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

使用 [`~datasets.Dataset.train_test_split`] 方法将数据集划分为训练集和测试集：

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

然后查看一个示例：

```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows: ...',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

您会用到的两个字段是：

- `text`：法案文本，将作为模型的输入。
- `summary`：`text` 的精简版本，将作为模型的目标输出。

## 预处理

下一步是加载 T5 分词器，处理 `text` 和 `summary`：

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

您要创建的预处理函数需要：

1. 在输入前添加提示词，让 T5 知道这是一个摘要任务。某些能够处理多种 NLP 任务的模型需要针对特定任务提示。
2. 在对标签进行分词时使用关键字参数 `text_target`。
3. 将序列截断至不超过 `max_length` 参数设置的最大长度。

```py
>>> prefix = "summarize: "


>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

使用 🤗 Datasets 的 [`~datasets.Dataset.map`] 方法将预处理函数应用于整个数据集。通过设置 `batched=True` 一次处理数据集的多个元素，可以加速 `map` 函数：

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

现在使用 [`DataCollatorForSeq2Seq`] 创建一批样本。在整理时将句子*动态填充*至批次中的最长长度，比将整个数据集填充至最大长度更高效。

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

## 评估

在训练过程中加入评估指标有助于评估模型的性能。您可以使用 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载评估方法。对于此任务，加载 [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) 指标（参阅 🤗 Evaluate [快速教程](https://huggingface.co/docs/evaluate/a_quick_tour)，了解更多关于加载和计算指标的信息）：

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

然后创建一个函数，将您的预测结果和标签传递给 [`~evaluate.EvaluationModule.compute`] 来计算 ROUGE 指标：

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
...     result["gen_len"] = np.mean(prediction_lens)

...     return {k: round(v, 4) for k, v in result.items()}
```

您的 `compute_metrics` 函数已准备就绪，在设置训练时会用到它。

## 训练

<Tip>

如果您不熟悉使用 [`Trainer`] 微调模型，请查看[这里](../training#train-with-pytorch-trainer)的基础教程！

</Tip>

现在可以开始训练模型了！使用 [`AutoModelForSeq2SeqLM`] 加载 T5：

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

此时，只剩三个步骤：

1. 在 [`Seq2SeqTrainingArguments`] 中定义训练超参数。唯一必需的参数是 `output_dir`，它指定保存模型的位置。通过设置 `push_to_hub=True`，将模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。每个 epoch 结束时，[`Trainer`] 将评估 ROUGE 指标并保存训练检查点。
2. 将训练参数传递给 [`Seq2SeqTrainer`]，同时传入模型、数据集、分词器、数据整理器和 `compute_metrics` 函数。
3. 调用 [`~Trainer.train`] 微调您的模型。

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_billsum_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=4,
...     predict_with_generate=True,
...     fp16=True, #change to bf16=True for XPU
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将模型分享到 Hub，让所有人都能使用您的模型：

```py
>>> trainer.push_to_hub()
```

<Tip>

如需了解如何微调摘要模型的更深入示例，请参阅相应的
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)。

</Tip>

## 推断

很好，现在您已经微调了模型，可以用它进行推断了！

准备一些您想要生成摘要的文本。对于 T5，您需要根据所处理的任务为输入添加前缀。对于摘要任务，前缀如下所示：

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

对文本进行分词并将 `input_ids` 作为 PyTorch 张量返回：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

使用 [`~generation.GenerationMixin.generate`] 方法创建摘要。有关不同文本生成策略和控制生成参数的更多详情，请查阅[文本生成](../main_classes/text_generation) API。

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

将生成的词元 id 解码回文本：

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it\'s the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
