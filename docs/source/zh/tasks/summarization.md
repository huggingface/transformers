<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在许可证的以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不提供任何明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含了我们的文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确呈现。
-->

# 摘要

[[在 Colab 中打开]]

<Youtube id="yHnr5Dk2zCI"/>

摘要创建了一个较短的文档或文章版本，捕捉所有重要信息。与翻译一样，它是可以被定义为序列到序列任务的另一个示例。摘要可以是:

- 抽取式：从文档中提取出最相关的信息。- 归纳式：生成包含最相关信息的新文本。

本指南将向您展示如何:

1. 在加利福尼亚州法案的 BillSum 数据集的基础上，对 [T5](https://huggingface.co/t5-small) 进行微调，用于归纳式摘要。
2. 使用您微调的模型进行推理。
<Tip> 本教程中介绍的任务受以下模型架构的支持:
<!--此提示由`make fix-copies`自动生成，不要手动填写！-->
[BART](../model_doc/bart), [BigBird-Pegasus](../model_doc/bigbird_pegasus), [Blenderbot](../model_doc/blenderbot), [BlenderbotSmall](../model_doc/blenderbot-small), [Encoder decoder](../model_doc/encoder-decoder), [FairSeq Machine-Translation](../model_doc/fsmt), [GPTSAN-japanese](../model_doc/gptsan-japanese), [LED](../model_doc/led), [LongT5](../model_doc/longt5), [M2M100](../model_doc/m2m_100), [Marian](../model_doc/marian), [mBART](../model_doc/mbart), [MT5](../model_doc/mt5), [MVP](../model_doc/mvp), [NLLB](../model_doc/nllb), [NLLB-MOE](../model_doc/nllb-moe), [Pegasus](../model_doc/pegasus), [PEGASUS-X](../model_doc/pegasus_x), [PLBart](../model_doc/plbart), [ProphetNet](../model_doc/prophetnet), [SwitchTransformers](../model_doc/switch_transformers), [T5](../model_doc/t5), [XLM-ProphetNet](../model_doc/xlm-prophetnet)
<!--生成提示的结束-->
</Tip>

开始之前，请确保已安装所有必要的库:
```bash
pip install transformers datasets evaluate rouge_score
```

我们鼓励您登录到您的 Hugging Face 帐户，这样您就可以与社区上传和共享您的模型。在提示时，输入您的令牌进行登录:
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 BillSum 数据集

首先，使用🤗 Datasets 库加载较小的加利福尼亚州法案子集的 BillSum 数据集:
```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

使用 [`~datasets.Dataset.train_test_split`] 方法将数据集拆分为训练集和测试集:
```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

然后查看一个示例:
```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

有两个字段您将要使用:

- `text`：法案的文本，将作为模型的输入。
- `summary`：`text` 的简化版本，将作为模型的目标。

## 预处理

下一步是加载一个 T5 分词器 (Tokenizer)来处理 `text` 和 `summary`:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

您要创建的预处理函数需要执行以下操作:

1. 使用提示为输入添加前缀，以便 T5 知道这是一个摘要任务。某些能够执行多个 NLP 任务的模型需要为特定任务提供提示。
2. 在标记化标签时使用 `text_target` 关键字参数
3. 将序列截断为不超过 `max_length` 参数设置的最大长度。

```py
>>> prefix = "summarize: "


>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

要在整个数据集上应用预处理函数，使用🤗 Datasets [`~datasets.Dataset.map`] 方法。

通过将 `batched=True` 设置为一次处理数据集的多个元素，可以加快 `map` 函数的速度:

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

现在使用 [`DataCollatorForSeq2Seq`] 创建一个示例批次。在整理过程中，将句子动态填充到批次中的最长长度，而不是将整个数据集填充到最大长度。

<frameworkcontent> 
<pt>

 ```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```
</pt> 
<tf> 

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```
</tf>
</frameworkcontent>


## 评估

在训练过程中包含一个度量标准通常有助于评估模型的性能。您可以使用🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载一个评估方法。对于此任务，加载 [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) 度量标准（查看🤗 Evaluate [快速导览](https://huggingface.co/docs/evaluate/a_quick_tour) 以了解如何加载和计算度量标准）:

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

然后创建一个函数，将您的预测和标签传递给 [`~evaluate.EvaluationModule.compute`] 以计算 ROUGE 度量标准:
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

您的 `compute_metrics` 函数已准备就绪，在设置训练时将返回到它。

## 训练

<frameworkcontent>
<pt>
<Tip>

如果您对使用 [`Trainer`] 进行模型微调不熟悉，请查看基本教程 [此处](../training#train-with-pytorch-trainer)！
</Tip>
现在，您已准备好开始训练模型了！使用 [`AutoModelForSeq2SeqLM`] 加载 T5:
```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

此时，仅剩下三个步骤:

1. 在 [`Seq2SeqTrainingArguments`] 中定义您的训练超参数。唯一需要的参数是 `output_dir`，用于指定保存模型的位置。通过设置 `push_to_hub=True` 将该模型推送到 Hub（需要登录到 Hugging Face 才能上传模型）。在每个时期结束时，[`Trainer`] 将评估 ROUGE 度量标准并保存训练检查点。
2. 将训练参数与模型、数据集、分词器 (Tokenizer)、数据整理器和 `compute_metrics` 函数一起传递给 [`Seq2SeqTrainer`]。
3. 调用 [`~Trainer.train`] 以微调您的模型。

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_billsum_model",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=4,
...     predict_with_generate=True,
...     fp16=True,
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

完成训练后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub，以便所有人都可以使用您的模型:
```py
>>> trainer.push_to_hub()
```
</pt> 
<tf> 

<Tip>

如果您对使用 Keras 微调模型不熟悉，请查看基本教程 [此处](../training#train-a-tensorflow-model-with-keras)！
</Tip> 要在 TensorFlow 中微调模型，请首先设置优化器函数、学习率调度和一些训练超参数:
```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

然后，您可以使用 [`TFAutoModelForSeq2SeqLM`] 加载 T5:
```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

使用 [`~transformers.TFPreTrainedModel.prepare_tf_dataset`] 将数据集转换为 `tf.data.Dataset` 格式:
```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_billsum["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_billsum["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

使用 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 配置模型进行训练。请注意，Transformers 模型都具有默认的与任务相关的损失函数，因此您不需要指定除非您想要使用其他损失函数:
```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

在开始训练之前，最后两件事是从预测中计算 ROUGE 分数，并提供一种推送模型到 Hub 的方式。这两件事都是通过使用 [Keras 回调](../main_classes/keras_callbacks) 来完成的。

将您的 `compute_metrics` 函数传递给 [`~transformers.KerasMetricCallback`]:
```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

在 [`~transformers.PushToHubCallback`] 中指定将模型和分词器 (Tokenizer)推送到的位置:
```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_billsum_model",
...     tokenizer=tokenizer,
... )
```

然后将您的回调函数捆绑在一起：
```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

最后，您可以开始训练模型了！调用 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 函数并传入训练和验证数据集、训练周期数以及回调函数来微调模型：
```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

训练完成后，您的模型将自动上传到 Hub，以供大家使用！
</tf>
</frameworkcontent>
<Tip>

要了解有关如何为摘要微调模型的更详细示例，请参阅相应的 [PyTorch 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb) 或者 [TensorFlow 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)。
</Tip>

## 推断

太棒了，现在您已经微调了一个模型，可以用它进行推断了！

想出一些您想要进行摘要的文本。对于 T5 模型，您需要根据您正在处理的任务为输入添加前缀。对于摘要任务，您应该按照下面所示的方式为输入添加前缀：

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

尝试使用 [`pipeline`] 是尝试推断您微调后的模型的最简单方法。

使用您的模型实例化一个用于摘要的 `pipeline`，并将文本传递给它：

```py
>>> from transformers import pipeline

>>> summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
>>> summarizer(text)
[{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
```

如果您愿意，还可以手动复制 `pipeline` 的结果：


<frameworkcontent> 
<pt> 

 将文本进行标记化并将 `input_ids` 作为 PyTorch 张量返回：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

使用 [`~transformers.generation_utils.GenerationMixin.generate`] 方法创建摘要。有关不同文本生成策略和参数控制生成的更多详细信息，请查看 [Text Generation](../main_classes/text_generation) API。
```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

将生成的标记 ID 解码回文本：
```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</pt> 
<tf> 

将文本进行标记化并将 `input_ids` 作为 TensorFlow 张量返回：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

使用 [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] 方法创建摘要。有关不同文本生成策略和参数控制生成的更多详细信息，请查看 [Text Generation](../main_classes/text_generation) API。
```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

将生成的标记 ID 解码回文本：
```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</tf>
</frameworkcontent>

