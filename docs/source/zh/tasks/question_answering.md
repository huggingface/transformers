<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；您除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。-->
⚠️请注意，该文件是 Markdown 格式的，但包含了我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中不能正常显示。
-->

# 问答

[[在 Colab 中打开]]
<Youtube id="ajPx5LwJD-I"/>

问答任务是在给定问题的情况下返回一个答案。如果您曾经询问过像 Alexa、Siri 或 Google 这样的虚拟助手天气如何，那么您以前肯定使用过问答模型。有两种常见的问答任务类型：

- 抽取型：从给定的上下文中提取答案。- 生成型：根据上下文生成能正确回答问题的答案。

本指南将向您展示如何：

1. 在 [SQuAD](https://huggingface.co/datasets/squad) 数据集上对 [DistilBERT](https://huggingface.co/distilbert-base-uncased) 进行微调，以实现抽取型问答。

2. 使用您微调的模型进行推理。

<Tip>

 本教程中演示的任务由以下模型架构支持：
<!--此提示由`make fix-copies`自动生成，请勿手动填写！-->
[ALBERT](../model_doc/albert)，[BART](../model_doc/bart)，[BERT](../model_doc/bert)，[BigBird](../model_doc/big_bird)，[BigBird-Pegasus](../model_doc/bigbird_pegasus)，[BLOOM](../model_doc/bloom)，[CamemBERT](../model_doc/camembert)，[CANINE](../model_doc/canine)，[ConvBERT](../model_doc/convbert)，[Data2VecText](../model_doc/data2vec-text)，[DeBERTa](../model_doc/deberta)，[DeBERTa-v2](../model_doc/deberta-v2)，[DistilBERT](../model_doc/distilbert)，[ELECTRA](../model_doc/electra)，[ERNIE](../model_doc/ernie)，[ErnieM](../model_doc/ernie_m)，[FlauBERT](../model_doc/flaubert)，[FNet](../model_doc/fnet)，[Funnel Transformer](../model_doc/funnel)，[OpenAI GPT-2](../model_doc/gpt2)，[GPT Neo](../model_doc/gpt_neo)，[GPT NeoX](../model_doc/gpt_neox)，[GPT-J](../model_doc/gptj)，[I-BERT](../model_doc/ibert)，[LayoutLMv2](../model_doc/layoutlmv2)，[LayoutLMv3](../model_doc/layoutlmv3)，[LED](../model_doc/led)，[LiLT](../model_doc/lilt)，[Longformer](../model_doc/longformer)，[LUKE](../model_doc/luke)，[LXMERT](../model_doc/lxmert)，[MarkupLM](../model_doc/markuplm)，[mBART](../model_doc/mbart)，[MEGA](../model_doc/mega)，[Megatron-BERT](../model_doc/megatron-bert)，[MobileBERT](../model_doc/mobilebert)，[MPNet](../model_doc/mpnet)，[MVP](../model_doc/mvp)，[Nezha](../model_doc/nezha)，[Nystr ö mformer](../model_doc/nystromformer)，[OPT](../model_doc/opt)，[QDQBert](../model_doc/qdqbert)，[Reformer](../model_doc/reformer)，[RemBERT](../model_doc/rembert)，[RoBERTa](../model_doc/roberta)，[RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm)，[RoCBert](../model_doc/roc_bert)，[RoFormer](../model_doc/roformer)，[Splinter](../model_doc/splinter)，[SqueezeBERT](../model_doc/squeezebert)，[XLM](../model_doc/xlm)，[XLM-RoBERTa](../model_doc/xlm-roberta)，[XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl)，[XLNet](../model_doc/xlnet)，[X-MOD](../model_doc/xmod)，[YOSO](../model_doc/yoso)

<!--生成提示的末尾-->
</Tip>

开始之前，请确保已安装所有必要的库：
```bash
pip install transformers datasets evaluate
```

我们鼓励您登录您的 Hugging Face 账户，这样您就可以上传和分享您的模型。在提示时，输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 SQuAD 数据集

首先，从🤗 Datasets 库中加载 SQuAD 数据集的一个较小子集。这样您就有机会在使用完整数据集进行更长时间的训练之前进行实验和确保一切正常。
```py
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

使用 [`~datasets.Dataset.train_test_split`] 方法将数据集的“train”拆分为训练集和测试集：
```py
>>> squad = squad.train_test_split(test_size=0.2)
```

然后看一个示例：
```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

这里有几个重要的字段：

- `answers`：答案标记的起始位置和答案文本。
- `context`：模型需要从中提取答案的背景信息。
- `question`：模型应该回答的问题。

## 预处理
<Youtube id="qgaM0weJHpA"/>

下一步是加载一个 DistilBERT 分词器 (Tokenizer)来处理 `question` 和 `context` 字段：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

对于问答任务，有几个特定的预处理步骤需要注意：
1. 数据集中的某些示例可能具有非常长的 `context`，超过模型的最大输入长度。为了处理更长的序列，只截断 `context`，设置 `truncation="only_second"`。
2. 接下来，通过设置 `return_offset_mapping=True`，将答案的起始位置和结束位置映射到原始的 `context`。
3. 有了映射之后，现在您可以找到答案的起始和结束标记。使用 [`~tokenizers.Encoding.sequence_ids`] 方法，找到偏移的哪部分对应 `question`，哪部分对应 `context`。

这是您可以创建一个函数来截断和映射 `answer` 的起始和结束标记到 `context` 的方法：  

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

...         # Find the start and end of the context
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # If the answer is not fully inside the context, label it (0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # Otherwise it's the start and end token positions
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

要在整个数据集上应用预处理函数，使用🤗 Datasets 的 [`~datasets.Dataset.map`] 函数。

您可以通过设置 `batched=True` 来加速 `map` 函数，以同时处理数据集的多个元素。删除您不需要的任何列：

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

现在使用 [`DefaultDataCollator`] 创建一批示例。与🤗 Transformers 中的其他数据整理器不同，[`DefaultDataCollator`] 不会应用任何额外的预处理，例如填充。

<frameworkcontent> 
<pt> 

 ```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt> 
<tf> 

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>


## 训练

<frameworkcontent>
<pt>
<Tip>

如果您对使用 [`Trainer`] 进行模型微调不熟悉，请查看基本教程 [此处](../training#train-with-pytorch-trainer)！
</Tip>

现在您可以加载 DistilBERT 模型 [`AutoModelForQuestionAnswering`]：
```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```

此时，只剩下三个步骤：

1. 在 [`TrainingArguments`] 中定义您的训练超参数。唯一必需的参数是 `output_dir`，用于指定保存模型的位置。通过设置 `push_to_hub=True` 将此模型推送到 Hub（您需要登录 Hugging Face 才能上传模型）。
2. 将训练参数与模型、数据集、分词器 (Tokenizer)和数据整理器一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 来进行模型微调。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     evaluation_strategy="epoch",
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
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

完成训练后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub，以便每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```
</pt>
<tf>

<Tip>

如果您对使用 Keras 进行模型微调不熟悉，请查看基本教程 [此处](../training#train-a-tensorflow-model-with-keras)！

</Tip> 

为了在 TensorFlow 中微调模型，首先需要设置优化器函数、学习率调度和一些训练超参数:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 2
>>> total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
>>> optimizer, schedule = create_optimizer(
...     init_lr=2e-5,
...     num_warmup_steps=0,
...     num_train_steps=total_train_steps,
... )
```

然后，可以使用 [`TFAutoModelForQuestionAnswering`] 加载 DistilBERT:

```py

>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering("distilbert-base-uncased")
```

使用 [`~transformers.TFPreTrainedModel.prepare_tf_dataset`] 将数据集转换为 `tf.data.Dataset` 格式:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_squad["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_squad["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

使用 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 配置模型进行训练:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

在开始训练之前，最后一件要做的是提供一种将模型和标记器推送到 Hub 的方式。

可以通过 [`~transformers.PushToHubCallback`] 指定将模型和标记器推送到何处:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_qa_model",
...     tokenizer=tokenizer,
... )
```

最后，准备好开始训练模型了！使用训练和验证数据集、迭代次数和微调模型的回调函数调用 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 进行训练:
```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=[callback])
```

训练完成后，模型会自动上传到 Hub，这样每个人都可以使用它！</tf> </frameworkcontent>
<Tip>

要了解如何对问题回答模型进行更深入的微调示例，请参考相应的 [PyTorch 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) 或 [TensorFlow 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)。
</Tip>

## 评估

问题回答的评估需要大量的后处理。为了不占用太多时间，本指南跳过了评估步骤。[`Trainer`] 仍然会在训练过程中计算评估损失，因此您不会完全不了解模型的性能。
如果您有更多时间，并且对如何评估问题回答模型感兴趣，请参考🤗 Hugging Face 课程中的 [问题回答](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) 章节！

## 推理

很好，现在您已经微调了一个模型，可以用它进行推理！
提出一个问题和一些上下文，希望模型预测:

```py
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

尝试使用 [`pipeline`] 来进行微调模型的推理是最简单的方法。使用您的模型实例化一个用于问题回答的 `pipeline`，并将文本传递给它:
```py
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401,
 'start': 10,
 'end': 95,
 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
```

如果愿意，也可以手动复制 `pipeline` 的结果:

<frameworkcontent>
<pt>

 对文本进行分词并返回 PyTorch 张量:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

将输入传递给模型并返回 `logits`:
```py
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

从模型输出中获取开始和结束位置的最高概率:
```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

解码预测的标记以获取答案:
```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</pt>
<tf>

对文本进行分词并返回 TensorFlow 张量:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, text, return_tensors="tf")
```

将输入传递给模型并返回 `logits`:
```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> outputs = model(**inputs)
```

从模型输出中获取开始和结束位置的最高概率:
```py
>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
```

解码预测的标记以获取答案:
```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</tf>
</frameworkcontent>
