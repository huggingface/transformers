<!--版权所有 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版许可（“许可证”），您除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的，不附带任何形式的担保或条件。请参阅许可证以获取特定语言中有关权限和限制的权限。-->
⚠️请注意，此文件是 Markdown 格式的，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能无法正确地在您的 Markdown 查看器中呈现。
-->

# 掩码语言建模 Masked language modeling

[[在 Google Colab 中打开]]
<Youtube id="mqElG5QJWUg"/>

掩码语言建模是预测序列中的掩码标记，模型可以双向关注标记。这意味着模型可以完全访问左右的标记。掩码语言建模非常适合需要对整个序列进行良好上下文理解的任务。BERT 就是掩码语言模型的一个示例。本指南将向您展示如何：
1. 在 [ELI5](https://huggingface.co/datasets/eli5) 数据集的 [r/askscience](https://www.reddit.com/r/askscience/) 子集上微调 [DistilRoBERTa](https://huggingface.co/distilroberta-base)。

2. 使用您微调的模型进行推理。
<Tip> 您可以按照本指南中的相同步骤微调其他架构的掩码语言建模。
<Tip>

请选择以下架构之一：
<!--此提示由`make fix-copies`自动生成，请勿手动填写！-->
[ALBERT](../model_doc/albert), [BART](../model_doc/bart), [BERT](../model_doc/bert), [BigBird](../model_doc/big_bird), [CamemBERT](../model_doc/camembert), [ConvBERT](../model_doc/convbert), [Data2VecText](../model_doc/data2vec-text), [DeBERTa](../model_doc/deberta), [DeBERTa-v2](../model_doc/deberta-v2), [DistilBERT](../model_doc/distilbert), [ELECTRA](../model_doc/electra), [ERNIE](../model_doc/ernie), [ESM](../model_doc/esm), [FlauBERT](../model_doc/flaubert), [FNet](../model_doc/fnet), [Funnel Transformer](../model_doc/funnel), [I-BERT](../model_doc/ibert), [LayoutLM](../model_doc/layoutlm), [Longformer](../model_doc/longformer), [LUKE](../model_doc/luke), [mBART](../model_doc/mbart), [MEGA](../model_doc/mega), [Megatron-BERT](../model_doc/megatron-bert), [MobileBERT](../model_doc/mobilebert), [MPNet](../model_doc/mpnet), [MVP](../model_doc/mvp), [Nezha](../model_doc/nezha), [Nystr ö mformer](../model_doc/nystromformer), [Perceiver](../model_doc/perceiver), [QDQBert](../model_doc/qdqbert), [Reformer](../model_doc/reformer), [RemBERT](../model_doc/rembert), [RoBERTa](../model_doc/roberta), [RoBERTa-PreLayerNorm](../model_doc/roberta-prelayernorm), [RoCBert](../model_doc/roc_bert), [RoFormer](../model_doc/roformer), [SqueezeBERT](../model_doc/squeezebert), [TAPAS](../model_doc/tapas), [Wav2Vec2](../model_doc/wav2vec2), [XLM](../model_doc/xlm), [XLM-RoBERTa](../model_doc/xlm-roberta), [XLM-RoBERTa-XL](../model_doc/xlm-roberta-xl), [X-MOD](../model_doc/xmod), [YOSO](../model_doc/yoso)
<!--End of the generated tip-->

</Tip>

开始之前，请确保已安装所有必需的库：
```bash
pip install transformers datasets evaluate
```

我们鼓励您登录您的 Hugging Face 账户，这样您就可以与社区共享和上传您的模型。在提示时，输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 ELI5 数据集

首先从🤗 Datasets 库中加载 r/askscience 子集的较小数据集。这样可以让您有机会进行实验，并确保一切正常，然后再在完整数据集上进行更长时间的训练。使用 [`~datasets.Dataset.train_test_split`] 方法将数据集的 `train_asks` 拆分为训练集和测试集：
```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5", split="train_asks[:5000]")
```

然后查看一个示例：
```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

Then take a look at an example:

```py
>>> eli5["train"][0]
{'answers': {'a_id': ['c3d1aib', 'c3d4lya'],
  'score': [6, 3],
  'text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
   "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"]},
 'answers_urls': {'url': []},
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls': {'url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg']},
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls': {'url': []}}
```

虽然这看起来可能很多，但您实际上只对 `text` 字段感兴趣。语言建模任务的有趣之处在于您不需要标签（也称为无监督任务），因为下一个词就是标签。

## 预处理
<Youtube id="8PmhEIXhBvI"/>

对于掩码语言建模，下一步是加载一个 DistilRoBERTa tokenizer 来处理 `text` 子字段：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
```

从上面的示例中，您会注意到 `text` 字段实际上是嵌套在 `answers` 中的。这意味着您需要使用 [`flatten`](https://huggingface.co/docs/datasets/process.html#flatten) 方法从嵌套结构中提取 `text` 子字段：

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'answers.a_id': ['c3d1aib', 'c3d4lya'],
 'answers.score': [6, 3],
 'answers.text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
  "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"],
 'answers_urls.url': [],
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls.url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg'],
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls.url': []}
```

现在，每个子字段都成为一个单独的列，由 `answers` 前缀指示，`text` 字段现在是一个列表。与单独对每个句子进行标记化不同，将列表转换为字符串，以便可以同时对它们进行标记化。

这是一个第一个预处理函数，用于连接每个示例的字符串列表并对结果进行标记化：
```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]])
```

要在整个数据集上应用此预处理函数，使用🤗 Datasets [`~datasets.Dataset.map`] 方法。您可以通过将 `batched=True` 设置为一次处理数据集的多个元素，并使用 `num_proc` 增加进程的数量来加快 `map` 函数的速度。删除您不需要的任何列：
```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

此数据集包含标记序列，但其中一些序列长度超过了模型的最大输入长度。

现在，您可以使用第二个预处理函数来- 连接所有序列- 将连接的序列分割为 `block_size` 定义的较短块，该块应既短于最大输入长度又短到足够适应您的 GPU RAM。
```py
>>> block_size = 128


>>> def group_texts(examples):
...     # Concatenate all texts.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
...     # customize this part to your needs.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # Split by chunks of block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     result["labels"] = result["input_ids"].copy()
...     return result
```

对整个数据集应用 `group_texts` 函数：
```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

现在使用 [`DataCollatorForLanguageModeling`] 创建一批示例。在整理期间，将句子动态填充到批次中的最长长度，而不是将整个数据集填充到最大长度。

<frameworkcontent> 
<pt> 

使用结束序列标记作为填充标记，并指定 `mlm_probability` 以在每次迭代数据时随机屏蔽标记：
```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```
</pt> <tf>

使用结束序列标记作为填充标记，并指定 `mlm_probability` 以在每次迭代数据时随机屏蔽标记：

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
```
</tf>
</frameworkcontent>


## 训练

<frameworkcontent> 
<pt> 
<Tip>

如果您对使用 [`Trainer`] 微调模型不熟悉，请查看 [此处](../training#train-with-pytorch-trainer) 的基本教程！
</Tip>

现在，您可以开始训练模型了！使用 [`AutoModelForMaskedLM`] 加载 DistilRoBERTa：
```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
```

此时，只剩下三个步骤：
1. 在 [`TrainingArguments`] 中定义您的训练超参数。唯一必需的参数是 `output_dir`，用于指定保存模型的位置。通过设置 `push_to_hub=True` 将该模型推送到 Hub（您需要登录 Hugging Face 以上传模型）。
2. 将训练参数与模型、数据集和数据整理器一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 来微调您的模型。
```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_mlm_model",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.evaluate`] 方法评估模型并获取其困惑度：
```py
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 8.76
```

然后使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型共享到 Hub 上，这样每个人都可以使用您的模型：
```py
>>> trainer.push_to_hub()
```
</pt> 
<tf> 

<Tip>
如果您对使用 Keras 进行模型微调不熟悉，请查看基础教程 [这里](../training#train-a-tensorflow-model-with-keras)！
</Tip> 要在 TensorFlow 中微调模型，请首先设置优化器函数、学习率调度和一些训练超参数：
```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

然后，您可以使用 [`TFAutoModelForMaskedLM`] 加载 DistilRoBERTa：
```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained("distilroberta-base")
```

将您的数据集转换为 `tf.data.Dataset` 格式，使用 [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]：
```py
>>> tf_train_set = model.prepare_tf_dataset(
...     lm_dataset["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     lm_dataset["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

使用 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 配置模型进行训练。请注意，Transformers 模型都有一个默认的与任务相关的损失函数，所以您不需要指定损失函数，除非您想要自定义：
```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

可以通过在 [`~transformers.PushToHubCallback`] 中指定模型和分词器 (Tokenizer)的推送位置来实现：
```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_eli5_mlm_model",
...     tokenizer=tokenizer,
... )
```

最后，您可以开始训练您的模型了！使用您的训练和验证数据集、训练周期数和回调函数调用 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 来微调模型：
```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])
```

训练完成后，您的模型会自动上传到 Hub，这样每个人都可以使用它！
</tf>
</frameworkcontent>
<Tip>

有关如何为遮蔽语言建模微调模型的更详细示例，请查看相应的 [PyTorch 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) 或者 [TensorFlow 笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)。
</Tip>

## 推理

太棒了，现在您已经微调了一个模型，可以用它进行推理了！
想出一些您希望模型填充空白的文本，并使用特殊的 `<mask>` 标记来表示空白处：

```py
>>> text = "The Milky Way is a <mask> galaxy."
```

尝试使用微调后的模型进行推理的最简单方法是在 [`pipeline`] 中使用它。使用您的模型实例化一个用于填充遮蔽的 `pipeline`，并将文本传递给它。如果需要，您可以使用 `top_k` 参数来指定返回多少个预测结果：
```py
>>> from transformers import pipeline

>>> mask_filler = pipeline("fill-mask", "stevhliu/my_awesome_eli5_mlm_model")
>>> mask_filler(text, top_k=3)
[{'score': 0.5150994658470154,
  'token': 21300,
  'token_str': ' spiral',
  'sequence': 'The Milky Way is a spiral galaxy.'},
 {'score': 0.07087188959121704,
  'token': 2232,
  'token_str': ' massive',
  'sequence': 'The Milky Way is a massive galaxy.'},
 {'score': 0.06434620916843414,
  'token': 650,
  'token_str': ' small',
  'sequence': 'The Milky Way is a small galaxy.'}]
```


<frameworkcontent> 
<pt> 

 将文本进行分词，并将 `input_ids` 返回为 PyTorch 张量。您还需要指定 `<mask>` 标记的位置：
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="pt")
>>> mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
```

将输入传递给模型，并返回遮蔽标记的 `logits`：
```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

然后返回最高概率的三个遮蔽标记，并打印出来：
```py
>>> top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
</pt> 
<tf> 

将文本进行分词，并将 `input_ids` 返回为 TensorFlow 张量。您还需要指定 `<mask>` 标记的位置：

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="tf")
>>> mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
```

将输入传递给模型，并返回遮蔽标记的 `logits`：
```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

然后返回最高概率的三个遮蔽标记，并打印出来：
```py
>>> top_3_tokens = tf.math.top_k(mask_token_logits, 3).indices.numpy()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
</tf>
</frameworkcontent>

