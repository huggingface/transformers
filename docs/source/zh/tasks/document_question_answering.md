<!--版权 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）的规定，您不得使用此文件，除非符合许可证。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”分发的基础，没有任何明示或暗示的担保或条件。请参阅许可证具体语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示。
-->

# 文档问答

[[在 Colab 中打开]]
文档问答，又称为文档视觉问答，是一项涉及提供关于文档图像的问题的答案的任务。支持此任务的模型的输入通常是图像和问题的组合，输出是用自然语言表示的答案。这些模型利用多种模态，包括文本、单词位置（边界框）和图像本身。

本指南说明了如何：

- 在 [LayoutLMv2](../model_doc/layoutlmv2) 上对 [DocVQA 数据集](https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut) 进行微调。- 使用您微调的模型进行推理。

<Tip>

本教程演示的任务由以下模型架构支持：
<!--此提示由`make fix-copies`自动生成，请勿手动填写！-->
[LayoutLM](../model_doc/layoutlm)，[LayoutLMv2](../model_doc/layoutlmv2)，[LayoutLMv3](../model_doc/layoutlmv3)
<!--生成提示的末尾-->

</Tip>

LayoutLMv2 通过在令牌的最终隐藏状态之上添加一个问答头来解决文档问答任务，以预测答案的开始和结束令牌的位置。换句话说，问题被视为抽取性问答：给定上下文，提取回答问题的信息片段。上下文来自 OCR 引擎的输出，这里使用的是 Google 的 Tesseract。开始之前，请确保已安装所有必要的库。LayoutLMv2 依赖于 detectron2、torchvision 和 tesseract。

```bash
pip install -q transformers datasets
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torchvision
```

```bash
sudo apt install tesseract-ocr
pip install -q pytesseract
```

安装完所有依赖项后，请重新启动运行时。
我们鼓励您与社区分享您的模型。

登录您的 Hugging Face 帐户将其上传到🤗 Hub。提示时，输入您的令牌以登录：
```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

让我们定义一些全局变量。
```py
>>> model_checkpoint = "microsoft/layoutlmv2-base-uncased"
>>> batch_size = 4
```

## 加载数据

在本指南中，我们使用一个预处理过的 DocVQA 的小样本，您可以在🤗 Hub 上找到。

如果您想使用完整的 DocVQA 数据集，可以在 [DocVQA 主页](https://rrc.cvc.uab.es/?ch=17) 上注册并下载。如果您这样做，要继续进行本指南，请查看 [如何将文件加载到🤗数据集](https://huggingface.co/docs/datasets/loading#local-and-remote-files)。

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("nielsr/docvqa_1200_examples")
>>> dataset
DatasetDict({
    train: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
        num_rows: 200
    })
})
```

正如您所见，数据集已经分为训练集和测试集。随机查看一个示例以熟悉其中的特征。
```py
>>> dataset["train"].features
```

以下是各个字段的含义：* `id`：示例的 ID* `image`：包含文档图像的 PIL.Image.Image 对象 * `query`：问题字符串-自然语言提问，以多种语言提问* `answers`：由人类标注者提供的正确答案列表 * `words` 和 `bounding_boxes`：OCR 的结果，在这里我们不会使用* `answer`：由其他模型匹配的答案，在这里我们不会使用
让我们仅保留英文问题，并删除似乎包含另一个模型预测的 `answer` 特征。我们还将从标注者提供的答案集中选择第一个答案。

或者，您可以随机抽样。
```py
>>> updated_dataset = dataset.map(lambda example: {"question": example["query"]["en"]}, remove_columns=["query"])
>>> updated_dataset = updated_dataset.map(
...     lambda example: {"answer": example["answers"][0]}, remove_columns=["answer", "answers"]
... )
```

请注意，我们在本指南中使用的 LayoutLMv2 检查点已经进行了 `max_position_embeddings = 512` 的训练（您可以在 [检查点的 `config.json` 文件](https://huggingface.co/microsoft/layoutlmv2-base-uncased/blob/main/config.json#L18) 中找到这些信息）。

我们可以截断示例，以避免嵌入可能比 512 还长的情况，在这里，我们将删除几个示例，其中嵌入可能超过 512。如果您的数据集中的大多数文档都很长，您可以实施滑动窗口策略-详细信息请参见 [此笔记本](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb)。

```py
>>> updated_dataset = updated_dataset.filter(lambda x: len(x["words"]) + len(x["question"].split()) < 512)
```

此时，让我们还从此数据集中删除 OCR 功能。这些是另一个模型的微调的 OCR 结果。如果我们要使用它们，它们仍然需要进行一些处理，因为它们不符合我们在本指南中使用的模型的输入要求。相反，我们可以在原始数据上使用 [`LayoutLMv2Processor`] 进行 OCR 和标记化处理。这样，我们将获得与模型期望输入匹配的输入。

如果您想手动处理图像，请查看 [`LayoutLMv2` 模型文档](../model_doc/layoutlmv2) 以了解模型期望的输入格式。

```py
>>> updated_dataset = updated_dataset.remove_columns("words")
>>> updated_dataset = updated_dataset.remove_columns("bounding_boxes")
```

最后，如果我们不查看图像示例，数据探索将不完整。
```py
>>> updated_dataset["train"][11]["image"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/docvqa_example.jpg" alt="DocVQA Image Example"/>
 </div>

## 预处理数据

文档问答任务是一项多模态任务，您需要确保每种模态的输入按照模型的预期进行预处理。让我们从加载 
[`LayoutLMv2Processor`] 开始，该处理器在内部结合了可以处理图像数据的图像处理器 (Image Processor)和可以编码文本数据的标记器。
```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### 预处理文档图像

首先，让我们通过处理器中的 `image_processor` 为模型准备文档图像。默认情况下，图像处理器 (Image Processor)将图像调整大小为 224x224，确保它们具有正确的颜色通道顺序，使用 tesseract 应用 OCR 以获取单词和规范化边界框。在本教程中，所有这些默认值恰好符合我们的需求。

编写一个将默认图像处理应用于一批图像并返回 OCR 结果的函数。

```py
>>> image_processor = processor.image_processor


>>> def get_ocr_words_and_boxes(examples):
...     images = [image.convert("RGB") for image in examples["image"]]
...     encoded_inputs = image_processor(images)

...     examples["image"] = encoded_inputs.pixel_values
...     examples["words"] = encoded_inputs.words
...     examples["boxes"] = encoded_inputs.boxes

...     return examples
```

要快速将此预处理应用于整个数据集，请使用 [`~datasets.Dataset.map`]。
```py
>>> dataset_with_ocr = updated_dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)
```

### 预处理文本数据

一旦我们对图像应用了 OCR，我们就需要对数据集的文本部分进行编码，以准备模型输入。这涉及将我们在上一步中获得的单词和边界框转换为令牌级别的 `input_ids`、`attention_mask`，`token_type_ids` 和 `bbox`。对于文本预处理，我们将需要处理器中的 `tokenizer`。
```py
>>> tokenizer = processor.tokenizer
```

除了上述预处理之外，我们还需要为模型添加标签。对于 `xxxForQuestionAnswering` 模型在🤗 Transformers 中，标签由 `start_positions` 和 `end_positions` 组成，表示答案的起始和结束的标记。让我们从这开始。定义一个能在较大列表（词列表）中找到子列表（答案拆分为单词）的辅助函数。

该函数将以两个列表 `words_list` 和 `answer_list` 作为输入。然后，它将遍历 `words_list` 并检查
当前词是否等于 `words_list` 中的第一个词（words_list [i]），并且长度与 `answer_list` 相等的以当前词为起始的子列表是否等于 `answer_list`。如果条件成立，表示找到了匹配项，函数将记录匹配项及其起始索引（idx）和结束索引（idx + len(answer_list) - 1)。如果找到了多个匹配项，则函数只返回第一个。如果未找到匹配项，函数返回(`None`，0 和 0)。If no match is found, the function returns (`None`, 0, and 0).

```py
>>> def subfinder(words_list, answer_list):
...     matches = []
...     start_indices = []
...     end_indices = []
...     for idx, i in enumerate(range(len(words_list))):
...         if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
...             matches.append(answer_list)
...             start_indices.append(idx)
...             end_indices.append(idx + len(answer_list) - 1)
...     if matches:
...         return matches[0], start_indices[0], end_indices[0]
...     else:
...         return None, 0, 0
```

为了说明该函数如何找到答案的位置，让我们以一个例子来使用它：
```py
>>> example = dataset_with_ocr["train"][1]
>>> words = [word.lower() for word in example["words"]]
>>> match, word_idx_start, word_idx_end = subfinder(words, example["answer"].lower().split())
>>> print("Question: ", example["question"])
>>> print("Words:", words)
>>> print("Answer: ", example["answer"])
>>> print("start_index", word_idx_start)
>>> print("end_index", word_idx_end)
Question:  Who is in  cc in this letter?
Words: ['wie', 'baw', 'brown', '&', 'williamson', 'tobacco', 'corporation', 'research', '&', 'development', 'internal', 'correspondence', 'to:', 'r.', 'h.', 'honeycutt', 'ce:', 't.f.', 'riehl', 'from:', '.', 'c.j.', 'cook', 'date:', 'may', '8,', '1995', 'subject:', 'review', 'of', 'existing', 'brainstorming', 'ideas/483', 'the', 'major', 'function', 'of', 'the', 'product', 'innovation', 'graup', 'is', 'to', 'develop', 'marketable', 'nove!', 'products', 'that', 'would', 'be', 'profitable', 'to', 'manufacture', 'and', 'sell.', 'novel', 'is', 'defined', 'as:', 'of', 'a', 'new', 'kind,', 'or', 'different', 'from', 'anything', 'seen', 'or', 'known', 'before.', 'innovation', 'is', 'defined', 'as:', 'something', 'new', 'or', 'different', 'introduced;', 'act', 'of', 'innovating;', 'introduction', 'of', 'new', 'things', 'or', 'methods.', 'the', 'products', 'may', 'incorporate', 'the', 'latest', 'technologies,', 'materials', 'and', 'know-how', 'available', 'to', 'give', 'then', 'a', 'unique', 'taste', 'or', 'look.', 'the', 'first', 'task', 'of', 'the', 'product', 'innovation', 'group', 'was', 'to', 'assemble,', 'review', 'and', 'categorize', 'a', 'list', 'of', 'existing', 'brainstorming', 'ideas.', 'ideas', 'were', 'grouped', 'into', 'two', 'major', 'categories', 'labeled', 'appearance', 'and', 'taste/aroma.', 'these', 'categories', 'are', 'used', 'for', 'novel', 'products', 'that', 'may', 'differ', 'from', 'a', 'visual', 'and/or', 'taste/aroma', 'point', 'of', 'view', 'compared', 'to', 'canventional', 'cigarettes.', 'other', 'categories', 'include', 'a', 'combination', 'of', 'the', 'above,', 'filters,', 'packaging', 'and', 'brand', 'extensions.', 'appearance', 'this', 'category', 'is', 'used', 'for', 'novel', 'cigarette', 'constructions', 'that', 'yield', 'visually', 'different', 'products', 'with', 'minimal', 'changes', 'in', 'smoke', 'chemistry', 'two', 'cigarettes', 'in', 'cne.', 'emulti-plug', 'te', 'build', 'yaur', 'awn', 'cigarette.', 'eswitchable', 'menthol', 'or', 'non', 'menthol', 'cigarette.', '*cigarettes', 'with', 'interspaced', 'perforations', 'to', 'enable', 'smoker', 'to', 'separate', 'unburned', 'section', 'for', 'future', 'smoking.', '«short', 'cigarette,', 'tobacco', 'section', '30', 'mm.', '«extremely', 'fast', 'buming', 'cigarette.', '«novel', 'cigarette', 'constructions', 'that', 'permit', 'a', 'significant', 'reduction', 'iretobacco', 'weight', 'while', 'maintaining', 'smoking', 'mechanics', 'and', 'visual', 'characteristics.', 'higher', 'basis', 'weight', 'paper:', 'potential', 'reduction', 'in', 'tobacco', 'weight.', '«more', 'rigid', 'tobacco', 'column;', 'stiffing', 'agent', 'for', 'tobacco;', 'e.g.', 'starch', '*colored', 'tow', 'and', 'cigarette', 'papers;', 'seasonal', 'promotions,', 'e.g.', 'pastel', 'colored', 'cigarettes', 'for', 'easter', 'or', 'in', 'an', 'ebony', 'and', 'ivory', 'brand', 'containing', 'a', 'mixture', 'of', 'all', 'black', '(black', 'paper', 'and', 'tow)', 'and', 'ail', 'white', 'cigarettes.', '499150498']
Answer:  T.F. Riehl
start_index 17
end_index 18
```

然而，一旦对示例进行编码，它们将变成这样：
```py
>>> encoding = tokenizer(example["question"], example["words"], example["boxes"])
>>> tokenizer.decode(encoding["input_ids"])
[CLS] who is in cc in this letter? [SEP] wie baw brown & williamson tobacco corporation research & development ...
```

我们需要找到编码输入中答案的位置。* `token_type_ids` 告诉我们哪些标记属于问题，哪些属于文档的词。* `tokenizer.cls_token_id` 将帮助找到输入开头的特殊标记。* `word_ids` 将帮助将原始 `words` 中找到的答案与完整编码输入中的相同答案匹配，并确定编码输入中答案的起始/结束位置。
有了这个想法后，我们来创建一个批量编码数据集的函数：
```py
>>> def encode_dataset(examples, max_length=512):
...     questions = examples["question"]
...     words = examples["words"]
...     boxes = examples["boxes"]
...     answers = examples["answer"]

...     # encode the batch of examples and initialize the start_positions and end_positions
...     encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
...     start_positions = []
...     end_positions = []

...     # loop through the examples in the batch
...     for i in range(len(questions)):
...         cls_index = encoding["input_ids"][i].index(tokenizer.cls_token_id)

...         # find the position of the answer in example's words
...         words_example = [word.lower() for word in words[i]]
...         answer = answers[i]
...         match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())

...         if match:
...             # if match is found, use `token_type_ids` to find where words start in the encoding
...             token_type_ids = encoding["token_type_ids"][i]
...             token_start_index = 0
...             while token_type_ids[token_start_index] != 1:
...                 token_start_index += 1

...             token_end_index = len(encoding["input_ids"][i]) - 1
...             while token_type_ids[token_end_index] != 1:
...                 token_end_index -= 1

...             word_ids = encoding.word_ids(i)[token_start_index : token_end_index + 1]
...             start_position = cls_index
...             end_position = cls_index

...             # loop over word_ids and increase `token_start_index` until it matches the answer position in words
...             # once it matches, save the `token_start_index` as the `start_position` of the answer in the encoding
...             for id in word_ids:
...                 if id == word_idx_start:
...                     start_position = token_start_index
...                 else:
...                     token_start_index += 1

...             # similarly loop over `word_ids` starting from the end to find the `end_position` of the answer
...             for id in word_ids[::-1]:
...                 if id == word_idx_end:
...                     end_position = token_end_index
...                 else:
...                     token_end_index -= 1

...             start_positions.append(start_position)
...             end_positions.append(end_position)

...         else:
...             start_positions.append(cls_index)
...             end_positions.append(cls_index)

...     encoding["image"] = examples["image"]
...     encoding["start_positions"] = start_positions
...     encoding["end_positions"] = end_positions

...     return encoding
```

现在我们有了这个预处理函数，我们可以对整个数据集进行编码：
```py
>>> encoded_train_dataset = dataset_with_ocr["train"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["train"].column_names
... )
>>> encoded_test_dataset = dataset_with_ocr["test"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["test"].column_names
... )
```

让我们看看编码数据集的特征是什么样的：
```py
>>> encoded_train_dataset.features
{'image': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='uint8', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None),
 'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
 'token_type_ids': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
 'bbox': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'start_positions': Value(dtype='int64', id=None),
 'end_positions': Value(dtype='int64', id=None)}
```

## 评估

文档问答评估需要大量的后处理工作。为了节省您的时间，本指南跳过了评估步骤。[`Trainer`] 仍然会在训练过程中计算评估损失，因此您不会对模型的性能完全一无所知。抽取式问答通常使用 F1/完全匹配进行评估。如果您希望自己实现它，请查看 [Hugging Face 课程的问答章节](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) 以获取灵感。

## 训练

恭喜！您已成功完成本指南最艰难的部分，现在您已经准备好训练自己的模型了。

训练包括以下步骤：
* 使用与预处理相同的检查点加载模型 [`AutoModelForDocumentQuestionAnswering`]。
* 在 [`TrainingArguments`] 中定义您的训练超参数。* 定义一个将示例批处理在一起的函数，这里使用 [`DefaultDataCollator`] 即可。
* 将训练参数与模型、数据集和数据收集器一起传递给 [`Trainer`]。
* 调用 [`~Trainer.train`] 来微调您的模型。


```py
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
```

在 [`TrainingArguments`] 中使用 `output_dir` 来指定保存模型的位置，并根据需要配置超参数。

如果您希望与社区分享您的模型，请将 `push_to_hub` 设置为 `True`（您必须登录 Hugging Face 才能上传模型）。在这种情况下，`output_dir` 也将是您的模型检查点将被推送到的仓库的名称。

```py
>>> from transformers import TrainingArguments

>>> # REPLACE THIS WITH YOUR REPO ID
>>> repo_id = "MariaK/layoutlmv2-base-uncased_finetuned_docvqa"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     evaluation_strategy="steps",
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

定义一个简单的数据收集器，将示例批处理在一起。
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

最后，将所有内容整合在一起，并调用 [`~Trainer.train`]：
```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=encoded_train_dataset,
...     eval_dataset=encoded_test_dataset,
...     tokenizer=processor,
... )

>>> trainer.train()
```

要将最终模型添加到🤗 Hub，请创建一个模型卡并调用 `push_to_hub`：
```py
>>> trainer.create_model_card()
>>> trainer.push_to_hub()
```

## 推理

现在，您已经微调了一个 LayoutLMv2 模型，并将其上传到了🤗 Hub，您可以用它进行推理。试用您微调的模型进行推理的最简单的方法是在 [`Pipeline`] 中使用它。
让我们来看个例子：

```py
>>> example = dataset["test"][2]
>>> question = example["query"]["en"]
>>> image = example["image"]
>>> print(question)
>>> print(example["answers"])
'Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?'
['TRRF Vice President', 'lee a. waller']
```

接下来，使用您的模型为文档问答实例化一个管道，并将图像+问题组合传递给它。
```py
>>> from transformers import pipeline

>>> qa_pipeline = pipeline("document-question-answering", model="MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> qa_pipeline(image, question)
[{'score': 0.9949808120727539,
  'answer': 'Lee A. Waller',
  'start': 55,
  'end': 57}]
```

如果您愿意，您还可以手动复制管道的结果：
1. 取一张图片和一个问题，使用您模型的处理器将它们准备好。
2. 将结果或处理结果传递给模型。
3. 模型返回 `start_logits` 和 `end_logits`，它们表示答案的起始标记和结束标记。两者的形状都是（batch_size，sequence_length）。
4. 对 `start_logits` 和 `end_logits` 的最后一个维度进行 argmax，得到预测的 `start_idx` 和 `end_idx`。
5. 使用分词器 (Tokenizer)解码答案。
```py
>>> import torch
>>> from transformers import AutoProcessor
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")

>>> with torch.no_grad():
...     encoding = processor(image.convert("RGB"), question, return_tensors="pt")
...     outputs = model(**encoding)
...     start_logits = outputs.start_logits
...     end_logits = outputs.end_logits
...     predicted_start_idx = start_logits.argmax(-1).item()
...     predicted_end_idx = end_logits.argmax(-1).item()

>>> processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1])
'lee a. waller'
```