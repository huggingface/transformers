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

# 🤗 Transformers 能做什么

🤗 Transformers是一个用于自然语言处理（NLP）、计算机视觉和音频和语音处理任务的预训练模型库。该库不仅包含Transformer模型，还包括用于计算机视觉任务的现代卷积网络等非Transformer模型。如果您看看今天最受欢迎的一些消费产品，比如智能手机、应用程序和电视，很可能背后都有某种深度学习技术的支持。想要从您智能手机拍摄的照片中删除背景对象吗？这里是一个全景分割任务的例子（如果您还不了解这是什么意思，我们将在以下部分进行描述！）。

本页面提供了使用🤗 Transformers库仅用三行代码解决不同的语音和音频、计算机视觉和NLP任务的概述！


## 音频
音频和语音处理任务与其他模态略有不同，主要是因为音频作为输入是一个连续的信号。与文本不同，原始音频波形不能像句子可以被划分为单词那样被整齐地分割成离散的块。为了解决这个问题，通常在固定的时间间隔内对原始音频信号进行采样。如果在每个时间间隔内采样更多样本，采样率就会更高，音频更接近原始音频源。

以前的方法是预处理音频以从中提取有用的特征。现在更常见的做法是直接将原始音频波形输入到特征编码器中，以提取音频表示。这样可以简化预处理步骤，并允许模型学习最重要的特征。

### 音频分类

音频分类是一项将音频数据从预定义的类别集合中进行标记的任务。这是一个广泛的类别，具有许多具体的应用，其中一些包括：

* 声学场景分类：使用场景标签（"办公室"、"海滩"、"体育场"）对音频进行标记。
* 声学事件检测：使用声音事件标签（"汽车喇叭声"、"鲸鱼叫声"、"玻璃破碎声"）对音频进行标记。
* 标记：对包含多种声音的音频进行标记（鸟鸣、会议中的说话人识别）。
* 音乐分类：使用流派标签（"金属"、"嘻哈"、"乡村"）对音乐进行标记。

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### 自动语音识别

自动语音识别（ASR）将语音转录为文本。这是最常见的音频任务之一，部分原因是因为语音是人类交流的自然形式。如今，ASR系统嵌入在智能技术产品中，如扬声器、电话和汽车。我们可以要求虚拟助手播放音乐、设置提醒和告诉我们天气。

但是，Transformer架构帮助解决的一个关键挑战是低资源语言。通过在大量语音数据上进行预训练，仅在一个低资源语言的一小时标记语音数据上进行微调，仍然可以产生与以前在100倍更多标记数据上训练的ASR系统相比高质量的结果。

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## 计算机视觉

计算机视觉任务中最早成功之一是使用卷积神经网络（[CNN](glossary#convolution)）识别邮政编码数字图像。图像由像素组成，每个像素都有一个数值。这使得将图像表示为像素值矩阵变得容易。每个像素值组合描述了图像的颜色。

计算机视觉任务可以通过以下两种通用方式解决：

1. 使用卷积来学习图像的层次特征，从低级特征到高级抽象特征。
2. 将图像分成块，并使用Transformer逐步学习每个图像块如何相互关联以形成图像。与CNN偏好的自底向上方法不同，这种方法有点像从一个模糊的图像开始，然后逐渐将其聚焦清晰。

### 图像分类

图像分类将整个图像从预定义的类别集合中进行标记。像大多数分类任务一样，图像分类有许多实际用例，其中一些包括：

* 医疗保健：标记医学图像以检测疾病或监测患者健康状况
* 环境：标记卫星图像以监测森林砍伐、提供野外管理信息或检测野火
* 农业：标记农作物图像以监测植物健康或用于土地使用监测的卫星图像
* 生态学：标记动物或植物物种的图像以监测野生动物种群或跟踪濒危物种

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### 目标检测

与图像分类不同，目标检测在图像中识别多个对象以及这些对象在图像中的位置（由边界框定义）。目标检测的一些示例应用包括：

* 自动驾驶车辆：检测日常交通对象，如其他车辆、行人和红绿灯
* 遥感：灾害监测、城市规划和天气预报
* 缺陷检测：检测建筑物中的裂缝或结构损坏，以及制造业产品缺陷


```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### 图像分割

图像分割是一项像素级任务，将图像中的每个像素分配给一个类别。它与使用边界框标记和预测图像中的对象的目标检测不同，因为分割更加精细。分割可以在像素级别检测对象。有几种类型的图像分割：

* 实例分割：除了标记对象的类别外，还标记每个对象的不同实例（“dog-1”，“dog-2”）
* 全景分割：语义分割和实例分割的组合； 它使用语义类为每个像素标记并标记每个对象的不同实例

分割任务对于自动驾驶车辆很有帮助，可以创建周围世界的像素级地图，以便它们可以在行人和其他车辆周围安全导航。它还适用于医学成像，其中任务的更精细粒度可以帮助识别异常细胞或器官特征。图像分割也可以用于电子商务，通过您的相机在现实世界中覆盖物体来虚拟试穿衣服或创建增强现实体验。

```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### 深度估计

深度估计预测图像中每个像素到相机的距离。这个计算机视觉任务对于场景理解和重建尤为重要。例如，在自动驾驶汽车中，车辆需要了解行人、交通标志和其他车辆等物体的距离，以避免障碍物和碰撞。深度信息还有助于从2D图像构建3D表示，并可用于创建生物结构或建筑物的高质量3D表示。

有两种方法可以进行深度估计：

* stereo（立体）：通过比较同一图像的两个略微不同角度的图像来估计深度
* monocular（单目）：从单个图像中估计深度


```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```

## 自然语言处理

NLP任务是最常见的类型之一，因为文本是我们进行交流的自然方式。为了让文本变成模型识别的格式，需要对其进行分词。这意味着将一段文本分成单独的单词或子词（`tokens`），然后将这些`tokens`转换为数字。因此，可以将一段文本表示为一系列数字，一旦有了一系列的数字，就可以将其输入到模型中以解决各种NLP任务！

### 文本分类

像任何模态的分类任务一样，文本分类将一段文本（可以是句子级别、段落或文档）从预定义的类别集合中进行标记。文本分类有许多实际应用，其中一些包括：

* 情感分析：根据某些极性（如`积极`或`消极`）对文本进行标记，可以支持政治、金融和营销等领域的决策制定
* 内容分类：根据某些主题对文本进行标记，有助于组织和过滤新闻和社交媒体提要中的信息（`天气`、`体育`、`金融`等）


```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### Token分类

在任何NLP任务中，文本都经过预处理，将文本序列分成单个单词或子词。这些被称为[tokens](/glossary#token)。Token分类将每个`token`分配一个来自预定义类别集的标签。

两种常见的Token分类是：

* 命名实体识别（NER）：根据实体类别（如组织、人员、位置或日期）对`token`进行标记。NER在生物医学设置中特别受欢迎，可以标记基因、蛋白质和药物名称。
* 词性标注（POS）：根据其词性（如名词、动词或形容词）对标记进行标记。POS对于帮助翻译系统了解两个相同的单词如何在语法上不同很有用（作为名词的银行与作为动词的银行）。

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

### 问答

问答是另一个`token-level`的任务，返回一个问题的答案，有时带有上下文（开放领域），有时不带上下文（封闭领域）。每当我们向虚拟助手提出问题时，例如询问一家餐厅是否营业，就会发生这种情况。它还可以提供客户或技术支持，并帮助搜索引擎检索您要求的相关信息。

有两种常见的问答类型：

* 提取式：给定一个问题和一些上下文，答案是从模型必须提取的上下文中的一段文本跨度。
* 抽象式：给定一个问题和一些上下文，答案从上下文中生成；这种方法由[`Text2TextGenerationPipeline`]处理，而不是下面显示的[`QuestionAnsweringPipeline`]。


```py
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="What is the name of the repository?",
...     context="The name of the repository is huggingface/transformers",
... )
>>> print(
...     f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
... )
score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

### 摘要

摘要从较长的文本中创建一个较短的版本，同时尽可能保留原始文档的大部分含义。摘要是一个序列到序列的任务；它输出比输入更短的文本序列。有许多长篇文档可以进行摘要，以帮助读者快速了解主要要点。法案、法律和财务文件、专利和科学论文等文档可以摘要，以节省读者的时间并作为阅读辅助工具。

像问答一样，摘要有两种类型：

* 提取式：从原始文本中识别和提取最重要的句子
* 抽象式：从原始文本生成目标摘要（可能包括不在输入文档中的新单词）；[`SummarizationPipeline`]使用抽象方法。


```py
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
... )
[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

### 翻译

翻译将一种语言的文本序列转换为另一种语言。它对于帮助来自不同背景的人们相互交流、帮助翻译内容以吸引更广泛的受众，甚至成为学习工具以帮助人们学习一门新语言都非常重要。除了摘要之外，翻译也是一个序列到序列的任务，意味着模型接收输入序列并返回目标输出序列。

在早期，翻译模型大多是单语的，但最近，越来越多的人对可以在多种语言之间进行翻译的多语言模型感兴趣。

```py
>>> from transformers import pipeline

>>> text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
>>> translator = pipeline(task="translation", model="google-t5/t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

### 语言模型

语言模型是一种预测文本序列中单词的任务。它已成为一种非常流行的NLP任务，因为预训练的语言模型可以微调用于许多其他下游任务。最近，人们对大型语言模型（LLMs）表现出了极大的兴趣，这些模型展示了`zero learning`或`few-shot learning`的能力。这意味着模型可以解决它未被明确训练过的任务！语言模型可用于生成流畅和令人信服的文本，但需要小心，因为文本可能并不总是准确的。

有两种类型的话语模型：

* causal：模型的目标是预测序列中的下一个`token`，而未来的`tokens`被遮盖。
  

    ```py
    >>> from transformers import pipeline

    >>> prompt = "Hugging Face is a community-based open-source platform for machine learning."
    >>> generator = pipeline(task="text-generation")
    >>> generator(prompt)  # doctest: +SKIP
    ```

*  masked：模型的目标是预测序列中被遮蔽的`token`，同时具有对序列中所有`tokens`的完全访问权限。

    
    ```py
    >>> text = "Hugging Face is a community-based open-source <mask> for machine learning."
    >>> fill_mask = pipeline(task="fill-mask")
    >>> preds = fill_mask(text, top_k=1)
    >>> preds = [
    ...     {
    ...         "score": round(pred["score"], 4),
    ...         "token": pred["token"],
    ...         "token_str": pred["token_str"],
    ...         "sequence": pred["sequence"],
    ...     }
    ...     for pred in preds
    ... ]
    >>> preds
    [{'score': 0.2236,
      'token': 1761,
      'token_str': ' platform',
      'sequence': 'Hugging Face is a community-based open-source platform for machine learning.'}]
    ```

## 多模态

多模态任务要求模型处理多种数据模态（文本、图像、音频、视频）以解决特定问题。图像描述是一个多模态任务的例子，其中模型将图像作为输入并输出描述图像或图像某些属性的文本序列。

虽然多模态模型处理不同的数据类型或模态，但内部预处理步骤帮助模型将所有数据类型转换为`embeddings`（向量或数字列表，包含有关数据的有意义信息）。对于像图像描述这样的任务，模型学习图像嵌入和文本嵌入之间的关系。

### 文档问答

文档问答是从文档中回答自然语言问题的任务。与`token-level`问答任务不同，文档问答将包含问题的文档的图像作为输入，并返回答案。文档问答可用于解析结构化文档并从中提取关键信息。在下面的例子中，可以从收据中提取总金额和找零金额。

```py
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/hf-internal-testing/example-documents/resolve/main/jpeg_images/2.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
>>> preds = doc_question_answerer(
...     question="What is the total amount?",
...     image=image,
... )
>>> preds
[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

希望这个页面为您提供了一些有关每种模态中所有类型任务的背景信息以及每个任务的实际重要性。在[下一节](tasks_explained)中，您将了解Transformers如何解决这些任务。
