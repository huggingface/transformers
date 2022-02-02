<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!---
A useful guide for English-Chinese translation of Hugging Face documentation
- Add space around English words and numbers when they appear between Chinese characters. E.g., 共 100 多种语言; 使用 transformers 库。
- Use square quotes, e.g.,「引用」

Dictionary

Hugging Face: 抱抱脸
token: 词符（并用括号标注原英文）
tokenize: 词符化（并用括号标注原英文）
tokenizer: 词符化器（并用括号标注原英文）
transformer: transformer（不翻译）
pipeline: 流水线
API: API (不翻译）
inference: 推理
Trainer: 训练器。当作为类名出现时不翻译。
pretrained/pretrain: 预训练
finetune: 微调
community: 社区
example: 当特指仓库中 example 目录时翻译为「用例」
Python data structures (e.g., list, set, dict): 翻译为列表，集合，词典，并用括号标注原英文
NLP/Natural Language Processing: 以 NLP 出现时不翻译，以 Natural Language Processing 出现时翻译为自然语言处理
checkpoint: 检查点
-->

<p align="center">
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <b>简体中文</b> |
        <a href="https://github.com/huggingface/transformers/blob/master/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/master/README_ko.md">한국어</a>
    <p>
</h4>

<h3 align="center">
    <p>为 Jax、PyTorch 和 TensorFlow 打造的先进的自然语言处理</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers 提供了数以千计的预训练模型，支持 100 多种语言的文本分类、信息抽取、问答、摘要、翻译、文本生成。它的宗旨让最先进的 NLP 技术人人易用。

🤗 Transformers 提供了便于快速下载和使用的API，让你可以把预训练模型用在给定文本、在你的数据集上微调然后通过 [model hub](https://huggingface.co/models) 与社区共享。同时，每个定义的 Python 模块均完全独立，方便修改和快速研究实验。

🤗 Transformers 支持三个最热门的深度学习库： [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) — 并与之无缝整合。你可以直接使用一个框架训练你的模型然后用另一个加载和推理。

## 在线演示

你可以直接在模型页面上测试大多数 [model hub](https://huggingface.co/models) 上的模型。 我们也提供了 [私有模型托管、模型版本管理以及推理API](https://huggingface.co/pricing)。

这里是一些例子：
- [用 BERT 做掩码填词](https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [用 Electra 做命名实体识别](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [用 GPT-2 做文本生成](https://huggingface.co/gpt2?text=A+long+time+ago%2C+)
- [用 RoBERTa 做自然语言推理](https://huggingface.co/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [用 BART 做文本摘要](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [用 DistilBERT 做问答](https://huggingface.co/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [用 T5 做翻译](https://huggingface.co/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

**[Write With Transformer](https://transformer.huggingface.co)**，由抱抱脸团队打造，是一个文本生成的官方 demo。

## 如果你在寻找由抱抱脸团队提供的定制化支持服务

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://huggingface.co/front/thumbnails/support.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## 快速上手

我们为快速使用模型提供了 `pipeline` （流水线）API。流水线聚合了预训练模型和对应的文本预处理。下面是一个快速使用流水线去判断正负面情绪的例子：

```python
>>> from transformers import pipeline

# 使用情绪分析流水线
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

第二行代码下载并缓存了流水线使用的预训练模型，而第三行代码则在给定的文本上进行了评估。这里的答案“正面” (positive) 具有 99 的置信度。

许多的 NLP 任务都有开箱即用的预训练流水线。比如说，我们可以轻松的从给定文本中抽取问题答案：

``` python
>>> from transformers import pipeline

# 使用问答流水线
>>> question_answerer = pipeline('question-answering')
>>> question_answerer({
...     'question': 'What is the name of the repository ?',
...     'context': 'Pipeline has been included in the huggingface/transformers repository'
... })
{'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}

```

除了给出答案，预训练模型还给出了对应的置信度分数、答案在词符化 (tokenized) 后的文本中开始和结束的位置。你可以从[这个教程](https://huggingface.co/docs/transformers/task_summary)了解更多流水线API支持的任务。

要在你的任务上下载和使用任意预训练模型也很简单，只需三行代码。这里是 PyTorch 版的示例：
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = AutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```
这里是等效的 TensorFlow 代码：
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

词符化器 (tokenizer) 为所有的预训练模型提供了预处理，并可以直接对单个字符串进行调用（比如上面的例子）或对列表 (list) 调用。它会输出一个你可以在下游代码里使用或直接通过 `**` 解包表达式传给模型的词典 (dict)。

模型本身是一个常规的 [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 或 [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)（取决于你的后端），可以常规方式使用。 [这个教程](https://huggingface.co/transformers/training.html)解释了如何将这样的模型整合到经典的 PyTorch 或 TensorFlow 训练循环中，或是如何使用我们的 `Trainer` 训练器）API 来在一个新的数据集上快速微调。

## 为什么要用 transformers？

1. 便于使用的先进模型：
    - NLU 和 NLG 上表现优越
    - 对教学和实践友好且低门槛
    - 高级抽象，只需了解三个类
    - 对所有模型统一的API

1. 更低计算开销，更少的碳排放：
    - 研究人员可以分享亿训练的模型而非次次从头开始训练
    - 工程师可以减少计算用时和生产环境开销
    - 数十种模型架构、两千多个预训练模型、100多种语言支持

1. 对于模型生命周期的每一个部分都面面俱到：
    - 训练先进的模型，只需 3 行代码
    - 模型在不同深度学习框架间任意转移，随你心意
    - 为训练、评估和生产选择最适合的框架，衔接无缝

1. 为你的需求轻松定制专属模型和用例：
    - 我们为每种模型架构提供了多个用例来复现原论文结果
    - 模型内部结构保持透明一致
    - 模型文件可单独使用，方便魔改和快速实验

## 什么情况下我不该用 transformers？

- 本库并不是模块化的神经网络工具箱。模型文件中的代码特意呈若璞玉，未经额外抽象封装，以便研究人员快速迭代魔改而不致溺于抽象和文件跳转之中。
- `Trainer` API 并非兼容任何模型，只为本库之模型优化。若是在寻找适用于通用机器学习的训练循环实现，请另觅他库。
- 尽管我们已尽力而为，[examples 目录](https://github.com/huggingface/transformers/tree/master/examples)中的脚本也仅为用例而已。对于你的特定问题，它们并不一定开箱即用，可能需要改几行代码以适之。

## 安装

### 使用 pip

这个仓库已在 Python 3.6+、Flax 0.3.2+、PyTorch 1.3.1+ 和 TensorFlow 2.3+ 下经过测试。

你可以在[虚拟环境](https://docs.python.org/3/library/venv.html)中安装 🤗 Transformers。如果你还不熟悉 Python 的虚拟环境，请阅此[用户说明](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)。

首先，用你打算使用的版本的 Python 创建一个虚拟环境并激活。

然后，你需要安装 Flax、PyTorch 或 TensorFlow 其中之一。关于在你使用的平台上安装这些框架，请参阅 [TensorFlow 安装页](https://www.tensorflow.org/install/), [PyTorch 安装页](https://pytorch.org/get-started/locally/#start-locally) 或 [Flax 安装页](https://github.com/google/flax#quick-install)。

当这些后端之一安装成功后， 🤗 Transformers 可依此安装：

```bash
pip install transformers
```

如果你想要试试用例或者想在正式发布前使用最新的开发中代码，你得[从源代码安装](https://huggingface.co/docs/transformers/installation#installing-from-source)。

### 使用 conda

自 Transformers 4.0.0 版始，我们有了一个 conda 频道： `huggingface`。

🤗 Transformers 可以通过 conda 依此安装：

```shell script
conda install -c huggingface transformers
```

要通过 conda 安装 Flax、PyTorch 或 TensorFlow 其中之一，请参阅它们各自安装页的说明。

## 模型架构

**🤗 Transformers 支持的[所有的模型检查点](https://huggingface.co/models)** 由[用户](https://huggingface.co/users)和[组织](https://huggingface.co/organizations)上传，均与 huggingface.co [model hub](https://huggingface.co) 无缝整合。

目前的检查点数量： ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers 目前支持如下的架构（模型概述请阅[这里](https://huggingface.co/docs/transformers/model_summary)）：

1. **[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)** (来自 Google Research and the Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。
1. **[BART](https://huggingface.co/docs/transformers/model_doc/bart)** (来自 Facebook) 伴随论文 [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) 由 Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer 发布。
1. **[BARThez](https://huggingface.co/docs/transformers/model_doc/barthez)** (来自 École polytechnique) 伴随论文 [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://arxiv.org/abs/2010.12321) 由 Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis 发布。
1. **[BARTpho](https://huggingface.co/docs/transformers/model_doc/bartpho)** (来自 VinAI Research) 伴随论文 [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese](https://arxiv.org/abs/2109.09701) 由 Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen 发布。
1. **[BEiT](https://huggingface.co/docs/transformers/model_doc/beit)** (来自 Microsoft) 伴随论文 [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) 由 Hangbo Bao, Li Dong, Furu Wei 发布。
1. **[BERT](https://huggingface.co/docs/transformers/model_doc/bert)** (来自 Google) 伴随论文 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 由 Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova 发布。
1. **[BERT For Sequence Generation](https://huggingface.co/docs/transformers/model_doc/bert-generation)** (来自 Google) 伴随论文 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) 由 Sascha Rothe, Shashi Narayan, Aliaksei Severyn 发布。
1. **[BERTweet](https://huggingface.co/docs/transformers/model_doc/bertweet)** (来自 VinAI Research) 伴随论文 [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/) 由 Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen 发布。
1. **[BigBird-Pegasus](https://huggingface.co/docs/transformers/model_doc/bigbird_pegasus)** (来自 Google Research) 伴随论文 [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) 由 Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed 发布。
1. **[BigBird-RoBERTa](https://huggingface.co/docs/transformers/model_doc/big_bird)** (来自 Google Research) 伴随论文 [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) 由 Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed 发布。
1. **[Blenderbot](https://huggingface.co/docs/transformers/model_doc/blenderbot)** (来自 Facebook) 伴随论文 [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637) 由 Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston 发布。
1. **[BlenderbotSmall](https://huggingface.co/docs/transformers/model_doc/blenderbot-small)** (来自 Facebook) 伴随论文 [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637) 由 Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston 发布。
1. **[BORT](https://huggingface.co/docs/transformers/model_doc/bort)** (来自 Alexa) 伴随论文 [Optimal Subarchitecture Extraction For BERT](https://arxiv.org/abs/2010.10499) 由 Adrian de Wynter and Daniel J. Perry 发布。
1. **[ByT5](https://huggingface.co/docs/transformers/model_doc/byt5)** (来自 Google Research) 伴随论文 [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) 由 Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel 发布。
1. **[CamemBERT](https://huggingface.co/docs/transformers/model_doc/camembert)** (来自 Inria/Facebook/Sorbonne) 伴随论文 [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) 由 Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot 发布。
1. **[CANINE](https://huggingface.co/docs/transformers/model_doc/canine)** (来自 Google Research) 伴随论文 [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874) 由 Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting 发布。
1. **[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)** (来自 OpenAI) 伴随论文 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) 由 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever 发布。
1. **[ConvBERT](https://huggingface.co/docs/transformers/model_doc/convbert)** (来自 YituTech) 伴随论文 [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496) 由 Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan 发布。
1. **[CPM](https://huggingface.co/docs/transformers/model_doc/cpm)** (来自 Tsinghua University) 伴随论文 [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413) 由 Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun 发布。
1. **[CTRL](https://huggingface.co/docs/transformers/model_doc/ctrl)** (来自 Salesforce) 伴随论文 [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) 由 Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher 发布。
1. **[DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta)** (来自 Microsoft) 伴随论文 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) 由 Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen 发布。
1. **[DeBERTa-v2](https://huggingface.co/docs/transformers/model_doc/deberta-v2)** (来自 Microsoft) 伴随论文 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) 由 Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen 发布。
1. **[DeiT](https://huggingface.co/docs/transformers/model_doc/deit)** (来自 Facebook) 伴随论文 [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) 由 Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou 发布。
1. **[DETR](https://huggingface.co/docs/transformers/model_doc/detr)** (来自 Facebook) 伴随论文 [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) 由 Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko 发布。
1. **[DialoGPT](https://huggingface.co/docs/transformers/model_doc/dialogpt)** (来自 Microsoft Research) 伴随论文 [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) 由 Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan 发布。
1. **[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)** (来自 HuggingFace), 伴随论文 [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) 由 Victor Sanh, Lysandre Debut and Thomas Wolf 发布。 同样的方法也应用于压缩 GPT-2 到 [DistilGPT2](https://github.com/huggingface/transformers/tree/master/examples/distillation), RoBERTa 到 [DistilRoBERTa](https://github.com/huggingface/transformers/tree/master/examples/distillation), Multilingual BERT 到 [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) 和德语版 DistilBERT。
1. **[DPR](https://huggingface.co/docs/transformers/model_doc/dpr)** (来自 Facebook) 伴随论文 [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) 由 Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih 发布。
1. **[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)** (来自 Google Research/Stanford University) 伴随论文 [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) 由 Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning 发布。
1. **[EncoderDecoder](https://huggingface.co/docs/transformers/model_doc/encoder-decoder)** (来自 Google Research) 伴随论文 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) 由 Sascha Rothe, Shashi Narayan, Aliaksei Severyn 发布。
1. **[FlauBERT](https://huggingface.co/docs/transformers/model_doc/flaubert)** (来自 CNRS) 伴随论文 [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) 由 Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab 发布。
1. **[FNet](https://huggingface.co/docs/transformers/model_doc/fnet)** (来自 Google Research) 伴随论文 [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) 由 James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon 发布。
1. **[Funnel Transformer](https://huggingface.co/docs/transformers/model_doc/funnel)** (来自 CMU/Google Brain) 伴随论文 [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://arxiv.org/abs/2006.03236) 由 Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le 发布。
1. **[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)** (来自 OpenAI) 伴随论文 [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) 由 Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever 发布。
1. **[GPT Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)** (来自 EleutherAI) 随仓库 [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) 发布。作者为 Sid Black, Stella Biderman, Leo Gao, Phil Wang and Connor Leahy 发布。
1. **[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)** (来自 OpenAI) 伴随论文 [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) 由 Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever** 发布。
1. **[GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)** (来自 EleutherAI) 伴随论文 [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) 由 Ben Wang and Aran Komatsuzaki 发布。
1. **[Hubert](https://huggingface.co/docs/transformers/model_doc/hubert)** (来自 Facebook) 伴随论文 [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) 由 Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed 发布。
1. **[I-BERT](https://huggingface.co/docs/transformers/model_doc/ibert)** (来自 Berkeley) 伴随论文 [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321) 由 Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer 发布。
1. **[ImageGPT](https://huggingface.co/docs/transformers/master/model_doc/imagegpt)** (来自 OpenAI) 伴随论文 [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt/) 由 Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever 发布。
1. **[LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)** (来自 Microsoft Research Asia) 伴随论文 [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) 由 Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou 发布。
1. **[LayoutLMv2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)** (来自 Microsoft Research Asia) 伴随论文 [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740) 由 Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou 发布。
1. **[LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)** (来自 Microsoft Research Asia) 伴随论文 [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836) 由 Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei 发布。
1. **[LED](https://huggingface.co/docs/transformers/model_doc/led)** (来自 AllenAI) 伴随论文 [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) 由 Iz Beltagy, Matthew E. Peters, Arman Cohan 发布。
1. **[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer)** (来自 AllenAI) 伴随论文 [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) 由 Iz Beltagy, Matthew E. Peters, Arman Cohan 发布。
1. **[LUKE](https://huggingface.co/docs/transformers/model_doc/luke)** (来自 Studio Ousia) 伴随论文 [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057) 由 Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yuji Matsumoto 发布。
1. **[LXMERT](https://huggingface.co/docs/transformers/model_doc/lxmert)** (来自 UNC Chapel Hill) 伴随论文 [LXMERT: Learning Cross-Modality Encoder Representations from Transformers for Open-Domain Question Answering](https://arxiv.org/abs/1908.07490) 由 Hao Tan and Mohit Bansal 发布。
1. **[M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100)** (来自 Facebook) 伴随论文 [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125) 由 Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin 发布。
1. **[MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)** 用 [OPUS](http://opus.nlpl.eu/) 数据训练的机器翻译模型由 Jörg Tiedemann 发布。[Marian Framework](https://marian-nmt.github.io/) 由微软翻译团队开发。
1. **[MBart](https://huggingface.co/docs/transformers/model_doc/mbart)** (来自 Facebook) 伴随论文 [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) 由 Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer 发布。
1. **[MBart-50](https://huggingface.co/docs/transformers/model_doc/mbart)** (来自 Facebook) 伴随论文 [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401) 由 Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan 发布。
1. **[Megatron-BERT](https://huggingface.co/docs/transformers/model_doc/megatron-bert)** (来自 NVIDIA) 伴随论文 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 由 Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper and Bryan Catanzaro 发布。
1. **[Megatron-GPT2](https://huggingface.co/docs/transformers/model_doc/megatron_gpt2)** (来自 NVIDIA) 伴随论文 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 由 Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper and Bryan Catanzaro 发布。
1. **[mLUKE](https://huggingface.co/docs/transformers/model_doc/mluke)** (来自 Studio Ousia) 伴随论文 [mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models](https://arxiv.org/abs/2110.08151) 由 Ryokan Ri, Ikuya Yamada, and Yoshimasa Tsuruoka 发布。
1. **[MPNet](https://huggingface.co/docs/transformers/model_doc/mpnet)** (来自 Microsoft Research) 伴随论文 [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297) 由 Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu 发布。
1. **[MT5](https://huggingface.co/docs/transformers/model_doc/mt5)** (来自 Google AI) 伴随论文 [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) 由 Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel 发布。
1. **[Nyströmformer](https://huggingface.co/docs/transformers/master/model_doc/nystromformer)** (来自 the University of Wisconsin - Madison) 伴随论文 [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902) 由 Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh 发布。
1. **[Pegasus](https://huggingface.co/docs/transformers/model_doc/pegasus)** (来自 Google) 伴随论文 [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777) 由 Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu 发布。
1. **[Perceiver IO](https://huggingface.co/docs/transformers/model_doc/perceiver)** (来自 Deepmind) 伴随论文 [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) 由 Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira 发布。
1. **[PhoBERT](https://huggingface.co/docs/transformers/model_doc/phobert)** (来自 VinAI Research) 伴随论文 [PhoBERT: Pre-trained language models for Vietnamese](https://www.aclweb.org/anthology/2020.findings-emnlp.92/) 由 Dat Quoc Nguyen and Anh Tuan Nguyen 发布。
1. **[ProphetNet](https://huggingface.co/docs/transformers/model_doc/prophetnet)** (来自 Microsoft Research) 伴随论文 [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) 由 Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou 发布。
1. **[QDQBert](https://huggingface.co/docs/transformers/model_doc/qdqbert)** (来自 NVIDIA) 伴随论文 [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602) 由 Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev and Paulius Micikevicius 发布。
1. **[REALM](https://huggingface.co/transformers/model_doc/realm.html)** (来自 Google Research) 伴随论文 [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) 由 Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat and Ming-Wei Chang 发布。
1. **[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer)** (来自 Google Research) 伴随论文 [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) 由 Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya 发布。
1. **[RemBERT](https://huggingface.co/docs/transformers/model_doc/rembert)** (来自 Google Research) 伴随论文 [Rethinking embedding coupling in pre-trained language models](https://arxiv.org/pdf/2010.12821.pdf) 由 Hyung Won Chung, Thibault Févry, Henry Tsai, M. Johnson, Sebastian Ruder 发布。
1. **[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)** (来自 Facebook), 伴随论文 [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) 由 Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov 发布。
1. **[RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer)** (来自 ZhuiyiTechnology), 伴随论文 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v1.pdf) 由 Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu 发布。
1. **[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)** (来自 NVIDIA) 伴随论文 [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) 由 Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo 发布。
1. **[SEW](https://huggingface.co/docs/transformers/model_doc/sew)** (来自 ASAPP) 伴随论文 [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) 由 Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi 发布。
1. **[SEW-D](https://huggingface.co/docs/transformers/model_doc/sew_d)** (来自 ASAPP) 伴随论文 [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) 由 Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi 发布。
1. **[SpeechToTextTransformer](https://huggingface.co/docs/transformers/model_doc/speech_to_text)** (来自 Facebook), 伴随论文 [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) 由 Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino 发布。
1. **[SpeechToTextTransformer2](https://huggingface.co/docs/transformers/model_doc/speech_to_text_2)** (来自 Facebook) 伴随论文 [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) 由 Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau 发布。
1. **[Splinter](https://huggingface.co/docs/transformers/model_doc/splinter)** (来自 Tel Aviv University) 伴随论文 [Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/abs/2101.00438) 由 Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy 发布。
1. **[SqueezeBert](https://huggingface.co/docs/transformers/model_doc/squeezebert)** (来自 Berkeley) 伴随论文 [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) 由 Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer 发布。
1. **[Swin Transformer](https://huggingface.co/docs/transformers/master/model_doc/swin)** (来自 Microsoft) 伴随论文 [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) 由 Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo 发布。
1. **[T5](https://huggingface.co/docs/transformers/model_doc/t5)** (来自 Google AI) 伴随论文 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) 由 Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu 发布。
1. **[T5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1)** (来自 Google AI) 伴随论文 [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511) 由 Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu 发布。
1. **[TAPAS](https://huggingface.co/docs/transformers/model_doc/tapas)** (来自 Google AI) 伴随论文 [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/abs/2004.02349) 由 Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno and Julian Martin Eisenschlos 发布。
1. **[Transformer-XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl)** (来自 Google/CMU) 伴随论文 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) 由 Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov 发布。
1. **[TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)** (来自 Microsoft) 伴随论文 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) 由 Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei 发布。
1. **[UniSpeech](https://huggingface.co/docs/transformers/model_doc/unispeech)** (来自 Microsoft Research) 伴随论文 [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) 由 Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang 发布。
1. **[UniSpeechSat](https://huggingface.co/docs/transformers/model_doc/unispeech-sat)** (来自 Microsoft Research) 伴随论文 [UNISPEECH-SAT: UNIVERSAL SPEECH REPRESENTATION LEARNING WITH SPEAKER AWARE PRE-TRAINING](https://arxiv.org/abs/2110.05752) 由 Sanyuan Chen, Yu Wu, Chengyi Wang, Zhengyang Chen, Zhuo Chen, Shujie Liu, Jian Wu, Yao Qian, Furu Wei, Jinyu Li, Xiangzhan Yu 发布。
1. **[ViLT)](https://huggingface.co/docs/transformers/master/model_doc/vilt)** (来自 NAVER AI Lab/Kakao Enterprise/Kakao Brain) 伴随论文 [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) 由 Wonjae Kim, Bokyung Son, Ildoo Kim 发布。
1. **[Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)** (来自 Google AI) 伴随论文 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) 由 Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby 发布。
1. **[VisualBERT](https://huggingface.co/docs/transformers/model_doc/visual_bert)** (来自 UCLA NLP) 伴随论文 [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557) 由 Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang 发布。
1. **[ViTMAE)](https://huggingface.co/docs/transformers/master/model_doc/vit_mae)** (来自 Meta AI) 伴随论文 [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) 由 Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick 发布。
1. **[Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)** (来自 Facebook AI) 伴随论文 [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) 由 Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli 发布。
1. **[Wav2Vec2Phoneme](https://huggingface.co/docs/master/transformers/model_doc/wav2vec2_phoneme)** (来自 Facebook AI) 伴随论文 [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition](https://arxiv.org/abs/2109.11680) 由 Qiantong Xu, Alexei Baevski, Michael Auli 发布。
1. **[WavLM](https://huggingface.co/docs/transformers/master/model_doc/wavlm)** (from Microsoft Research) released with the paper [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) by Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Furu Wei.
1. **[XGLM](https://huggingface.co/docs/master/transformers/model_doc/xglm)** (From Facebook AI) released with the paper [Few-shot Learning with Multilingual Language Models](https://arxiv.org/abs/2112.10668) by Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li. 
1. **[XLM](https://huggingface.co/docs/transformers/model_doc/xlm)** (来自 Facebook) 伴随论文 [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) 由 Guillaume Lample and Alexis Conneau 发布。
1. **[XLM-ProphetNet](https://huggingface.co/docs/transformers/model_doc/xlm-prophetnet)** (来自 Microsoft Research) 伴随论文 [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) 由 Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou 发布。
1. **[XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)** (来自 Facebook AI), 伴随论文 [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) 由 Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov 发布。
1. **[XLM-RoBERTa-XL](https://huggingface.co/docs/transformers/master/model_doc/xlm-roberta-xl)** (来自 Facebook AI) 伴随论文 [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://arxiv.org/abs/2105.00572) 由 Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, Alexis Conneau 发布。
1. **[XLNet](https://huggingface.co/docs/transformers/model_doc/xlnet)** (来自 Google/CMU) 伴随论文 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) 由 Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le 发布。
1. **[XLS-R](https://huggingface.co/docs/master/transformers/model_doc/xls_r)** (来自 Facebook AI) 伴随论文 [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://arxiv.org/abs/2111.09296) 由 Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli 发布。
1. **[XLSR-Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/xlsr_wav2vec2)** (来自 Facebook AI) 伴随论文 [Unsupervised Cross-Lingual Representation Learning For Speech Recognition](https://arxiv.org/abs/2006.13979) 由 Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli 发布。
1. **[YOSO](https://huggingface.co/docs/transformers/master/model_doc/yoso)** (来自 the University of Wisconsin - Madison) 伴随论文 [You Only Sample (Almost) 由 Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh 发布。
1. 想要贡献新的模型？我们这里有一份**详细指引和模板**来引导你添加新的模型。你可以在 [`templates`](./templates) 目录中找到他们。记得查看 [贡献指南](./CONTRIBUTING.md) 并在开始写 PR 前联系维护人员或开一个新的 issue 来获得反馈。

要检查某个模型是否已有 Flax、PyTorch 或 TensorFlow 的实现，或其是否在 🤗 Tokenizers 库中有对应词符化器（tokenizer），敬请参阅[此表](https://huggingface.co/docs/transformers/index#supported-frameworks)。

这些实现均已于多个数据集测试（请参看用例脚本）并应于原版实现表现相当。你可以在用例文档的[此节](https://huggingface.co/docs/transformers/examples)中了解表现的细节。


## 了解更多

| 章节 | 描述 |
|-|-|
| [文档](https://huggingface.co/transformers/) | 完整的 API 文档和教程 |
| [任务总结](https://huggingface.co/docs/transformers/task_summary) | 🤗 Transformers 支持的任务 |
| [预处理教程](https://huggingface.co/docs/transformers/preprocessing) | 使用 `Tokenizer` 来为模型准备数据 |
| [训练和微调](https://huggingface.co/docstransformers/training) | 在 PyTorch/TensorFlow 的训练循环或 `Trainer` API 中使用 🤗 Transformers 提供的模型 |
| [快速上手：微调和用例脚本](https://github.com/huggingface/transformers/tree/master/examples) | 为各种任务提供的用例脚本 |
| [模型分享和上传](https://huggingface.co/docs/transformers/model_sharing) | 和社区上传和分享你微调的模型 |
| [迁移](https://huggingface.co/docs/transformers/migration) | 从 `pytorch-transformers` 或 `pytorch-pretrained-bert` 迁移到 🤗 Transformers |

## 引用

我们已将此库的[论文](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)正式发表，如果你使用了 🤗 Transformers 库，请引用:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
