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
pipeline: 流水线
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
</p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <b>简体中文</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_vi.md">Tiếng Việt</a> |
    </p>
</h4>

<h3 align="center">
    <p>为 Jax、PyTorch 和 TensorFlow 打造的先进的自然语言处理</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers 提供了数以千计的预训练模型，支持 100 多种语言的文本分类、信息抽取、问答、摘要、翻译、文本生成。它的宗旨是让最先进的 NLP 技术人人易用。

🤗 Transformers 提供了便于快速下载和使用的API，让你可以把预训练模型用在给定文本、在你的数据集上微调然后通过 [model hub](https://huggingface.co/models) 与社区共享。同时，每个定义的 Python 模块均完全独立，方便修改和快速研究实验。

🤗 Transformers 支持三个最热门的深度学习库： [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) 以及 [TensorFlow](https://www.tensorflow.org/) — 并与之无缝整合。你可以直接使用一个框架训练你的模型然后用另一个加载和推理。

## 在线演示

你可以直接在模型页面上测试大多数 [model hub](https://huggingface.co/models) 上的模型。 我们也提供了 [私有模型托管、模型版本管理以及推理API](https://huggingface.co/pricing)。

这里是一些例子：
- [用 BERT 做掩码填词](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [用 Electra 做命名实体识别](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [用 GPT-2 做文本生成](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [用 RoBERTa 做自然语言推理](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [用 BART 做文本摘要](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [用 DistilBERT 做问答](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [用 T5 做翻译](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

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

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```
这里是等效的 TensorFlow 代码：
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

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
    - 研究人员可以分享已训练的模型而非每次从头开始训练
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
- 尽管我们已尽力而为，[examples 目录](https://github.com/huggingface/transformers/tree/main/examples)中的脚本也仅为用例而已。对于你的特定问题，它们并不一定开箱即用，可能需要改几行代码以适之。

## 安装

### 使用 pip

这个仓库已在 Python 3.8+、Flax 0.4.1+、PyTorch 1.11+ 和 TensorFlow 2.6+ 下经过测试。

你可以在[虚拟环境](https://docs.python.org/3/library/venv.html)中安装 🤗 Transformers。如果你还不熟悉 Python 的虚拟环境，请阅此[用户说明](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)。

首先，用你打算使用的版本的 Python 创建一个虚拟环境并激活。

然后，你需要安装 Flax、PyTorch 或 TensorFlow 其中之一。关于在你使用的平台上安装这些框架，请参阅 [TensorFlow 安装页](https://www.tensorflow.org/install/), [PyTorch 安装页](https://pytorch.org/get-started/locally/#start-locally) 或 [Flax 安装页](https://github.com/google/flax#quick-install)。

当这些后端之一安装成功后， 🤗 Transformers 可依此安装：

```bash
pip install transformers
```

如果你想要试试用例或者想在正式发布前使用最新的开发中代码，你得[从源代码安装](https://huggingface.co/docs/transformers/installation#installing-from-source)。

### 使用 conda

🤗 Transformers 可以通过 conda 依此安装：

```shell script
conda install conda-forge::transformers
```

> **_笔记:_** 从 `huggingface` 渠道安装 `transformers` 已被废弃。

要通过 conda 安装 Flax、PyTorch 或 TensorFlow 其中之一，请参阅它们各自安装页的说明。

## 模型架构

🤗 Transformers 支持的[**所有的模型检查点**](https://huggingface.co/models)由[用户](https://huggingface.co/users)和[组织](https://huggingface.co/organizations)上传，均与 huggingface.co [model hub](https://huggingface.co) 无缝整合。

目前的检查点数量： ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers 目前支持如下的架构（模型概述请阅[这里](https://huggingface.co/docs/transformers/model_summary)）：

1. **[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)** (来自 Google Research and the Toyota Technological Institute at Chicago) 伴随论文 [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), 由 Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut 发布。
1. **[ALIGN](https://huggingface.co/docs/transformers/model_doc/align)** (来自 Google Research) 伴随论文 [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) 由 Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig 发布。
1. **[AltCLIP](https://huggingface.co/docs/transformers/model_doc/altclip)** (来自 BAAI) 伴随论文 [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679) 由 Chen, Zhongzhi and Liu, Guang and Zhang, Bo-Wen and Ye, Fulong and Yang, Qinghong and Wu, Ledell 发布。
1. **[Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)** (来自 MIT) 伴随论文 [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) 由 Yuan Gong, Yu-An Chung, James Glass 发布。
1. **[Autoformer](https://huggingface.co/docs/transformers/model_doc/autoformer)** (from Tsinghua University) released with the paper [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
1. **[Bark](https://huggingface.co/docs/transformers/model_doc/bark)** (from Suno) released in the repository [suno-ai/bark](https://github.com/suno-ai/bark) by Suno AI team.
1. **[BART](https://huggingface.co/docs/transformers/model_doc/bart)** (来自 Facebook) 伴随论文 [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) 由 Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer 发布。
1. **[BARThez](https://huggingface.co/docs/transformers/model_doc/barthez)** (来自 École polytechnique) 伴随论文 [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://arxiv.org/abs/2010.12321) 由 Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis 发布。
1. **[BARTpho](https://huggingface.co/docs/transformers/model_doc/bartpho)** (来自 VinAI Research) 伴随论文 [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese](https://arxiv.org/abs/2109.09701) 由 Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen 发布。
1. **[BEiT](https://huggingface.co/docs/transformers/model_doc/beit)** (来自 Microsoft) 伴随论文 [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) 由 Hangbo Bao, Li Dong, Furu Wei 发布。
1. **[BERT](https://huggingface.co/docs/transformers/model_doc/bert)** (来自 Google) 伴随论文 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 由 Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova 发布。
1. **[BERT For Sequence Generation](https://huggingface.co/docs/transformers/model_doc/bert-generation)** (来自 Google) 伴随论文 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) 由 Sascha Rothe, Shashi Narayan, Aliaksei Severyn 发布。
1. **[BERTweet](https://huggingface.co/docs/transformers/model_doc/bertweet)** (来自 VinAI Research) 伴随论文 [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/) 由 Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen 发布。
1. **[BigBird-Pegasus](https://huggingface.co/docs/transformers/model_doc/bigbird_pegasus)** (来自 Google Research) 伴随论文 [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) 由 Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed 发布。
1. **[BigBird-RoBERTa](https://huggingface.co/docs/transformers/model_doc/big_bird)** (来自 Google Research) 伴随论文 [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) 由 Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed 发布。
1. **[BioGpt](https://huggingface.co/docs/transformers/model_doc/biogpt)** (来自 Microsoft Research AI4Science) 伴随论文 [BioGPT: generative pre-trained transformer for biomedical text generation and mining](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9) 由 Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu 发布。
1. **[BiT](https://huggingface.co/docs/transformers/model_doc/bit)** (来自 Google AI) 伴随论文 [Big Transfer (BiT) 由 Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby 发布。
1. **[Blenderbot](https://huggingface.co/docs/transformers/model_doc/blenderbot)** (来自 Facebook) 伴随论文 [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637) 由 Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston 发布。
1. **[BlenderbotSmall](https://huggingface.co/docs/transformers/model_doc/blenderbot-small)** (来自 Facebook) 伴随论文 [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637) 由 Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston 发布。
1. **[BLIP](https://huggingface.co/docs/transformers/model_doc/blip)** (来自 Salesforce) 伴随论文 [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) 由 Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi 发布。
1. **[BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)** (来自 Salesforce) 伴随论文 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) 由 Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi 发布。
1. **[BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom)** (from BigScience workshop) released by the [BigScience Workshop](https://bigscience.huggingface.co/).
1. **[BORT](https://huggingface.co/docs/transformers/model_doc/bort)** (来自 Alexa) 伴随论文 [Optimal Subarchitecture Extraction For BERT](https://arxiv.org/abs/2010.10499) 由 Adrian de Wynter and Daniel J. Perry 发布。
1. **[BridgeTower](https://huggingface.co/docs/transformers/model_doc/bridgetower)** (from Harbin Institute of Technology/Microsoft Research Asia/Intel Labs) released with the paper [BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning](https://arxiv.org/abs/2206.08657) by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
1. **[BROS](https://huggingface.co/docs/transformers/model_doc/bros)** (来自 NAVER CLOVA) 伴随论文 [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539) 由 Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park 发布。
1. **[ByT5](https://huggingface.co/docs/transformers/model_doc/byt5)** (来自 Google Research) 伴随论文 [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) 由 Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel 发布。
1. **[CamemBERT](https://huggingface.co/docs/transformers/model_doc/camembert)** (来自 Inria/Facebook/Sorbonne) 伴随论文 [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) 由 Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot 发布。
1. **[CANINE](https://huggingface.co/docs/transformers/model_doc/canine)** (来自 Google Research) 伴随论文 [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874) 由 Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting 发布。
1. **[Chinese-CLIP](https://huggingface.co/docs/transformers/model_doc/chinese_clip)** (来自 OFA-Sys) 伴随论文 [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) 由 An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou 发布。
1. **[CLAP](https://huggingface.co/docs/transformers/model_doc/clap)** (来自 LAION-AI) 伴随论文 [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) 由 Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov 发布。
1. **[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)** (来自 OpenAI) 伴随论文 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) 由 Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever 发布。
1. **[CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)** (来自 University of Göttingen) 伴随论文 [Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003) 由 Timo Lüddecke and Alexander Ecker 发布。
1. **[CLVP](https://huggingface.co/docs/transformers/model_doc/clvp)** released with the paper [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243) by James Betker.
1. **[CodeGen](https://huggingface.co/docs/transformers/model_doc/codegen)** (来自 Salesforce) 伴随论文 [A Conversational Paradigm for Program Synthesis](https://arxiv.org/abs/2203.13474) 由 Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong 发布。
1. **[CodeLlama](https://huggingface.co/docs/transformers/model_doc/llama_code)** (来自 MetaAI) 伴随论文 [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) 由 Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve 发布。
1. **[Conditional DETR](https://huggingface.co/docs/transformers/model_doc/conditional_detr)** (来自 Microsoft Research Asia) 伴随论文 [Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152) 由 Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang 发布。
1. **[ConvBERT](https://huggingface.co/docs/transformers/model_doc/convbert)** (来自 YituTech) 伴随论文 [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496) 由 Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan 发布。
1. **[ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext)** (来自 Facebook AI) 伴随论文 [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) 由 Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie 发布。
1. **[ConvNeXTV2](https://huggingface.co/docs/transformers/model_doc/convnextv2)** (from Facebook AI) released with the paper [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) by Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
1. **[CPM](https://huggingface.co/docs/transformers/model_doc/cpm)** (来自 Tsinghua University) 伴随论文 [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://arxiv.org/abs/2012.00413) 由 Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun 发布。
1. **[CPM-Ant](https://huggingface.co/docs/transformers/model_doc/cpmant)** (from OpenBMB) released by the [OpenBMB](https://www.openbmb.org/).
1. **[CTRL](https://huggingface.co/docs/transformers/model_doc/ctrl)** (来自 Salesforce) 伴随论文 [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) 由 Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher 发布。
1. **[CvT](https://huggingface.co/docs/transformers/model_doc/cvt)** (来自 Microsoft) 伴随论文 [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) 由 Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang 发布。
1. **[Data2Vec](https://huggingface.co/docs/transformers/model_doc/data2vec)** (来自 Facebook) 伴随论文 [Data2Vec:  A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555) 由 Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli 发布。
1. **[DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta)** (来自 Microsoft) 伴随论文 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) 由 Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen 发布。
1. **[DeBERTa-v2](https://huggingface.co/docs/transformers/model_doc/deberta-v2)** (来自 Microsoft) 伴随论文 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) 由 Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen 发布。
1. **[Decision Transformer](https://huggingface.co/docs/transformers/model_doc/decision_transformer)** (来自 Berkeley/Facebook/Google) 伴随论文 [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) 由 Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch 发布。
1. **[Deformable DETR](https://huggingface.co/docs/transformers/model_doc/deformable_detr)** (来自 SenseTime Research) 伴随论文 [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) 由 Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai 发布。
1. **[DeiT](https://huggingface.co/docs/transformers/model_doc/deit)** (来自 Facebook) 伴随论文 [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) 由 Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou 发布。
1. **[DePlot](https://huggingface.co/docs/transformers/model_doc/deplot)** (来自 Google AI) 伴随论文 [DePlot: One-shot visual language reasoning by plot-to-table translation](https://arxiv.org/abs/2212.10505) 由 Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun 发布。
1. **[Depth Anything](https://huggingface.co/docs/transformers/model_doc/depth_anything)** (来自 University of Hong Kong and TikTok) 伴随论文 [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891) 由 Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao 发布。
1. **[DETA](https://huggingface.co/docs/transformers/model_doc/deta)** (来自 The University of Texas at Austin) 伴随论文 [NMS Strikes Back](https://arxiv.org/abs/2212.06137) 由 Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl 发布。
1. **[DETR](https://huggingface.co/docs/transformers/model_doc/detr)** (来自 Facebook) 伴随论文 [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) 由 Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko 发布。
1. **[DialoGPT](https://huggingface.co/docs/transformers/model_doc/dialogpt)** (来自 Microsoft Research) 伴随论文 [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) 由 Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan 发布。
1. **[DiNAT](https://huggingface.co/docs/transformers/model_doc/dinat)** (来自 SHI Labs) 伴随论文 [Dilated Neighborhood Attention Transformer](https://arxiv.org/abs/2209.15001) 由 Ali Hassani and Humphrey Shi 发布。
1. **[DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2)** (来自 Meta AI) 伴随论文 [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) 由 Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski 发布。
1. **[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)** (来自 HuggingFace), 伴随论文 [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) 由 Victor Sanh, Lysandre Debut and Thomas Wolf 发布。 同样的方法也应用于压缩 GPT-2 到 [DistilGPT2](https://github.com/huggingface/transformers/tree/main/examples/distillation), RoBERTa 到 [DistilRoBERTa](https://github.com/huggingface/transformers/tree/main/examples/distillation), Multilingual BERT 到 [DistilmBERT](https://github.com/huggingface/transformers/tree/main/examples/distillation) 和德语版 DistilBERT。
1. **[DiT](https://huggingface.co/docs/transformers/model_doc/dit)** (来自 Microsoft Research) 伴随论文 [DiT: Self-supervised Pre-training for Document Image Transformer](https://arxiv.org/abs/2203.02378) 由 Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei 发布。
1. **[Donut](https://huggingface.co/docs/transformers/model_doc/donut)** (来自 NAVER) 伴随论文 [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) 由 Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park 发布。
1. **[DPR](https://huggingface.co/docs/transformers/model_doc/dpr)** (来自 Facebook) 伴随论文 [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) 由 Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih 发布。
1. **[DPT](https://huggingface.co/docs/transformers/master/model_doc/dpt)** (来自 Intel Labs) 伴随论文 [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) 由 René Ranftl, Alexey Bochkovskiy, Vladlen Koltun 发布。
1. **[EfficientFormer](https://huggingface.co/docs/transformers/model_doc/efficientformer)** (来自 Snap Research) 伴随论文 [EfficientFormer: Vision Transformers at MobileNetSpeed](https://arxiv.org/abs/2206.01191) 由 Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren 发布。
1. **[EfficientNet](https://huggingface.co/docs/transformers/model_doc/efficientnet)** (from Google Brain) released with the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) by Mingxing Tan, Quoc V. Le.
1. **[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)** (来自 Google Research/Stanford University) 伴随论文 [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) 由 Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning 发布。
1. **[EnCodec](https://huggingface.co/docs/transformers/model_doc/encodec)** (来自 Meta AI) 伴随论文 [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) 由 Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi 发布。
1. **[EncoderDecoder](https://huggingface.co/docs/transformers/model_doc/encoder-decoder)** (来自 Google Research) 伴随论文 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) 由 Sascha Rothe, Shashi Narayan, Aliaksei Severyn 发布。
1. **[ERNIE](https://huggingface.co/docs/transformers/model_doc/ernie)** (来自 Baidu) 伴随论文 [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223) by Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu 发布。
1. **[ErnieM](https://huggingface.co/docs/transformers/model_doc/ernie_m)** (来自 Baidu) 伴随论文 [ERNIE-M: Enhanced Multilingual Representation by Aligning Cross-lingual Semantics with Monolingual Corpora](https://arxiv.org/abs/2012.15674) 由 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang 发布。
1. **[ESM](https://huggingface.co/docs/transformers/model_doc/esm)** (from Meta AI) are transformer protein language models.  **ESM-1b** was released with the paper [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/content/118/15/e2016239118) by Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. **ESM-1v** was released with the paper [Language models enable zero-shot prediction of the effects of mutations on protein function](https://doi.org/10.1101/2021.07.09.450648) by Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu and Alexander Rives. **ESM-2** was released with the paper [Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://doi.org/10.1101/2022.07.20.500902) by Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido, Alexander Rives.
1. **[Falcon](https://huggingface.co/docs/transformers/model_doc/falcon)** (from Technology Innovation Institute) by Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and Cappelli, Alessandro and Cojocaru, Ruxandra and Debbah, Merouane and Goffinet, Etienne and Heslow, Daniel and Launay, Julien and Malartic, Quentin and Noune, Badreddine and Pannier, Baptiste and Penedo, Guilherme.
1. **[FastSpeech2Conformer](model_doc/fastspeech2_conformer)** (来自 ESPnet and Microsoft Research) 伴随论文 [Fastspeech 2: Fast And High-quality End-to-End Text To Speech](https://arxiv.org/pdf/2006.04558.pdf) 由 Pengcheng Guo, Florian Boyer, Xuankai Chang, Tomoki Hayashi, Yosuke Higuchi, Hirofumi Inaguma, Naoyuki Kamo, Chenda Li, Daniel Garcia-Romero, Jiatong Shi, Jing Shi, Shinji Watanabe, Kun Wei, Wangyou Zhang, and Yuekai Zhang 发布。
1. **[FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)** (from Google AI) released in the repository [google-research/t5x](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints) by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
1. **[FLAN-UL2](https://huggingface.co/docs/transformers/model_doc/flan-ul2)** (from Google AI) released in the repository [google-research/t5x](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-ul2-checkpoints) by Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei
1. **[FlauBERT](https://huggingface.co/docs/transformers/model_doc/flaubert)** (来自 CNRS) 伴随论文 [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) 由 Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab 发布。
1. **[FLAVA](https://huggingface.co/docs/transformers/model_doc/flava)** (来自 Facebook AI) 伴随论文 [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482) 由 Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela 发布。
1. **[FNet](https://huggingface.co/docs/transformers/model_doc/fnet)** (来自 Google Research) 伴随论文 [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) 由 James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon 发布。
1. **[FocalNet](https://huggingface.co/docs/transformers/model_doc/focalnet)** (来自 Microsoft Research) 伴随论文 [Focal Modulation Networks](https://arxiv.org/abs/2203.11926) 由 Jianwei Yang, Chunyuan Li, Xiyang Dai, Lu Yuan, Jianfeng Gao 发布。
1. **[Funnel Transformer](https://huggingface.co/docs/transformers/model_doc/funnel)** (来自 CMU/Google Brain) 伴随论文 [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://arxiv.org/abs/2006.03236) 由 Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le 发布。
1. **[Fuyu](https://huggingface.co/docs/transformers/model_doc/fuyu)** (来自 ADEPT) 伴随论文 [blog post](https://www.adept.ai/blog/fuyu-8b) 由 Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, Sağnak Taşırlar 发布。
1. **[Gemma](https://huggingface.co/docs/transformers/main/model_doc/gemma)** (来自 Google) 伴随论文 [Gemma: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/gemma-open-models/) 由 the Gemma Google team 发布。
1. **[GIT](https://huggingface.co/docs/transformers/model_doc/git)** (来自 Microsoft Research) 伴随论文 [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) 由 Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, Lijuan Wang 发布。
1. **[GLPN](https://huggingface.co/docs/transformers/model_doc/glpn)** (来自 KAIST) 伴随论文 [Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://arxiv.org/abs/2201.07436) 由 Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim 发布。
1. **[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)** (来自 OpenAI) 伴随论文 [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised/) 由 Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever 发布。
1. **[GPT Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)** (来自 EleutherAI) 随仓库 [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) 发布。作者为 Sid Black, Stella Biderman, Leo Gao, Phil Wang and Connor Leahy 发布。
1. **[GPT NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox)** (from EleutherAI) released with the paper [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745) by Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, Samuel Weinbach
1. **[GPT NeoX Japanese](https://huggingface.co/docs/transformers/model_doc/gpt_neox_japanese)** (来自 ABEJA) 由 Shinya Otani, Takayoshi Makabe, Anuj Arora, Kyo Hattori。
1. **[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)** (来自 OpenAI) 伴随论文 [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models/) 由 Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever 发布。
1. **[GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)** (来自 EleutherAI) 伴随论文 [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) 由 Ben Wang and Aran Komatsuzaki 发布。
1. **[GPT-Sw3](https://huggingface.co/docs/transformers/model_doc/gpt-sw3)** (from AI-Sweden) released with the paper [Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf) by Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman, Fredrik Carlsson, Magnus Sahlgren.
1. **[GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode)** (来自 BigCode) 伴随论文 [SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988) 由 Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane Anderson, Yangtian Zi, Joel Lamy Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky, Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes, Daniel Fried, Arjun Guha, Harm de Vries, Leandro von Werra 发布。
1. **[GPTSAN-japanese](https://huggingface.co/docs/transformers/model_doc/gptsan-japanese)** released in the repository [tanreinama/GPTSAN](https://github.com/tanreinama/GPTSAN/blob/main/report/model.md) by 坂本俊之(tanreinama).
1. **[Graphormer](https://huggingface.co/docs/transformers/model_doc/graphormer)** (from Microsoft) released with the paper [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234) by Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu.
1. **[GroupViT](https://huggingface.co/docs/transformers/model_doc/groupvit)** (来自 UCSD, NVIDIA) 伴随论文 [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094) 由 Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang 发布。
1. **[HerBERT](https://huggingface.co/docs/transformers/model_doc/herbert)** (来自 Allegro.pl, AGH University of Science and Technology) 伴随论文 [KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://www.aclweb.org/anthology/2020.acl-main.111.pdf) 由 Piotr Rybak, Robert Mroczkowski, Janusz Tracz, Ireneusz Gawlik 发布。
1. **[Hubert](https://huggingface.co/docs/transformers/model_doc/hubert)** (来自 Facebook) 伴随论文 [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) 由 Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed 发布。
1. **[I-BERT](https://huggingface.co/docs/transformers/model_doc/ibert)** (来自 Berkeley) 伴随论文 [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321) 由 Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer 发布。
1. **[IDEFICS](https://huggingface.co/docs/transformers/model_doc/idefics)** (from HuggingFace) released with the paper [OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents](https://huggingface.co/papers/2306.16527) by Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh.
1. **[ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt)** (来自 OpenAI) 伴随论文 [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt/) 由 Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever 发布。
1. **[Informer](https://huggingface.co/docs/transformers/model_doc/informer)** (from Beihang University, UC Berkeley, Rutgers University, SEDD Company) released with the paper [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
1. **[InstructBLIP](https://huggingface.co/docs/transformers/model_doc/instructblip)** (来自 Salesforce) 伴随论文 [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) 由 Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi 发布。
1. **[Jukebox](https://huggingface.co/docs/transformers/model_doc/jukebox)** (from OpenAI) released with the paper [Jukebox: A Generative Model for Music](https://arxiv.org/pdf/2005.00341.pdf) by Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, Ilya Sutskever.
1. **[KOSMOS-2](https://huggingface.co/docs/transformers/model_doc/kosmos-2)** (from Microsoft Research Asia) released with the paper [Kosmos-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/abs/2306.14824) by Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei.
1. **[LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)** (来自 Microsoft Research Asia) 伴随论文 [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) 由 Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou 发布。
1. **[LayoutLMv2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)** (来自 Microsoft Research Asia) 伴随论文 [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740) 由 Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou 发布。
1. **[LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)** (来自 Microsoft Research Asia) 伴随论文 [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387) 由 Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei 发布。
1. **[LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutxlm)** (来自 Microsoft Research Asia) 伴随论文 [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836) 由 Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei 发布。
1. **[LED](https://huggingface.co/docs/transformers/model_doc/led)** (来自 AllenAI) 伴随论文 [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) 由 Iz Beltagy, Matthew E. Peters, Arman Cohan 发布。
1. **[LeViT](https://huggingface.co/docs/transformers/model_doc/levit)** (来自 Meta AI) 伴随论文 [LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/abs/2104.01136) 由 Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze 发布。
1. **[LiLT](https://huggingface.co/docs/transformers/model_doc/lilt)** (来自 South China University of Technology) 伴随论文 [LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding](https://arxiv.org/abs/2202.13669) 由 Jiapeng Wang, Lianwen Jin, Kai Ding 发布。
1. **[LLaMA](https://huggingface.co/docs/transformers/model_doc/llama)** (来自 The FAIR team of Meta AI) 伴随论文 [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) 由 Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample 发布。
1. **[Llama2](https://huggingface.co/docs/transformers/model_doc/llama2)** (来自 The FAIR team of Meta AI) 伴随论文 [Llama2: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/XXX) 由 Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushka rMishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing EllenTan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom. 发布。
1. **[LLaVa](https://huggingface.co/docs/transformers/model_doc/llava)** (来自 Microsoft Research & University of Wisconsin-Madison) 伴随论文 [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) 由 Haotian Liu, Chunyuan Li, Yuheng Li and Yong Jae Lee 发布。
1. **[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer)** (来自 AllenAI) 伴随论文 [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) 由 Iz Beltagy, Matthew E. Peters, Arman Cohan 发布。
1. **[LongT5](https://huggingface.co/docs/transformers/model_doc/longt5)** (来自 Google AI) released 伴随论文 [LongT5: Efficient Text-To-Text Transformer for Long Sequences](https://arxiv.org/abs/2112.07916) 由 Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, Yinfei Yang 发布。
1. **[LUKE](https://huggingface.co/docs/transformers/model_doc/luke)** (来自 Studio Ousia) 伴随论文 [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057) 由 Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yuji Matsumoto 发布。
1. **[LXMERT](https://huggingface.co/docs/transformers/model_doc/lxmert)** (来自 UNC Chapel Hill) 伴随论文 [LXMERT: Learning Cross-Modality Encoder Representations from Transformers for Open-Domain Question Answering](https://arxiv.org/abs/1908.07490) 由 Hao Tan and Mohit Bansal 发布。
1. **[M-CTC-T](https://huggingface.co/docs/transformers/model_doc/mctct)** (来自 Facebook) 伴随论文 [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161) 由 Loren Lugosch, Tatiana Likhomanenko, Gabriel Synnaeve, and Ronan Collobert 发布。
1. **[M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100)** (来自 Facebook) 伴随论文 [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125) 由 Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin 发布。
1. **[MADLAD-400](https://huggingface.co/docs/transformers/model_doc/madlad-400)** (from Google) released with the paper [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](https://arxiv.org/abs/2309.04662) by Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A. Choquette-Choo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, Orhan Firat.
1. **[Mamba](https://huggingface.co/docs/transformers/main/model_doc/mamba)** (来自 Albert Gu and Tri Dao) 伴随论文 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) 由 Albert Gu and Tri Dao 发布。
1. **[MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)** 用 [OPUS](http://opus.nlpl.eu/) 数据训练的机器翻译模型由 Jörg Tiedemann 发布。[Marian Framework](https://marian-nmt.github.io/) 由微软翻译团队开发。
1. **[MarkupLM](https://huggingface.co/docs/transformers/model_doc/markuplm)** (来自 Microsoft Research Asia) 伴随论文 [MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding](https://arxiv.org/abs/2110.08518) 由 Junlong Li, Yiheng Xu, Lei Cui, Furu Wei 发布。
1. **[Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former)** (来自 FAIR and UIUC) 伴随论文 [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) 由 Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar 发布。
1. **[MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer)** (from Meta and UIUC) released with the paper [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278) by Bowen Cheng, Alexander G. Schwing, Alexander Kirillov
1. **[MatCha](https://huggingface.co/docs/transformers/model_doc/matcha)** (来自 Google AI) 伴随论文 [MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering](https://arxiv.org/abs/2212.09662) 由 Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, Julian Martin Eisenschlos 发布。
1. **[mBART](https://huggingface.co/docs/transformers/model_doc/mbart)** (来自 Facebook) 伴随论文 [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) 由 Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer 发布。
1. **[mBART-50](https://huggingface.co/docs/transformers/model_doc/mbart)** (来自 Facebook) 伴随论文 [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401) 由 Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan 发布。
1. **[MEGA](https://huggingface.co/docs/transformers/model_doc/mega)** (来自 Facebook) 伴随论文 [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655) 由 Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer 发布。
1. **[Megatron-BERT](https://huggingface.co/docs/transformers/model_doc/megatron-bert)** (来自 NVIDIA) 伴随论文 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 由 Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper and Bryan Catanzaro 发布。
1. **[Megatron-GPT2](https://huggingface.co/docs/transformers/model_doc/megatron_gpt2)** (来自 NVIDIA) 伴随论文 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 由 Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper and Bryan Catanzaro 发布。
1. **[MGP-STR](https://huggingface.co/docs/transformers/model_doc/mgp-str)** (来自 Alibaba Research) 伴随论文 [Multi-Granularity Prediction for Scene Text Recognition](https://arxiv.org/abs/2209.03592) 由 Peng Wang, Cheng Da, and Cong Yao 发布。
1. **[Mistral](https://huggingface.co/docs/transformers/model_doc/mistral)** (from Mistral AI) by The Mistral AI team: Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed..
1. **[Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral)** (from Mistral AI) by The [Mistral AI](https://mistral.ai) team: Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.
1. **[mLUKE](https://huggingface.co/docs/transformers/model_doc/mluke)** (来自 Studio Ousia) 伴随论文 [mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models](https://arxiv.org/abs/2110.08151) 由 Ryokan Ri, Ikuya Yamada, and Yoshimasa Tsuruoka 发布。
1. **[MMS](https://huggingface.co/docs/transformers/model_doc/mms)** (来自 Facebook) 伴随论文 [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516) 由 Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, Michael Auli 发布。
1. **[MobileBERT](https://huggingface.co/docs/transformers/model_doc/mobilebert)** (来自 CMU/Google Brain) 伴随论文 [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) 由 Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou 发布。
1. **[MobileNetV1](https://huggingface.co/docs/transformers/model_doc/mobilenet_v1)** (来自 Google Inc.) 伴随论文 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) 由 Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam 发布。
1. **[MobileNetV2](https://huggingface.co/docs/transformers/model_doc/mobilenet_v2)** (来自 Google Inc.) 伴随论文 [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) 由 Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen 发布。
1. **[MobileViT](https://huggingface.co/docs/transformers/model_doc/mobilevit)** (来自 Apple) 伴随论文 [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) 由 Sachin Mehta and Mohammad Rastegari 发布。
1. **[MobileViTV2](https://huggingface.co/docs/transformers/model_doc/mobilevitv2)** (来自 Apple) 伴随论文 [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) 由 Sachin Mehta and Mohammad Rastegari 发布。
1. **[MPNet](https://huggingface.co/docs/transformers/model_doc/mpnet)** (来自 Microsoft Research) 伴随论文 [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297) 由 Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu 发布。
1. **[MPT](https://huggingface.co/docs/transformers/model_doc/mpt)** (来自 MosaiML) 伴随论文 [llm-foundry](https://github.com/mosaicml/llm-foundry/) 由 the MosaicML NLP Team 发布。
1. **[MRA](https://huggingface.co/docs/transformers/model_doc/mra)** (来自 the University of Wisconsin - Madison) 伴随论文 [Multi Resolution Analysis (MRA)](https://arxiv.org/abs/2207.10284) 由 Zhanpeng Zeng, Sourav Pal, Jeffery Kline, Glenn M Fung, Vikas Singh 发布。
1. **[MT5](https://huggingface.co/docs/transformers/model_doc/mt5)** (来自 Google AI) 伴随论文 [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) 由 Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel 发布。
1. **[MusicGen](https://huggingface.co/docs/transformers/model_doc/musicgen)** (from Meta) released with the paper [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi and Alexandre Défossez.
1. **[MVP](https://huggingface.co/docs/transformers/model_doc/mvp)** (来自 中国人民大学 AI Box) 伴随论文 [MVP: Multi-task Supervised Pre-training for Natural Language Generation](https://arxiv.org/abs/2206.12131) 由 Tianyi Tang, Junyi Li, Wayne Xin Zhao and Ji-Rong Wen 发布。
1. **[NarrowBert](https://huggingface.co/docs/transformers/main/model_doc/narrow_bert)** (来自 University of Washington) 伴随论文 [NarrowBERT: Accelerating Masked Language Model Pretraining and Inference](https://arxiv.org/abs/2301.04761) 由 Haoxin Li, Phillip Keung, Daniel Cheng, Jungo Kasai, and Noah A. Smith 发布。
1. **[NAT](https://huggingface.co/docs/transformers/model_doc/nat)** (来自 SHI Labs) 伴随论文 [Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143) 由 Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi 发布。
1. **[Nezha](https://huggingface.co/docs/transformers/model_doc/nezha)** (来自华为诺亚方舟实验室) 伴随论文 [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204) 由 Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen and Qun Liu 发布。
1. **[NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)** (来自 Meta) 伴随论文 [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) 由 the NLLB team 发布。
1. **[NLLB-MOE](https://huggingface.co/docs/transformers/model_doc/nllb-moe)** (来自 Meta) 伴随论文 [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) 由 the NLLB team 发布。
1. **[Nougat](https://huggingface.co/docs/transformers/model_doc/nougat)** (来自 Meta AI) 伴随论文 [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418) 由 Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic 发布。
1. **[Nyströmformer](https://huggingface.co/docs/transformers/model_doc/nystromformer)** (来自 the University of Wisconsin - Madison) 伴随论文 [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902) 由 Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh 发布。
1. **[OneFormer](https://huggingface.co/docs/transformers/model_doc/oneformer)** (来自 SHI Labs)  伴随论文 [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220) 由 Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi 发布。
1. **[OpenLlama](https://huggingface.co/docs/transformers/model_doc/open-llama)** (来自 [s-JoL](https://huggingface.co/s-JoL)) 由 GitHub (现已删除).
1. **[OPT](https://huggingface.co/docs/transformers/master/model_doc/opt)** (来自 Meta AI) 伴随论文 [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) 由 Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen et al 发布。
1. **[OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)** (来自 Google AI) 伴随论文 [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) 由 Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby 发布。
1. **[OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2)** (来自 Google AI) 伴随论文 [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) 由 Matthias Minderer, Alexey Gritsenko, Neil Houlsby 发布。
1. **[PatchTSMixer](https://huggingface.co/docs/transformers/model_doc/patchtsmixer)** (来自  IBM Research) 伴随论文 [TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting](https://arxiv.org/pdf/2306.09364.pdf) 由 Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, Jayant Kalagnanam 发布。
1. **[PatchTST](https://huggingface.co/docs/transformers/model_doc/patchtst)** (来自 IBM) 伴随论文 [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/pdf/2211.14730.pdf) 由 Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam 发布。
1. **[Pegasus](https://huggingface.co/docs/transformers/model_doc/pegasus)** (来自 Google) 伴随论文 [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777) 由 Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu 发布。
1. **[PEGASUS-X](https://huggingface.co/docs/transformers/model_doc/pegasus_x)** (来自 Google) 伴随论文 [Investigating Efficiently Extending Transformers for Long Input Summarization](https://arxiv.org/abs/2208.04347) 由 Jason Phang, Yao Zhao, Peter J. Liu 发布。
1. **[Perceiver IO](https://huggingface.co/docs/transformers/model_doc/perceiver)** (来自 Deepmind) 伴随论文 [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) 由 Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira 发布。
1. **[Persimmon](https://huggingface.co/docs/transformers/model_doc/persimmon)** (来自 ADEPT) 伴随论文 [blog post](https://www.adept.ai/blog/persimmon-8b) 由 Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, Arushi Somani 发布。
1. **[Phi](https://huggingface.co/docs/transformers/model_doc/phi)** (from Microsoft) released with the papers - [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) by Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee and Yuanzhi Li, [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463) by Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar and Yin Tat Lee.
1. **[PhoBERT](https://huggingface.co/docs/transformers/model_doc/phobert)** (来自 VinAI Research) 伴随论文 [PhoBERT: Pre-trained language models for Vietnamese](https://www.aclweb.org/anthology/2020.findings-emnlp.92/) 由 Dat Quoc Nguyen and Anh Tuan Nguyen 发布。
1. **[Pix2Struct](https://huggingface.co/docs/transformers/model_doc/pix2struct)** (来自 Google) 伴随论文 [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) 由 Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova 发布。
1. **[PLBart](https://huggingface.co/docs/transformers/model_doc/plbart)** (来自 UCLA NLP) 伴随论文 [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) 由 Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang 发布。
1. **[PoolFormer](https://huggingface.co/docs/transformers/model_doc/poolformer)** (来自 Sea AI Labs) 伴随论文 [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) 由 Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng 发布。
1. **[Pop2Piano](https://huggingface.co/docs/transformers/model_doc/pop2piano)** released with the paper [Pop2Piano : Pop Audio-based Piano Cover Generation](https://arxiv.org/abs/2211.00895) by Jongho Choi, Kyogu Lee.
1. **[ProphetNet](https://huggingface.co/docs/transformers/model_doc/prophetnet)** (来自 Microsoft Research) 伴随论文 [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) 由 Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou 发布。
1. **[PVT](https://huggingface.co/docs/transformers/model_doc/pvt)** (来自 Nanjing University, The University of Hong Kong etc.) 伴随论文 [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf) 由 Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao 发布。
1. **[QDQBert](https://huggingface.co/docs/transformers/model_doc/qdqbert)** (来自 NVIDIA) 伴随论文 [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602) 由 Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev and Paulius Micikevicius 发布。
1. **[Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)** (来自 the Qwen team, Alibaba Group) 伴随论文 [Qwen Technical Report](https://arxiv.org/abs/2309.16609) 由 Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou and Tianhang Zhu 发布。
1. **[RAG](https://huggingface.co/docs/transformers/model_doc/rag)** (来自 Facebook) 伴随论文 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) 由 Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela 发布。
1. **[REALM](https://huggingface.co/docs/transformers/model_doc/realm.html)** (来自 Google Research) 伴随论文 [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) 由 Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat and Ming-Wei Chang 发布。
1. **[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer)** (来自 Google Research) 伴随论文 [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) 由 Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya 发布。
1. **[RegNet](https://huggingface.co/docs/transformers/model_doc/regnet)** (from META Research) released with the paper [Designing Network Design Space](https://arxiv.org/abs/2003.13678) by Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár.
1. **[RemBERT](https://huggingface.co/docs/transformers/model_doc/rembert)** (来自 Google Research) 伴随论文 [Rethinking embedding coupling in pre-trained language models](https://arxiv.org/pdf/2010.12821.pdf) 由 Hyung Won Chung, Thibault Févry, Henry Tsai, M. Johnson, Sebastian Ruder 发布。
1. **[ResNet](https://huggingface.co/docs/transformers/model_doc/resnet)** (from Microsoft Research) released with the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
1. **[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)** (来自 Facebook), 伴随论文 [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) 由 Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov 发布。
1. **[RoBERTa-PreLayerNorm](https://huggingface.co/docs/transformers/model_doc/roberta-prelayernorm)** (来自 Facebook) 伴随论文 [fairseq: A Fast, Extensible Toolkit for Sequence Modeling](https://arxiv.org/abs/1904.01038) 由 Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli 发布。
1. **[RoCBert](https://huggingface.co/docs/transformers/model_doc/roc_bert)** (来自 WeChatAI), 伴随论文 [RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining](https://aclanthology.org/2022.acl-long.65.pdf) 由 HuiSu, WeiweiShi, XiaoyuShen, XiaoZhou, TuoJi, JiaruiFang, JieZhou 发布。
1. **[RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer)** (来自 ZhuiyiTechnology), 伴随论文 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v1.pdf) 由 Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu 发布。
1. **[RWKV](https://huggingface.co/docs/transformers/model_doc/rwkv)** (来自 Bo Peng) 伴随论文 [this repo](https://github.com/BlinkDL/RWKV-LM) 由 Bo Peng 发布。
1. **[SeamlessM4T](https://huggingface.co/docs/transformers/model_doc/seamless_m4t)** (from Meta AI) released with the paper [SeamlessM4T — Massively Multilingual & Multimodal Machine Translation](https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf) by the Seamless Communication team.
1. **[SeamlessM4Tv2](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)** (from Meta AI) released with the paper [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team.
1. **[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)** (来自 NVIDIA) 伴随论文 [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) 由 Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo 发布。
1. **[SegGPT](https://huggingface.co/docs/transformers/main/model_doc/seggpt)** (来自 Beijing Academy of Artificial Intelligence (BAAI) 伴随论文 [SegGPT: Segmenting Everything In Context](https://arxiv.org/abs/2304.03284) 由 Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, Tiejun Huang 发布。
1. **[Segment Anything](https://huggingface.co/docs/transformers/model_doc/sam)** (来自 Meta AI) 伴随论文 [Segment Anything](https://arxiv.org/pdf/2304.02643v1.pdf) 由 Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick 发布。
1. **[SEW](https://huggingface.co/docs/transformers/model_doc/sew)** (来自 ASAPP) 伴随论文 [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) 由 Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi 发布。
1. **[SEW-D](https://huggingface.co/docs/transformers/model_doc/sew_d)** (来自 ASAPP) 伴随论文 [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/2109.06870) 由 Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi 发布。
1. **[SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)** (来自 Google AI) 伴随论文 [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) 由 Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer 发布。
1. **[SpeechT5](https://huggingface.co/docs/transformers/model_doc/speecht5)** (来自 Microsoft Research) 伴随论文 [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) 由 Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei 发布。
1. **[SpeechToTextTransformer](https://huggingface.co/docs/transformers/model_doc/speech_to_text)** (来自 Facebook), 伴随论文 [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) 由 Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino 发布。
1. **[SpeechToTextTransformer2](https://huggingface.co/docs/transformers/model_doc/speech_to_text_2)** (来自 Facebook) 伴随论文 [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) 由 Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau 发布。
1. **[Splinter](https://huggingface.co/docs/transformers/model_doc/splinter)** (来自 Tel Aviv University) 伴随论文 [Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/abs/2101.00438) 由 Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy 发布。
1. **[SqueezeBERT](https://huggingface.co/docs/transformers/model_doc/squeezebert)** (来自 Berkeley) 伴随论文 [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) 由 Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W. Keutzer 发布。
1. **[StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm)** (from Stability AI) released with the paper [StableLM 3B 4E1T (Technical Report)](https://stability.wandb.io/stability-llm/stable-lm/reports/StableLM-3B-4E1T--VmlldzoyMjU4?accessToken=u3zujipenkx5g7rtcj9qojjgxpconyjktjkli2po09nffrffdhhchq045vp0wyfo) by  Jonathan Tow, Marco Bellagente, Dakota Mahan, Carlos Riquelme Ruiz, Duy Phung, Maksym Zhuravinskyi, Nathan Cooper, Nikhil Pinnaparaju, Reshinth Adithyan, and James Baicoianu. 
1. **[Starcoder2](https://huggingface.co/docs/transformers/main/model_doc/starcoder2)** (from BigCode team) released with the paper [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/abs/2402.19173) by Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.
1. **[SwiftFormer](https://huggingface.co/docs/transformers/model_doc/swiftformer)** (来自 MBZUAI) 伴随论文 [SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications](https://arxiv.org/abs/2303.15446) 由 Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, Fahad Shahbaz Khan 发布。
1. **[Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin)** (来自 Microsoft) 伴随论文 [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) 由 Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo 发布。
1. **[Swin Transformer V2](https://huggingface.co/docs/transformers/model_doc/swinv2)** (来自 Microsoft) 伴随论文 [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) 由 Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo 发布。
1. **[Swin2SR](https://huggingface.co/docs/transformers/model_doc/swin2sr)** (来自 University of Würzburg) 伴随论文 [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345) 由 Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte 发布。
1. **[SwitchTransformers](https://huggingface.co/docs/transformers/model_doc/switch_transformers)** (from Google) released with the paper [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) by William Fedus, Barret Zoph, Noam Shazeer.
1. **[T5](https://huggingface.co/docs/transformers/model_doc/t5)** (来自 Google AI) 伴随论文 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) 由 Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu 发布。
1. **[T5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1)** (来自 Google AI) 伴随论文 [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511) 由 Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu 发布。
1. **[Table Transformer](https://huggingface.co/docs/transformers/model_doc/table-transformer)** (来自 Microsoft Research) 伴随论文 [PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents](https://arxiv.org/abs/2110.00061) 由 Brandon Smock, Rohith Pesala, Robin Abraham 发布。
1. **[TAPAS](https://huggingface.co/docs/transformers/model_doc/tapas)** (来自 Google AI) 伴随论文 [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/abs/2004.02349) 由 Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno and Julian Martin Eisenschlos 发布。
1. **[TAPEX](https://huggingface.co/docs/transformers/model_doc/tapex)** (来自 Microsoft Research) 伴随论文 [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/abs/2107.07653) 由 Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, Jian-Guang Lou 发布。
1. **[Time Series Transformer](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)** (from HuggingFace).
1. **[TimeSformer](https://huggingface.co/docs/transformers/model_doc/timesformer)** (from Facebook) released with the paper [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) by Gedas Bertasius, Heng Wang, Lorenzo Torresani.
1. **[Trajectory Transformer](https://huggingface.co/docs/transformers/model_doc/trajectory_transformers)** (from the University of California at Berkeley) released with the paper [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039) by Michael Janner, Qiyang Li, Sergey Levine
1. **[Transformer-XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl)** (来自 Google/CMU) 伴随论文 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) 由 Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov 发布。
1. **[TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)** (来自 Microsoft) 伴随论文 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) 由 Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei 发布。
1. **[TVLT](https://huggingface.co/docs/transformers/model_doc/tvlt)** (来自 UNC Chapel Hill) 伴随论文 [TVLT: Textless Vision-Language Transformer](https://arxiv.org/abs/2209.14156) 由 Zineng Tang, Jaemin Cho, Yixin Nie, Mohit Bansal 发布。
1. **[TVP](https://huggingface.co/docs/transformers/model_doc/tvp)** (来自 Intel) 伴随论文 [Text-Visual Prompting for Efficient 2D Temporal Video Grounding](https://arxiv.org/abs/2303.04995) 由 Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, Ke Ding 发布.
1. **[UDOP](https://huggingface.co/docs/transformers/main/model_doc/udop)** (来自 Microsoft Research) 伴随论文 [Unifying Vision, Text, and Layout for Universal Document Processing](https://arxiv.org/abs/2212.02623) 由 Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal 发布。
1. **[UL2](https://huggingface.co/docs/transformers/model_doc/ul2)** (from Google Research) released with the paper [Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131v1) by Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, Donald Metzler
1. **[UMT5](https://huggingface.co/docs/transformers/model_doc/umt5)** (来自 Google Research) 伴随论文 [UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining](https://openreview.net/forum?id=kXwdL1cWOAi) 由 Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, Noah Constant 发布。
1. **[UniSpeech](https://huggingface.co/docs/transformers/model_doc/unispeech)** (来自 Microsoft Research) 伴随论文 [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://arxiv.org/abs/2101.07597) 由 Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang 发布。
1. **[UniSpeechSat](https://huggingface.co/docs/transformers/model_doc/unispeech-sat)** (来自 Microsoft Research) 伴随论文 [UNISPEECH-SAT: UNIVERSAL SPEECH REPRESENTATION LEARNING WITH SPEAKER AWARE PRE-TRAINING](https://arxiv.org/abs/2110.05752) 由 Sanyuan Chen, Yu Wu, Chengyi Wang, Zhengyang Chen, Zhuo Chen, Shujie Liu, Jian Wu, Yao Qian, Furu Wei, Jinyu Li, Xiangzhan Yu 发布。
1. **[UnivNet](https://huggingface.co/docs/transformers/model_doc/univnet)** (from Kakao Corporation) released with the paper [UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation](https://arxiv.org/abs/2106.07889) by Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kim, and Juntae Kim.
1. **[UPerNet](https://huggingface.co/docs/transformers/model_doc/upernet)** (来自 Peking University) 伴随论文 [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) 由 Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, Jian Sun 发布。
1. **[VAN](https://huggingface.co/docs/transformers/model_doc/van)** (来自 Tsinghua University and Nankai University) 伴随论文 [Visual Attention Network](https://arxiv.org/pdf/2202.09741.pdf) 由 Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu 发布。
1. **[VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)** (来自 Multimedia Computing Group, Nanjing University) 伴随论文 [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) 由 Zhan Tong, Yibing Song, Jue Wang, Limin Wang 发布。
1. **[ViLT](https://huggingface.co/docs/transformers/model_doc/vilt)** (来自 NAVER AI Lab/Kakao Enterprise/Kakao Brain) 伴随论文 [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) 由 Wonjae Kim, Bokyung Son, Ildoo Kim 发布。
1. **[VipLlava](https://huggingface.co/docs/transformers/model_doc/vipllava)** (来自 University of Wisconsin–Madison) 伴随论文 [Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/abs/2312.00784) 由 Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, Yong Jae Lee 发布。
1. **[Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)** (来自 Google AI) 伴随论文 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) 由 Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby 发布。
1. **[VisualBERT](https://huggingface.co/docs/transformers/model_doc/visual_bert)** (来自 UCLA NLP) 伴随论文 [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557) 由 Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang 发布。
1. **[ViT Hybrid](https://huggingface.co/docs/transformers/model_doc/vit_hybrid)** (来自 Google AI) 伴随论文 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) 由 Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby 发布。
1. **[VitDet](https://huggingface.co/docs/transformers/model_doc/vitdet)** (来自 Meta AI) 伴随论文 [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527) 由 Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He 发布。
1. **[ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae)** (来自 Meta AI) 伴随论文 [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) 由 Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick 发布。
1. **[ViTMatte](https://huggingface.co/docs/transformers/model_doc/vitmatte)** (来自 HUST-VL) 伴随论文 [ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers](https://arxiv.org/abs/2305.15272) 由 Jingfeng Yao, Xinggang Wang, Shusheng Yang, Baoyuan Wang 发布。
1. **[ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn)** (来自 Meta AI) 伴随论文 [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141) by Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas 发布.
1. **[VITS](https://huggingface.co/docs/transformers/model_doc/vits)** (来自 Kakao Enterprise) 伴随论文 [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) 由 Jaehyeon Kim, Jungil Kong, Juhee Son 发布。
1. **[ViViT](https://huggingface.co/docs/transformers/model_doc/vivit)** (来自 Google Research) released with the paper [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) 由 Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid.
1. **[Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)** (来自 Facebook AI) 伴随论文 [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) 由 Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli 发布。
1. **[Wav2Vec2-BERT](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert)** (from Meta AI) released with the paper [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team.
1. **[Wav2Vec2-Conformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer)** (来自 Facebook AI) 伴随论文 [FAIRSEQ S2T: Fast Speech-to-Text Modeling with FAIRSEQ](https://arxiv.org/abs/2010.05171) 由 Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino 发布。
1. **[Wav2Vec2Phoneme](https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme)** (来自 Facebook AI) 伴随论文 [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition](https://arxiv.org/abs/2109.11680) 由 Qiantong Xu, Alexei Baevski, Michael Auli 发布。
1. **[WavLM](https://huggingface.co/docs/transformers/model_doc/wavlm)** (from Microsoft Research) released with the paper [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) by Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Furu Wei.
1. **[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)** (来自 OpenAI) 伴随论文 [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) 由 Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever 发布。
1. **[X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)** (来自 Microsoft Research) 伴随论文 [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/abs/2208.02816) 由 Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, Haibin Ling 发布。
1. **[X-MOD](https://huggingface.co/docs/transformers/model_doc/xmod)** (来自 Meta AI) 伴随论文 [Lifting the Curse of Multilinguality by Pre-training Modular Transformers](http://dx.doi.org/10.18653/v1/2022.naacl-main.255) 由 Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, Mikel Artetxe 发布。
1. **[XGLM](https://huggingface.co/docs/transformers/model_doc/xglm)** (From Facebook AI) released with the paper [Few-shot Learning with Multilingual Language Models](https://arxiv.org/abs/2112.10668) by Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.
1. **[XLM](https://huggingface.co/docs/transformers/model_doc/xlm)** (来自 Facebook) 伴随论文 [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) 由 Guillaume Lample and Alexis Conneau 发布。
1. **[XLM-ProphetNet](https://huggingface.co/docs/transformers/model_doc/xlm-prophetnet)** (来自 Microsoft Research) 伴随论文 [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) 由 Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang and Ming Zhou 发布。
1. **[XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)** (来自 Facebook AI), 伴随论文 [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) 由 Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov 发布。
1. **[XLM-RoBERTa-XL](https://huggingface.co/docs/transformers/model_doc/xlm-roberta-xl)** (来自 Facebook AI) 伴随论文 [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://arxiv.org/abs/2105.00572) 由 Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, Alexis Conneau 发布。
1. **[XLM-V](https://huggingface.co/docs/transformers/model_doc/xlm-v)** (来自 Meta AI) 伴随论文 [XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models](https://arxiv.org/abs/2301.10472) 由 Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, Madian Khabsa 发布。
1. **[XLNet](https://huggingface.co/docs/transformers/model_doc/xlnet)** (来自 Google/CMU) 伴随论文 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) 由 Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le 发布。
1. **[XLS-R](https://huggingface.co/docs/transformers/model_doc/xls_r)** (来自 Facebook AI) 伴随论文 [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://arxiv.org/abs/2111.09296) 由 Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli 发布。
1. **[XLSR-Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/xlsr_wav2vec2)** (来自 Facebook AI) 伴随论文 [Unsupervised Cross-Lingual Representation Learning For Speech Recognition](https://arxiv.org/abs/2006.13979) 由 Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli 发布。
1. **[YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos)** (来自 Huazhong University of Science & Technology) 伴随论文 [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/abs/2106.00666) 由 Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu 发布。
1. **[YOSO](https://huggingface.co/docs/transformers/model_doc/yoso)** (来自 the University of Wisconsin - Madison) 伴随论文 [You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](https://arxiv.org/abs/2111.09714) 由 Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh 发布。
1. 想要贡献新的模型？我们这里有一份**详细指引和模板**来引导你添加新的模型。你可以在 [`templates`](./templates) 目录中找到他们。记得查看 [贡献指南](./CONTRIBUTING.md) 并在开始写 PR 前联系维护人员或开一个新的 issue 来获得反馈。

要检查某个模型是否已有 Flax、PyTorch 或 TensorFlow 的实现，或其是否在 🤗 Tokenizers 库中有对应词符化器（tokenizer），敬请参阅[此表](https://huggingface.co/docs/transformers/index#supported-frameworks)。

这些实现均已于多个数据集测试（请参看用例脚本）并应于原版实现表现相当。你可以在用例文档的[此节](https://huggingface.co/docs/transformers/examples)中了解表现的细节。


## 了解更多

| 章节 | 描述 |
|-|-|
| [文档](https://huggingface.co/docs/transformers/) | 完整的 API 文档和教程 |
| [任务总结](https://huggingface.co/docs/transformers/task_summary) | 🤗 Transformers 支持的任务 |
| [预处理教程](https://huggingface.co/docs/transformers/preprocessing) | 使用 `Tokenizer` 来为模型准备数据 |
| [训练和微调](https://huggingface.co/docs/transformers/training) | 在 PyTorch/TensorFlow 的训练循环或 `Trainer` API 中使用 🤗 Transformers 提供的模型 |
| [快速上手：微调和用例脚本](https://github.com/huggingface/transformers/tree/main/examples) | 为各种任务提供的用例脚本 |
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
