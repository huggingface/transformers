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

# 分词器的摘要
[[open-in-colab]]

在这个页面，我们来仔细研究分词的知识。
<Youtube id="VFp38yj8h3A"/>

正如我们在[the preprocessing tutorial](preprocessing)所看到的那样，对文本进行分词就是将一段文本分割成很多单词或者子单词，
这些单词或者子单词然后会通过一个查询表格被转换到id，将单词或者子单词转换到id是很直截了当的，也就是一个简单的映射，
所以这么来看，我们主要关注将一段文本分割成很多单词或者很多子单词（像：对一段文本进行分词），更加准确的来说，我们将关注
在🤗 Transformers内用到的三种主要类型的分词器：[Byte-Pair Encoding (BPE)](#byte-pair-encoding), [WordPiece](#wordpiece), 
and [SentencePiece](#sentencepiece)，并且给出了示例，哪个模型用到了哪种类型的分词器。

注意到在每个模型的主页，你可以查看文档上相关的分词器，就可以知道预训练模型使用了哪种类型的分词器。
举个例子，如果我们查看[`BertTokenizer`]，我们就能看到模型使用了[WordPiece](#wordpiece)。

## 介绍
将一段文本分词到小块是一个比它看起来更加困难的任务，并且有很多方式来实现分词，举个例子，让我们看看这个句子
`"Don't you love 🤗 Transformers? We sure do."`

<Youtube id="nhJxYji1aho"/>

对这段文本分词的一个简单方式，就是使用空格来分词，得到的结果是：

```
["Don't", "you", "love", "🤗", "Transformers?", "We", "sure", "do."]
```

上面的分词是一个明智的开始，但是如果我们查看token `"Transformers?"` 和 `"do."`，我们可以观察到标点符号附在单词`"Transformer"` 
和 `"do"`的后面，这并不是最理想的情况。我们应该将标点符号考虑进来，这样一个模型就没必要学习一个单词和每个可能跟在后面的
标点符号的不同的组合，这么组合的话，模型需要学习的组合的数量会急剧上升。将标点符号也考虑进来，对范例文本进行分词的结果就是：

```
["Don", "'", "t", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

分词的结果更好了，然而，这么做也是不好的，分词怎么处理单词`"Don't"`，`"Don't"`的含义是`"do not"`，所以这么分词`["Do", "n't"]`
会更好。现在开始事情就开始变得复杂起来了，部分的原因是每个模型都有它自己的分词类型。依赖于我们应用在文本分词上的规则，
相同的文本会产生不同的分词输出。用在训练数据上的分词规则，被用来对输入做分词操作，一个预训练模型才会正确的执行。

[spaCy](https://spacy.io/) and [Moses](http://www.statmt.org/moses/?n=Development.GetStarted) 是两个受欢迎的基于规则的
分词器。将这两个分词器应用在示例文本上，*spaCy* 和 *Moses*会输出类似下面的结果：

```
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

可见上面的分词使用到了空格和标点符号的分词方式，以及基于规则的分词方式。空格和标点符号分词以及基于规则的分词都是单词分词的例子。
不那么严格的来说，单词分词的定义就是将句子分割到很多单词。然而将文本分割到更小的块是符合直觉的，当处理大型文本语料库时，上面的
分词方法会导致很多问题。在这种情况下，空格和标点符号分词通常会产生一个非常大的词典（使用到的所有不重复的单词和tokens的集合）。
像：[Transformer XL](model_doc/transformerxl)使用空格和标点符号分词，结果会产生一个大小是267,735的词典！

这么大的一个词典容量，迫使模型有着一个巨大的embedding矩阵，以及巨大的输入和输出层，这会增加内存使用量，也会提高时间复杂度。通常
情况下，transformers模型几乎没有词典容量大于50,000的，特别是只在一种语言上预训练的模型。

所以如果简单的空格和标点符号分词让人不满意，为什么不简单的对字符分词？

<Youtube id="ssLq_EK2jLE"/>

尽管字符分词是非常简单的，并且能极大的减少内存使用，降低时间复杂度，但是这样做会让模型很难学到有意义的输入表达。像：
比起学到单词`"today"`的一个有意义的上下文独立的表达，学到字母`"t"`的一个有意义的上下文独立的表达是相当困难的。因此，
字符分词经常会伴随着性能的下降。所以为了获得最好的结果，transformers模型在单词级别分词和字符级别分词之间使用了一个折中的方案
被称作**子词**分词。

## 子词分词

<Youtube id="zHvTiHr506c"/>

子词分词算法依赖这样的原则：频繁使用的单词不应该被分割成更小的子词，但是很少使用的单词应该被分解到有意义的子词。举个例子：
`"annoyingly"`能被看作一个很少使用的单词，能被分解成`"annoying"`和`"ly"`。`"annoying"`和`"ly"`作为独立地子词，出现
的次数都很频繁，而且与此同时单词`"annoyingly"`的含义可以通过组合`"annoying"`和`"ly"`的含义来获得。在粘合和胶水语言上，
像Turkish语言，这么做是相当有用的，在这样的语言里，通过线性组合子词，大多数情况下你能形成任意长的复杂的单词。

子词分词允许模型有一个合理的词典大小，而且能学到有意义的上下文独立地表达。除此以外，子词分词可以让模型处理以前从来没见过的单词，
方式是通过分解这些单词到已知的子词，举个例子：[`~transformers.BertTokenizer`]对句子`"I have a new GPU!"`分词的结果如下：

```py
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> tokenizer.tokenize("I have a new GPU!")
["i", "have", "a", "new", "gp", "##u", "!"]
```

因为我们正在考虑不区分大小写的模型，句子首先被转换成小写字母形式。我们可以见到单词`["i", "have", "a", "new"]`在分词器
的词典内，但是这个单词`"gpu"`不在词典内。所以，分词器将`"gpu"`分割成已知的子词`["gp" and "##u"]`。`"##"`意味着剩下的
token应该附着在前面那个token的后面，不带空格的附着（分词的解码或者反向）。

另外一个例子，[`~transformers.XLNetTokenizer`]对前面的文本例子分词结果如下：

```py
>>> from transformers import XLNetTokenizer

>>> tokenizer = XLNetTokenizer.from_pretrained("xlnet/xlnet-base-cased")
>>> tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do.")
["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]
```

当我们查看[SentencePiece](#sentencepiece)时会回过头来解释这些`"▁"`符号的含义。正如你能见到的，很少使用的单词
`"Transformers"`能被分割到更加频繁使用的子词`"Transform"`和`"ers"`。

现在让我们来看看不同的子词分割算法是怎么工作的，注意到所有的这些分词算法依赖于某些训练的方式，这些训练通常在语料库上完成，
相应的模型也是在这个语料库上训练的。

<a id='byte-pair-encoding'></a>

### Byte-Pair Encoding (BPE)

Byte-Pair Encoding (BPE)来自于[Neural Machine Translation of Rare Words with Subword Units (Sennrich et
al., 2015)](https://arxiv.org/abs/1508.07909)。BPE依赖于一个预分词器，这个预分词器会将训练数据分割成单词。预分词可以是简单的
空格分词，像：：[GPT-2](model_doc/gpt2)，[RoBERTa](model_doc/roberta)。更加先进的预分词方式包括了基于规则的分词，像： [XLM](model_doc/xlm)，[FlauBERT](model_doc/flaubert)，FlauBERT在大多数语言使用了Moses，或者[GPT](model_doc/gpt)，GPT
使用了Spacy和ftfy，统计了训练语料库中每个单词的频次。

在预分词以后，生成了单词的集合，也确定了训练数据中每个单词出现的频次。下一步，BPE产生了一个基础词典，包含了集合中所有的符号，
BPE学习融合的规则-组合基础词典中的两个符号来形成一个新的符号。BPE会一直学习直到词典的大小满足了期望的词典大小的要求。注意到
期望的词典大小是一个超参数，在训练这个分词器以前就需要人为指定。

举个例子，让我们假设在预分词以后，下面的单词集合以及他们的频次都已经确定好了：

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

所以，基础的词典是`["b", "g", "h", "n", "p", "s", "u"]`。将所有单词分割成基础词典内的符号，就可以获得：

```
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```
BPE接着会统计每个可能的符号对的频次，然后挑出出现最频繁的的符号对，在上面的例子中，`"h"`跟了`"u"`出现了10 + 5 = 15次
（10次是出现了10次`"hug"`，5次是出现了5次`"hugs"`）。然而，最频繁的符号对是`"u"`后面跟了个`"g"`，总共出现了10 + 5 + 5
= 20次。因此，分词器学到的第一个融合规则是组合所有的`"u"`后面跟了个`"g"`符号。下一步，`"ug"`被加入到了词典内。单词的集合
就变成了：

```
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

BPE接着会统计出下一个最普遍的出现频次最大的符号对。也就是`"u"`后面跟了个`"n"`，出现了16次。`"u"`，`"n"`被融合成了`"un"`。
也被加入到了词典中，再下一个出现频次最大的符号对是`"h"`后面跟了个`"ug"`，出现了15次。又一次这个符号对被融合成了`"hug"`，
也被加入到了词典中。

在当前这步，词典是`["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]`，我们的单词集合则是：

```
("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)
```

假设，the Byte-Pair Encoding在这个时候停止训练，学到的融合规则并应用到其他新的单词上（只要这些新单词不包括不在基础词典内的符号
就行）。举个例子，单词`"bug"`会被分词到`["b", "ug"]`，但是`"mug"`会被分词到`["<unk>", "ug"]`，因为符号`"m"`不在基础词典内。
通常来看的话，单个字母像`"m"`不会被`"<unk>"`符号替换掉，因为训练数据通常包括了每个字母，每个字母至少出现了一次，但是在特殊的符号
中也可能发生像emojis。

就像之前提到的那样，词典的大小，举个例子，基础词典的大小 + 融合的数量，是一个需要配置的超参数。举个例子：[GPT](model_doc/gpt)
的词典大小是40,478，因为GPT有着478个基础词典内的字符，在40,000次融合以后选择了停止训练。

#### Byte-level BPE

一个包含了所有可能的基础字符的基础字典可能会非常大，如果考虑将所有的unicode字符作为基础字符。为了拥有一个更好的基础词典，[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)使用了字节
作为基础词典，这是一个非常聪明的技巧，迫使基础词典是256大小，而且确保了所有基础字符包含在这个词典内。使用了其他的规则
来处理标点符号，这个GPT2的分词器能对每个文本进行分词，不需要使用到<unk>符号。[GPT-2](model_doc/gpt)有一个大小是50,257
的词典，对应到256字节的基础tokens，一个特殊的文本结束token，这些符号经过了50,000次融合学习。

<a id='wordpiece'></a>

### WordPiece

WordPiece是子词分词算法，被用在[BERT](model_doc/bert)，[DistilBERT](model_doc/distilbert)，和[Electra](model_doc/electra)。
这个算法发布在[Japanese and Korean
Voice Search (Schuster et al., 2012)](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
和BPE非常相似。WordPiece首先初始化一个词典，这个词典包含了出现在训练数据中的每个字符，然后递进的学习一个给定数量的融合规则。和BPE相比较，
WordPiece不会选择出现频次最大的符号对，而是选择了加入到字典以后能最大化训练数据似然值的符号对。

所以这到底意味着什么？参考前面的例子，最大化训练数据的似然值，等价于找到一个符号对，它们的概率除以这个符号对中第一个符号的概率，
接着除以第二个符号的概率，在所有的符号对中商最大。像：如果`"ug"`的概率除以`"u"`除以`"g"`的概率的商，比其他任何符号对更大，
这个时候才能融合`"u"`和`"g"`。直觉上，WordPiece，和BPE有点点不同，WordPiece是评估融合两个符号会失去的量，来确保这么做是值得的。

<a id='unigram'></a>

### Unigram

Unigram是一个子词分词器算法，介绍见[Subword Regularization: Improving Neural Network Translation
Models with Multiple Subword Candidates (Kudo, 2018)](https://arxiv.org/pdf/1804.10959.pdf)。和BPE或者WordPiece相比较
，Unigram使用大量的符号来初始化它的基础字典，然后逐渐的精简每个符号来获得一个更小的词典。举例来看基础词典能够对应所有的预分词
的单词以及最常见的子字符串。Unigram没有直接用在任何transformers的任何模型中，但是和[SentencePiece](#sentencepiece)一起联合使用。

在每个训练的步骤，Unigram算法在当前词典的训练数据上定义了一个损失函数（经常定义为log似然函数的），还定义了一个unigram语言模型。
然后，对词典内的每个符号，算法会计算如果这个符号从词典内移除，总的损失会升高多少。Unigram然后会移除百分之p的符号，这些符号的loss
升高是最低的（p通常是10%或者20%），像：这些在训练数据上对总的损失影响最小的符号。重复这个过程，直到词典已经达到了期望的大小。
为了任何单词都能被分词，Unigram算法总是保留基础的字符。

因为Unigram不是基于融合规则（和BPE以及WordPiece相比较），在训练以后算法有几种方式来分词，如果一个训练好的Unigram分词器
的词典是这个：

```
["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"],
```
`"hugs"`可以被分词成`["hug", "s"]`, `["h", "ug", "s"]`或者`["h", "u", "g", "s"]`。所以选择哪一个呢？Unigram在保存
词典的时候还会保存训练语料库内每个token的概率，所以在训练以后可以计算每个可能的分词结果的概率。实际上算法简单的选择概率
最大的那个分词结果，但是也会提供概率来根据分词结果的概率来采样一个可能的分词结果。

分词器在损失函数上训练，这些损失函数定义了这些概率。假设训练数据包含了这些单词 $x_{1}$, $\dots$, $x_{N}$，一个单词$x_{i}$
的所有可能的分词结果的集合定义为$S(x_{i})$，然后总的损失就可以定义为：

$$\mathcal{L} = -\sum_{i=1}^{N} \log \left ( \sum_{x \in S(x_{i})} p(x) \right )$$

<a id='sentencepiece'></a>

### SentencePiece
目前为止描述的所有分词算法都有相同的问题：它们都假设输入的文本使用空格来分开单词。然而，不是所有的语言都使用空格来分开单词。
一个可能的解决方案是使用某种语言特定的预分词器。像：[XLM](model_doc/xlm)使用了一个特定的中文、日语和Thai的预分词器。
为了更加广泛的解决这个问题，[SentencePiece: A simple and language independent subword tokenizer and
detokenizer for Neural Text Processing (Kudo et al., 2018)](https://arxiv.org/pdf/1808.06226.pdf)
将输入文本看作一个原始的输入流，因此使用的符合集合中也包括了空格。SentencePiece然后会使用BPE或者unigram算法来产生合适的
词典。

举例来说，[`XLNetTokenizer`]使用了SentencePiece，这也是为什么上面的例子中`"▁"`符号包含在词典内。SentencePiece解码是非常容易的，因为所有的tokens能被concatenate起来，然后将`"▁"`替换成空格。

库内所有使用了SentencePiece的transformers模型，会和unigram组合起来使用，像：使用了SentencePiece的模型是[ALBERT](model_doc/albert), 
[XLNet](model_doc/xlnet)，[Marian](model_doc/marian)，和[T5](model_doc/t5)。
