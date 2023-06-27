<!--版权所有2020年HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合本许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件以“按现状”分发，不附带任何明示或暗示的保证或条件。有关特定语言下权限和限制的详细信息，请参阅许可证。⚠️请注意，本文件是 Markdown 格式，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能无法在 Markdown 查看器中正确渲染。
-->
## BertJapanese## 概述

用于日文文本的 BERT 模型。

有两种不同的标记化方法的模型：
- 使用 MeCab 和 WordPiece 进行标记化。这需要一些额外的依赖项，[fugashi](https://github.com/polm/fugashi)，它是 [MeCab](https://taku910.github.io/mecab/) 的封装。

- 字符标记化。

要使用 *MecabTokenizer*，您应该 `pip install transformers["ja"]`（或者如果您从源代码安装，则为 `pip install -e .["ja"]`）以安装依赖项。
详见 [cl-tohoku 存储库的详细信息](https://github.com/cl-tohoku/bert-japanese)。

使用具有 MeCab 和 WordPiece 标记化的模型的示例：


```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

使用字符标记化的模型的示例：
```python
>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

>>> ## Input Japanese Text
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

提示：
- 除了标记化方法外，此实现与 BERT 相同。有关更多使用示例，请参阅 [BERT 文档](bert)。

此模型由 [cl-tohoku](https://huggingface.co/cl-tohoku) 贡献。

## BertJapaneseTokenizer

[[autodoc]] BertJapaneseTokenizer
