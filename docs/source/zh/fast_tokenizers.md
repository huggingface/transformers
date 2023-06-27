<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证，第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“原样”基础分发的，不提供任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件为 Markdown 格式，但包含特定的语法以供我们的文档构建器（类似于 MDX）使用，这可能无法在您的 Markdown 查看器中正确渲染。
-->

# 使用🤗 Tokenizers

[`PreTrainedTokenizerFast`] 依赖于 [🤗 Tokenizers](https://huggingface.co/docs/tokenizers) 库。从 🤗 Tokenizers 库中获取的 tokenizer 可以非常简单地加载到 🤗 Transformers 中。

在进入详细信息之前，让我们首先通过几行代码创建一个虚拟 tokenizer：
```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> from tokenizers.pre_tokenizers import Whitespace

>>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
>>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

>>> tokenizer.pre_tokenizer = Whitespace()
>>> files = [...]
>>> tokenizer.train(files, trainer)
```

现在我们有一个在我们定义的文件上训练的 tokenizer。我们可以在该运行时继续使用它，也可以将其保存到 JSON 文件中以供将来重用。

## 直接从 tokenizer 对象加载

让我们看看如何在🤗 Transformers 库中利用这个 tokenizer 对象。[`PreTrainedTokenizerFast`] 类允许通过接受实例化的 *tokenizer* 对象作为参数来轻松实例化：

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

现在，这个对象可以与 🤗 Transformers tokenizers 共享的所有方法一起使用！请访问 [tokenizer 页面](main_classes/tokenizer) 了解更多信息。

## 从 JSON 文件加载

为了从 JSON 文件加载 tokenizer，让我们首先保存我们的 tokenizer：

```python
>>> tokenizer.save("tokenizer.json")
```

我们保存此文件的路径可以通过 [`PreTrainedTokenizerFast`] 初始化方法的 `tokenizer_file` 参数传递：
```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

现在，这个对象可以与 🤗 Transformers tokenizers 共享的所有方法一起使用！请访问 [tokenizer 页面](main_classes/tokenizer) 了解更多信息。