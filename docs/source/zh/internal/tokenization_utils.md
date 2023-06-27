<!--版权所有2020年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“AS IS” BASIS 分发的，没有任何形式的保证或条件。请参阅许可证以获取特定语言下权限和限制的详细信息。⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。

-->


# Tokenizers 的实用工具

此页面列出了所有标记器使用的实用函数，主要是实现了 [`~tokenization_utils_base.PreTrainedTokenizerBase`] 类的公共方法，该类实现了 [`PreTrainedTokenizer`]、[`PreTrainedTokenizerFast`] 和 [`~tokenization_utils_base.SpecialTokensMixin`] 混入类。

大多数这些函数只在您研究库中的标记器代码时才有用。

## PreTrainedTokenizerBase

[[autodoc]] tokenization_utils_base.PreTrainedTokenizerBase
    - __call__
    - all

## SpecialTokensMixin

[[autodoc]] tokenization_utils_base.SpecialTokensMixin

## Enums and namedtuples

[[autodoc]] tokenization_utils_base.TruncationStrategy

[[autodoc]] tokenization_utils_base.CharSpan

[[autodoc]] tokenization_utils_base.TokenSpan
