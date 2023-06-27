<!--版权所有2021年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）的规定，除非符合许可证的规定，否则不得使用本文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取适用于特定语言的权限和限制。⚠️请注意，此文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
渲染。
-->
# LayoutXLM
## 概述
LayoutXLM 是由 Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, ChaZhang, Furu Wei 提出的 [LayoutXLM: 多模态预训练用于多语言视觉丰富文档理解](https://arxiv.org/abs/2104.08836) 的多语言扩展，使用了 53 种语言进行训练。
论文的摘要如下：
*最近，在文本、布局和图像的多模态预训练中，对于视觉丰富文档理解任务取得了 SOTA 性能，这表明了跨不同模态之间的联合学习的巨大潜力。在本文中，我们提出了 LayoutXLM，这是一个用于多语言文档理解的多模态预训练模型，旨在弥合视觉丰富文档理解的语言障碍。为了准确评估 LayoutXLM，我们还 this paper, we present LayoutXLM, a multimodal pre-trained model for multilingual document understanding, which aims to
引入了一个名为 XFUN 的多语言表单理解基准数据集，其中包括 7 种语言（中文、日语、西班牙语、法语、意大利语、德语、葡萄牙语）的表单理解样本，introduce a multilingual form understanding benchmark dataset named XFUN, which includes form understanding samples in
并为每种语言手动标记了键值对。实验结果表明，LayoutXLM 模型在 XFUN 数据集上显著优于现有的 SOTA 跨语言预训练模型。*for each language. Experiment results show that the LayoutXLM model has significantly outperformed the existing SOTA
可以直接将 LayoutXLM 的权重插入到 LayoutLMv2 模型中，如下所示：
请注意，LayoutXLM 有自己的标记器，基于
```python
from transformers import LayoutLMv2Model

model = LayoutLMv2Model.from_pretrained("microsoft/layoutxlm-base")
```

Note that LayoutXLM has its own tokenizer, based on
[`LayoutXLMTokenizer`]/[`LayoutXLMTokenizerFast`]。您可以按照以下方式进行初始化：follows:

```python
from transformers import LayoutXLMTokenizer

tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
```

与 LayoutLMv2 类似，您可以使用 [`LayoutXLMProcessor`]（它内部应用了 [`LayoutLMv2FeatureExtractor`] 和 [`LayoutXLMTokenizer`]/[`LayoutXLMTokenizerFast`]）来为模型准备所有数据。data for the model.

由于 LayoutXLM 的架构与 LayoutLMv2 相同，因此可以参考 [LayoutLMv2 的文档页面](layoutlmv2) 获取所有提示、代码示例和笔记本。
此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可在 [此处](https://github.com/microsoft/unilm) 找到。

## LayoutXLMTokenizer
[[autodoc]] LayoutXLMTokenizer    - __call__    - build_inputs_with_special_tokens    - get_special_tokens_mask    - create_token_type_ids_from_sequences    - save_vocabulary
## LayoutXLMTokenizerFast
[[autodoc]] LayoutXLMTokenizerFast    - __call__
## LayoutXLMProcessor
[[autodoc]] LayoutXLMProcessor    - __call__