<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证，版本 2.0（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法正常渲染在您的 Markdown 阅读器中。
# BERT 学派 rendered properly in your Markdown viewer.

-->

有一个日益发展的研究领域，致力于研究像 BERT 这样的大规模 Transformer 模型的内部工作（有人称之为“BERT 学派”）。这个领域的一些优秀例子包括：
- 《BERT 重新发现经典 NLP 流水线》（Ian Tenney, Dipanjan Das, Ellie Pavlick 著）：  https://arxiv.org/abs/1905.05950

- 《十六个注意力头是否真的比一个好？》（Paul Michel, Omer Levy, Graham Neubig 著）：https://arxiv.org/abs/1905.10650  https://arxiv.org/abs/1905.05950
- Are Sixteen Heads Really Better than One? by Paul Michel, Omer Levy, Graham Neubig: https://arxiv.org/abs/1905.10650
- 《BERT 看什么？BERT 的注意力分析》（Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning 著）：https://arxiv.org/abs/1906.04341  Manning: https://arxiv.org/abs/1906.04341

- 《CAT-probing：一种基于度量的解释预训练编程语言模型如何关注代码结构的方法》：https://arxiv.org/abs/2210.04633
为了帮助这个新领域的发展，我们在 BERT/GPT/GPT-2 模型中增加了一些额外的功能，以帮助人们访问内部表示，这主要是基于 Paul Michel 的优秀工作（https://arxiv.org/abs/1905.10650）：(https://arxiv.org/abs/1905.10650):


- 访问 BERT/GPT/GPT-2 的所有隐藏状态,- 访问 BERT/GPT/GPT-2 每个注意力头的所有注意力权重,- 检索头部输出值和梯度，以便计算头部重要性分数和修剪头部，详细说明请参见  https://arxiv.org/abs/1905.10650.

为了帮助您理解和使用这些功能，我们添加了一个特定的示例脚本：[bertology.py](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology/run_bertology.py)，用于提取信息和修剪在 GLUE 上预训练的模型。