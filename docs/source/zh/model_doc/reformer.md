<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。注意：此文件是 Markdown 格式的，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在 Markdown 查看器中正确呈现。
⚠️ 请注意，本文件是 Markdown 格式，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在 Markdown 查看器中正确呈现。注意：此文件是 Markdown 格式的，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在 Markdown 查看器中正确呈现。
-->
# Reformer
<div class="flex flex-wrap space-x-1"> <a href="https://huggingface.co/models?filter=reformer"> <img alt="Models" src="https://img.shields.io/badge/All_model_pages-reformer-blueviolet"> </a> <a href="https://huggingface.co/spaces/docs-demos/reformer-crime-and-punishment"> <img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> </a> </div>

**免责声明：** 此模型仍在开发中，如果您发现任何异常，请提交 [GitHub 问题](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

## 概览

Reformer 模型是由 Nikita Kitaev, Ł ukasz Kaiser, Anselm Levskaya 在论文 [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451.pdf) 中提出的。

以下是来自论文的摘要：

*大型 Transformer 模型通常在许多任务上取得最先进的结果，但训练这些模型的成本往往很高，尤其是对于长序列来说。为了提高 Transformer 的效率，我们提出了两种技术。首先，我们将点积注意力替换为使用局部敏感哈希的注意力，将其复杂度从 O(L^2)降低到 O(Llog(L))，其中 L 是序列的长度。此外，我们使用可逆残差层代替标准残差层，在训练过程中只需存储一次激活，而不是 N 次，其中 N 是层数。由此得到的模型 Reformer 在性能上与 Transformer 模型相当，但在处理长序列时更节省内存且速度更快。* 

此模型由 [patrickvonplaten](https://huggingface.co/patrickvonplaten) 贡献。作者的代码可以在此处找到 [here](https://github.com/google/trax/tree/master/trax/models/reformer)。

提示：

- 由于 PyTorch 中的错误，Reformer **不** 与 *torch.nn.DataParallel* 兼容，请参阅 [问题＃ 36035](https://github.com/pytorch/pytorch/issues/36035)。- 使用轴向位置编码（详见下文）。这是一种避免大型位置编码矩阵（当序列长度非常大时）的机制，将其分解为较小的矩阵。

- 将传统的注意力替换为 LSH（局部敏感哈希）注意力（详见下文）。这是一种在注意力层中避免计算完整的查询-键乘积的技术。- 通过使用可逆 Transformer 层在反向传播过程中获得每层的中间结果（从输入中减去残差），或者在给定层内重新计算它们的结果（比存储它们不太高效，但节省内存）。

- 将前馈操作分块计算，而不是整个批次一起计算。

## 轴向位置编码 Axial Positional Encodings

轴向位置编码首次在 Google 的 [trax 库](https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29) 中实现，并由本模型论文的作者进一步开发。在处理非常长的输入序列的模型中，传统的位置 id 编码为每个位置\\(i, \ldots, n_s\\)存储一个大小为\\(d\\)（即 `config.hidden_size`）的嵌入向量，其中\\(n_s\\)为 `config.max_embedding_size`。这意味着当序列长度为\\(n_s = 2^{19} \approx 0.5M\\)且 `config.hidden_size` 为\\(d = 2^{10} \approx 1000\\)时，将会得到一个位置编码矩阵：

这个矩阵本身就有超过 5 亿个参数需要存储。而轴向位置编码将\\(X_{i, j}\\)分解为两个矩阵：
$$X^{1}_{i,j}, \text{ where } i \in \left[1,\ldots, d^1\right] \text{ and } j \in \left[1,\ldots, n_s^1\right]$$
和
$$X^{2}_{i,j}, \text{ where } i \in \left[1,\ldots, d^2\right] \text{ and } j \in \left[1,\ldots, n_s^2\right]$$
其中：
$$d = d^1 + d^2 \text{ and } n_s = n_s^1 \times n_s^2 .$$

因此有以下关系：

$$X_{i,j} = \begin{cases}X^{1}_{i, k}, & \text{if }\ i < d^1 \text{ with } k = j \mod n_s^1 \\X^{2}_{i - d^1, l}, & \text{if } i \ge d^1 \text{ with } l = \lfloor\frac{j}{n_s^1}\rfloor\end{cases}$$

直观地说，位置嵌入向量\\(x_j \in \mathbb{R}^{d}\\)现在由两个因式分解的嵌入向量组成：\\(x^1_{k, l} + x^2_{l, k}\\)，其中\\(k\\)和\\(l\\)是 `config.max_embedding_size` 维\\(j\\)的分解。这种设计确保了每个位置嵌入向量\\(x_j\\)的唯一性。


再次以上述示例为例，轴向位置编码使用\\(d^1 = 2^9, d^2 = 2^9, n_s^1 = 2^9, n_s^2 = 2^{10}\\)可以将参数数量从 5 亿减少到大约 78 万个参数，这意味着减少了 85%的内存使用量。

在实践中，参数 `config.axial_pos_embds_dim` 设置为元组\\((d^1, d^2)\\)的形式，其和必须等于 `config.hidden_size`，参数 `config.axial_pos_shape` 设置为元组\\((n_s^1, n_s^2)\\)的形式，其乘积必须等于 `config.max_embedding_size`，在训练过程中必须与 `input_ids` 的 *序列长度* 相等。

## LSH 自注意力

在局部敏感哈希（LSH）自注意力中，键和查询投影权重是相同的。因此，键查询嵌入向量也是相同的。LSH 自注意力使用 [Practical and Optimal LSH for Angular Distance](https://arxiv.org/abs/1509.02897) 中提出的局部敏感哈希机制，将每个相同的键查询嵌入向量分配给可能的 `config.num_buckets` 个存储桶之一。其基本思想是，*余弦相似度* 越高的键查询嵌入向量，它们被分配到同一个存储桶的可能性就越大。

通过增加 `config.num_hashes` 或直接增加前向函数的 `num_hashes` 参数，可以提高 LSH 机制的准确性，以便 LSH 自注意力的输出更好地近似于 "正常" 全自注意力的输出。然后对存储桶进行排序，并将其分块为查询键嵌入向量块。

每个长度为 `config.lsh_chunk_length` 的块，查询嵌入向量都会关注其键向量（它们与自身相连）以及 `config.lsh_num_chunks_before` 个之前的相邻块和 `config.lsh_num_chunks_after` 个之后的相邻块。
更多信息，请参阅 [原论文](https://arxiv.org/abs/2001.04451) 或这篇很棒的 [博客文章](https://www.pragmatic.ml/reformer-deep-dive/)。

请注意，`config.num_buckets` 也可以分解为一组\\((n_{\text{buckets}}^1, n_{\text{buckets}}^2)\\)。这样，将查询键嵌入向量分配给\\((1,\ldots, n_{\text{buckets}})\\)之一，它们将被分配给\\((1-1,\ldots, n_{\text{buckets}}^1-1, \ldots,1-n_{\text{buckets}}^2, \ldots, n_{\text{buckets}}^1-n_{\text{buckets}}^2)\\)之一。这对于非常长的序列非常重要以节省内存。

在从头开始训练模型时，建议将 `config.num_buckets=None`，这样可以根据序列长度实时计算出 `num_buckets` 的良好值。然后将自动保存该值在配置文件中，并在推理时重用。

使用 LSH 自注意力，查询-键矩阵乘法操作的内存和时间复杂度可以从
\\(\mathcal{O}(n_s \times n_s)\\)降低到\\(\mathcal{O}(n_s \times \log(n_s))\\)，这通常是 Transformer 模型中的内存和时间瓶颈，其中\\(n_s\\)是序列长度。

## 本地自注意力
本地自注意力本质上是一个“普通”的自注意力层，具有键、查询和值的投影，但是它是分块的，每个长度为 `config.local_chunk_length` 的块中，查询嵌入向量只关注其块内的键嵌入向量，并关注 `config.local_num_chunks_before` 个之前的相邻块和 `config.local_num_chunks_after` 个之后的相邻块。

使用本地自注意力，查询-键矩阵乘法操作的内存和时间复杂度可以从\\(\mathcal{O}(n_s \times n_s)\\)降低到\\(\mathcal{O}(n_s \times \log(n_s))\\)，这通常是 Transformer 模型中的内存和时间瓶颈，其中\\(n_s\\)是序列长度。

## 训练

在训练过程中，必须确保将序列长度设置为可以被 `config.lsh_chunk_length` 和 `config.local_chunk_length` 的最小公倍数整除，并且 Axial 位置编码的参数设置正确，如上所述。Reformer 非常高效，因此模型可以 Positional Encodings are correctly set as described above. Reformer is very memory efficient so that the model can
轻松地在长达 64000 个标记的序列上进行训练。
对于训练，应按以下方式使用 [`ReformerModelWithLMHead`]：

```python
input_ids = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")
loss = model(input_ids, labels=input_ids)[0]
```

## 文档资源

- [文本分类任务指南](../tasks/sequence_classification)
- [问答任务指南](../tasks/question_answering)- [因果语言建模任务指南](../tasks/language_modeling)
- [掩码语言建模任务指南](../tasks/masked_language_modeling)

## ReformerConfig

[[autodoc]] ReformerConfig

## ReformerTokenizer

[[autodoc]] ReformerTokenizer
    - save_vocabulary

## ReformerTokenizerFast

[[autodoc]] ReformerTokenizerFast

## ReformerModel

[[autodoc]] ReformerModel
    - forward

## ReformerModelWithLMHead

[[autodoc]] ReformerModelWithLMHead
    - forward

## ReformerForMaskedLM

[[autodoc]] ReformerForMaskedLM
    - forward

## ReformerForSequenceClassification

[[autodoc]] ReformerForSequenceClassification
    - forward

## ReformerForQuestionAnswering

[[autodoc]] ReformerForQuestionAnswering
    - forward
