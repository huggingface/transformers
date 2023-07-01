<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；您除非遵守许可证，否则不得使用此文件。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，不提供任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言规定和限制。特别说明：此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确显示。-->

# GLPN

<Tip>

这是一个最近引入的模型，因此 API 尚未经过大量测试。可能会出现一些错误或轻微的破坏性更改，以便在将来修复。如果您注意到任何奇怪的问题，请提交 [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title)。

</Tip>

## 概述

GLPN 模型是由 Doyeon Kim，Woonghyun Ga，Pyungwhan Ahn，Donggyu Joo，Sehwan Chun，Junmo Kim 在 [Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://arxiv.org/abs/2201.07436) 中提出的。GLPN 将 [SegFormer](segformer) 的分层混合 Transformer 与用于单眼深度估计的轻量级解码器相结合。

所提出的解码器显示出比先前提出的解码器更好的性能，并且计算复杂度明显降低。该论文的摘要如下：

*从单张图像中估计深度是一项重要任务，可应用于计算机视觉的各个领域，并且随着卷积神经网络的发展而迅速发展。在本文中，我们提出了一种新颖的结构和训练策略，用于进一步提高网络的预测精度。我们部署了分层 Transformer 编码器来捕捉和传递全局上下文，并设计了一个轻量级但功能强大的解码器，以在考虑局部连通性的同时生成估计的深度图。通过使用我们提出的选择性特征融合模块在多尺度局部特征和全局解码流之间构建连接路径，网络可以集成两种表示并恢复细节。此外，所提出的解码器显示出比先前提出的解码器更好的性能，并且计算复杂度明显降低。此外，我们通过利用深度估计中的一个重要观察结果改进了深度特定的数据增强方法，以增强模型。我们的网络在具有挑战性的深度数据集 NYU Depth V2 上实现了最先进的性能。我们进行了大量实验证明和验证了所提出方法的有效性。最后，我们的模型显示出比其他比较模型更好的泛化能力和鲁棒性。* 

*请注意，使用我们的分词器需要安装 sentencepiece 库，可以使用以下命令进行安装：* `pip install transformers[sentencepiece]` 或 `pip install sentencepiece`

示例用法：

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-356m")
>>> model = AutoModelForCausalLM.from_pretrained("AI-Sweden/gpt-sw3-356m")

>>> input_ids = tokenizer("Träd är fina för att", return_tensors="pt")["input_ids"]

>>> generated_token_ids = model.generate(inputs=input_ids, max_new_tokens=10, do_sample=True)[0]

>>> print(tokenizer.decode(generated_token_ids))
Träd är fina för att de är färgstarka. Men ibland är det fint
```





## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Causal language modeling task guide](../tasks/language_modeling)

## GPTSw3Tokenizer

[[autodoc]] GPTSw3Tokenizer
    - save_vocabulary