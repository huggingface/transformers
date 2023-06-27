<!-- 版权所有2022年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在许可证的网址处获取许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，以“按原样”分发的软件根据许可证分发，不附带任何形式的担保或条件。请参阅许可证了解更多信息。
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们 doc-builder 的特定语法（类似于 MDX），可能无法在 Markdown 查看器中正确渲染。请注意。
-->
# 在多个 GPU 上进行高效推理
本文档包含有关如何在多个 GPU 上进行高效推理的信息。<Tip>
注意：多 GPU 设置可以使用单个 GPU 部分中描述的大多数策略。但是，您必须了解一些简单的技术，以便更好地利用。
</Tip>
## 通过 `BetterTransformer` 进行更快的推理
我们最近在文本、图像和音频模型的多 GPU 上集成了 `BetterTransformer`，以实现更快的推理。有关此集成的详细信息，请查看 [此处](https://huggingface.co/docs/optimum/bettertransformer/overview) 的文档。