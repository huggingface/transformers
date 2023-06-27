<!--版权所有2021年HuggingFace团队保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可，除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解许可证下的特定语言规定和限制。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
请注意，此文件是 Markdown 格式的，但包含我们的文档构建器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档构建器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。请注意，此文件是 Markdown 格式的，但包含我们的文档构建器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。
-->
# ViLT
## 概述
ViLT 模型是由 Wonjae Kim，Bokyung Son 和 Ildoo Kim 在 [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) 中提出的。ViLT 将文本嵌入集成到 Vision Transformer（ViT）中，使其具有最小的设计用于视觉和语言预训练（VLP）。以下是来自论文的摘要：for Vision-and-Language Pre-training (VLP).

*视觉与语言预训练（VLP）在各种联合视觉与语言的下游任务上提高了性能。当前的 VLP 方法在很大程度上依赖于图像特征提取过程，其中大部分涉及区域监督（例如，物体检测）和卷积架构（例如，ResNet）。虽然文献中忽视了这一点，但我们认为这在（1）效率/速度方面是有问题的，因为仅提取输入特征比多模态交互步骤需要更多计算；以及（2）表达能力方面存在问题，因为它的上限取决于视觉嵌入器及其预定义的视觉词汇表的表达能力。在本文中，我们提出了一个简化的 VLP 模型，Vision-and-Language Transformer（ViLT），在视觉输入的处理上只是以与文本输入相同的无卷积方式简化处理。我们展示了 ViLT 比以前的 VLP 模型快上几十倍，同时具有竞争力或更好的下游任务性能。*
*Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks.
Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision
(e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we
find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more
computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive
power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model,
Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically
simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of
times faster than previous VLP models, yet with competitive or better downstream task performance.*

提示：
- 快速开始使用 ViLT 的最简单方法是查看 [示例笔记本](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViLT)（展示推理和对自定义数据的微调）。  (which showcase both inference and fine-tuning on custom data).
- ViLT 是一个同时接受 `pixel_values` 和 `input_ids` 作为输入的模型。可以使用 [`ViltProcessor`] 来为模型准备数据。  This processor wraps a feature extractor (for the image modality) and a tokenizer (for the language modality) into one.
- ViLT 使用各种尺寸的图像进行训练：作者将输入图像的较短边调整为 384，并将较长边限制在小于 640 的范围内，同时保持纵横比。为了批处理图像，作者使用了一个 `pixel_mask` 来指示哪些像素值是真实的，哪些是填充的。[`ViltProcessor`] 会自动为您创建这个。  under 640 while preserving the aspect ratio. To make batching of images possible, the authors use a `pixel_mask` that indicates
  which pixel values are real and which are padding. [`ViltProcessor`] automatically creates this for you.
- ViLT 的设计与标准的 Vision Transformer（ViT）非常相似。唯一的区别是该模型包含了用于语言模态的额外嵌入层。  additional embedding layers for the language modality.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vilt_architecture.jpg"alt="drawing" width="600"/> alt = "drawing" width = "600"/>

<small> ViLT 架构。取自 <a href="https://arxiv.org/abs/2102.03334"> 原始论文 </a>。</small>
此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可以在 [此处](https://github.com/dandelin/ViLT) 找到。

提示：
- 此模型的 PyTorch 版本仅适用于 torch 1.10 及更高版本。
## ViltConfig
[[autodoc]] ViltConfig
## ViltFeatureExtractor
[[autodoc]] ViltFeatureExtractor    - __call__
## ViltImageProcessor
[[autodoc]] ViltImageProcessor    - preprocess
## ViltProcessor
[[autodoc]] ViltProcessor    - __call__
## ViltModel
[[autodoc]] ViltModel    - forward
## ViltForMaskedLM
[[autodoc]] ViltForMaskedLM    - forward
## ViltForQuestionAnswering
[[autodoc]] ViltForQuestionAnswering    - forward
## ViltForImagesAndTextClassification
[[autodoc]] ViltForImagesAndTextClassification    - forward
## ViltForImageAndTextRetrieval
[[autodoc]] ViltForImageAndTextRetrieval    - forward
## ViltForTokenClassification
[[autodoc]] ViltForTokenClassification    - forward