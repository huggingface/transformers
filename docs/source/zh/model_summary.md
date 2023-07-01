<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可; 除非符合许可证，否则不得使用此文件。您可以在以下位置获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的基础，不附带任何明示或暗示的担保或条件。有关许可证的详细信息特定语言下的权限和限制，请参阅许可证。
⚠️请注意，此文件为 Markdown 格式，但包含特定的语法，用于我们的 doc-builder（类似于 MDX），在您的 Markdown 查看器中可能无法正确地呈现。
-->

# Transformer 模型系列

自 2017 年推出以来，[原始 Transformer](https://arxiv.org/abs/1706.03762) 模型启发了许多新颖且令人兴奋的模型，超越了自然语言处理（NLP）任务。有用于 [预测蛋白质的折叠结构](https://huggingface.co/blog/deep-learning-with-proteins)，[训练猎豹奔跑](https://huggingface.co/blog/train-decision-transformers) 和 [时间序列预测](https://huggingface.co/blog/time-series-transformers) 的模型。由于 Transformer 变体如此之多，很容易错过更大的画面。所有这些模型的共同之处是它们都基于原始的 Transformer 架构。有些模型只使用编码器或解码器，而其他模型则同时使用两者。这为对 Transformer 家族中的模型进行分类和研究提供了有用的分类法，并有助于您理解之前未遇到的 Transformer。

如果您对原始 Transformer 模型不熟悉或需要复习，请查看 Hugging Face 课程的 [Transformer 是如何工作的](https://huggingface.co/course/chapter1/4?fw=pt) 章节。
<div align="center">
    < iframe width =" 560 " height =" 315 " src =" https://www.youtube.com/embed/H39Z_720T5s " title =" YouTube video player "
    frameborder = "0" allow =" accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
    picture-in-picture " allowfullscreen > </iframe>
</div>

## 计算机视觉

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FacQBpeFBVvrDUlzFlkejoz%2FModelscape-timeline%3Fnode-id%3D0%253A1%26t%3Dm0zJ7m2BQ9oe0WtO-1" allowfullscreen>
 </iframe> 

### 卷积网络（CNN）

长时间以来，卷积网络（CNN）一直是计算机视觉任务的主导范例，直到 [视觉 Transformer](https://arxiv.org/abs/2010.11929) 展示了其可扩展性和效率。即使如此，一些 CNN 的最佳特性，如平移不变性，是如此强大（特别是对于某些任务），以至于一些 Transformer 在其架构中使用了卷积。[ConvNeXt](model_doc/convnext) 颠倒了这种交换，并融入了 Transformer 的设计选择以现代化 CNN。例如，ConvNeXt 使用非重叠滑动窗口将图像分为补丁，并使用较大的内核来增加其全局感受野。ConvNeXt 还进行了几个层的设计选择，以提高内存效率和性能，因此它与 Transformer 竞争力相当！

### 编码器 [[cv-encoder]]

[视觉 Transformer（ViT）](model_doc/vit) 打开了计算机视觉任务的大门，无需卷积。ViT 使用标准的 Transformer 编码器，但它的主要突破在于如何处理图像。它将图像分割为固定大小的补丁，并使用它们创建嵌入，就像将句子分割为标记一样。ViT 利用 Transformer 的高效架构，在需要更少资源进行训练的同时，展示了与当时的 CNN 相竞争的结果。随后出现了其他处理密集视觉任务（如分割和检测）的视觉模型。

其中之一是 [Swin](model_doc/swin) Transformer。它从较小的补丁构建分层特征图（类似于 CNN 👀，与 ViT 不同），并在更深的层中将其与相邻的补丁合并。注意力仅在局部窗口内计算，并且窗口在注意力层之间移动，以创建有助于模型学习的连接。由于 Swin Transformer 可以生成分层特征图，因此它是密集预测任务（如分割和检测）的良好候选模型。[SegFormer](model_doc/segformer) 也使用 Transformer 编码器构建分层特征图，但它在顶部添加了一个简单的多层感知机（MLP）解码器，以组合所有特征图并进行预测。

其他视觉模型，如 BeIT 和 ViTMAE，从 BERT 的预训练目标中汲取灵感。[BeIT](model_doc/beit) 通过 *遮蔽图像建模（MIM）* 进行预训练; 图像补丁被随机遮蔽，并且图像也被划分为视觉标记。 BeIT 被训练以预测与遮蔽补丁对应的视觉标记。[ViTMAE](model_doc/vitmae) 具有类似的预训练目标，除了它必须预测像素而不是视觉标记。不寻常的是，75 ％的图像补丁被遮蔽！解码器通过遮蔽的标记和编码补丁来重建像素。在预训练之后，解码器被丢弃，编码器准备用于下游任务。

### 解码器 [[cv-decoder]]

仅具有解码器的视觉模型很少，因为大多数视觉模型依赖编码器来学习图像表示。但是对于像图像生成这样的用例，解码器是一个自然的选择，正如我们从 GPT-2 等文本生成模型中看到的那样。[ImageGPT](model_doc/imagegpt) 使用与 GPT-2 相同的架构，但是它预测图像中的下一个像素，而不是预测序列中的下一个标记。除了图像生成，ImageGPT 还可以进行图像分类的微调。

### 编码器-解码器 [[cv-encoder-decoder]]

视觉模型通常使用编码器（也称为骨干）在将重要的图像特征传递给 Transformer 解码器之前提取这些特征。[DETR](model_doc/detr) 具有预训练的骨干，但它还使用完整的 Transformer 编码器-解码器架构进行对象检测。编码器学习图像表示并将其与对象查询（每个对象查询是专注于图像中的区域或对象的学习嵌入）在解码器中结合。DETR 预测每个对象查询的边界框坐标和类别标签。

## 自然语言处理

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FUhbQAZDlpYW5XEpdFy6GoG%2Fnlp-model-timeline%3Fnode-id%3D0%253A1%26t%3D4mZMr4r1vDEYGJ50-1" allowfullscreen> </iframe>

### 编码器 [[nlp-encoder]]

[BERT](model_doc/bert) 是一个仅编码器的 Transformer，它随机屏蔽输入中的某些标记，以避免看到其他标记，从而“作弊”。预训练目标是基于上下文预测屏蔽的标记。这使得 BERT 能够充分利用左右上下文来帮助学习更深入、更丰富的输入表示。然而，BERT 的预训练策略仍有改进的空间。[RoBERTa](model_doc/roberta) 通过引入新的预训练配方改进了这一点，该配方包括在每个时期随机屏蔽标记，而不仅仅是在预处理期间屏蔽一次，并且取消了下一句预测目标。

提高性能的主要策略是增加模型大小。但是训练大型模型在计算上是昂贵的。一种降低计算成本的方法是使用较小的模型，如 [DistilBERT](model_doc/distilbert)。DistilBERT 使用 [知识蒸馏](https://arxiv.org/abs/1503.02531)——一种压缩技术——创建了一个较小版本的 BERT，同时几乎保留了其语言理解能力。

然而，大多数 Transformer 模型继续趋向于更多的参数，导致新的模型专注于提高训练效率。[ALBERT](model_doc/albert) 通过两种方式降低参数数量来减少内存消耗：将更大的词汇嵌入分为两个较小的矩阵，并允许层共享参数。[DeBERTa](model_doc/deberta) 添加了一个解耦注意机制，其中单词和其位置分别编码为两个向量。注意力从这些单独的向量计算，而不是从包含单词和位置嵌入的单个向量计算。[Longformer](model_doc/longformer) 也着重于使注意力更加高效，特别是用于处理具有较长序列长度的文档。它使用局部窗口注意力（仅计算固定窗口大小周围每个标记的注意力）和全局注意力（仅用于特定任务标记，如分类的 `[CLS]`）来创建一个稀疏的注意力矩阵，而不是完整的注意力矩阵。

### 解码器 [[nlp-decoder]]

[GPT-2](model_doc/gpt2) 是一个仅解码器的 Transformer，它预测序列中的下一个单词。它屏蔽右侧的标记，以防止模型通过向前查看来“作弊”。通过在大量文本上进行预训练，即使文本有时不准确或不真实，GPT-2 仍然非常擅长生成文本。但是 GPT-2 缺乏 BERT 预训练的双向上下文，这使得它在某些任务中不适用。[XLNET](model_doc/xlnet) 结合了 BERT 和 GPT-2 的预训练目标的优点，使用排列语言建模（PLM）目标进行训练，使其能够学习双向上下文。

在 GPT-2 之后，语言模型变得更大，现在被称为“大型语言模型（LLMs）”。如果在足够大的数据集上进行预训练，LLMs 可以展示零-或少样本学习。[GPT-J](model_doc/gptj) 是一个具有 60 亿参数的 LLM，训练数据达到 4000 亿标记。随后发布的是 [OPT](model_doc/opt)，它是一系列仅解码器的模型，其中最大的模型有 1750 亿参数，训练数据为 1800 亿标记。[BLOOM](model_doc/bloom) 在同一时间发布，该系列中最大的模型有 1760 亿参数，训练数据为 3660 亿标记，涵盖了 46 种语言和 13 种编程语言。

### 编码器-解码器 [[nlp-encoder-decoder]]

[BART](model_doc/bart) 保留了原始的 Transformer 架构，但使用“文本填充”损坏修改了预训练目标，其中一些文本片段被替换为单个“mask”标记。解码器预测未损坏的标记（未来的标记被屏蔽），并使用编码器的隐藏状态来帮助预测。[Pegasus](model_doc/pegasus) 类似于 BART，但 Pegasus 屏蔽整个句子而不是文本片段。除了屏蔽语言建模，Pegasus 还通过间隔句子生成（GSG）进行预训练。GSG 目标屏蔽了文档中重要的整个句子，并用“mask”标记替换它们。解码器必须从剩余的句子中生成输出。[T5](model_doc/t5) 是一个更独特的模型，它将所有 NLP 任务都转化为文本到文本的问题，并使用特定的前缀。例如，前缀“Summarize:”表示摘要任务。T5 通过监督训练（GLUE 和 SuperGLUE）和自监督训练（随机采样并删除 15% 的标记）进行预训练。

## 音频

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2Fvrchl8jDV9YwNVPWu2W0kK%2Fspeech-and-audio-model-timeline%3Fnode-id%3D0%253A1%26t%3DmM4H8pPMuK23rClL-1" allowfullscreen> </iframe>

### 编码器 [[audio-encoder]]

[Wav2Vec2](model_doc/wav2vec2) 使用 Transformer 编码器直接从原始音频波形中学习语音表示。它通过对一组错误的语音表示中确定真实语音表示的对比任务进行预训练。[HuBERT](model_doc/hubert) 类似于 Wav2Vec2，但其训练过程有所不同。目标标签由聚类步骤创建，其中将相似音频片段分配给一个聚类，该聚类成为隐藏单元。隐藏单元被映射到一个嵌入以进行预测。

### 编码器-解码器 [[audio-encoder-decoder]]

[Speech2Text](model_doc/speech_to_text) 是专为自动语音识别（ASR）和语音翻译而设计的语音模型。该模型接受从音频波形中提取的对数梅尔滤波器组特征，并进行预训练以生成转录或翻译。[Whisper](model_doc/whisper) 也是一个 ASR 模型，但与许多其他语音模型不同，它在大量✨有标签的✨音频转录数据上进行预训练，以实现零样本性能。该数据集的大部分还包含非英语语言，这意味着 Whisper 也可用于低资源语言。在结构上，Whisper 类似于 Speech2Text。音频信号被转换为由编码器编码的对数梅尔频谱图。解码器从编码器的隐藏状态和先前的标记中自回归地生成转录。


## 多模态

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FcX125FQHXJS2gxeICiY93p%2Fmultimodal%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen> </iframe>

### 编码器 [[mm-encoder]]

[VisualBERT](model_doc/visual_bert) 是一个用于视觉语言任务的多模态模型，它在 BERT 之后不久发布。它将 BERT 和预训练的目标检测系统结合起来，将图像特征提取为视觉嵌入，并与文本嵌入一起传递给 BERT。VisualBERT 根据未屏蔽的文本和视觉嵌入预测屏蔽的文本，并且还需要预测文本是否与图像对齐。当 ViT 发布时，[ViLT](model_doc/vilt) 采用了 ViT 的架构，因为这样更容易获得图像嵌入。图像嵌入与文本嵌入一起进行联合处理。从那时起，ViLT 通过图像文本匹配、屏蔽语言建模和整词屏蔽进行预训练。

[CLIP](model_doc/clip) 采用了一种不同的方法，对(`image`, `text`)进行了一对预测。通过在一个包含 4 亿个(`image`, `text`)对的数据集上联合训练图像编码器（ViT）和文本编码器（Transformer），最大化了(`image`, `text`)对的图像和文本嵌入之间的相似性。预训练后，您可以使用自然语言指示 CLIP 根据图像预测文本，或者反之亦然。[OWL-ViT](model_doc/owlvit) 在 CLIP 的基础上构建了零样本目标检测。在预训练后，添加了一个目标检测头部，对(`class`, `bounding box`)对进行集合预测。

### 编码器-解码器 [[mm-encoder-decoder]]

光学字符识别（OCR）是一项长期存在的文本识别任务，通常涉及多个组件来理解图像并生成文本。[TrOCR](model_doc/trocr) 使用端到端 Transformer 简化了这个过程。编码器是一个用于图像理解的 ViT 风格模型，并将图像处理为固定大小的 patch。解码器接受编码器的隐藏状态，并自回归地生成文本。[Donut](model_doc/donut) 是一个更通用的视觉文档理解模型，不依赖于基于 OCR 的方法。它使用 Swin Transformer 作为编码器，多语言 BART 作为解码器。Donut 通过预测基于图像和文本注释的下一个单词来预训练以读取文本。解码器根据提示生成一个令牌序列。提示由每个下游任务的特殊令牌表示。例如，文档解析有一个特殊的 `parsing` 令牌，它与编码器的隐藏状态结合在一起将文档解析为结构化的输出格式（JSON）。

## 强化学习

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FiB3Y6RvWYki7ZuKO6tNgZq%2Freinforcement-learning%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen> </iframe>

### 解码器 [[rl-decoder]]

决策与轨迹 Transformer 将状态、动作和奖励视为序列建模问题。[决策 Transformer](model_doc/decision_transformer) 根据回报、过去状态和动作生成一系列导致未来期望回报的动作。

在最后的 *K* 个时间步长中，这三种模态都被转换为令牌嵌入，并由类似 GPT 的模型进行处理，以预测未来的动作令牌。[轨迹 Transformer](model_doc/trajectory_transformer) 也对状态、动作和奖励进行标记，并使用 GPT 结构进行处理。与决策 Transformer 专注于奖励条件不同，轨迹 Transformer 使用波束搜索生成未来的动作。