<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”提供的，不附带任何明示或暗示的担保或条件。请查看许可证以了解
⚠️请注意，此文件采用 Markdown 格式，但包含了特定的语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确渲染。
-->

# 如何将模型添加到🤗 Transformers？
🤗 Transformers 库通常能够通过社区贡献者提供新的模型。但这可能是一个具有挑战性的项目，需要深入了解🤗 Transformers 库和要实现的模型。在 Hugging Face，我们试图授权更多社区成员积极添加模型，并为此编写了本指南，以指导您完成添加 PyTorch 模型的过程（确保您已安装 [PyTorch](https://pytorch.org/get-started/locally/)）。
<Tip>
如果您有兴趣实施 TensorFlow 模型，请查看 [如何将🤗 Transformers 模型转换为 TensorFlow](add_tensorflow_model) 指南！
</Tip>

在此过程中，您将：
- 了解开源最佳实践的见解- 了解最受欢迎的深度学习库背后的设计原则- 学习如何高效测试大型模型- 学习如何集成 Python 工具（如 `black`、`ruff` 和 `make fix-copies`），以确保代码整洁和可读性
🤗团队成员将在整个过程中为您提供帮助，您将永远不会感到孤单。🤗❤️

要开始，请为您想要在🤗 Transformers 中看到的模型打开一个 [New model addition](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&template=new-model-addition.yml) 问题。如果您对贡献特定模型没有特别要求，可以按 [New model 标签](https://github.com/huggingface/transformers/labels/New%20model) 进行过滤，查看是否有任何未认领的模型请求并开始处理。
一旦您打开了新模型请求，第一步是熟悉🤗 Transformers（如果您还不熟悉）！

## 🤗 Transformers 的概述

首先，您应该对🤗 Transformers 有一个总体概述。🤗 Transformers 是一个非常主观的库，所以有可能您不同意库的某些哲学或设计选择。然而，根据我们的经验，我们发现库的基本设计选择和哲学对于有效扩展🤗 Transformers 并保持合理的维护成本至关重要。

更好地了解该库的一个很好的起点是阅读我们的 [代码哲学文档](philosophy)。由于我们的工作方式，有一些选择我们试图应用于所有模型：发现库的基本设计选择和理念对于有效地扩展🤗至关重要。由于我们的工作方式，我们尝试将一些选择应用于所有模型：

- 通常更喜欢组合而不是抽象。
- 复制代码并不总是坏事，如果它能大大提高模型的可读性或可访问性

- 模型文件尽可能自包含，这样当您阅读特定模型的代码时，理想情况下只需要查看相应的 `modeling_....py` 文件。

在我们看来，库的代码不仅仅是提供产品的手段，例如使用 BERT 进行推理的能力，而且也是我们要改进的产品本身。因此，当添加模型时，用户不仅是将使用您的模型的人，还有所有将阅读、尝试理解并可能调整您的代码的人。

考虑到这一点，让我们更深入地了解一下通用库设计。

### 模型概述

要成功添加模型，重要的是要了解您的模型与其配置 [`PreTrainedModel`] 和 [`PretrainedConfig`] 之间的交互。为了示范目的，我们将 [`PreTrainedModel`], and [`PretrainedConfig`].
要添加到🤗 Transformers 的模型称为 `BrandNewBert`。
让我们来看一下：
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_overview.png"/>

正如您所看到的，在🤗 Transformers 中，我们确实使用了继承，但我们将抽象级别保持到最低限度。库中任何模型的抽象级别永远不会超过两个。`BrandNewBertModel` 继承自 `BrandNewBertPreTrainedModel`，后者又继承自 [`PreTrainedModel`]，就是这样。通常情况下，我们希望确保新模型仅依赖于 [`PreTrainedModel`]。自动为每个新模型提供的重要功能包括 [`~PreTrainedModel.from_pretrained`] 和 [`~PreTrainedModel.save_pretrained`]，用于序列化和反序列化。所有其他重要功能，如 `BrandNewBertModel.forward`，应完全在新的 `modeling_brand_new_bert.py` 脚本中定义。接下来，我们要确保具有特定头部层的模型（例如 `BrandNewBertForMaskedLM`）不继承自 `BrandNewBertModel`，而是将 `BrandNewBertModel` 作为组件，在其前向传递中进行调用，以保持抽象级别的低。每个新模型都需要一个配置类，称为 `BrandNewBertConfig`。这个配置始终作为 [`PreTrainedModel`] 中的一个属性存储，因此可以通过 `config` 属性访问所有继承自 `BrandNewBertPreTrainedModel` 的类。

```python
model = BrandNewBertModel.from_pretrained("brandy/brand_new_bert")
model.config  # model has access to its config
```

与模型类似，配置从 [`PretrainedConfig`] 继承基本的序列化和反序列化功能。

请注意，配置和模型始终以两种不同的格式进行序列化 - 模型序列化为 *pytorch_model.bin* 文件，配置序列化为 *config.json* 文件。调用 [`~PreTrainedModel.save_pretrained`] will automatically call
[`~PreTrainedModel.save_pretrained`] 将自动调用

### 代码风格
在编写新模型时，请记住 Transformers 是一个主观的库，我们对代码编写有一些自己的特点 :-)

1. 您的模型的前向传递应完全在建模文件中编写，同时与库中的其他模型完全独立。如果要重用另一个模型的代码块，请复制代码并在顶部添加 `# Copied from` 注释（参见 [此处](https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/roberta/modeling_roberta.py#L160)作为很好的示例）。

2. 代码应易于理解，即使对于非英语母语的人也是如此。这意味着您应选择具有描述性的变量名并避免使用缩写。 强烈不建议使用单个字母的变量名，除非它是 for 循环中的索引。
3. 更一般地说，我们更倾向于使用更长、更明确的代码，而不是过于简短和神秘的代码。
4. 避免在 PyTorch 中子类化 `nn.Sequential`，而是子类化 `nn.Module` 并编写前向传递函数，这样任何人都   可以通过添加打印语句或断点来快速调试您的代码。

5. 函数签名应进行类型注解。对于其余部分，良好的变量名称比类型注解更易读和   理解。

### 分词器 (Tokenizer)概述 tokenizers
还不太准备好 :-( 该部分即将添加！

## 逐步将模型添加到🤗 Transformers

每个人对于如何移植模型都有不同的喜好，因此查看其他贡献者如何将模型移植到 Hugging Face 可能非常有帮助。下面是关于如何移植模型的社区博客文章的列表：1. [移植 GPT2 模型](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28) by [Thomas](https://huggingface.co/thomwolf)
1. [Porting GPT2 Model](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28) by [Thomas](https://huggingface.co/thomwolf)
2. [移植 WMT19 MT 模型](https://huggingface.co/blog/porting-fsmt) by [Stas](https://huggingface.co/stas)

根据经验，我们可以告诉您在添加模型时要牢记的最重要的事情是：

- 不要重复造轮子！您将为新的🤗 Transformers模型添加的大部分代码已经存在于🤗 Transformers中的某个地方。花些时间寻找类似的，已经存在的模型和分词器 (Tokenizer)，您可以从中复制。  [grep](https://www.gnu.org/software/grep/) 和 [rg](https://github.com/BurntSushi/ripgrep) 是您的  朋友。请注意，您的模型的分词器 (Tokenizer)很可能基于一个模型实现，而您的模型的建模代码则基于另一个模型实现。例如，FSMT 的建模代码是基于 BART 的，而 FSMT 的分词器 (Tokenizer)代码  则基于 XLM。
- 这更多是一个工程挑战而不是科学挑战。您应该花更多的时间创建一个有效的调试环境，而不是试图理解论文中的所有理论方面。
- 当您遇到困难时，请寻求帮助！模型是🤗 Transformers的核心组件，因此我们在 Hugging Face 非常乐意帮助您每一步添加模型。如果您发现自己没有取得进展，请  不要犹豫，随时提问。  progress.

接下来，我们尝试为您提供一般的移植模型到🤗 Transformers的步骤。以下列表是添加模型所需完成的所有工作的摘要，您可以将其用作待办事项清单：

以下是要完成的所有工作摘要的总结列表：

☐ （可选）理解模型的理论方面 <br> 
☐ 准备🤗 Transformers开发环境 <br> 
☐ 设置原始存储库的调试环境 <br> 
☐ 创建能够成功运行原始存储库和检查点的 `forward()` 函数的脚本 <br> 
☐ 成功将模型框架添加到🤗 Transformers <br> 
☐ 成功将原始检查点转换为🤗 Transformers检查点 <br> ☐ 成功在🤗 Transformers中运行与原始检查点输出完全相同的 `forward()` 函数 <br> 
☐ 完成🤗 Transformers中的模型测试 <br> 
☐ 成功在🤗 Transformers中添加分词器 (Tokenizer) <br> 
☐ 运行端到端集成测试 <br> 
☐ 完成文档 <br> 
☐ 将模型权重上传到 Hub <br> 
☐ 提交拉取请求 <br> 
☐ （可选）添加演示笔记本

首先，我们通常建议您先对 `BrandNewBert` 的理论有一个很好的理解。但是，如果您更喜欢在工作中理解模型的理论方面，那么直接深入研究 `BrandNewBert` 的代码库也是可以的。如果您的工程技能优于理论技能，如果您在理解 `BrandNewBert` 的论文方面遇到困难，或者如果您只是更喜欢编程而不是阅读科学论文，那么这个选择可能更适合您。

### 1. （可选）BrandNewBert 的理论方面
如果存在 *BrandNewBert* 的论文，请花些时间阅读它。论文中可能有一些难以理解的部分。如果是这种情况，不要担心！目标不是深入理解论文，而是提取实现模型在🤗 Transformers中所需的必要信息。也就是说，您不必花太多时间在理论方面，而是更关注实践方面，即：

- *brand_new_bert* 是什么类型的模型？类似 BERT 的仅编码器模型？类似 GPT2 的仅解码器模型？类似 BART 的  编码器-解码器模型？如果对它们之间的差异不熟悉，请查看 [model_summary](model_summary)。

- *brand_new_bert* 的应用是什么？文本分类？文本生成？Seq2Seq 任务，例如  摘要？- *brand_new_bert* 与 BERT/GPT-2/BART 有何不同的新功能？- 已有的 [🤗 Transformers模型](https://huggingface.co/transformers/#contents) 中与 *brand_new_bert* 最  相似的是哪个？

- 使用了什么类型的分词器 (Tokenizer)？SentencePiece 分词器 (Tokenizer)？WordPiece 分词器 (Tokenizer)？与 BERT 或 BART 使用的分词器 (Tokenizer)相同吗？  similar to *brand_new_bert*?

在您对模型的架构有了很好的概述之后，您可能希望向 Hugging Face 团队提问您可能遇到的任何问题。这可能包括有关模型的架构、注意力层等的问题。我们将非常乐意为您提供帮助。  

### 2. 接下来，准备您的环境

1. 单击存储库页面上的'Fork'按钮，将 [存储库](https://github.com/huggingface/transformers) fork 到您的 GitHub 用户帐户下，这将在您的 GitHub 用户帐户下创建代码的副本。

2. 将您的 `transformers` fork 克隆到本地磁盘，并将基本存储库添加为远程存储库：

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

3. 设置开发环境，例如运行以下命令：
```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

根据您的操作系统以及由于 Transformers 的可选依赖项数量不断增加，您可能会遇到此命令失败的情况。如果是这种情况，请确保安装了您正在使用的深度学习框架（PyTorch、TensorFlow 和/或 Flax），然后执行以下操作：

```bash
pip install -e ".[quality]"
```

这对于大多数情况来说应该足够了。然后您可以返回到父目录
```bash
cd ..
```

4. 我们建议将 *brand_new_bert* 的 PyTorch 版本添加到 Transformers。要安装 PyTorch，请按照 https://pytorch.org/get-started/locally/上的说明进行操作。   instructions on https://pytorch.org/get-started/locally/.

**注意：** 您不需要安装 CUDA。使新模型在 CPU 上工作就足够了。

5. 要移植 *brand_new_bert*，您还需要访问其原始存储库：
```bash
git clone https://github.com/org_that_created_brand_new_bert_org/brand_new_bert.git
cd brand_new_bert
pip install -e .
```

现在，您已经设置了一个开发环境，可以将 *brand_new_bert* 移植到🤗 Transformers。

### 3.-4. 使用原始存储库运行预训练检查点
首先，您将在原始 *brand_new_bert* 存储库上工作。通常，原始实现非常“researchy”。这意味着文档可能不完善，代码可能很难理解。但这正是您重新实现 *brand_new_bert* 的动力所在。在 Hugging Face，我们的主要目标之一是 *让人们站在巨人的肩膀上*，这在这里非常好地体现出来，即采用现有的工作模型并重新编写，使其尽可能 **易于理解、用户友好且美观**。这是重新实现模型的首要动力。我们尝试将模型转换为🤗 Transformers，以便将复杂的新 NLP 技术变得 **向所有人开放**。

因此，您应该首先深入研究原始代码库。成功运行原始代码库中的官方预训练模型通常是 **最困难** 的一步。根据我们的经验，花一些时间熟悉原始代码库非常重要。您需要弄清楚以下几点：
- 如何找到预训练的权重？
- 如何将预训练的权重加载到相应的模型中？
- 如何独立于模型运行分词器 (Tokenizer)？
- 追踪一次前向传递，以了解哪些类和函数需要进行简单的前向传递。通常情况下，您只需要重新实现这些函数。
- 能够找到模型的重要组件：模型的类在哪里？是否有模型的子类，例如 EncoderModel、DecoderModel？自注意力层在哪里？是否有多个不同的注意力层，例如自注意力、交叉注意力等？
- 如何在存储库的原始环境中调试模型？是否需要添加打印语句？是否可以使用类似于 ipdb 的交互式调试器？或者是否应该使用高效的 IDE（如 PyCharm）来调试模型？

在开始移植过程之前，能够在原始代码库中 **高效地** 调试代码非常重要！

同时，请记住，您正在使用的是一个开源库，因此请不要犹豫在原始代码库中提交问题或拉取请求。这个代码库的维护者很可能非常高兴有人审查他们的代码！  

此时，您可以根据自己的喜好选择调试环境和策略，以调试原始模型。我们强烈建议您在开始工作时，不要设置昂贵的 GPU 环境，而是在 CPU 上进行操作，无论是深入研究原始代码库还是开始编写🤗 Transformers 的模型时都是如此。只有在模型已经成功移植到🤗 Transformers 之后，才应该验证模型在 GPU 上是否按预期工作。

一般来说，运行原始模型有两种可能的调试环境
- [Jupyter notebooks](https://jupyter.org/) / [google colab](https://colab.research.google.com/notebooks/intro.ipynb)- 本地 python 脚本。
Jupyter notebooks 的优点是它们允许逐个单元格执行，这有助于更好地将逻辑组件分开，并且具有更快的调试周期，因为可以存储中间结果。此外，与其他贡献者共享笔记本通常更容易，这在向 Hugging Face 团队寻求帮助时可能非常有帮助。如果您熟悉 Jupyter notebooks，我们强烈建议您使用它们。

Jupyter notebooks 的明显缺点是，如果您不习惯使用它们，您将不得不花一些时间适应新的编程环境，并且可能无法再使用您熟悉的调试工具，如 `ipdb`。

对于每个代码库，一个很好的第一步始终是加载一个 **小型** 的预训练检查点，并能够使用虚拟整数向量作为输入来复现单个前向传递。这样的脚本可能如下所示（伪代码）：


```python
model = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = [0, 4, 5, 2, 3, 7, 9]  # vector of input ids
original_output = model.predict(input_ids)
```

接下来，关于调试策略，通常有几种选择：
- 将原始模型分解为许多小的可测试组件，并对每个组件运行前向传递以进行验证  verification
- 仅将原始模型分解为原始的 *tokenizer* 和原始的 *model*，对它们进行前向传递，并使用中间的打印语句或断点进行验证

再次强调，选择哪种策略完全取决于您。根据原始代码库的不同，其中一种策略可能更有优势。

如果原始代码库允许您将模型分解为较小的子组件，例如如果原始代码库可以在即时模式下轻松运行，那么通常值得这样做。这样做有一些重要的优点：
- 在稍后将原始模型与🤗 Transformers 实现进行比较时，您可以自动验证每个组件是否与🤗 Transformers 实现的相应组件匹配，而不是依赖于通过打印语句进行视觉比较 
- 它可以帮助您将移植模型的大问题分解为只移植单个组件的小问题，从而更好地组织您的工作。
- 将模型分解为逻辑上有意义的组件，有助于您更好地了解模型的设计
- 在稍后的阶段，这些逐个组件的测试有助于确保在继续更改代码时不会发生退化

[Lysandre's](https://gist.github.com/LysandreJik/db4c948f6b4483960de5cbac598ad4ed) 针对 ELECTRA 的集成检查提供了一个很好的示例。
然而，如果原始代码库非常复杂，或者只允许在编译模式下运行中间组件，分解模型为可测试的子组件可能会耗费太多时间，甚至是不可能的。一个很好的例子是 [T5 的 MeshTensorFlow](https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow) 库，它非常复杂，没有简单的方法将模型分解为其子组件。对于这样的库，人们通常依靠验证打印语句。
无论您选择哪种策略，推荐的步骤通常是相同的，即应先从开始的层进行调试，最后再调试结束的层。

建议您按以下顺序检索输出，可以通过打印语句或子组件函数来实现：

1. 检索传递给模型的输入 ID
2. 检索词嵌入 
3. 检索第一个 Transformer 层的输入 
4. 检索第一个 Transformer 层的输出 
5. 检索后续 n-1 个 Transformer 层的输出 
6. 检索整个 BrandNewBert 模型的输出
输入 ID 应包含一个整数数组，例如 `input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]`
后续层的输出通常由多维浮点数组组成，可能如下所示：
```
[[
 [-0.1465, -0.6501,  0.1993,  ...,  0.1451,  0.3430,  0.6024],
 [-0.4417, -0.5920,  0.3450,  ..., -0.3062,  0.6182,  0.7132],
 [-0.5009, -0.7122,  0.4548,  ..., -0.3662,  0.6091,  0.7648],
 ...,
 [-0.5613, -0.6332,  0.4324,  ..., -0.3792,  0.7372,  0.9288],
 [-0.5416, -0.6345,  0.4180,  ..., -0.3564,  0.6992,  0.9191],
 [-0.5334, -0.6403,  0.4271,  ..., -0.3339,  0.6533,  0.8694]]],
```

我们期望每个添加到🤗 Transformers的模型都通过了一些集成测试，这意味着原始模型和🤗 Transformers中重新实现的版本必须在精度为0.001的范围内给出完全相同的输出！由于同一个模型在不同的库中编写可能会产生略微不同的输出，这是正常的，我们接受1e-3（0.001）的误差容限。如果模型几乎给出了相同的输出，这还不够，它们必须几乎完全相同。因此，您肯定会多次将🤗 Transformers版本的中间输出与原始*brand_new_bert*实现的中间输出进行比较，在这种情况下，原始存储库的**高效**调试环境非常重要。下面是一些建议，可让您的调试环境尽可能高效：

- 找到调试中间结果的最佳方式。原始存储库是使用PyTorch编写的吗？那么您可能需要花时间编写一个较长的脚本，将原始模型分解为较小的子组件，以检索中间值。原始存储库是使用TensorFlow 1编写的吗？那么您可能需要依赖于像 [tf.print](https://www.tensorflow.org/api_docs/python/tf/print) 这样的TensorFlow打印操作来输出中间值。原始存储库是使用Jax编写的吗？在运行前向传递时，请确保模型**未经过即时编译**，例如，请参考[此链接](https://github.com/google/jax/issues/196)。
- 使用尽可能小的预训练检查点。检查点越小，您的调试周期就越快。如果您的预训练模型太大，以至于前向传递需要超过10秒的时间，那么这是低效的。如果只有非常大的检查点可用，更有意义的做法可能是在新环境中创建一个具有随机初始化权重的虚拟模型，并保存这些权重以与🤗 Transformers版本的模型进行比较。
- 确保您正在使用在原始存储库中最简单的调用前向传递的方法。理想情况下，您希望找到在原始存储库中**仅**调用单个前向传递的函数，即通常称为 `predict`、`evaluate`、`forward` 或 `__call__` 的函数。您不希望调试多次调用 `forward` 的函数，例如用于生成文本的函数，如 `autoregressive_sample`、`generate`。
- 尝试将分词与模型的*forward*传递分离如果原始存储库显示了需要输入字符串的示例，请尝试找出在前向传递的哪个位置将字符串输入更改为输入ID，并从该点开始。这可能意味着您可能需要自己编写一个小脚本或更改原始代码，以便直接输入ID而不是输入字符串。
确保您的调试设置中的模型不处于训练模式，这通常会导致模型由于模型中的多个dropout层而产生随机输出。确保调试环境中的前向传递是确定性的，以便不使用dropout层。或者，如果新旧实现位于相同的框架中，可以使用 transformers.utils.set_seed。
下面的部分将为您提供有关如何为brand_new_bert执行上述操作的更具体细节和提示。

5.-14. 将BrandNewBert移植到🤗 Transformers
接下来，您终于可以开始向🤗 Transformers添加新代码了。进入您的🤗 Transformers fork 的克隆目录：
```bash
cd transformers
```

在特殊情况下，如果要添加的模型的体系结构与现有模型的体系结构完全匹配只需按照 [此部分](#write-a-conversion-script) 中描述的方式添加转换脚本即可。在这种情况下，您可以直接重用已经存在模型的整个体系结构。

否则，让我们开始生成一个新模型。您有两个选择：

- `transformers-cli add-new-model-like`，以添加类似于现有模型的新模型

- `transformers-cli add-new-model`，以添加来自我们的模板的新模型（将类似于 BERT 或 Bart，具体取决于所选模型的类型）

在这两种情况下，您将被要求填写有关您的模型的基本信息的问卷调查。第二个命令需要安装 `cookiecutter`，您可以在此处找到更多信息 [here](https://github.com/huggingface/transformers/tree/main/templates/adding_a_new_model)。

**在主要的 huggingface/transformers 仓库上打开一个拉取请求（Pull Request）**

在开始适应自动生成的代码之前，现在是时候在🤗 Transformers 上打开一个“正在进行中（WIP）”的拉取请求，例如“[WIP] Add *brand_new_bert*”，以便您和 Hugging Face 团队可以并行地将模型集成到🤗 Transformers 中。

您应该执行以下操作：

1. 从您的主分支创建一个具有描述性名称的分支
```bash
git checkout -b add_brand_new_bert
```

2. 提交自动生成的代码：
```bash
git add .
git commit
```

3. 拉取并将其与当前主分支合并：
```bash
git fetch upstream
git rebase upstream/main
```

4. 将更改推送到您的账户中：
```bash
git push -u origin a-descriptive-name-for-my-changes
```

5. 一旦您满意，转到 GitHub 上的您的分支的网页。点击“拉取请求”。确保将 Hugging Face 团队的成员的 GitHub 账号添加为   审阅人员，以便 Hugging Face 团队能够收到有关未来更改的通知。   

6. 单击 GitHub 拉取请求网页右侧的“转换为草稿（Convert to draft）”按钮，将 PR 更改为草稿状态。
接下来，每当您取得一些进展时，不要忘记提交您的工作并将其推送到您的账户中，以便在拉取请求中显示。此外，您应该确保定期使用以下命令将您的工作与当前主分支更新：

```bash
git fetch upstream
git merge upstream/main
```

一般来说，您对模型或实现的所有问题都应在您的 PR 中提出，并在 PR 中进行讨论/解决。这样，Hugging Face 团队将始终在您提交新代码或您有问题时收到通知。指向您添加的代码对于 Hugging Face 团队高效理解您的问题或疑问通常非常有帮助。

为此，您可以转到“Files changed”选项卡，其中显示了您的所有更改，转到您想要提问的相关行，并单击“+”符号添加评论。每当问题或问题得到解决时，您可以单击创建的评论的 “Resolve” 按钮。

同样，Hugging Face 团队在审查您的代码时会提出评论。我们建议在 GitHub 的 PR 上提出大多数问题。对于一些非常常规且对公众不太有用的问题，请随时通过 Slack 或电子邮件与 Hugging Face 团队联系。

**5. 为 brand_new_bert 调整生成的模型代码**
首先，我们将仅关注模型本身，不关心分词器 (Tokenizer)。所有相关的代码应该在生成的文件 `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` 和 `src/transformers/models/brand_new_bert/configuration_brand_new_bert.py`。
现在你终于可以开始编码了：）。生成的代码在 `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` 中将具有与 BERT 相同的架构，如果它是一个仅编码器模型，或者是一个 BART 模型，如果它是一个编码器-解码器模型。此时，你应该回想起你在开始时关于模型的理论方面学到的内容：*该模型与 BERT 或 BART 有何不同？* "。实现这些变化通常意味着改变 *self-attention* 层，规范化层的顺序等等。再次说明，查看 Transformers 中已有模型的类似架构通常是很有帮助的，以便更好地了解如何实现你的模型。

**注意**，此时你不必非常确定你的代码是否完全正确或干净。相反，建议将原始代码的第一个“不干净的”复制粘贴版本添加到
`src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`，直到你感觉已经添加了所有必要的代码。根据我们的经验，通过转换脚本快速添加所需代码的第一个版本，然后进行迭代改进/修正代码要更加高效。此时唯一需要确保的是你可以实例化 🤗 Transformers 实现的 *brand_new_bert*，即以下命令应该能够正常运行：

```python
from transformers import BrandNewBertModel, BrandNewBertConfig

model = BrandNewBertModel(BrandNewBertConfig())
```

上述命令将根据 `BrandNewBertConfig()` 中定义的默认参数创建一个模型，使用随机权重，从而确保所有组件的 `init()` 方法正常工作。

请注意，所有随机初始化应在  `BrandnewBertPreTrainedModel`类的 `_init_weights`方法中进行。该方法应根据配置的变量初始化所有叶子模块。以下是具有BERT `_init_weights`方法的示例：

```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

如果您需要为某些模块进行特殊初始化，您可以添加一些自定义方案。例如，在 `Wav2Vec2ForPreTraining` 中，最后两个线性层需要使用常规的 PyTorch nn.Linear 初始化，而其他所有层都应使用上述的初始化方法。代码如下所示：
```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstnace(module, Wav2Vec2ForPreTraining):
        module.project_hid.reset_parameters()
        module.project_q.reset_parameters()
        module.project_hid._is_hf_initialized = True
        module.project_q._is_hf_initialized = True
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
```

`_is_hf_initialized` 标志在内部用于确保我们只初始化一次子模块。通过为 `module.project_q` 和 `module.project_hid` 设置为 `True`，我们确保我们做的自定义初始化后续不会被覆盖，`_init_weights` 函数不会被应用到它们上。

**6. Write a conversion script**

请继续翻译：接下来，您应该编写一个转换脚本，用于将您在原始存储库中用于调试 *brand_new_bert* 的检查点转换为与您刚创建的🤗 Transformers实现的 *brand_new_bert* 兼容的检查点。不建议从头开始编写转换脚本，而是浏览🤗 Transformers中已经存在的转换脚本，找到一个已经用于转换类似于在相同框架中编写的 *brand_new_bert* 的模型的脚本。通常情况下，复制一个已经存在的转换脚本并对其进行轻微的调整以适应您的用例就足够了。请随时向Hugging Face团队询问是否有一个类似的现有转换脚本适用于您的模型。

- 如果您正在将模型从TensorFlow转换到PyTorch，可以从BERT的转换脚本 [here](https://github.com/huggingface/transformers/blob/7acfa95afb8194f8f9c1f4d2c6028224dbed35a2/src/transformers/models/bert/modeling_bert.py#L91) 开始
- 如果您正在将模型从PyTorch转换到PyTorch，可以从BART的转换脚本 [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py) 开始

接下来，我们将快速解释PyTorch模型如何存储层权重和定义层名称。在PyTorch中，层的名称由您为该层给定的类属性的名称定义。让我们以PyTorch中的 `SimpleModel` 为例定义一个虚拟模型：

```python
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 10)
        self.intermediate = nn.Linear(10, 10)
        self.layer_norm = nn.LayerNorm(10)
```

现在，我们可以创建该模型定义的实例，它将使用随机权重填充 `dense`、`intermediate`、`layer_norm` 层。我们可以打印模型以查看其架构：

```python
model = SimpleModel()

print(model)
```

这将输出以下内容：

```
SimpleModel(
  (dense): Linear(in_features=10, out_features=10, bias=True)
  (intermediate): Linear(in_features=10, out_features=10, bias=True)
  (layer_norm): LayerNorm((10,), eps=1e-05, elementwise_affine=True)
)
```

我们可以看到，PyTorch中的层名称由类属性的名称定义。您可以打印特定层的权重值：


```python
print(model.dense.weight.data)
```

to see that the weights were randomly initialized

```
tensor([[-0.0818,  0.2207, -0.0749, -0.0030,  0.0045, -0.1569, -0.1598,  0.0212,
         -0.2077,  0.2157],
        [ 0.1044,  0.0201,  0.0990,  0.2482,  0.3116,  0.2509,  0.2866, -0.2190,
          0.2166, -0.0212],
        [-0.2000,  0.1107, -0.1999, -0.3119,  0.1559,  0.0993,  0.1776, -0.1950,
         -0.1023, -0.0447],
        [-0.0888, -0.1092,  0.2281,  0.0336,  0.1817, -0.0115,  0.2096,  0.1415,
         -0.1876, -0.2467],
        [ 0.2208, -0.2352, -0.1426, -0.2636, -0.2889, -0.2061, -0.2849, -0.0465,
          0.2577,  0.0402],
        [ 0.1502,  0.2465,  0.2566,  0.0693,  0.2352, -0.0530,  0.1859, -0.0604,
          0.2132,  0.1680],
        [ 0.1733, -0.2407, -0.1721,  0.1484,  0.0358, -0.0633, -0.0721, -0.0090,
          0.2707, -0.2509],
        [-0.1173,  0.1561,  0.2945,  0.0595, -0.1996,  0.2988, -0.0802,  0.0407,
          0.1829, -0.1568],
        [-0.1164, -0.2228, -0.0403,  0.0428,  0.1339,  0.0047,  0.1967,  0.2923,
          0.0333, -0.0536],
        [-0.1492, -0.1616,  0.1057,  0.1950, -0.2807, -0.2710, -0.1586,  0.0739,
          0.2220,  0.2358]]).
```

在转换脚本中，您应该使用检查点中相应层的精确权重填充这些随机初始化的权重。例如，

```python
# retrieve matching layer weights, e.g. by
# recursive algorithm
layer_name = "dense"
pretrained_weight = array_of_dense_layer

model_pointer = getattr(model, "dense")

model_pointer.weight.data = torch.from_numpy(pretrained_weight)
```


在执行此操作时，您必须验证PyTorch模型的每个随机初始化权重与其对应的预训练检查点权重在形状和名称上完全匹配。为了做到这一点，必须为形状添加断言语句，并打印出检查点权重的名称。例如，您应该添加类似以下的语句：

```python
assert (
    model_pointer.weight.shape == pretrained_weight.shape
), f"Pointer shape of random weight {model_pointer.shape} and array shape of checkpoint weight {pretrained_weight.shape} mismatched"
```

此外，您还应该打印出两个权重的名称，以确保它们匹配，例如：

```python
logger.info(f"Initialize PyTorch weight {layer_name} from {pretrained_weight.name}")
```

如果形状或名称不匹配，那么您可能将错误的检查点权重分配给了🤗 Transformers实现的随机初始化层。

形状不匹配很可能是由于在 `BrandNewBertConfig()` 中错误设置了配置参数，这些参数与要转换的检查点使用的参数不完全匹配。然而，也有可能是由于PyTorch中的层实现在之前需要对权重进行转置。

最后，您还应该检查是否**初始化了所有**必需的权重，并打印出所有未用于初始化的检查点权重，以确保模型转换正确。如果转换失败，并且出现错误的形状声明或错误的名称分配，这是完全正常的。这很可能是因为您在 `BrandNewBertConfig()` 中使用了错误的参数，🤗 Transformers实现中存在错误的架构，🤗 Transformers实现的某个组件的 `init()` 函数存在错误，或者需要对其中一个检查点权重进行转置。

直到将检查点的所有权重正确加载到Transformers模型中，应重复执行此步骤。在将检查点正确加载到🤗 Transformers实现后，您可以将模型保存在您选择的文件夹 `/path/to/converted/checkpoint/folder` 下，该文件夹应包含一个 `pytorch_model.bin` 文件和一个 `config.json` 文件：

```python
model.save_pretrained("/path/to/converted/checkpoint/folder")
```

**7. Implement the forward pass**

成功将预训练权重正确加载到🤗 Transformers实现后，现在您应该确保前向传递被正确实现。在 [熟悉原始存储库](#34-run-a-pretrained-checkpoint-using-the-original-repository) 中，您已经创建了一个使用原始存储库运行模型前向传递的脚本。现在，您应该编写一个类似的脚本，但是使用🤗 Transformers实现代替原始实现。脚本应如下所示：

```python
model = BrandNewBertModel.from_pretrained("/path/to/converted/checkpoint/folder")
input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]
output = model(input_ids).last_hidden_states
```

很可能🤗 Transformers实现和原始模型实现在第一次运行时无法给出完全相同的输出，或者前向传递会引发错误。不要失望 - 这是正常的！首先，您应该确保前向传递不会引发任何错误。经常出现的情况是使用了错误的维度，导致*维度不匹配*错误，或者使用了错误的数据类型对象，例如使用 `torch.long` 而不是 `torch.float32`。如果您无法解决某些错误，请随时向Hugging Face团队寻求帮助。

确保🤗 Transformers实现正确工作的最后一部分是确保输出的精度达到`1e-3`。首先，您应该确保输出的形状是相同的，即脚本的🤗 Transformers实现和原始实现的输出形状应该是相同的。接下来，您应该确保输出的值也是相同的。这是添加新模型最困难的部分之一。输出不相同的常见错误包括：

- 某些层没有添加，即未添加*激活*层，或者遗忘了残差连接
- 单词嵌入矩阵未绑定
- 使用了错误的位置嵌入，因为原始实现使用了偏移量
- 在前向传递过程中应用了dropout。要解决这个问题，请确保 *model.training 为 False*，并且在前向传递过程中没有误激活任何dropout层，即将 *self.training* 传递给[PyTorch的functional.dropout](https://pytorch.org/docs/stable/nn.functional.html?highlight=dropout#torch.nn.functional.dropout)

通常修复问题的最佳方法是同时查看原始实现和🤗 Transformers实现的前向传递，并检查是否存在任何差异。理想情况下，您应该调试/打印出前向传递的中间输出，以找到🤗 Transformers实现与原始实现产生不同输出的确切位置。首先，请确保两个脚本中硬编码的 `input_ids` 是相同的。接下来，验证 `input_ids` 的第一个转换的输出（通常是单词嵌入）是否相同。然后逐层向上工作，直到达到网络的最后一层。在某个点上，您将注意到两个实现之间的差异，这应该指向🤗 Transformers实现中的错误。根据我们的经验，一种简单有效的方法是在原始实现和🤗 Transformers实现中添加许多打印语句，分别在相同的网络位置，并逐步删除显示相同中间表示值的打印语句。

当您确信两个实现产生相同的输出

，并使用`torch.allclose(original_output, output, atol=1e-3)`验证输出时，您已经完成了最困难的部分！恭喜您 - 剩下的工作将会变得轻松愉快 😊。
**8. Adding all necessary model tests**

在这一步，您已经成功添加了一个新的模型。然而，很有可能模型还不完全符合所需的设计。为了确保实现与🤗 Transformers完全兼容，所有常见测试都应该通过。Cookiecutter应该自动为您的模型添加了一个测试文件，可能位于相同的目录下 `tests/models/brand_new_bert/test_modeling_brand_new_bert.py`。运行此测试文件以验证所有常见测试是否通过：

```bash
pytest tests/models/brand_new_bert/test_modeling_brand_new_bert.py
```

在修复了所有常见测试之后，现在至关重要的是确保您所做的所有工作都经过了良好的测试，以便：

- a) 社区可以通过查看 *brand_new_bert* 的特定测试轻松理解您的工作
- b) 对您的模型进行的未来更改不会破坏模型的任何重要功能。

首先，应该添加集成测试。这些集成测试基本上与您之前用于将模型移植到🤗 Transformers的调试脚本相同。Cookiecutter已经添加了这些模型测试的模板，称为 `BrandNewBertModelIntegrationTests`，您只需要填写它。为了确保这些测试通过，运行以下命令：

```bash
RUN_SLOW=1 pytest -sv tests/models/brand_new_bert/test_modeling_brand_new_bert.py::BrandNewBertModelIntegrationTests
```

<Tip>

In case you are using Windows, you should replace `RUN_SLOW=1` with `SET RUN_SLOW=1`

</Tip>

其次，应该在`BrandNewBertModelTester`/`BrandNewBertModelTest`下单独对*brand_new_bert*的特殊功能进行额外的测试。这一部分经常被忽略，但在两个方面非常有用：

- 它帮助将您在模型添加过程中获得的知识传递给社区，展示*brand_new_bert*的特殊功能应该如何工作。
- 未来的贡献者可以通过运行这些特殊测试来快速测试对模型的更改。

**9. 实现分词器 (Tokenizer)**

接下来，我们应该添加*brand_new_bert*的分词器 (Tokenizer)。通常情况下，分词器 (Tokenizer)与🤗 Transformers中已经存在的分词器 (Tokenizer)等效或非常相似。

找到/提取原始的分词器 (Tokenizer)文件并成功加载该文件到🤗 Transformers的分词器 (Tokenizer)实现中非常重要。

为了确保分词器 (Tokenizer)正常工作，建议首先在原始存储库中创建一个输入字符串并返回`input_ids`的脚本。伪代码示例如下：
```python
input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
model = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = model.tokenize(input_str)
```

您可能需要再次深入查看原始存储库，以找到正确的分词器 (Tokenizer)函数，甚至可能需要对原始存储库的克隆进行更改，以仅输出`input_ids`。编写了一个使用原始存储库的功能性分词脚本后，应该创建一个类似的🤗 Transformers脚本。它应该类似于以下内容：

```python
from transformers import BrandNewBertTokenizer

input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."

tokenizer = BrandNewBertTokenizer.from_pretrained("/path/to/tokenizer/folder/")

input_ids = tokenizer(input_str).input_ids
```
当两个`input_ids`产生相同的值时，作为最后一步，还应添加一个分词器 (Tokenizer)测试文件。

与*brand_new_bert*的建模测试文件类似，*brand_new_bert*的分词测试文件应包含一些硬编码的集成测试。

**10. 运行端到端集成测试**

在添加了分词器 (Tokenizer)之后，您还应添加一些端到端的集成测试，使用模型和分词器 (Tokenizer)在🤗 Transformers的 `tests/models/brand_new_bert/test_modeling_brand_new_bert.py` 中进行。这样的测试应该在一个有意义的文本对样本中展示🤗 Transformers实现按预期工作。有意义的文本对样本可以包括源到目标的翻译对、文章到摘要的对、问题到答案的对等。如果没有任何移植检查点在下游任务上进行了微调，只需仅依赖模型测试即可。最后，为了确保模型完全可用，建议您在GPU上运行所有测试。有可能您忘记为模型的内部张量添加一些`.to(self.device)`语句，在这种测试中会显示错误。如果您无法访问GPU，Hugging Face团队可以为您运行这些测试。

**11. 添加文档字符串**

现在，*brand_new_bert*所需的所有功能都已添加 - 您即将完成！唯一剩下的就是添加一个完善的文档字符串和文档页面。Cookiecutter应该已经添加了一个模板文件，称为 `docs/source/model_doc/brand_new_bert.md`，您需要填写该文件。用户在使用您的模型之前通常会首先查看此页面。因此，文档必须易于理解且简明扼要。为了帮助社区，向文档中添加一些*提示*来显示如何使用模型是非常有用的。如果有关文档字符串的问题，请随时联系Hugging Face团队。

接下来，请确保添加到 `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` 的文档字符串是正确的，并包含所有必要的输入和输出。我们有一份关于撰写文档和我们的文档字符串格式的详细指南 [here](writing-documentation)。始终要记住，与🤗 Transformers中的代码一样，文档应至少要像代码一样仔细对待，因为文档通常是社区与模型的第一个接触点。

**代码重构**

太棒了，您现在已经添加了*brand_new_bert*所需的所有代码。此时，您应该通过运行以下命令来纠正一些潜在的不正确代码风格：

```bash
make style
```

并验证您的代码风格是否通过质量检查：

```bash
make quality
```

在🤗 Transformers中还可能存在

一些其他非常严格的设计测试，可能仍然失败，这会在您的拉取请求的测试中显示出来。这通常是由于文档字符串中缺少某些信息或命名不正确。如果您遇到困难，Hugging Face团队肯定会帮助您。

最后，在确保代码正确工作之后，重新整理代码总是一个好主意。在所有测试都通过的情况下，现在是时候回顾所添加的代码并进行一些重构了。

恭喜您完成了编码部分！🎉 您真棒！😎
**12. 将模型上传到模型中心**

在这最后一部分，您应该转换并上传所有检查点到模型中心，并为每个上传的模型检查点添加一个模型卡片。您可以通过阅读我们的[模型分享和上传页面](model_sharing)来熟悉模型中心的功能。在这里，您应该与Hugging Face团队合作，为每个检查点决定一个合适的名称，并获得所需的访问权限，以便将模型上传到*brand_new_bert*作者组织下。`transformers`中的所有模型都具有`push_to_hub`方法，这是将检查点快速有效地推送到中心的方法。下面是一个简短的示例：

```python
brand_new_bert.push_to_hub("brand_new_bert")
# Uncomment the following line to push to an organization.
# brand_new_bert.push_to_hub("<organization>/brand_new_bert")
```

值得花一些时间为每个检查点创建合适的模型卡片。模型卡片应突出显示此特定检查点的特点，例如：检查点在哪个数据集上进行了预训练/微调？该模型应该用于哪个下游任务？还应包含有关如何正确使用模型的一些代码。

**13.（可选）添加笔记本**

添加一个笔记本，详细展示了如何在*brand_new_bert*上进行推理和/或在下游任务上进行微调，这非常有帮助。这不是合并您的PR所必需的，但对社区非常有用。

**14. 提交您完成的PR**

您现在已经完成了编程工作，可以进入最后一步，将您的PR合并到主分支中。通常，此时Hugging Face团队应该已经帮助您了，但值得花些时间为您的完成的PR添加一个良好的描述，并在需要向审查人员指出某些设计选择时添加注释。

### 分享您的工作！！

现在，是时候从社区获得对您工作的认可了！完成一个模型添加是对Transformers和整个NLP社区的重大贡献。您的代码和移植的预训练模型肯定会被数百甚至上千的开发人员和研究人员使用。您应该为自己的工作感到自豪，并与社区分享您的成就。

**您又创建了一个对社区中的每个人都非常易于访问的模型！🤯**

