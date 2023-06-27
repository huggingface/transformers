<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。具体语言下的权限和限制。
⚠️ 请注意，此文件以 Markdown 格式编写，但包含特定于我们的文档生成器（类似于 MDX）的语法，可能在您的 Markdown 查看器中无法正确渲染。
-->

# 处理器

处理器在 Transformers 库中可以表示两个不同的含义：
- 用于多模态模型（例如 [Wav2Vec2](../model_doc/wav2vec2)（语音和文本）或者 [CLIP](../model_doc/clip)（文本和图像））的  输入预处理对象
- 用于旧版本库中预处理 GLUE 或 SQUAD 数据的废弃对象

## 多模态处理器

任何多模态模型都需要一个用于编码或解码数据的对象，该对象将多个模态（文本、视觉和音频）组合在一起。这由称为处理器的对象处理，它将多个处理对象（如文本模态的标记器、视觉模态的图像处理器 (Image Processor)和音频模态的特征提取器）组合在一起。这些处理器继承以下实现保存和加载功能的基类：

[[autodoc]] ProcessorMixin

[[autodoc]] ProcessorMixin

## 废弃处理器

所有处理器都遵循相同的架构，即 [`~data.processors.utils.DataProcessor`]。处理器返回一个 [`~data.processors.utils.InputExample`] 列表。

这些 [`~data.processors.utils.InputExample`] 可以转换为 [`~data.processors.utils.InputFeatures`] 以输入模型。

[[autodoc]] data.processors.utils.DataProcessor
[[autodoc]] data.processors.utils.InputExample
[[autodoc]] data.processors.utils.InputFeatures
## GLUE

[通用语言理解评估（GLUE）](https://gluebenchmark.com/) 是一个评估模型在各种现有 NLU 任务上性能的基准。它与文章 [GLUE: A 多任务基准和分析平台自然语言理解](https://openreview.net/pdf?id=rJ4km2R5t7) 一起发布 multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id = rJ4km2R5t7)

本库提供了 10 个处理器，用于以下任务：MRPC、MNLI、MNLI（不匹配）、CoLA、SST2、STSB、QQP、QNLI、RTE 和 WNLI。QQP, QNLI, RTE and WNLI.

这些处理器如下：

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]

此外，可以使用以下方法从数据文件加载值并将其转换为 [`~data.processors.utils.InputExample`] 列表。
[[autodoc]] data.processors.glue.glue_convert_examples_to_features

## XNLI
[跨语言自然语言推理语料库（XNLI）](https://www.nyu.edu/projects/bowman/xnli/) 是一个评估跨语言文本表示质量的基准。XNLI 是基于 [*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/) 的众包数据集：文本对标注有 15 种不同语言的文本蕴含注释（包括英语等高资源语言和斯瓦希里等低资源语言）。

它与文章 [XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053) 一起发布

本库提供了加载 XNLI 数据的处理器：

- [`~data.processors.utils.XnliProcessor`]
请注意，由于测试集上提供了黄金标签，因此评估是在测试集上进行的。

使用这些处理器的示例在 [run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/text-classification/run_xnli.py) 脚本中给出。

## SQuAD

[斯坦福问答数据集（SQuAD）](https://rajpurkar.github.io/SQuAD-explorer//) 是一个评估模型在问答任务上性能的基准。有两个版本可用，v1.1 和 v2.0。第一个版本（v1.1）与文章 [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) 一起发布。第二个版本（v2.0）与文章 [Know What You Don'tKnow: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822) 一起发布。
本库提供了两个版本的处理器：

###  处理器（Processor）

这些处理器如下：

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]
They both inherit from the abstract class [`~data.processors.utils.SquadProcessor`]

[[autodoc]] data.processors.squad.SquadProcessor
    - all

此外，可以使用以下方法将 SQuAD 示例转换为 [`~data.processors.utils.SquadFeatures`]，以用作模型输入。
[[autodoc]] data.processors.squad.squad_convert_examples_to_features

这些处理器以及上述方法可以与包含数据的文件和 *tensorflow_datasets* 包一起使用。以下是示例。

### 示例用法

下面是使用处理器和数据文件进行转换的示例：

```python
# Loading a V2 processor
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# Loading a V1 processor
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

使用 *tensorflow_datasets* 与使用数据文件一样简单：
```python
# tensorflow_datasets only handle Squad V1.
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

在 [run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) 脚本中还有另一个使用这些处理器的示例。