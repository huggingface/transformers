<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Processors

在 Transformers 库中，processors可以有两种不同的含义：
- 为多模态模型，例如[Wav2Vec2](../model_doc/wav2vec2)（语音和文本）或[CLIP](../model_doc/clip)（文本和视觉）预处理输入的对象
- 在库的旧版本中用于预处理GLUE或SQUAD数据的已弃用对象。

## 多模态processors

任何多模态模型都需要一个对象来编码或解码将多个模态（包括文本、视觉和音频）组合在一起的数据。这由称为processors的对象处理，这些processors将两个或多个处理对象组合在一起，例如tokenizers（用于文本模态），image processors（用于视觉）和feature extractors（用于音频）。

这些processors继承自以下实现保存和加载功能的基类：


[[autodoc]] ProcessorMixin

## 已弃用的processors

所有processor都遵循与 [`~data.processors.utils.DataProcessor`] 相同的架构。processor返回一个 [`~data.processors.utils.InputExample`] 列表。这些 [`~data.processors.utils.InputExample`] 可以转换为 [`~data.processors.utils.InputFeatures`] 以供输送到模型。

[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE

[General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) 是一个基准测试，评估模型在各种现有的自然语言理解任务上的性能。它与论文 [GLUE: A multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7) 一同发布。

该库为以下任务提供了总共10个processor：MRPC、MNLI、MNLI（mismatched）、CoLA、SST2、STSB、QQP、QNLI、RTE 和 WNLI。

这些processor是：

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]

此外，还可以使用以下方法从数据文件加载值并将其转换为 [`~data.processors.utils.InputExample`] 列表。

[[autodoc]] data.processors.glue.glue_convert_examples_to_features


## XNLI

[跨语言NLI语料库（XNLI）](https://www.nyu.edu/projects/bowman/xnli/) 是一个评估跨语言文本表示质量的基准测试。XNLI是一个基于[*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/)的众包数据集：”文本对“被标记为包含15种不同语言（包括英语等高资源语言和斯瓦希里语等低资源语言）的文本蕴涵注释。

它与论文 [XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053) 一同发布。

该库提供了加载XNLI数据的processor：

- [`~data.processors.utils.XnliProcessor`]

请注意，由于测试集上有“gold”标签，因此评估是在测试集上进行的。

使用这些processor的示例在 [run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py) 脚本中提供。


## SQuAD

[斯坦福问答数据集（SQuAD）](https://rajpurkar.github.io/SQuAD-explorer//) 是一个评估模型在问答上性能的基准测试。有两个版本，v1.1 和 v2.0。第一个版本（v1.1）与论文 [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) 一同发布。第二个版本（v2.0）与论文 [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822) 一同发布。

该库为两个版本各自提供了一个processor：

### Processors

这两个processor是：

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]

它们都继承自抽象类 [`~data.processors.utils.SquadProcessor`]。

[[autodoc]] data.processors.squad.SquadProcessor
    - all

此外，可以使用以下方法将 SQuAD 示例转换为可用作模型输入的 [`~data.processors.utils.SquadFeatures`]。

[[autodoc]] data.processors.squad.squad_convert_examples_to_features


这些processor以及前面提到的方法可以与包含数据的文件以及tensorflow_datasets包一起使用。下面给出了示例。


### Example使用

以下是使用processor以及使用数据文件的转换方法的示例：

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

使用 *tensorflow_datasets* 就像使用数据文件一样简单：

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

另一个使用这些processor的示例在 [run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) 脚本中提供。