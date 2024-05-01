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

# 导出为 ONNX

在生产环境中部署 🤗 Transformers 模型通常需要或者能够受益于，将模型导出为可在专门的运行时和硬件上加载和执行的序列化格式。

🤗 Optimum 是 Transformers 的扩展，可以通过其 `exporters` 模块将模型从 PyTorch 或 TensorFlow 导出为 ONNX 及 TFLite 等序列化格式。🤗 Optimum 还提供了一套性能优化工具，可以在目标硬件上以最高效率训练和运行模型。

本指南演示了如何使用 🤗 Optimum 将 🤗 Transformers 模型导出为 ONNX。有关将模型导出为 TFLite 的指南，请参考 [导出为 TFLite 页面](tflite)。

## 导出为 ONNX

[ONNX (Open Neural Network eXchange 开放神经网络交换)](http://onnx.ai) 是一个开放的标准，它定义了一组通用的运算符和一种通用的文件格式，用于表示包括 PyTorch 和 TensorFlow 在内的各种框架中的深度学习模型。当一个模型被导出为 ONNX时，这些运算符被用于构建计算图（通常被称为*中间表示*），该图表示数据在神经网络中的流动。

通过公开具有标准化运算符和数据类型的图，ONNX使得模型能够轻松在不同深度学习框架间切换。例如，在 PyTorch 中训练的模型可以被导出为 ONNX，然后再导入到 TensorFlow（反之亦然）。

导出为 ONNX 后，模型可以：
- 通过 [图优化（graph optimization）](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) 和 [量化（quantization）](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization) 等技术进行推理优化。 
- 通过 [`ORTModelForXXX` 类](https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort) 使用 ONNX Runtime 运行，它同样遵循你熟悉的 Transformers 中的 `AutoModel` API。
- 使用 [优化推理流水线（pipeline）](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines) 运行，其 API 与 🤗 Transformers 中的 [`pipeline`] 函数相同。

🤗 Optimum 通过利用配置对象提供对 ONNX 导出的支持。多种模型架构已经有现成的配置对象，并且配置对象也被设计得易于扩展以适用于其他架构。

现有的配置列表请参考 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/onnx/overview)。

有两种方式可以将 🤗 Transformers 模型导出为 ONNX，这里我们展示这两种方法：

- 使用 🤗 Optimum 的 CLI（命令行）导出。
- 使用 🤗 Optimum 的 `optimum.onnxruntime` 模块导出。

### 使用 CLI 将 🤗 Transformers 模型导出为 ONNX

要将 🤗 Transformers 模型导出为 ONNX，首先需要安装额外的依赖项：

```bash
pip install optimum[exporters]
```

请参阅 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli) 以查看所有可用参数，或者在命令行中查看帮助：

```bash
optimum-cli export onnx --help
```

运行以下命令，以从 🤗 Hub 导出模型的检查点（checkpoint），以 `distilbert/distilbert-base-uncased-distilled-squad` 为例：

```bash
optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

你应该能在日志中看到导出进度以及生成的 `model.onnx` 文件的保存位置，如下所示：

```bash
Validating ONNX model distilbert_base_uncased_squad_onnx/model.onnx...
	-[✓] ONNX model output names match reference model (start_logits, end_logits)
	- Validating ONNX Model output "start_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
	- Validating ONNX Model output "end_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: distilbert_base_uncased_squad_onnx
```

上面的示例说明了从 🤗 Hub 导出检查点的过程。导出本地模型时，首先需要确保将模型的权重和分词器文件保存在同一目录（`local_path`）中。在使用 CLI 时，将 `local_path` 传递给 `model` 参数，而不是 🤗 Hub 上的检查点名称，并提供 `--task` 参数。你可以在 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/task_manager)中查看支持的任务列表。如果未提供 `task` 参数，将默认导出不带特定任务头的模型架构。

```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

生成的 `model.onnx` 文件可以在支持 ONNX 标准的 [许多加速引擎（accelerators）](https://onnx.ai/supported-tools.html#deployModel) 之一上运行。例如，我们可以使用 [ONNX Runtime](https://onnxruntime.ai/) 加载和运行模型，如下所示：

```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

从 Hub 导出 TensorFlow 检查点的过程也一样。例如，以下是从 [Keras 组织](https://huggingface.co/keras-io) 导出纯 TensorFlow 检查点的命令：

```bash
optimum-cli export onnx --model keras-io/transformers-qa distilbert_base_cased_squad_onnx/
```

### 使用 `optimum.onnxruntime` 将 🤗 Transformers 模型导出为 ONNX

除了 CLI 之外，你还可以使用代码将 🤗 Transformers 模型导出为 ONNX，如下所示：

```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> # 从 transformers 加载模型并将其导出为 ONNX
>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> # 保存 onnx 模型以及分词器
>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```

### 导出尚未支持的架构的模型

如果你想要为当前无法导出的模型添加支持，请先检查 [`optimum.exporters.onnx`](https://huggingface.co/docs/optimum/exporters/onnx/overview) 是否支持该模型，如果不支持，你可以 [直接为 🤗 Optimum 贡献代码](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute)。

### 使用 `transformers.onnx` 导出模型

<Tip warning={true}>

`tranformers.onnx` 不再进行维护，请如上所述，使用 🤗 Optimum 导出模型。这部分内容将在未来版本中删除。

</Tip>

要使用 `tranformers.onnx` 将 🤗 Transformers 模型导出为 ONNX，请安装额外的依赖项：

```bash
pip install transformers[onnx]
```

将 `transformers.onnx` 包作为 Python 模块使用，以使用现成的配置导出检查点：

```bash
python -m transformers.onnx --model=distilbert/distilbert-base-uncased onnx/
```

以上代码将导出由 `--model` 参数定义的检查点的 ONNX 图。传入任何 🤗 Hub 上或者存储与本地的检查点。生成的 `model.onnx` 文件可以在支持 ONNX 标准的众多加速引擎上运行。例如，使用 ONNX Runtime 加载并运行模型，如下所示：

```python
>>> from transformers import AutoTokenizer
>>> from onnxruntime import InferenceSession

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
>>> session = InferenceSession("onnx/model.onnx")
>>> # ONNX Runtime expects NumPy arrays as input
>>> inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
>>> outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

可以通过查看每个模型的 ONNX 配置来获取所需的输出名（例如 `["last_hidden_state"]`）。例如，对于 DistilBERT，可以用以下代码获取输出名称：

```python
>>> from transformers.models.distilbert import DistilBertConfig, DistilBertOnnxConfig

>>> config = DistilBertConfig()
>>> onnx_config = DistilBertOnnxConfig(config)
>>> print(list(onnx_config.outputs.keys()))
["last_hidden_state"]
```

从 Hub 导出 TensorFlow 检查点的过程也一样。导出纯 TensorFlow 检查点的示例代码如下：

```bash
python -m transformers.onnx --model=keras-io/transformers-qa onnx/
```

要导出本地存储的模型，请将模型的权重和分词器文件保存在同一目录中（例如 `local-pt-checkpoint`），然后通过将 `transformers.onnx` 包的 `--model` 参数指向该目录，将其导出为 ONNX：

```bash
python -m transformers.onnx --model=local-pt-checkpoint onnx/
```