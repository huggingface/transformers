<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
适用法律要求或书面同意前提下，根据许可证分发的软件是按 "按原样" 的基础分发的，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含了我们 doc-builder 的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确显示。
-->

# 导出为 ONNX

在生产环境中部署🤗 Transformers 模型经常需要将模型导出为序列化格式，以便在专用运行时和硬件上加载和执行。🤗 Optimum 是 Transformers 的扩展，通过其 `exporters` 模块，可以将模型从 PyTorch 或 TensorFlow 导出为 ONNX 和 TFLite 等序列化格式。此外，🤗 Optimum 还提供了一套性能优化工具，以实现在目标硬件上以最大效率训练和运行模型。

本指南演示了如何使用🤗 Optimum 将🤗 Transformers 模型导出为 ONNX。如果要了解将模型导出为 TFLite 的指南，请参阅 [导出为 TFLite 页面](tflite)。


## 导出为 ONNX

[ONNX (Open Neural Network eXchange)](http://onnx.ai) 是一种开放标准，定义了一组通用的操作符和一种通用的文件格式，用于在各种框架（包括 PyTorch 和 TensorFlow）中表示深度学习模型。当模型导出到 ONNX 格式时，这些操作符被用于构建一个计算图（通常称为_中间表示_），表示数据在神经网络中的流动。


通过公开具有标准化操作符和数据类型的图表，ONNX 使得在不同框架之间切换变得容易。例如，可以将在 PyTorch 中训练的模型导出为 ONNX 格式，然后在 TensorFlow 中导入（反之亦然）。

将模型导出为 ONNX 格式后，可以进行以下操作：

- 通过技术手段（例如 [图优化](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) 和 [量化](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)）对推理进行优化。
- 通过 [`ORTModelForXXX` 类](https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort) 使用 ONNX Runtime 运行模型。这些类与🤗 Transformers 中的 `AutoModel` API 相同。
- 使用 [优化的推理流水线](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines) 运行模型，该流水线与🤗 Transformers 中的 [`pipeline`] 函数具有相同的 API。


🤗 Optimum 通过利用配置对象来支持 ONNX 导出。这些配置对象为许多模型架构提供了现成的支持，并且被设计成易于扩展到其他架构。

有两种将🤗 Transformers 模型导出为 ONNX 的方法，我们在这里都展示一下：


- 通过 CLI 使用🤗 Optimum 进行导出。- 使用 `optimum.onnxruntime` 使用🤗 Optimum 进行导出。
### 使用 CLI 将🤗 Transformers 模型导出为 ONNX
要将🤗 Transformers 模型导出为 ONNX，首先安装一个额外的依赖项：
```bash
pip install optimum[exporters]
```

要查看所有可用参数，请参阅 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli)，或在命令行中查看帮助：
```bash
optimum-cli export onnx --help
```

要从🤗 Hub 导出模型的检查点，例如 `distilbert-base-uncased-distilled-squad`，运行以下命令：
```bash
optimum-cli export onnx --model distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

您应该看到日志显示进度，并显示保存结果的 `model.onnx` 的位置，如下所示：
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

上面的示例演示了从🤗 Hub 导出检查点的过程。当导出本地模型时，请确保将模型的权重和标记器文件保存在同一个目录（`local_path`）中。使用 CLI 时，将 `local_path` 作为 `model` 参数传递，而不是在🤗 Hub 上的检查点名称，并提供 `--task` 参数。

您可以在 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/task_manager) 中查看支持的任务列表。

如果未提供 `task` 参数，它将默认为不带任何任务特定头的模型架构。
```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

然后，可以在支持 ONNX 标准的 [多个加速器](https://onnx.ai/supported-tools.html#deployModel) 上运行生成的 `model.onnx` 文件。
例如，我们可以使用 [ONNXRuntime](https://onnxruntime.ai/) 加载和运行模型，如下所示：Runtime](https://onnxruntime.ai/) as follows:

```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

使用 TensorFlow Hub 上的 TensorFlow 检查点的过程相同。

例如，以下是如何从 [Keras 组织](https://huggingface.co/keras-io) 导出纯 TensorFlow 检查点：
```bash
optimum-cli export onnx --model keras-io/transformers-qa distilbert_base_cased_squad_onnx/
```

### 使用 `optimum.onnxruntime` 将🤗 Transformers 模型导出为 ONNX

除了 CLI，您还可以通过编程方式将🤗 Transformers 模型导出为 ONNX，如下所示：
```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> # Load a model from transformers and export it to ONNX
>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> # Save the onnx model and tokenizer
>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```

### 导出不受支持的架构的模型

如果要通过添加对当前无法导出的模型的支持来进行贡献，首先应检查是否在 [`optimum.exporters.onnx`](https://huggingface.co/docs/optimum/exporters/onnx/overview) 中支持该模型，如果不支持，请直接 [向🤗 Optimum 贡献](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute)。directly.

### 使用 `transformers.onnx` 导出模型

<Tip warning={true}>

`tranformers.onnx` 不再维护，请按上述方法使用🤗 Optimum 导出模型。此部分将在将来的版本中删除。
</Tip>
要使用 `tranformers.onnx` 将🤗 Transformers 模型导出为 ONNX，请安装额外的依赖项：
```bash
pip install transformers[onnx]
```

使用 `transformers.onnx` 包作为 Python 模块，使用现成的配置导出检查点：
```bash
python -m transformers.onnx --model=distilbert-base-uncased onnx/
```

这将导出由 `--model` 参数定义的检查点的 ONNX 图。传递🤗 Hub 上的任何检查点或本地存储的检查点。然后，可以在支持 ONNX 标准的许多加速器上运行生成的 `model.onnx` 文件。例如，使用 ONNX Runtime 加载和运行模型，如下所示：
```python
>>> from transformers import AutoTokenizer
>>> from onnxruntime import InferenceSession

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
>>> session = InferenceSession("onnx/model.onnx")
>>> # ONNX Runtime expects NumPy arrays as input
>>> inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
>>> outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

所需的输出名称（如 `["last_hidden_state"]`）可以通过查看每个模型的 ONNX 配置来获得。

例如，对于 DistilBERT，我们有：each model. For example, for DistilBERT we have:

```python
>>> from transformers.models.distilbert import DistilBertConfig, DistilBertOnnxConfig

>>> config = DistilBertConfig()
>>> onnx_config = DistilBertOnnxConfig(config)
>>> print(list(onnx_config.outputs.keys()))
["last_hidden_state"]
```

TensorFlow Hub 上的 TensorFlow 检查点的过程相同。例如，导出纯 TensorFlow 检查点的操作如下所示：
```bash
python -m transformers.onnx --model=keras-io/transformers-qa onnx/
```

要导出本地存储的模型，请将模型的权重和标记器文件保存在同一个目录中（例如 `local-pt-checkpoint`），然后通过将 `transformers.onnx` 包的 `--model` 参数指向所需目录，将其导出为 ONNX：
```bash
python -m transformers.onnx --model=local-pt-checkpoint onnx/
```