<!--版权所有2023年HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以获取特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含适用于我们 doc-builder 的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。
-->

# 导出到 TFLite

[TensorFlow Lite](https://www.tensorflow.org/lite/guide) 是一个轻量级的框架，用于在资源受限的设备上部署机器学习模型，例如移动电话、嵌入式系统和物联网（IoT）设备。TFLite 旨在通过有限的计算能力、内存和功耗，在这些设备上高效地优化和运行模型。
TensorFlow Lite 模型以一种特殊的高效可移植格式表示，该格式由 `.tflite` 文件扩展名标识。

🤗 Optimum 通过 `exporters.tflite` 模块提供将🤗 Transformers 模型导出为 TFLite 的功能。

有关支持的模型架构列表，请参阅 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/tflite/overview)。

要将模型导出到 TFLite，请安装所需的依赖项： 

```bash
pip install optimum[exporters-tf]
```

要查看所有可用参数，请参阅 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model)，或在命令行中查看帮助：
```bash
optimum-cli export tflite --help
```

要从🤗 Hub 导出模型的检查点，例如 `bert-base-uncased`，请运行以下命令：
```bash
optimum-cli export tflite --model bert-base-uncased --sequence_length 128 bert_tflite/
```

您应该看到显示进度并显示生成的 `model.tflite` 保存位置的日志，例如：
```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
```
上面的示例演示了如何从🤗 Hub导出一个检查点。当导出本地模型时，请确保将模型的权重和分词器文件保存在同一个目录(`local_path`)中。当使用命令行界面时，将`local_path`传递给`model`参数，而不是传递🤗 Hub上的检查点名称。