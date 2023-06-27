<!--版权所有2023年The HuggingFace Team。保留所有权利。
根据 Apache 许可证，版本 2.0（“许可证”）进行许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以获取特定语言下许可证的权限和限制。
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
请注意，这个文件是 Markdown 格式的，但包含了我们的文档生成器（类似 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确呈现。specific language governing permissions and limitations under the License.



-->
# FLAN-UL2

## 概述

Flan-UL2 是基于 T5 架构的编码解码模型。它使用与去年初发布的 UL2 模型相同的配置。它使用了“Flan”提示微调和数据集收集进行微调。与 `Flan-T5` 类似，您可以直接使用 FLAN-UL2 的权重而无需微调模型：

根据原始博客，以下是值得注意的改进：

- 原始的 UL2 模型只用 512 的感受野进行训练，这使得它在 N 较大的 N-shot 提示中非理想。- Flan-UL2 检查点使用 2048 的感受野，使其更适用于少样本上下文学习。- 原始的 UL2 模型也有模式切换标记，这在获得良好性能时是必需的。然而，它们有点繁琐，因为这在推理或微调过程中经常需要进行一些更改。在此更新/更改中，我们在应用 Flan 指令微调之前，继续对 UL2 20B 进行额外的 10 万步（使用小批量）训练以忽略“模式标记”。这个 Flan-UL2 检查点不再需要模式标记。Google 发布了以下变体：

您可以参考 [T5 的文档页面](t5) 获取有关模型的所有提示、代码示例和笔记本。以及有关模型训练和评估的 FLAN-T5 模型卡片的更多详细信息。

原始检查点可以在 [这里](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-ul2-checkpoints) 找到。

## 在低资源设备上运行

该模型非常庞大（半精度约为 40GB），因此如果您只是想运行模型，请确保以 8 位加载模型，并使用 `device_map="auto"` 确保您没有任何 OOM 问题！

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", load_in_8bit=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

>>> inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['In a large skillet, brown the ground beef and onion over medium heat. Add the garlic']
```

## 推断

推断协议与任何 `T5` 模型完全相同，请参阅 [T5 的文档页面](t5) 获取更多详细信息。