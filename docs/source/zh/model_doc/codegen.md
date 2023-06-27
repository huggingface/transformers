<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2 版（“许可证”）的规定，您只能在符合许可证的情况下使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS，无论是明示还是暗示，都没有任何形式的保证或条件。有关许可的详细语言规定了权限和限制。
⚠️ 请注意，此文件是 Markdown 格式的，但其中包含我们的文档生成器（类似于 MDX）的特定语法，可能不会在您的 Markdown 查看器中正确渲染。
-->
# 代码生成
## 概述
CodeGen 模型由 Erik Nijkamp、Bo Pang、Hiroaki Hayashi、Lifu Tu、Huan Wang、Yingbo Zhou、Silvio Savarese 和 Caiming Xiong 在 [A Conversational Paradigm for Program Synthesis](https://arxiv.org/abs/2203.13474) 中提出。
CodeGen 是一个自回归语言模型，用于程序合成，通过对 [The Pile](https://pile.eleuther.ai/)、BigQuery 和 BigPython 进行顺序训练。
论文摘要如下：
*程序合成旨在生成一个计算机程序，作为给定问题规范的解决方案。我们提出了一种通过大型语言模型进行对话式程序合成的方法，该方法解决了之前方法中在广泛的程序空间搜索和用户意图规范方面面临的挑战。我们的新方法将编写规范和程序的过程视为用户与系统之间的多轮对话。它将程序合成视为一个序列预测问题，其中规范用自然语言表示，所需程序是有条件地采样的。我们使用自然语言和编程语言数据训练了一系列大型语言模型，称为 CodeGen。在数据的弱监督和数据规模以及模型大小的扩大下，对话能力从简单的自回归语言建模中产生。为了研究模型在对话式程序合成中的行为，我们开发了一个多轮编程基准（MTPB），在该基准中，通过用户和模型之间的多轮对话解决每个问题都需要多步合成。我们的研究结果显示出对话能力的出现以及所提出的对话式程序合成范式的有效性。此外，我们的 CodeGen 模型（使用 TPU-v4 上训练的多达 16B 参数）在 HumanEval 基准测试中优于 OpenAI 的 Codex。我们将包括检查点在内的训练库 JaxFormer 作为开源贡献提供：[this https URL](https://github.com/salesforce/codegen)。*
此模型由 [Hiroaki Hayashi](https://huggingface.co/rooa) 贡献。原始代码可在 [此处](https://github.com/salesforce/codegen) 找到。
## 检查点命名
* CodeGen 模型 [checkpoints](https://huggingface.co/models?other=codegen) 可在不同的预训练数据上使用，大小可变。* 格式为：`Salesforce/codegen-{size}-{data}`，其中  * `size`: `350M`、`2B`、`6B`、`16B`  * `data`:     * `nl`: 在 Pile 上进行预训练    * `multi`: 使用 `nl` 进行初始化，然后在多种编程语言数据上进一步预训练    * `mono`: 使用 `multi` 进行初始化，然后在 Python 数据上进一步预训练* 例如，`Salesforce/codegen-350M-mono` 提供了一个预先训练了 3.5 亿参数的检查点，顺序训练了 Pile、多种编程语言和 Python。
## 如何使用
```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> checkpoint = "Salesforce/codegen-350M-mono"
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

>>> text = "def hello_world():"

>>> completion = model.generate(**tokenizer(text, return_tensors="pt"))

>>> print(tokenizer.decode(completion[0]))
def hello_world():
    print("Hello World")

hello_world()
```

## 文档资源
- [因果语言建模任务指南](../tasks/language_modeling)
## CodeGenConfig
[[autodoc]] CodeGenConfig    - all
## CodeGenTokenizer
[[autodoc]] CodeGenTokenizer    - save_vocabulary
## CodeGenTokenizerFast
[[autodoc]] CodeGenTokenizerFast
## CodeGenModel
[[autodoc]] CodeGenModel    - forward
## CodeGenForCausalLM
[[autodoc]] CodeGenForCausalLM    - forward