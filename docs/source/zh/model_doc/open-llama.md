<!--版权 2023 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，不附带任何形式的明示或暗示担保或条件。请参阅许可证中的特定语言，以了解权限和限制。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
⚠️请注意，此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
⚠️请注意，此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。⚠️请注意，此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# Open-Llama

## 概述

Open-Llama 模型是由社区开发者 s-JoL 在 [Open-Llama 项目](https://github.com/s-JoL/Open-Llama) 中提出的。

该模型主要基于 LLaMA 进行了一些修改，包括来自 Xformers 的内存高效注意力、来自 Bloom 的稳定嵌入和来自 PaLM 的共享输入输出嵌入。该模型在中文和英文上进行了预训练，从而在中文语言任务上表现更好。

该模型由 [s-JoL](https://huggingface.co/s-JoL) 贡献。原始代码可以在 [Open-Llama](https://github.com/s-JoL/Open-Llama) 找到。检查点和用法可以在 [s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1) 找到。

## OpenLlamaConfig

[[autodoc]] OpenLlamaConfig

## OpenLlamaModel

[[autodoc]] OpenLlamaModel
    - forward

## OpenLlamaForCausalLM

[[autodoc]] OpenLlamaForCausalLM
    - forward

## OpenLlamaForSequenceClassification

[[autodoc]] OpenLlamaForSequenceClassification
    - forward