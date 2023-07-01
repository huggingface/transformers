<!--版权所有 2022 年 HuggingFace 团队和 OpenBMB 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，不附带任何明示或暗示的担保或条件。有关详细信息，请参阅许可证中的特定语言的权限和限制。⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。特别提示：
-->

# CPMAnt## 概述

CPM-Ant 是一个具有 100 亿参数的开源中文预训练语言模型（PLM）。它也是 CPM-Live 实时训练过程的第一个里程碑。训练过程具有成本效益且环境友好。CPM-Ant 还通过对 CUGE 基准进行增量调优取得了有希望的结果。除了提供全模型之外，我们还提供了各种压缩版本，以满足不同硬件配置的要求。[了解更多](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live)

## 提示：

此模型由 [OpenBMB](https://huggingface.co/openbmb) 贡献。原始代码可以在 [此处](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live) 找到。

⚙️ 训练和推理

- [CPM-Live](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live) 上的教程。

## CpmAntConfig

[[autodoc]] CpmAntConfig
    - all

## CpmAntTokenizer

[[autodoc]] CpmAntTokenizer
    - all

## CpmAntModel

[[autodoc]] CpmAntModel
    - all
    
## CpmAntForCausalLM

[[autodoc]] CpmAntForCausalLM
    - all