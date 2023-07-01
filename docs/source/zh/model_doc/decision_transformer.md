<!--版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
适用法律或书面同意的情况下，根据许可证分发的软件是按照“按原样”分发的基础上，不提供任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的 doc-builder（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 决策 Transformer （Decision Transformer）

## 概述

决策 Transformer 模型是由 Lili Chen、Kevin Lu、Aravind Rajeswaran、Kimin Lee、Aditya Grover、Michael Laskin、Pieter Abbeel、Aravind Srinivas 和 Igor Mordatch 在 [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) 中提出的。以下是论文中的摘要:

*我们引入了一个将强化学习（RL）抽象为序列建模问题的框架。
这使我们能够利用 Transformer 架构的简单性和可扩展性，以及 GPT-x 和 BERT 等语言建模的相关进展。特别是，我们提出了 Decision Transformer，这是一种将 RL 问题作为条件序列建模的架构。与先前的 RL 方法不同，它适配了值函数或计算策略梯度，Decision Transformer 仅通过利用具有因果遮蔽的 Transformer 来输出最佳动作。通过将自回归模型条件化为期望的回报（奖励）、过去的状态和动作，我们的 Decision Transformer 模型可以生成实现期望回报的未来动作。尽管简单，Decision Transformer 在 Atari、OpenAI Gym 和 Key-to-Door 任务上与最先进的无模型离线 RL 基线模型的性能相匹配或超过。* 

提示:
此版本的模型适用于状态为向量的任务，基于图像的状态将很快推出。
此模型由 [edbeeching](https://huggingface.co/edbeeching) 贡献。原始代码可以在 [此处](https://github.com/kzl/decision-transformer) 找到。
## DecisionTransformerConfig

[[autodoc]] DecisionTransformerConfig


## DecisionTransformerGPT2Model

[[autodoc]] DecisionTransformerGPT2Model
    - forward

## DecisionTransformerModel

[[autodoc]] DecisionTransformerModel
    - forward