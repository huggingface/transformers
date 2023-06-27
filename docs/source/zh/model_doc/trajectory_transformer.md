<!--版权 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，不附带任何形式的保证或条件。请查阅许可证以获取特定语言下权限和限制的详细信息。⚠️ 请注意，这个文件是 Markdown 格式的，但包含了我们的文档构建器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确渲染。
-->

#  Trajectory Transformer

## 概述

轨迹变压器 （Trajectory Transformer）模型是由 Michael Janner、Qiyang Li 和 Sergey Levine 在 [离线增强学习作为一个大型序列建模问题](https://arxiv.org/abs/2106.02039) 中提出的。

以下是论文中的摘要：

*增强学习（RL）通常关注估计稳定策略或单步模型，利用马尔可夫属性在时间上分解问题。然而，我们也可以将 RL 视为一个通用的序列建模问题，目标是生成一系列动作，以获得一系列高回报。从这个角度来看，我们很容易考虑在其他领域（如自然语言处理）中工作良好的高容量序列预测模型是否也能为 RL 问题提供有效的解决方案。为此，我们探索了如何使用序列建模工具来解决 RL 问题，使用 Transformer 架构对轨迹的分布进行建模，并将波束搜索用作规划算法。将 RL 构建为序列建模问题简化了一系列设计决策，使我们可以省去离线 RL 算法中常见的许多组件。我们展示了这种方法在长时间预测、模仿学习、目标条件 RL 和离线 RL 中的灵活性。此外，我们还展示了这种方法可以与现有的无模型算法结合使用，以在稀疏奖励和长时间预测任务中产生最先进的规划器。*

提示：

此变压器用于深度增强学习。要使用它，您需要从所有先前时间步的动作、状态和奖励创建序列。该模型将把所有这些元素一起视为一个大型序列（一条轨迹）。

此模型由 [CarlCochet](https://huggingface.co/CarlCochet) 贡献。原始代码可在 [此处](https://github.com/jannerm/trajectory-transformer) 找到。

## TrajectoryTransformerConfig

[[autodoc]] TrajectoryTransformerConfig


## TrajectoryTransformerModel

[[autodoc]] TrajectoryTransformerModel
    - forward