<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在下面获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不附带任何明示或暗示的担保或条件。有关许可证下的特定语言权限和限制，请参阅许可证。请注意，此文件是 Markdown 文件，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确呈现。
⚠️请注意，此文件是 Markdown 文件，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确呈现。渲染的。
-->

# Trainer 的实用工具

此页面列出了 [`Trainer`] 使用的所有实用函数。
如果您正在研究库中的 Trainer 的代码，则其中大部分只有在这种情况下才有用。

## Utilities

[[autodoc]] EvalPrediction

[[autodoc]] IntervalStrategy

[[autodoc]] enable_full_determinism

[[autodoc]] set_seed

[[autodoc]] torch_distributed_zero_first

## Callbacks internals

[[autodoc]] trainer_callback.CallbackHandler

## Distributed Evaluation

[[autodoc]] trainer_pt_utils.DistributedTensorGatherer

## Distributed Evaluation

[[autodoc]] HfArgumentParser

## Debug Utilities

[[autodoc]] debug_utils.DebugUnderflowOverflow