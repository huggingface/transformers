<!--版权所有2020年The HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可。除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”分发的，不附带任何明示或暗示的担保或条件。有关许可证下特定语言的权限和限制，请参阅许可证。
请注意，此文件采用 Markdown 格式，但包含我们文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正常显示。
⚠️请注意，此文件采用 Markdown 格式，但包含我们文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正常显示。
-->

# 优化 optimization

`.optimization` 模块提供了以下功能：

- 一个带有固定权重衰减的优化器，可用于微调模型。- 一些继承自 `_LRSchedule` 的调度对象：

- 一个梯度累积类，用于累积多个批次的梯度。

## AdamW（PyTorch）
[[autodoc]] AdamW
## AdaFactor（PyTorch）
[[autodoc]] Adafactor

## AdamWeightDecay（TensorFlow）
[[autodoc]] AdamWeightDecay
[[autodoc]] create_optimizer

## Schedules

### Learning Rate Schedules (Pytorch)

[[autodoc]] SchedulerType

[[autodoc]] get_scheduler

[[autodoc]] get_constant_schedule

[[autodoc]] get_constant_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png"/>

[[autodoc]] get_cosine_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png"/>

[[autodoc]] get_cosine_with_hard_restarts_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png"/>

[[autodoc]] get_linear_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png"/>

[[autodoc]] get_polynomial_decay_schedule_with_warmup

[[autodoc]] get_inverse_sqrt_schedule

### Warmup (TensorFlow)

[[autodoc]] WarmUp

## Gradient Strategies

### GradientAccumulator (TensorFlow)

[[autodoc]] GradientAccumulator
