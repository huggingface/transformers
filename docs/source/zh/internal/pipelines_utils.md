<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合该许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不带任何形式的保证或条件。请参阅许可证中的具体语言，以了解权限和限制。⚠️请注意，此文件是 Markdown 格式，但包含特定于我们文档构建器（类似于 MDX）的语法，可能无法在 Markdown 查看器中正确呈现。特定语言的权限和限制。
-->

# 管道工具

此页面列出了库提供的所有管道工具函数。

如果您正在研究库中的模型代码，这些工具函数大多是有用的。

## Argument handling

[[autodoc]] pipelines.ArgumentHandler

[[autodoc]] pipelines.ZeroShotClassificationArgumentHandler

[[autodoc]] pipelines.QuestionAnsweringArgumentHandler

## Data format

[[autodoc]] pipelines.PipelineDataFormat

[[autodoc]] pipelines.CsvPipelineDataFormat

[[autodoc]] pipelines.JsonPipelineDataFormat

[[autodoc]] pipelines.PipedPipelineDataFormat

## Utilities

[[autodoc]] pipelines.PipelineException