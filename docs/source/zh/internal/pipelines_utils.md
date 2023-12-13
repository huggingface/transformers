<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# pipelines的工具


此页面列出了库为pipelines提供的所有实用程序功能。

其中大多数只有在您研究库中模型的代码时才有用。


## 参数处理

[[autodoc]] pipelines.ArgumentHandler

[[autodoc]] pipelines.ZeroShotClassificationArgumentHandler

[[autodoc]] pipelines.QuestionAnsweringArgumentHandler

## 数据格式

[[autodoc]] pipelines.PipelineDataFormat

[[autodoc]] pipelines.CsvPipelineDataFormat

[[autodoc]] pipelines.JsonPipelineDataFormat

[[autodoc]] pipelines.PipedPipelineDataFormat

## 实用函数

[[autodoc]] pipelines.PipelineException
