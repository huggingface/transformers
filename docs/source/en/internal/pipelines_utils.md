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

# Utilities for pipelines

This page lists all the utility functions the library provides for pipelines.

Most of those are only useful if you are studying the code of the models in the library.

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
