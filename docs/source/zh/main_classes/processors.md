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

# Processors

## 多模态processors

任何多模态模型都需要一个对象来编码或解码将多个模态（包括文本、视觉和音频）组合在一起的数据。这由称为processors的对象处理，这些processors将两个或多个处理对象组合在一起，例如tokenizers（用于文本模态），image processors（用于视觉）和feature extractors（用于音频）。

这些processors继承自以下实现保存和加载功能的基类：


[[autodoc]] ProcessorMixin
