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

# Configuration

基类[`PretrainedConfig`]实现了从本地文件或目录加载/保存配置的常见方法，或下载库提供的预训练模型配置（从HuggingFace的AWS S3库中下载）。

每个派生的配置类都实现了特定于模型的属性。所有配置类中共同存在的属性有：`hidden_size`、`num_attention_heads` 和 `num_hidden_layers`。文本模型进一步添加了 `vocab_size`。


## PretrainedConfig

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all
