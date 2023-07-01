<!--版权所有 2022 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非遵守许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”方式分发，不提供任何明示或默示的担保或条件。参见许可证中的具体语言以了解权限和限制。特别注意，此文件是 Markdown 格式，但包含了特定于我们的文档生成器（类似于 MDX）的语法，您的 Markdown 查看器可能无法正确呈现。此外，请注意，该文件是 Markdown 格式，但包含了与我们的文档生成器（类似于 MDX）的特定语法，您的 Markdown 查看器可能无法正确呈现。
⚠️ 注意，此文件是 Markdown 格式的，但包含了我们的文档生成器（类似于 MDX）的特定语法，您的 Markdown 查看器可能无法正确渲染。渲染。
-->

# 配置

基类 [`PretrainedConfig`] 实现了加载/保存配置的常用方法，可以从本地文件或目录加载，也可以从库提供的预训练模型配置加载（从 HuggingFace 的 AWS S3 存储库下载）。每个派生的配置类都实现了特定于模型的属性。所有配置类中的共同属性包括：`hidden_size`，`num_attention_heads` 和 `num_hidden_layers`。文本模型还实现了：`vocab_size`。from HuggingFace's AWS S3 repository).



## PretrainedConfig

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all
