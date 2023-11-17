<!--
版权所有 2020 年 HuggingFace 团队保留所有权利。

根据 Apache 许可证，版本 2.0 进行许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”提供的，没有任何形式的担保或条件。有关许可证的具体语言，请参阅许可证。

⚠️ 请注意，此文件是Markdown格式，但包含特定于我们文档生成器（类似于MDX）的语法，可能在您的Markdown查看器中无法正确呈现。

-->

# 配置

基类[`PretrainedConfig`]实现了从本地文件或目录加载/保存配置的常用方法，也可以从库提供的预训练模型配置（从HuggingFace的 AWS S3 存储库下载）中加载。

每个派生配置类都实现了特定于模型的属性。所有配置类中都存在的通用属性是：`hidden_size`、`num_attention_heads` 和 `num_hidden_layers`。文本模型还实现了：`vocab_size`。

## PretrainedConfig

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all
