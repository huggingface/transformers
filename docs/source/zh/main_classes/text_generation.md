<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用本文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️ 请注意，该文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确地渲染。
-->

# 生成 Generation

每个框架都有一个在其相应的 `GenerationMixin` 类中实现的用于文本生成的生成方法：
- PyTorch [`~generation.GenerationMixin.generate`] 实现在 [`~generation.GenerationMixin`] 中。
- TensorFlow [`~generation.TFGenerationMixin.generate`] 实现在 [`~generation.TFGenerationMixin`] 中。
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] 实现在 [`~generation.FlaxGenerationMixin`] 中。

无论您选择的框架如何，您都可以使用 [`~generation.GenerationConfig`] 类的实例对生成方法进行参数化。请参考此类以获取完整的生成参数列表，这些参数控制生成方法的行为。要了解如何检查模型的生成配置、默认值是什么、如何临时更改参数以及如何创建和保存自定义的生成配置，请参阅 [文本生成策略指南](../generation_strategies)。该指南还解释了如何使用相关功能，如令牌流。

## 生成配置
[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained
## GenerationMixin
[[autodoc]] generation.GenerationMixin
	- generate
	- compute_transition_scores
	- greedy_search
	- sample
	- beam_search
	- beam_sample
	- contrastive_search
	- group_beam_search
	- constrained_beam_search
    
## TFGenerationMixin

[[autodoc]] generation.TFGenerationMixin
	- generate
	- compute_transition_scores

## FlaxGenerationMixin

[[autodoc]] generation.FlaxGenerationMixin
	- generate
