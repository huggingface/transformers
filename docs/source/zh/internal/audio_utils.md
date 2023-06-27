<!--版权所有 2023 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。您可以在以下网址获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”基础分发的，不附带任何形式的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含特定于我们文档构建器（类似于 MDX）的语法，您的 Markdown 查看器可能无法正确呈现。
-->

# 用于“FeatureExtractors”的实用工具

本页面列出了可以由音频 [`FeatureExtractor`] 使用的所有实用函数，以便使用常见算法（如 *短时傅里叶变换* 或 *对数梅尔频谱*）从原始音频计算特殊特征。
这些大多数只在您研究库中的音频处理器代码时有用。

## 音频转换
[[autodoc]] audio_utils.hertz_to_mel
[[autodoc]] audio_utils.mel_to_hertz
[[autodoc]] audio_utils.mel_filter_bank
[[autodoc]] audio_utils.optimal_fft_length
[[autodoc]] audio_utils.window_function
[[autodoc]] audio_utils.spectrogram
[[autodoc]] audio_utils.power_to_db
[[autodoc]] audio_utils.amplitude_to_db