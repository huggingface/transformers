
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以了解具体语言下的权限和限制。⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确呈现。特征提取器
特征提取器负责为音频或视觉模型准备输入特征。这包括从序列中提取特征，例如将音频文件预处理为 Log-Mel Spectrogram 特征，从图像中提取特征，例如裁剪图像文件，还包括填充、归一化和转换为 Numpy、PyTorch 和 TensorFlow 张量。rendered properly in your Markdown viewer.

-->

# Feature Extractor

特征提取器负责为音频或视觉模型准备输入特征。这包括从序列中提取特征，例如将音频文件预处理为 Log-Mel Spectrogram 特征，从图像中提取特征，例如裁剪图像文件，还包括填充、归一化和转换为 Numpy、PyTorch 和 TensorFlow 张量。


## FeatureExtractionMixin

[[autodoc]] feature_extraction_utils.FeatureExtractionMixin
    - from_pretrained
    - save_pretrained

## SequenceFeatureExtractor

[[autodoc]] SequenceFeatureExtractor
    - pad

## BatchFeature

[[autodoc]] BatchFeature

## ImageFeatureExtractionMixin

[[autodoc]] image_utils.ImageFeatureExtractionMixin