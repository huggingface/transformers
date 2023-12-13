<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

#  `FeatureExtractors`的工具

此页面列出了音频 [`FeatureExtractor`] 可以使用的所有实用函数，以便使用常见的算法（如 *Short Time Fourier Transform* 或 *log mel spectrogram*）从原始音频中计算特殊特征。

其中大多数仅在您研究库中音频processors的代码时有用。


## 音频转换

[[autodoc]] audio_utils.hertz_to_mel

[[autodoc]] audio_utils.mel_to_hertz

[[autodoc]] audio_utils.mel_filter_bank

[[autodoc]] audio_utils.optimal_fft_length

[[autodoc]] audio_utils.window_function

[[autodoc]] audio_utils.spectrogram

[[autodoc]] audio_utils.power_to_db

[[autodoc]] audio_utils.amplitude_to_db
