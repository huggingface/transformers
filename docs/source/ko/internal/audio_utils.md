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

# `FeatureExtractors`를 위한 유틸리티 [[utilities-for-featureextractors]]

이 페이지는 오디오 [`FeatureExtractor`]가 *단시간 푸리에 변환(Short Time Fourier Transform)* 또는 *로그 멜 스펙트로그램(log mel spectrogram)*과 같은 일반적인 알고리즘을 사용하여 원시 오디오에서 특수한 특성을 계산하는 데 사용할 수 있는 유틸리티 함수들을 나열합니다.

이 함수들 대부분은 라이브러리 내 오디오 처리 코드를 연구할 때에만 유용합니다.

## 오디오 변환 [[transformers.audio_utils.hertz_to_mel]]

[[autodoc]] audio_utils.hertz_to_mel

[[autodoc]] audio_utils.mel_to_hertz

[[autodoc]] audio_utils.mel_filter_bank

[[autodoc]] audio_utils.optimal_fft_length

[[autodoc]] audio_utils.window_function

[[autodoc]] audio_utils.spectrogram

[[autodoc]] audio_utils.power_to_db

[[autodoc]] audio_utils.amplitude_to_db
