<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Whisper [[whisper]]

## 개요 [[overview]]

휘스퍼 모델은 Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever에 의해 제안되었습니다. 이 모델은 [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)에서 설명되어 있습니다.

논문의 요약은 다음과 같습니다:

*우리는 인터넷 오디오의 대량 트랜스크립트를 예측하기만 하는 방식으로 훈련된 음성 처리 시스템의 능력을 연구합니다. 68만 시간 이상의 다국적 및 다중 작업 지도에 확장할 때 결과 모델은 표준 벤치마크에 대해 일반화되며 종종 이전의 완전히 감독된 결과와 경쟁력이 있습니다. 그러나 미세 조정이 필요하지 않는 제로샷 전송 환경에서입니다. 모델을 인간과 비교했을 때 모델은 정확도와 견고성에 접근합니다. 우리는 이 모델을 공개하고 인퍼런스 코드를 제공하여 견고한 음성 처리에 대한 추가 연구의 기반으로 제공합니다.*



팁:

- 이 모델은 일반적으로 미세 조정이 필요하지 않고 잘 작동합니다.
- 아키텍처는 전통적인 인코더-디코더 아키텍처를 따르므로 인퍼런스에 [`~generation.GenerationMixin.generate`] 함수를 사용합니다.
- 현재 인퍼런스는 짧은 형식에만 구현되어 있으며 오디오는 <=30초 세그먼트로 미리 분할됩니다. 장형 형식 (타임스탬프 포함)은 미래 릴리스에서 구현될 예정입니다.
- 모델을 위해 오디오를 준비하고 예측된 ID를 텍스트로 디코딩하려면 [`WhisperProcessor`]를 사용할 수 있습니다.

이 모델은 [Arthur Zucker](https://huggingface.co/ArthurZ)에 의해 제공되었습니다. 이 모델의 Tensorflow 버전은 [amyeroberts](https://huggingface.co/amyeroberts)에 의해 기여되었습니다.
원본 코드는 [여기](https://github.com/openai/whisper)에서 찾을 수 있습니다.



## WhisperConfig [[whisperconfig]]

[[autodoc]] WhisperConfig

## WhisperTokenizer [[whispertokenizer]]

[[autodoc]] WhisperTokenizer
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## WhisperTokenizerFast [[whispertokenizerfast]]

[[autodoc]] WhisperTokenizerFast
    - set_prefix_tokens
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## WhisperFeatureExtractor [[whisperfeatureextractor]]

[[autodoc]] WhisperFeatureExtractor
    - __call__

## WhisperProcessor [[whisperprocessor]]

[[autodoc]] WhisperProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## WhisperModel [[whispermodel]]

[[autodoc]] WhisperModel
    - forward
    - _mask_input_features

## WhisperForConditionalGeneration [[whisperforconditionalgeneration]]

[[autodoc]] WhisperForConditionalGeneration
    - forward

## WhisperForAudioClassification [[whisperforaudioclassification]]

[[autodoc]] WhisperForAudioClassification
    - forward



## TFWhisperModel [[tfwhispermodel]]

[[autodoc]] TFWhisperModel
    - call

## TFWhisperForConditionalGeneration [[tfwhisperforconditionalgeneration]]

[[autodoc]] TFWhisperForConditionalGeneration
    - call


## FlaxWhisperModel [[flaxwhispermodel]]

[[autodoc]] FlaxWhisperModel
    - __call__

## FlaxWhisperForConditionalGeneration [[flaxwhisperforconditionalgeneration]]

[[autodoc]] FlaxWhisperForConditionalGeneration
    - __call__

## FlaxWhisperForAudioClassification [[flaxwhisperforaudioclassification]]

[[autodoc]] FlaxWhisperForAudioClassification
    - __call__

