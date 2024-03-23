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

Whisper 모델은 Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever에 의해 [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)에서 제안되었습니다.

논문의 초록은 다음과 같습니다:

*우리는 인터넷에서 대량의 오디오를 글로 옮긴 것을 예측하도록 간단히 훈련된 음성 처리 시스템의 성능을 연구합니다. 68만 시간의 다국어 및 다중 작업 지도(multitask supervision)에 확장했을 때, 결과 모델은 표준 벤치마크에 잘 일반화되며, 미세 조정이 필요 없는 제로샷 전송 설정에서 이전의 완전히 지도된(fully-supervised) 결과와 경쟁할 수 있는 경우가 많습니다. 사람과 비교하면, 이 모델은 사람의 정확도와 견고성에 근접합니다. 우리는 강력한 음성 처리를 위한 추가 작업의 기반이 될 모델과 추론 코드를 공개합니다.*



팁:

- 이 모델은 일반적으로 별도의 미세 조정 없이도 잘 작동합니다.
- 아키텍처는 고전적인 인코더-디코더 아키텍처를 따르기 때문에, 추론을 위해 [`~generation.GenerationMixin.generate`] 함수를 사용합니다.
- 현재 추론은 짧은 형식에만 구현되어 있으며, 오디오는 30초 미만의 세그먼트로 미리 분할되어야 합니다. 타임스탬프를 포함한 긴 형식에 대한 추론은 향후 릴리스에서 구현될 예정입니다.
- [`WhisperProcessor`]를 사용하여 모델에 사용할 오디오를 준비하고, 예측된 ID를 텍스트로 디코딩할 수 있습니다.

- 모델과 프로세서를 변환하려면 다음을 사용하는 것이 좋습니다:

```bash
python src/transformers/models/whisper/convert_openai_to_hf.py --checkpoint_path "" --pytorch_dump_folder_path "Arthur/whisper-3" --convert_preprocessor True
```
스크립트는 OpenAI 체크포인트에서 필요한 모든 매개변수를 자동으로 결정합니다. OpenAI 변환을 수행하려면 `tiktoken` 라이브러리를 설치해야 합니다.
라이브러리를 설치해야 OpenAI 토큰화기를 `tokenizers` 버전으로 변환할 수 있습니다.

이 모델은 [Arthur Zucker](https://huggingface.co/ArthurZ)에 의해 제공되었습니다. 이 모델의 Tensorflow 버전은 [amyeroberts](https://huggingface.co/amyeroberts)에 의해 제공되었습니다.
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

