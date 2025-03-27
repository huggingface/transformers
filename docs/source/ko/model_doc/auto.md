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

# Auto 클래스[[auto-classes]]

많은 경우, 사용하려는 아키텍처는 `from_pretrained()` 메소드에서 제공하는 사전 훈련된 모델의 이름이나 경로로부터 유추할 수 있습니다. AutoClasses는 이 작업을 위해 존재하며, 사전 학습된 모델 가중치/구성/단어사전에 대한 이름/경로를 제공하면 자동으로 관련 모델을 가져오도록 도와줍니다.

[`AutoConfig`], [`AutoModel`], [`AutoTokenizer`] 중 하나를 인스턴스화하면 해당 아키텍처의 클래스를 직접 생성합니다. 예를 들어,


```python
model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

위 코드는 [`BertModel`]의 인스턴스인 모델을 생성합니다.

각 작업에 대해 하나의 `AutoModel` 클래스가 있으며, 각각의 백엔드(PyTorch, TensorFlow 또는 Flax)에 해당하는 클래스가 존재합니다.

## 자동 클래스 확장[[extending-the-auto-classes]]

각 자동 클래스는 사용자의 커스텀 클래스로 확장될 수 있는 메소드를 가지고 있습니다. 예를 들어, `NewModel`이라는 커스텀 모델 클래스를 정의했다면, `NewModelConfig`를 준비한 후 다음과 같이 자동 클래스에 추가할 수 있습니다:

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

이후에는 일반적으로 자동 클래스를 사용하는 것처럼 사용할 수 있습니다!

<Tip warning={true}>

만약 `NewModelConfig`가 [`~transformers.PretrainedConfig`]의 서브클래스라면, 해당 `model_type` 속성이 등록할 때 사용하는 키(여기서는 `"new-model"`)와 동일하게 설정되어 있는지 확인하세요.

마찬가지로, `NewModel`이 [`PreTrainedModel`]의 서브클래스라면, 해당 `config_class` 속성이 등록할 때 사용하는 클래스(여기서는 `NewModelConfig`)와 동일하게 설정되어 있는지 확인하세요.

</Tip>

## AutoConfig[[transformers.AutoConfig]]

[[autodoc]] AutoConfig

## AutoTokenizer[[transformers.AutoTokenizer]]

[[autodoc]] AutoTokenizer

## AutoFeatureExtractor[[transformers.AutoFeatureExtractor]]

[[autodoc]] AutoFeatureExtractor

## AutoImageProcessor[[transformers.AutoImageProcessor]]

[[autodoc]] AutoImageProcessor

## AutoProcessor[[transformers.AutoProcessor]]

[[autodoc]] AutoProcessor

## 일반적인 모델 클래스[[generic-model-classes]]

다음 자동 클래스들은 특정 헤드 없이 기본 모델 클래스를 인스턴스화하는 데 사용할 수 있습니다.

### AutoModel[[transformers.AutoModel]]

[[autodoc]] AutoModel

### TFAutoModel[[transformers.TFAutoModel]]

[[autodoc]] TFAutoModel

### FlaxAutoModel[[transformers.FlaxAutoModel]]

[[autodoc]] FlaxAutoModel

## 일반적인 사전 학습 클래스[[generic-pretraining-classes]]

다음 자동 클래스들은 사전 훈련 헤드가 포함된 모델을 인스턴스화하는 데 사용할 수 있습니다.

### AutoModelForPreTraining[[transformers.AutoModelForPreTraining]]

[[autodoc]] AutoModelForPreTraining

### TFAutoModelForPreTraining[[transformers.TFAutoModelForPreTraining]]

[[autodoc]] TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining[[transformers.FlaxAutoModelForPreTraining]]

[[autodoc]] FlaxAutoModelForPreTraining

## 자연어 처리[[natural-language-processing]]

다음 자동 클래스들은 아래의 자연어 처리 작업에 사용할 수 있습니다.

### AutoModelForCausalLM[[transformers.AutoModelForCausalLM]]

[[autodoc]] AutoModelForCausalLM

### TFAutoModelForCausalLM[[transformers.TFAutoModelForCausalLM]]

[[autodoc]] TFAutoModelForCausalLM

### FlaxAutoModelForCausalLM[[transformers.FlaxAutoModelForCausalLM]]

[[autodoc]] FlaxAutoModelForCausalLM

### AutoModelForMaskedLM[[transformers.AutoModelForMaskedLM]]

[[autodoc]] AutoModelForMaskedLM

### TFAutoModelForMaskedLM[[transformers.TFAutoModelForMaskedLM]]

[[autodoc]] TFAutoModelForMaskedLM

### FlaxAutoModelForMaskedLM[[transformers.FlaxAutoModelForMaskedLM]]

[[autodoc]] FlaxAutoModelForMaskedLM

### AutoModelForMaskGeneration[[transformers.AutoModelForMaskGeneration]]

[[autodoc]] AutoModelForMaskGeneration

### TFAutoModelForMaskGeneration[[transformers.TFAutoModelForMaskGeneration]]

[[autodoc]] TFAutoModelForMaskGeneration

### AutoModelForSeq2SeqLM[[transformers.AutoModelForSeq2SeqLM]]

[[autodoc]] AutoModelForSeq2SeqLM

### TFAutoModelForSeq2SeqLM[[transformers.TFAutoModelForSeq2SeqLM]]

[[autodoc]] TFAutoModelForSeq2SeqLM

### FlaxAutoModelForSeq2SeqLM[[transformers.FlaxAutoModelForSeq2SeqLM]]

[[autodoc]] FlaxAutoModelForSeq2SeqLM

### AutoModelForSequenceClassification[[transformers.AutoModelForSequenceClassification]]

[[autodoc]] AutoModelForSequenceClassification

### TFAutoModelForSequenceClassification[[transformers.TFAutoModelForSequenceClassification]]

[[autodoc]] TFAutoModelForSequenceClassification

### FlaxAutoModelForSequenceClassification[[transformers.FlaxAutoModelForSequenceClassification]]

[[autodoc]] FlaxAutoModelForSequenceClassification

### AutoModelForMultipleChoice[[transformers.AutoModelForMultipleChoice]]

[[autodoc]] AutoModelForMultipleChoice

### TFAutoModelForMultipleChoice[[transformers.TFAutoModelForMultipleChoice]]

[[autodoc]] TFAutoModelForMultipleChoice

### FlaxAutoModelForMultipleChoice[[transformers.FlaxAutoModelForMultipleChoice]]

[[autodoc]] FlaxAutoModelForMultipleChoice

### AutoModelForNextSentencePrediction[[transformers.AutoModelForNextSentencePrediction]]

[[autodoc]] AutoModelForNextSentencePrediction

### TFAutoModelForNextSentencePrediction[[transformers.TFAutoModelForNextSentencePrediction]]

[[autodoc]] TFAutoModelForNextSentencePrediction

### FlaxAutoModelForNextSentencePrediction[[transformers.FlaxAutoModelForNextSentencePrediction]]

[[autodoc]] FlaxAutoModelForNextSentencePrediction

### AutoModelForTokenClassification[[transformers.AutoModelForTokenClassification]]

[[autodoc]] AutoModelForTokenClassification

### TFAutoModelForTokenClassification[[transformers.TFAutoModelForTokenClassification]]

[[autodoc]] TFAutoModelForTokenClassification

### FlaxAutoModelForTokenClassification[[transformers.FlaxAutoModelForTokenClassification]]

[[autodoc]] FlaxAutoModelForTokenClassification

### AutoModelForQuestionAnswering[[transformers.AutoModelForQuestionAnswering]]

[[autodoc]] AutoModelForQuestionAnswering

### TFAutoModelForQuestionAnswering[[transformers.TFAutoModelForQuestionAnswering]]

[[autodoc]] TFAutoModelForQuestionAnswering

### FlaxAutoModelForQuestionAnswering[[transformers.FlaxAutoModelForQuestionAnswering]]

[[autodoc]] FlaxAutoModelForQuestionAnswering

### AutoModelForTextEncoding[[transformers.AutoModelForTextEncoding]]

[[autodoc]] AutoModelForTextEncoding

### TFAutoModelForTextEncoding[[transformers.TFAutoModelForTextEncoding]]

[[autodoc]] TFAutoModelForTextEncoding

## 컴퓨터 비전[[computer-vision]]

다음 자동 클래스들은 아래의 컴퓨터 비전 작업에 사용할 수 있습니다.

### AutoModelForDepthEstimation[[transformers.AutoModelForDepthEstimation]]

[[autodoc]] AutoModelForDepthEstimation

### AutoModelForImageClassification[[transformers.AutoModelForImageClassification]]

[[autodoc]] AutoModelForImageClassification

### TFAutoModelForImageClassification[[transformers.TFAutoModelForImageClassification]]

[[autodoc]] TFAutoModelForImageClassification

### FlaxAutoModelForImageClassification[[transformers.FlaxAutoModelForImageClassification]]

[[autodoc]] FlaxAutoModelForImageClassification

### AutoModelForVideoClassification[[transformers.AutoModelForVideoClassification]]

[[autodoc]] AutoModelForVideoClassification

### AutoModelForKeypointDetection[[transformers.AutoModelForKeypointDetection]]

[[autodoc]] AutoModelForKeypointDetection

### AutoModelForMaskedImageModeling[[transformers.AutoModelForMaskedImageModeling]]

[[autodoc]] AutoModelForMaskedImageModeling

### TFAutoModelForMaskedImageModeling[[transformers.TFAutoModelForMaskedImageModeling]]

[[autodoc]] TFAutoModelForMaskedImageModeling

### AutoModelForObjectDetection[[transformers.AutoModelForObjectDetection]]

[[autodoc]] AutoModelForObjectDetection

### AutoModelForImageSegmentation[[transformers.AutoModelForImageSegmentation]]

[[autodoc]] AutoModelForImageSegmentation

### AutoModelForImageToImage[[transformers.AutoModelForImageToImage]]

[[autodoc]] AutoModelForImageToImage

### AutoModelForSemanticSegmentation[[transformers.AutoModelForSemanticSegmentation]]

[[autodoc]] AutoModelForSemanticSegmentation

### TFAutoModelForSemanticSegmentation[[transformers.TFAutoModelForSemanticSegmentation]]

[[autodoc]] TFAutoModelForSemanticSegmentation

### AutoModelForInstanceSegmentation[[transformers.AutoModelForInstanceSegmentation]]

[[autodoc]] AutoModelForInstanceSegmentation

### AutoModelForUniversalSegmentation[[transformers.AutoModelForUniversalSegmentation]]

[[autodoc]] AutoModelForUniversalSegmentation

### AutoModelForZeroShotImageClassification[[transformers.AutoModelForZeroShotImageClassification]]

[[autodoc]] AutoModelForZeroShotImageClassification

### TFAutoModelForZeroShotImageClassification[[transformers.TFAutoModelForZeroShotImageClassification]]

[[autodoc]] TFAutoModelForZeroShotImageClassification

### AutoModelForZeroShotObjectDetection[[transformers.AutoModelForZeroShotObjectDetection]]

[[autodoc]] AutoModelForZeroShotObjectDetection

## 오디오[[audio]]

다음 자동 클래스들은 아래의 오디오 작업에 사용할 수 있습니다.

### AutoModelForAudioClassification[[transformers.AutoModelForAudioClassification]]

[[autodoc]] AutoModelForAudioClassification

### TFAutoModelForAudioClassification[[transformers.TFAutoModelForAudioClassification]]

[[autodoc]] TFAutoModelForAudioClassification

### AutoModelForAudioFrameClassification[[transformers.AutoModelForAudioFrameClassification]]

[[autodoc]] AutoModelForAudioFrameClassification

### AutoModelForCTC[[transformers.AutoModelForCTC]]

[[autodoc]] AutoModelForCTC

### AutoModelForSpeechSeq2Seq[[transformers.AutoModelForSpeechSeq2Seq]]

[[autodoc]] AutoModelForSpeechSeq2Seq

### TFAutoModelForSpeechSeq2Seq[[transformers.TFAutoModelForSpeechSeq2Seq]]

[[autodoc]] TFAutoModelForSpeechSeq2Seq

### FlaxAutoModelForSpeechSeq2Seq[[transformers.FlaxAutoModelForSpeechSeq2Seq]]

[[autodoc]] FlaxAutoModelForSpeechSeq2Seq

### AutoModelForAudioXVector[[transformers.AutoModelForAudioXVector]]

[[autodoc]] AutoModelForAudioXVector

### AutoModelForTextToSpectrogram[[transformers.AutoModelForTextToSpectrogram]]

[[autodoc]] AutoModelForTextToSpectrogram

### AutoModelForTextToWaveform[[transformers.AutoModelForTextToWaveform]]

[[autodoc]] AutoModelForTextToWaveform

## 멀티모달[[multimodal]]

다음 자동 클래스들은 아래의 멀티모달 작업에 사용할 수 있습니다.

### AutoModelForTableQuestionAnswering[[transformers.AutoModelForTableQuestionAnswering]]

[[autodoc]] AutoModelForTableQuestionAnswering

### TFAutoModelForTableQuestionAnswering[[transformers.TFAutoModelForTableQuestionAnswering]]

[[autodoc]] TFAutoModelForTableQuestionAnswering

### AutoModelForDocumentQuestionAnswering[[transformers.AutoModelForDocumentQuestionAnswering]]

[[autodoc]] AutoModelForDocumentQuestionAnswering

### TFAutoModelForDocumentQuestionAnswering[[transformers.TFAutoModelForDocumentQuestionAnswering]]

[[autodoc]] TFAutoModelForDocumentQuestionAnswering

### AutoModelForVisualQuestionAnswering[[transformers.AutoModelForVisualQuestionAnswering]]

[[autodoc]] AutoModelForVisualQuestionAnswering

### AutoModelForVision2Seq[[transformers.AutoModelForVision2Seq]]

[[autodoc]] AutoModelForVision2Seq

### TFAutoModelForVision2Seq[[transformers.TFAutoModelForVision2Seq]]

[[autodoc]] TFAutoModelForVision2Seq

### FlaxAutoModelForVision2Seq[[transformers.FlaxAutoModelForVision2Seq]]

[[autodoc]] FlaxAutoModelForVision2Seq
