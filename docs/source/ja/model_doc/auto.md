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

# Auto Classes

多くの場合、使用するアーキテクチャは、事前トレーニングされたモデルの名前またはパスから推測できます。
`from_pretrained()` メソッドに供給しています。 AutoClasses はこの仕事をあなたのために行います。
事前トレーニングされた重み/構成/語彙への名前/パスを指定すると、関連するモデルが自動的に取得されます。

[`AutoConfig`]、[`AutoModel`]、および
[`AutoTokenizer`] は関連するアーキテクチャのクラスを直接作成します。例えば

```python
model = AutoModel.from_pretrained("bert-base-cased")
```

[`BertModel`] のインスタンスであるモデルを作成します。

各タスクおよび各バックエンド (PyTorch、TensorFlow、または Flax) に対して`AutoModel`クラスが 1 つあります。

## Extending the Auto Classes

各自動クラスには、カスタム クラスで拡張できるメソッドがあります。たとえば、次のように定義した場合、
モデル `NewModel` のカスタム クラス。`NewModelConfig` があることを確認してください。そうすれば、それらを auto に追加できます。
このようなクラス:

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

これで、通常と同じように auto クラスを使用できるようになります。

<Tip warning={true}>

`NewModelConfig` が [`~transformer.PretrainedConfig`] のサブクラスである場合、そのサブクラスであることを確認してください。
`model_type` 属性は、構成を登録するときに使用するのと同じキーに設定されます (ここでは `"new-model"`)。

同様に、`NewModel` が [`PreTrainedModel`] のサブクラスである場合、そのサブクラスが
`config_class` 属性は、モデルの登録時に使用するのと同じクラスに設定されます (ここでは
`NewModelConfig`)。

</Tip>

## AutoConfig

[[autodoc]] AutoConfig

## AutoTokenizer

[[autodoc]] AutoTokenizer

## AutoFeatureExtractor

[[autodoc]] AutoFeatureExtractor

## AutoImageProcessor

[[autodoc]] AutoImageProcessor

## AutoProcessor

[[autodoc]] AutoProcessor

## Generic model classes

次の自動クラスは、特定のヘッドなしで基本モデル クラスをインスタンス化するために使用できます。

### AutoModel

[[autodoc]] AutoModel

### TFAutoModel

[[autodoc]] TFAutoModel

### FlaxAutoModel

[[autodoc]] FlaxAutoModel

## Generic pretraining classes

次の自動クラスは、事前トレーニング ヘッドを使用してモデルをインスタンス化するために使用できます。

### AutoModelForPreTraining

[[autodoc]] AutoModelForPreTraining

### TFAutoModelForPreTraining

[[autodoc]] TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining

[[autodoc]] FlaxAutoModelForPreTraining

## Natural Language Processing

次の自動クラスは、次の自然言語処理タスクで使用できます。

### AutoModelForCausalLM

[[autodoc]] AutoModelForCausalLM

### TFAutoModelForCausalLM

[[autodoc]] TFAutoModelForCausalLM

### FlaxAutoModelForCausalLM

[[autodoc]] FlaxAutoModelForCausalLM

### AutoModelForMaskedLM

[[autodoc]] AutoModelForMaskedLM

### TFAutoModelForMaskedLM

[[autodoc]] TFAutoModelForMaskedLM

### FlaxAutoModelForMaskedLM

[[autodoc]] FlaxAutoModelForMaskedLM

### AutoModelForMaskGeneration

[[autodoc]] AutoModelForMaskGeneration

### TFAutoModelForMaskGeneration

[[autodoc]] TFAutoModelForMaskGeneration

### AutoModelForSeq2SeqLM

[[autodoc]] AutoModelForSeq2SeqLM

### TFAutoModelForSeq2SeqLM

[[autodoc]] TFAutoModelForSeq2SeqLM

### FlaxAutoModelForSeq2SeqLM

[[autodoc]] FlaxAutoModelForSeq2SeqLM

### AutoModelForSequenceClassification

[[autodoc]] AutoModelForSequenceClassification

### TFAutoModelForSequenceClassification

[[autodoc]] TFAutoModelForSequenceClassification

### FlaxAutoModelForSequenceClassification

[[autodoc]] FlaxAutoModelForSequenceClassification

### AutoModelForMultipleChoice

[[autodoc]] AutoModelForMultipleChoice

### TFAutoModelForMultipleChoice

[[autodoc]] TFAutoModelForMultipleChoice

### FlaxAutoModelForMultipleChoice

[[autodoc]] FlaxAutoModelForMultipleChoice

### AutoModelForNextSentencePrediction

[[autodoc]] AutoModelForNextSentencePrediction

### TFAutoModelForNextSentencePrediction

[[autodoc]] TFAutoModelForNextSentencePrediction

### FlaxAutoModelForNextSentencePrediction

[[autodoc]] FlaxAutoModelForNextSentencePrediction

### AutoModelForTokenClassification

[[autodoc]] AutoModelForTokenClassification

### TFAutoModelForTokenClassification

[[autodoc]] TFAutoModelForTokenClassification

### FlaxAutoModelForTokenClassification

[[autodoc]] FlaxAutoModelForTokenClassification

### AutoModelForQuestionAnswering

[[autodoc]] AutoModelForQuestionAnswering

### TFAutoModelForQuestionAnswering

[[autodoc]] TFAutoModelForQuestionAnswering

### FlaxAutoModelForQuestionAnswering

[[autodoc]] FlaxAutoModelForQuestionAnswering

### AutoModelForTextEncoding

[[autodoc]] AutoModelForTextEncoding

### TFAutoModelForTextEncoding

[[autodoc]] TFAutoModelForTextEncoding

## Computer vision

次の自動クラスは、次のコンピューター ビジョン タスクで使用できます。

### AutoModelForDepthEstimation

[[autodoc]] AutoModelForDepthEstimation

### AutoModelForImageClassification

[[autodoc]] AutoModelForImageClassification

### TFAutoModelForImageClassification

[[autodoc]] TFAutoModelForImageClassification

### FlaxAutoModelForImageClassification

[[autodoc]] FlaxAutoModelForImageClassification

### AutoModelForVideoClassification

[[autodoc]] AutoModelForVideoClassification

### AutoModelForMaskedImageModeling

[[autodoc]] AutoModelForMaskedImageModeling

### TFAutoModelForMaskedImageModeling

[[autodoc]] TFAutoModelForMaskedImageModeling

### AutoModelForObjectDetection

[[autodoc]] AutoModelForObjectDetection

### AutoModelForImageSegmentation

[[autodoc]] AutoModelForImageSegmentation

### AutoModelForImageToImage

[[autodoc]] AutoModelForImageToImage

### AutoModelForSemanticSegmentation

[[autodoc]] AutoModelForSemanticSegmentation

### TFAutoModelForSemanticSegmentation

[[autodoc]] TFAutoModelForSemanticSegmentation

### AutoModelForInstanceSegmentation

[[autodoc]] AutoModelForInstanceSegmentation

### AutoModelForUniversalSegmentation

[[autodoc]] AutoModelForUniversalSegmentation

### AutoModelForZeroShotImageClassification

[[autodoc]] AutoModelForZeroShotImageClassification

### TFAutoModelForZeroShotImageClassification

[[autodoc]] TFAutoModelForZeroShotImageClassification

### AutoModelForZeroShotObjectDetection

[[autodoc]] AutoModelForZeroShotObjectDetection

## Audio

次の自動クラスは、次のオーディオ タスクで使用できます。

### AutoModelForAudioClassification

[[autodoc]] AutoModelForAudioClassification

### AutoModelForAudioFrameClassification

[[autodoc]] TFAutoModelForAudioClassification

### TFAutoModelForAudioFrameClassification

[[autodoc]] AutoModelForAudioFrameClassification

### AutoModelForCTC

[[autodoc]] AutoModelForCTC

### AutoModelForSpeechSeq2Seq

[[autodoc]] AutoModelForSpeechSeq2Seq

### TFAutoModelForSpeechSeq2Seq

[[autodoc]] TFAutoModelForSpeechSeq2Seq

### FlaxAutoModelForSpeechSeq2Seq

[[autodoc]] FlaxAutoModelForSpeechSeq2Seq

### AutoModelForAudioXVector

[[autodoc]] AutoModelForAudioXVector

### AutoModelForTextToSpectrogram

[[autodoc]] AutoModelForTextToSpectrogram

### AutoModelForTextToWaveform

[[autodoc]] AutoModelForTextToWaveform

## Multimodal

次の自動クラスは、次のマルチモーダル タスクで使用できます。

### AutoModelForTableQuestionAnswering

[[autodoc]] AutoModelForTableQuestionAnswering

### TFAutoModelForTableQuestionAnswering

[[autodoc]] TFAutoModelForTableQuestionAnswering

### AutoModelForDocumentQuestionAnswering

[[autodoc]] AutoModelForDocumentQuestionAnswering

### TFAutoModelForDocumentQuestionAnswering

[[autodoc]] TFAutoModelForDocumentQuestionAnswering

### AutoModelForVisualQuestionAnswering

[[autodoc]] AutoModelForVisualQuestionAnswering

### AutoModelForVision2Seq

[[autodoc]] AutoModelForVision2Seq

### TFAutoModelForVision2Seq

[[autodoc]] TFAutoModelForVision2Seq

### FlaxAutoModelForVision2Seq

[[autodoc]] FlaxAutoModelForVision2Seq
