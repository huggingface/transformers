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

多くの場合、`from_pretrained()`メソッドに与えられた事前学習済みモデルの名前やパスから、使用したいアーキテクチャを推測することができます。自動クラスはこの仕事をあなたに代わって行うためにここにありますので、事前学習済みの重み/設定/語彙への名前/パスを与えると自動的に関連するモデルを取得できます。

[`AutoConfig`]、[`AutoModel`]、[`AutoTokenizer`]のいずれかをインスタンス化すると、関連するアーキテクチャのクラスが直接作成されます。例えば、

```python
model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

これは[`BertModel`]のインスタンスであるモデルを作成します。

各タスクごと、そして各バックエンド（PyTorch、TensorFlow、またはFlax）ごとに`AutoModel`のクラスが存在します。

## 自動クラスの拡張

それぞれの自動クラスには、カスタムクラスで拡張するためのメソッドがあります。例えば、`NewModel`というモデルのカスタムクラスを定義した場合、`NewModelConfig`を確保しておけばこのようにして自動クラスに追加することができます：

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

その後、通常どおりauto classesを使用することができるようになります！

<Tip warning={true}>

あなたの`NewModelConfig`が[`~transformers.PretrainedConfig`]のサブクラスである場合、その`model_type`属性がコンフィグを登録するときに使用するキー（ここでは`"new-model"`）と同じに設定されていることを確認してください。

同様に、あなたの`NewModel`が[`PreTrainedModel`]のサブクラスである場合、その`config_class`属性がモデルを登録する際に使用するクラス（ここでは`NewModelConfig`）と同じに設定されていることを確認してください。

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

以下の自動クラスは、特定のヘッドを持たないベースモデルクラスをインスタンス化するために利用可能です。

### AutoModel

[[autodoc]] AutoModel

### TFAutoModel

[[autodoc]] TFAutoModel

### FlaxAutoModel

[[autodoc]] FlaxAutoModel

## Generic pretraining classes

以下の自動クラスは、事前学習ヘッドを持つモデルをインスタンス化するために利用可能です。

### AutoModelForPreTraining

[[autodoc]] AutoModelForPreTraining

### TFAutoModelForPreTraining

[[autodoc]] TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining

[[autodoc]] FlaxAutoModelForPreTraining

## Natural Language Processing

以下の自動クラスは、次の自然言語処理タスクに利用可能です。

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

以下の自動クラスは、次のコンピュータービジョンタスクに利用可能です。

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

以下の自動クラスは、次の音声タスクに利用可能です。

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

以下の自動クラスは、次のマルチモーダルタスクに利用可能です。

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

### AutoModelForImageTextToText

[[autodoc]] AutoModelForImageTextToText
