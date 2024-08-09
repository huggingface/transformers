# الفئات التلقائية 

في العديد من الحالات، يمكن تخمين البنية التي تريد استخدامها من اسم أو مسار النموذج المُدرب مسبقًا الذي تقوم بتمريره إلى طريقة `from_pretrained()`. والفئات التلقائية موجودة للقيام بهذه المهمة نيابة عنك، بحيث يمكنك استرجاع النموذج ذي الصلة تلقائيًا بناءً على اسم/مسار الأوزان المُدربة مسبقًا/الإعدادات/المفردات.

إن إنشاء مثيل واحد من [`AutoConfig`]`[`، و [`AutoModel`]`[`، و [`AutoTokenizer`]`[` سيقوم مباشرة بإنشاء فئة البنية ذات الصلة. على سبيل المثال:

```python
model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

سيقوم بإنشاء نموذج يكون مثالاً لفئة [`BertModel`]`[`.

هناك فئة واحدة من `AutoModel` لكل مهمة، ولكل منصة خلفية (PyTorch أو TensorFlow أو Flax).

## توسيع الفئات التلقائية

تمتلك كل فئة تلقائية طريقة لتوسيع نطاقها باستخدام الفئات المخصصة الخاصة بك. على سبيل المثال، إذا قمت بتعريف فئة مخصصة للنموذج باسم `NewModel`، فتأكد من وجود `NewModelConfig`، ثم يمكنك إضافة تلك الفئات إلى الفئات التلقائية على النحو التالي:

```python
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```

بعد ذلك، ستتمكن من استخدام الفئات التلقائية كما تفعل عادة!

<Tip warning={true}>

إذا كانت فئة `NewModelConfig` هي فئة فرعية من [`~transformers.PretrainedConfig`]`[`، فتأكد من تعيين سمة `model_type` الخاصة بها إلى نفس المفتاح الذي تستخدمه عند تسجيل الإعدادات (هنا `"new-model"`).

وبالمثل، إذا كانت فئة `NewModel` هي فئة فرعية من [`PreTrainedModel`]`[`، فتأكد من تعيين سمة `config_class` الخاصة بها إلى نفس الفئة التي تستخدمها عند تسجيل النموذج (هنا `NewModelConfig`).

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

## فئات النماذج العامة

تتوفر الفئات التلقائية التالية لإنشاء فئة نموذج أساسي بدون رأس محدد.

### AutoModel

[[autodoc]] AutoModel

### TFAutoModel

[[autodoc]] TFAutoModel

### FlaxAutoModel

[[autodoc]] FlaxAutoModel

## فئات التدريب التلقائي العامة

تتوفر الفئات التلقائية التالية لإنشاء نموذج برأس تدريب تلقائي.

### AutoModelForPreTraining

[[autodoc]] AutoModelForPreTraining

### TFAutoModelForPreTraining

[[autodoc]] TFAutoModelForPreTraining

### FlaxAutoModelForPreTraining

[[autodoc]] FlaxAutoModelForPreTraining

## معالجة اللغات الطبيعية

تتوفر الفئات التلقائية التالية لمهام معالجة اللغات الطبيعية التالية.

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

## رؤية الكمبيوتر

تتوفر الفئات التلقائية التالية لمهام رؤية الكمبيوتر التالية.

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

### AutoModelForKeypointDetection

[[autodoc]] AutoModelForKeypointDetection

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

## الصوت

تتوفر الفئات التلقائية التالية لمهام الصوت التالية.

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

## متعدد الوسائط

تتوفر الفئات التلقائية التالية لمهام متعددة الوسائط التالية.

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