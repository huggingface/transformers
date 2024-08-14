# مخرجات النموذج

تمتلك جميع النماذج مخرجات هي عبارة عن حالات من الفئات الفرعية لـ [`~utils.ModelOutput`]. وهذه عبارة عن هياكل بيانات تحتوي على جميع المعلومات التي يرجعها النموذج، ولكن يمكن استخدامها أيضًا كمجموعات أو قواميس.

دعونا نرى كيف يبدو هذا في مثال:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # حجم الدفعة 1
outputs = model(**inputs, labels=labels)
```

إن كائن `outputs` هو [`~ modeling_outputs.SequenceClassifierOutput`]، كما يمكننا أن نرى في توثيق تلك الفئة أدناه، وهذا يعني أنه يحتوي على خاصية `loss` اختيارية، و`logits`، و`hidden_states` اختيارية، و`attentions` اختيارية. هنا لدينا `loss` لأننا مررنا `labels`، لكننا لا نملك `hidden_states` و`attentions` لأننا لم نمرر `output_hidden_states=True` أو `output_attentions=True`.

<Tip>

عند تمرير `output_hidden_states=True`، قد تتوقع أن تتطابق `outputs.hidden_states[-1]` مع `outputs.last_hidden_states` بالضبط. ومع ذلك، هذا ليس هو الحال دائمًا. تطبق بعض النماذج التطبيع أو المعالجة اللاحقة على الحالة المخفية الأخيرة عند إرجاعها.

</Tip>

يمكنك الوصول إلى كل خاصية كما تفعل عادةً، وإذا لم ترجع تلك الخاصية من قبل النموذج، فستحصل على `None`. هنا على سبيل المثال، `outputs.loss` هي الخسارة التي يحسبها النموذج، و`outputs.attentions` هي `None`.

عند اعتبار كائن `outputs` الخاص بنا على أنه مجموعة، فإنه لا يأخذ في الاعتبار سوى الخصائص التي لا تحتوي على قيم `None`. هنا على سبيل المثال، لديه عنصرين، `loss` ثم `logits`، لذا

```python
outputs[:2]
```

سترجع المجموعة `(outputs.loss, outputs.logits)` على سبيل المثال.

عند اعتبار كائن `outputs` الخاص بنا كقاموس، فإنه لا يأخذ في الاعتبار سوى الخصائص التي لا تحتوي على قيم `None`. هنا على سبيل المثال، لديه مفتاحين هما `loss` و`logits`.

نوثق هنا مخرجات النموذج العامة التي يستخدمها أكثر من نوع نموذج واحد. يتم توثيق أنواع الإخراج المحددة على صفحة النموذج المقابلة.

## ModelOutput

[[autodoc]] utils.ModelOutput

- to_tuple

## BaseModelOutput

[[autodoc]] modeling_outputs.BaseModelOutput

## BaseModelOutputWithPooling

[[autodoc]] modeling_outputs.BaseModelOutputWithPooling

## BaseModelOutputWithCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithCrossAttentions

## BaseModelOutputWithPoolingAndCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions

## BaseModelOutputWithPast

[[autodoc]] modeling_outputs.BaseModelOutputWithPast

## BaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithPastAndCrossAttentions

## Seq2SeqModelOutput

[[autodoc]] modeling_outputs.Seq2SeqModelOutput

## CausalLMOutput

[[autodoc]] modeling_outputs.CausalLMOutput

## CausalLMOutputWithCrossAttentions

[[autodoc]] modeling_outputs.CausalLMOutputWithCrossAttentions

## CausalLMOutputWithPast

[[autodoc]] modeling_outputs.CausalLMOutputWithPast

## MaskedLMOutput

[[autodoc]] modeling_outputs.MaskedLMOutput

## Seq2SeqLMOutput

[[autodoc]] modeling_outputs.Seq2SeqLMOutput

## NextSentencePredictorOutput

[[autodoc]] modeling_outputs.NextSentencePredictorOutput

## SequenceClassifierOutput

[[autodoc]] modeling_outputs.SequenceClassifierOutput

## Seq2SeqSequenceClassifierOutput

[[autodoc]] modeling_outputs.Seq2SeqSequenceClassifierOutput

## MultipleChoiceModelOutput

[[autodoc]] modeling_outputs.MultipleChoiceModelOutput

## TokenClassifierOutput

[[autodoc]] modeling_outputs.TokenClassifierOutput

## QuestionAnsweringModelOutput

[[autodoc]] modeling_outputs.QuestionAnsweringModelOutput

## Seq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_outputs.Seq2SeqQuestionAnsweringModelOutput

## Seq2SeqSpectrogramOutput

[[autodoc]] modeling_outputs.Seq2SeqSpectrogramOutput

## SemanticSegmenterOutput

[[autodoc]] modeling_outputs.SemanticSegmenterOutput

## ImageClassifierOutput

[[autodoc]] modeling_outputs.ImageClassifierOutput

## ImageClassifierOutputWithNoAttention


[[autodoc]] modeling_outputs.ImageClassifierOutputWithNoAttention

## DepthEstimatorOutput

[[autodoc]] modeling_outputs.DepthEstimatorOutput

## Wav2Vec2BaseModelOutput

[[autodoc]] modeling_outputs.Wav2Vec2BaseModelOutput

## XVectorOutput

[[autodoc]] modeling_outputs.XVectorOutput

## Seq2SeqTSModelOutput

[[autodoc]] modeling_outputs.Seq2SeqTSModelOutput

## Seq2SeqTSPredictionOutput

[[autodoc]] modeling_outputs.Seq2SeqTSPredictionOutput

## SampleTSPredictionOutput

[[autodoc]] modeling_outputs.SampleTSPredictionOutput

## TFBaseModelOutput

[[autodoc]] modeling_tf_outputs.TFBaseModelOutput

## TFBaseModelOutputWithPooling

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPooling

## TFBaseModelOutputWithPoolingAndCrossAttentions

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions

## TFBaseModelOutputWithPast

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPast

## TFBaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPastAndCrossAttentions

## TFSeq2SeqModelOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqModelOutput

## TFCausalLMOutput

[[autodoc]] modeling_tf_outputs.TFCausalLMOutput

## TFCausalLMOutputWithCrossAttentions

[[autodoc]] modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions

## TFCausalLMOutputWithPast

[[autodoc]] modeling_tf_outputs.TFCausalLMOutputWithPast

## TFMaskedLMOutput

[[autodoc]] modeling_tf_outputs.TFMaskedLMOutput

## TFSeq2SeqLMOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqLMOutput

## TFNextSentencePredictorOutput

[[autodoc]] modeling_tf_outputs.TFNextSentencePredictorOutput

## TFSequenceClassifierOutput

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutput

## TFSeq2SeqSequenceClassifierOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqSequenceClassifierOutput

## TFMultipleChoiceModelOutput

[[autodoc]] modeling_tf_outputs.TFMultipleChoiceModelOutput

## TFTokenClassifierOutput

[[autodoc]] modeling_tf_outputs.TFTokenClassifierOutput

## TFQuestionAnsweringModelOutput

[[autodoc]] modeling_tf_outputs.TFQuestionAnsweringModelOutput

## TFSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqQuestionAnsweringModelOutput

## FlaxBaseModelOutput

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutput

## FlaxBaseModelOutputWithPast

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPast

## FlaxBaseModelOutputWithPooling

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPooling

## FlaxBaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions

## FlaxSeq2SeqModelOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqModelOutput

## FlaxCausalLMOutputWithCrossAttentions

[[autodoc]] modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions

## FlaxMaskedLMOutput

[[autodoc]] modeling_flax_outputs.FlaxMaskedLMOutput

## FlaxSeq2SeqLMOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqLMOutput

## FlaxNextSentencePredictorOutput

[[autodoc]] modeling_flax_outputs.FlaxNextSentencePredictorOutput

## FlaxSequenceClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxSequenceClassifierOutput

## FlaxSeq2SeqSequenceClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqSequenceClassifierOutput

## FlaxMultipleChoiceModelOutput

[[autodoc]] modeling_flax_outputs.FlaxMultipleChoiceModelOutput

## FlaxTokenClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxTokenClassifierOutput

## FlaxQuestionAnsweringModelOutput

[[autodoc]] modeling_flax_outputs.FlaxQuestionAnsweringModelOutput

## FlaxSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput