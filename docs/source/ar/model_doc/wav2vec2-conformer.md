# Wav2Vec2-Conformer

## نظرة عامة

تمت إضافة Wav2Vec2-Conformer إلى نسخة محدثة من [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) بواسطة Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino.

يمكن العثور على النتائج الرسمية للنموذج في الجدول 3 والجدول 4 من الورقة.

تم إصدار أوزان Wav2Vec2-Conformer بواسطة فريق Meta AI ضمن [مكتبة Fairseq](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models).

تمت المساهمة بهذا النموذج بواسطة [patrickvonplaten](https://huggingface.co/patrickvonplaten).

يمكن العثور على الكود الأصلي [هنا](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec).

ملاحظة: أصدرت Meta (FAIR) إصدارًا جديدًا من [Wav2Vec2-BERT 2.0](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) - وهو مدرب مسبقًا على 4.5 مليون ساعة من الصوت. نوصي بشكل خاص باستخدامه لمهام الضبط الدقيق، على سبيل المثال كما هو موضح في [هذا الدليل](https://huggingface.co/blog/fine-tune-w2v2-bert).

## نصائح الاستخدام

- يتبع Wav2Vec2-Conformer نفس بنية Wav2Vec2، ولكنه يستبدل كتلة "الانتباه" بكتلة "Conformer" كما هو مقدم في [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100).

- بالنسبة لنفس عدد الطبقات، يتطلب Wav2Vec2-Conformer عددًا أكبر من المعلمات مقارنة بـ Wav2Vec2، ولكنه يوفر أيضًا معدل خطأ كلمة محسن.

- يستخدم Wav2Vec2-Conformer نفس المحلل الرمزي ومستخرج الميزات مثل Wav2Vec2.

- يمكن لـ Wav2Vec2-Conformer استخدام إما عدم وجود تضمينات موضع نسبي، أو تضمينات موضع مثل Transformer-XL، أو تضمينات موضع دوارة عن طريق تعيين `config.position_embeddings_type` بشكل صحيح.

## الموارد

- [دليل مهمة تصنيف الصوت](../tasks/audio_classification)

- [دليل مهمة التعرف التلقائي على الكلام](../tasks/asr)

## Wav2Vec2ConformerConfig

[[autodoc]] Wav2Vec2ConformerConfig

## المخرجات الخاصة بـ Wav2Vec2Conformer

[[autodoc]] models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerForPreTrainingOutput

## Wav2Vec2ConformerModel

[[autodoc]] Wav2Vec2ConformerModel

- forward

## Wav2Vec2ConformerForCTC

[[autodoc]] Wav2Vec2ConformerForCTC

- forward

## Wav2Vec2ConformerForSequenceClassification

[[autodoc]] Wav2Vec2ConformerForSequenceClassification

- forward

## Wav2Vec2ConformerForAudioFrameClassification

[[autodoc]] Wav2Vec2ConformerForAudioFrameClassification

- forward

## Wav2Vec2ConformerForXVector

[[autodoc]] Wav2Vec2ConformerForXVector

- forward

## Wav2Vec2ConformerForPreTraining

[[autodoc]] Wav2Vec2ConformerForPreTraining

- forward