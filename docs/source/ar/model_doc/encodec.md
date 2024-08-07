# EnCodec

## نظرة عامة
تم اقتراح نموذج EnCodec neural codec model في ورقة "High Fidelity Neural Audio Compression" بواسطة Alexandre Défossez وJade Copet وGabriel Synnaeve وYossi Adi.

ملخص الورقة البحثية هو كما يلي:

*نحن نقدم أحدث تقنيات الترميز الصوتي في الوقت الفعلي وعالي الدقة والذي يعتمد على الشبكات العصبية. ويتكون من بنية تدفق الترميز فك الترميز مع مساحة كمية خفية يتم تدريبها بطريقة متكاملة. نقوم بتبسيط وتسريع التدريب باستخدام معارض متعددة النطاقات للتحليل الطيفي والتي تقلل بشكل فعال من التشوهات وتنتج عينات عالية الجودة. كما نقدم آلية موازنة خسارة جديدة لتثبيت التدريب: وزن الخسارة الآن يحدد نسبة التدرج الكلي الذي يجب أن تمثله، وبالتالي فصل اختيار هذا الفرط المعلمي عن النطاق النموذجي للخسارة. وأخيرًا، ندرس كيف يمكن استخدام نماذج المحول الخفيفة لضغط التمثيل الذي تم الحصول عليه بنسبة تصل إلى 40%، مع الحفاظ على سرعة أسرع من الوقت الفعلي. نقدم وصفًا تفصيليًا لخيارات التصميم الرئيسية للنموذج المقترح بما في ذلك: الهدف من التدريب والتغييرات المعمارية ودراسة لوظائف الخسارة الإدراكية المختلفة. نقدم تقييمًا ذاتيًا واسع النطاق (اختبارات MUSHRA) جنبًا إلى جنب مع دراسة إبطال لأحجام النطاق ومجالات الصوت المختلفة، بما في ذلك الكلام والخطاب الصاخب والصدى والموسيقى. تفوق طريقتنا طرق الخط الأساسي عبر جميع الإعدادات المقيمين، مع مراعاة كل من الصوت أحادي القناة 24 كيلو هرتز والستيريو 48 كيلو هرتز.*

تمت المساهمة في هذا النموذج بواسطة Matthijs وPatrick Von Platen وArthur Zucker. يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/encodec).

## مثال على الاستخدام
فيما يلي مثال سريع يوضح كيفية ترميز وفك ترميز الصوت باستخدام هذا النموذج:

```python
>>> from datasets import load_dataset, Audio
>>> from transformers import EncodecModel, AutoProcessor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> model = EncodecModel.from_pretrained("facebook/encodec_24khz")
>>> processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
>>> # أو المكافئ مع تمريرة للأمام
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

## EncodecConfig

[[autodoc]] EncodecConfig

## EncodecFeatureExtractor

[[autodoc]] EncodecFeatureExtractor

- __call__

## EncodecModel

[[autodoc]] EncodecModel

- decode

- encode

- forward