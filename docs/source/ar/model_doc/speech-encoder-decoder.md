# نماذج الترميز فك الترميز الصوتي 

يمكن استخدام [`SpeechEncoderDecoderModel`] لتهيئة نموذج تحويل الكلام إلى نص باستخدام أي نموذج ترميز ذاتي التعلم مسبقًا للكلام كترميز (*e.g.* [Wav2Vec2](wav2vec2)، [Hubert](hubert)) وأي نموذج توليدي مسبق التدريب كفك تشفير.

وقد تم إثبات فعالية تهيئة النماذج التسلسلية للكلام إلى تسلسلات نصية باستخدام نقاط تفتيش مسبقة التدريب للتعرف على الكلام وترجمته على سبيل المثال في [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) بواسطة Changhan Wang وAnne Wu وJuan Pino وAlexei Baevski وMichael Auli وAlexis Conneau.

يمكن الاطلاع على مثال حول كيفية استخدام [`SpeechEncoderDecoderModel`] للاستنتاج في [Speech2Text2](speech_to_text_2).

## تهيئة `SpeechEncoderDecoderModel` بشكل عشوائي من تكوينات النموذج.

يمكن تهيئة [`SpeechEncoderDecoderModel`] بشكل عشوائي من تكوين الترميز وفك الترميز. في المثال التالي، نوضح كيفية القيام بذلك باستخدام تكوين [`Wav2Vec2Model`] الافتراضي للترميز
وتكوين [`BertForCausalLM`] الافتراضي لفك الترميز.

```python
>>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

>>> config_encoder = Wav2Vec2Config()
>>> config_decoder = BertConfig()

>>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = SpeechEncoderDecoderModel(config=config)
```

## تهيئة `SpeechEncoderDecoderModel` من ترميز وفك تشفير مُدرَّب مسبقًا.

يمكن تهيئة [`SpeechEncoderDecoderModel`] من نقطة تفتيش ترميز مُدرَّب مسبقًا ونقطة تفتيش فك تشفير مُدرَّب مسبقًا. لاحظ أن أي نموذج كلام قائم على Transformer، مثل [Wav2Vec2](wav2vec2) أو [Hubert](hubert) يمكن أن يعمل كترميز، في حين يمكن استخدام كل من النماذج الذاتية الترميز المُدرَّبة مسبقًا، مثل BERT، ونماذج اللغة السببية المُدرَّبة مسبقًا، مثل GPT2، بالإضافة إلى الجزء المدرَّب مسبقًا من فك تشفير نماذج التسلسل إلى تسلسل، مثل فك تشفير BART، كفك تشفير.

اعتمادًا على البنية التي تختارها كفك تشفير، قد يتم تهيئة طبقات الاهتمام المتقاطع بشكل عشوائي.

تتطلب تهيئة [`SpeechEncoderDecoderModel`] من نقطة تفتيش ترميز وفك تشفير مُدرَّب مسبقًا تهيئة النموذج الدقيقة على مهمة أسفل النهر، كما هو موضح في [منشور المدونة *Warm-starting-encoder-decoder*](https://huggingface.co/blog/warm-starting-encoder-decoder).

للقيام بذلك، توفر فئة `SpeechEncoderDecoderModel` طريقة [`SpeechEncoderDecoderModel.from_encoder_decoder_pretrained`].

```python
>>> from transformers import SpeechEncoderDecoderModel

>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/hubert-large-ll60k", "google-bert/bert-base-uncased"
... )
```

## تحميل نقطة تفتيش `SpeechEncoderDecoderModel` موجودة والقيام بالاستدلال.

لتحميل نقاط تفتيش مُدرَّبة دقيقة من فئة `SpeechEncoderDecoderModel`، توفر [`SpeechEncoderDecoderModel`] طريقة `from_pretrained(...)` مثل أي بنية نموذج أخرى في Transformers.

للقيام بالاستدلال، يمكنك استخدام طريقة [`generate`]، والتي تسمح بتوليد النص بشكل تلقائي. تدعم هذه الطريقة أشكالًا مختلفة من فك التشفير، مثل الجشع، وبحث الشعاع، وأخذ العينات متعددة الحدود.

```python
>>> from transformers import Wav2Vec2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> # تحميل نموذج ترجمة كلام مُدرَّب دقيق ومعالج مطابق
>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
>>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

>>> # لنقم بالاستدلال على قطعة من الكلام باللغة الإنجليزية (والتي سنترجمها إلى الألمانية)
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # توليد النسخ النصي تلقائيًا (يستخدم فك التشفير الجشع بشكل افتراضي)
>>> generated_ids = model.generate(input_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.
```

## التدريب 

بمجرد إنشاء النموذج، يمكن ضبطه بشكل دقيق مثل BART أو T5 أو أي نموذج ترميز وفك تشفير آخر على مجموعة بيانات من أزواج (كلام، نص).

كما ترى، يلزم إدخالان فقط للنموذج من أجل حساب الخسارة: `input_values` (وهي إدخالات الكلام) و`labels` (وهي `input_ids` للتسلسل المستهدف المشفر).

```python
>>> from transformers import AutoTokenizer, AutoFeatureExtractor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset

>>> encoder_id = "facebook/wav2vec2-base-960h" # ترميز النموذج الصوتي
>>> decoder_id = "google-bert/bert-base-uncased" # فك تشفير النص

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
>>> tokenizer = AutoTokenizer.from_pretrained(decoder_id)
>>> # دمج الترميز المُدرَّب مسبقًا وفك الترميز المُدرَّب مسبقًا لتشكيل نموذج تسلسل إلى تسلسل
>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id)

>>> model.config.decoder_start_token_id = tokenizer.cls_token_
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> # تحميل إدخال صوتي ومعالجته مسبقًا (تطبيع المتوسط/الانحراف المعياري إلى 0/1)
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> # تحميل النسخ النصي المقابل له وتشفيره لإنشاء العلامات
>>> labels = tokenizer(ds[0]["text"], return_tensors="pt").input_ids

>>> # تقوم دالة التقديم تلقائيًا بإنشاء decoder_input_ids الصحيحة
>>> loss = model(input_values=input_values, labels=labels).loss
>>> loss.backward()
```

## SpeechEncoderDecoderConfig

[[autodoc]] SpeechEncoderDecoderConfig

## SpeechEncoderDecoderModel

[[autodoc]] SpeechEncoderDecoderModel

- forward
- from_encoder_decoder_pretrained

## FlaxSpeechEncoderDecoderModel

[[autodoc]] FlaxSpeechEncoderDecoderModel

- __call__
- from_encoder_decoder_pretrained