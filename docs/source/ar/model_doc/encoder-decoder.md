# Encoder Decoder Models

## Overview

يمكن استخدام [`EncoderDecoderModel`] لتهيئة نموذج تسلسل إلى تسلسل باستخدام أي نموذج ترميز ذاتي مسبق التدريب كترميز وأي نموذج توليدي ذاتي مسبق التدريب كفك تشفير.

وقد تم توضيح فعالية تهيئة نماذج تسلسل إلى تسلسل باستخدام نقاط تفتيش مسبقة التدريب لمهام توليد التسلسل في [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) بواسطة
Sascha Rothe، Shashi Narayan، Aliaksei Severyn.

بعد تدريب/ضبط نموذج [`EncoderDecoderModel`]، يمكن حفظه/تحميله تمامًا مثل أي نموذج آخر (راجع الأمثلة لمزيد من المعلومات).

يمكن أن يكون أحد تطبيقات هذه البنية هو الاستفادة من نموذجين [`BertModel`] مسبقين التدريب كترميز وفك تشفير لنموذج تلخيص كما هو موضح في: [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) بواسطة Yang Liu وMirella Lapata.

## التهيئة العشوائية لـ `EncoderDecoderModel` من تكوينات النموذج.

يمكن تهيئة [`EncoderDecoderModel`] بشكل عشوائي من تكوين الترميز وفك التشفير. في المثال التالي، نوضح كيفية القيام بذلك باستخدام تكوين [`BertModel`] الافتراضي للترميز وتكوين [`BertForCausalLM`] الافتراضي لفك التشفير.

```python
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

## تهيئة `EncoderDecoderModel` من ترميز مسبق التدريب وفك تشفير مسبق التدريب.

يمكن تهيئة [`EncoderDecoderModel`] من نقطة تفتيش ترميز مسبقة التدريب ونقطة تفتيش فك تشفير مسبقة التدريب. لاحظ أن أي نموذج ترميز ذاتي مسبق التدريب، مثل BERT، يمكن أن يعمل كترميز، ويمكن استخدام كل من نماذج الترميز الذاتي المسبقة التدريب، مثل BERT، ونماذج اللغة السببية المسبقة التدريب، مثل GPT2، بالإضافة إلى الجزء فك تشفير من نماذج تسلسل إلى تسلسل، مثل فك تشفير BART، كفك تشفير.

اعتمادًا على البنية التي تختارها كفك تشفير، قد يتم تهيئة طبقات الاهتمام المتقاطع بشكل عشوائي.

تتطلب تهيئة [`EncoderDecoderModel`] من نقطة تفتيش ترميز مسبقة التدريب ونقطة تفتيش فك تشفير مسبقة التدريب ضبط النموذج على مهمة أسفل البنية، كما هو موضح في [منشور المدونة *Warm-starting-encoder-decoder*](https://huggingface.co/blog/warm-starting-encoder-decoder).

للقيام بذلك، توفر فئة `EncoderDecoderModel` طريقة [`EncoderDecoderModel.from_encoder_decoder_pretrained`].

```python
>>> from transformers import EncoderDecoderModel, BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")
```

## تحميل نقطة تفتيش `EncoderDecoderModel` موجودة والقيام بالاستدلال.

لتحميل نقاط تفتيش ضبط الدقة لفئة `EncoderDecoderModel`، توفر [`EncoderDecoderModel`] طريقة `from_pretrained(...)` تمامًا مثل أي بنية نموذج أخرى في Transformers.

للاستدلال، يستخدم المرء طريقة [`generate`]، والتي تسمح بتوليد النص بشكل تلقائي. تدعم هذه الطريقة أشكالًا مختلفة من فك التشفير، مثل الجشع، وبحث الشعاع، وأخذ العينات متعددة الحدود.

```python
>>> from transformers import AutoTokenizer, EncoderDecoderModel

>>> # تحميل نموذج تسلسل إلى تسلسل مضبوط الدقة ومحلل الأجزاء المقابلة
>>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

>>> # لنقم بالاستدلال على قطعة نص طويلة
>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

>>> # توليد تلخيص تلقائيًا (يستخدم فك التشفير الجشع بشكل افتراضي)
>>> generated_ids = model.generate(input_ids)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
nearly 800 thousand customers were affected by the shutoffs. the aim is to reduce the risk of wildfires. nearly 800, 000 customers were expected to be affected by high winds amid dry conditions. pg & e said it scheduled the blackouts to last through at least midday tomorrow.

```

## تحميل نقطة تفتيش PyTorch في `TFEncoderDecoderModel`.

لا يدعم [`TFEncoderDecoderModel.from_pretrained`] حاليًا تهيئة النموذج من
نقطة تفتيش PyTorch. سيؤدي تمرير `from_pt=True` إلى هذه الطريقة إلى إلقاء استثناء. إذا كانت هناك نقاط تفتيش PyTorch فقط
لنموذج ترميز وفك تشفير معين، فإن الحل البديل هو:

```python
>>> # حل بديل لتحميل نقطة تفتيش PyTorch
>>> from transformers import EncoderDecoderModel, TFEncoderDecoderModel

>>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder"، encoder_from_pt=True، decoder_from_pt=True
... )
>>> # هذا فقط لنسخ بعض السمات المحددة لهذا النموذج المحدد.
>>> model.config = _model.config
```

## التدريب

بمجرد إنشاء النموذج، يمكن ضبط دقته مثل BART أو T5 أو أي نموذج ترميز وفك تشفير آخر.

كما ترون، هناك مدخلان فقط مطلوبان للنموذج من أجل حساب الخسارة: `input_ids` (التي هي
`input_ids` من تسلسل الإدخال المشفر) و`labels` (التي هي
`input_ids` من تسلسل الإخراج المشفر).

```python
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_Multiplier = tokenizer.pad_token_id

# >>> input_ids = tokenizer(
# ...     "يبلغ ارتفاع البرج 324 مترًا (1,063 قدمًا)، أي ما يعادل ارتفاع مبنى مكون من 81 طابقًا، وهو أطول هيكل في باريس. قاعدته مربعة، يبلغ طول كل جانب 125 مترًا (410 قدمًا). خلال بنائه، تجاوز برج إيفل نصب واشنطن التذكاري ليصبح أطول هيكل من صنع الإنسان في العالم، وهو لقب احتفظ به لمدة 41 عامًا حتى اكتمل مبنى كرايسلر في مدينة نيويورك في عام 1930. كان أول هيكل يصل إلى ارتفاع 300 متر. وبسبب إضافة هوائي بث في الجزء العلوي من البرج في عام 1957، فهو الآن أطول من مبنى كرايسلر بـ 5.2 متر (17 قدمًا). باستثناء أجهزة الإرسال، يعد برج إيفل ثاني أطول هيكل قائم بذاته في فرنسا بعد جسر ميلو"،
# ...     return_tensors="pt"،
# ... ).input_ids

# >>> labels = tokenizer(
# ...     "تجاوز برج إيفل نصب واشنطن التذكاري ليصبح أطول هيكل في العالم. كان أول هيكل يصل إلى ارتفاع 300 متر في باريس في عام 1930. إنه الآن أطول من مبنى كرايسلر بـ 5.2 متر (17 قدمًا) وهو ثاني أطول هيكل قائم بذاته في باريس.",
# ...     return_tensors="pt"،
# ... ).input_ids

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> # تقوم دالة forward تلقائيًا بإنشاء decoder_input_ids الصحيحة
>>> loss = model(input_ids=input_ids, labels=labels).loss
```

[تفاصيل](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl) حول التدريب.

تمت المساهمة بهذا النموذج بواسطة [thomwolf](https://github.com/thomwolf). تمت المساهمة في إصدارات TensorFlow وFlax من هذا النموذج
بواسطة [ydshieh](https://github.com/ydshieh).

## EncoderDecoderConfig

[[autodoc]] EncoderDecoderConfig

<frameworkcontent>
<pt>

## EncoderDecoderModel

[[autodoc]] EncoderDecoderModel

- forward
- from_encoder_decoder_pretrained

</pt>
<tf>

## TFEncoderDecoderModel

[[autodoc]] TFEncoderDecoderModel

- call
- from_encoder_decoder_pretrained

</tf>
<jax>

## FlaxEncoderDecoderModel

[[autodoc]] FlaxEncoderDecoderModel

- __call__
- from_encoder_decoder_pretrained

</jax>

</frameworkcontent>