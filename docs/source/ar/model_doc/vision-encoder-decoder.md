# نماذج الترميز فك الترميز البصري Vision Encoder Decoder Model

## نظرة عامة

يمكن استخدام [`VisionEncoderDecoderModel`] لتهيئة نموذج للصور إلى نص باستخدام أي نموذج رؤية قائم على محول تم تدريبه مسبقًا كترميز (*على سبيل المثال* [ViT](vit)، [BEiT](beit)، [DeiT](deit)، [Swin](swin)) وأي نموذج لغة تم تدريبه مسبقًا كفك تشفير (*على سبيل المثال* [RoBERTa](roberta)، [GPT2](gpt2)، [BERT](bert)، [DistilBERT](distilbert)).

وقد تم إثبات فعالية تهيئة النماذج التسلسلية للصور إلى نص باستخدام نقاط تفتيش مُدربة مسبقًا في (على سبيل المثال) [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) بواسطة Minghao Li، Tengchao Lv، Lei Cui، Yijuan Lu، Dinei Florencio، Cha Zhang، Zhoujun Li، Furu Wei.

بعد تدريب/ضبط نموذج [`VisionEncoderDecoderModel`]، يمكن حفظه/تحميله تمامًا مثل أي نماذج أخرى (راجع الأمثلة أدناه للحصول على مزيد من المعلومات).

ومن الأمثلة على التطبيقات وصف الصور، حيث يتم استخدام الترميز لتشفير الصورة، وبعد ذلك يقوم نموذج اللغة التنبؤي التلقائي بتوليد التعليق التوضيحي. ومن الأمثلة الأخرى التعرف الضوئي على الحروف. راجع [TrOCR](trocr)، وهو مثيل لنموذج [`VisionEncoderDecoderModel`].

## تهيئة `VisionEncoderDecoderModel` بشكل عشوائي من تكوينات النماذج.

يمكن تهيئة [`VisionEncoderDecoderModel`] بشكل عشوائي من تكوين الترميز وفك تشفير التكوين. في المثال التالي، نوضح كيفية القيام بذلك باستخدام تكوين [`ViTModel`] الافتراضي للترميز
وتكوين [`BertForCausalLM`] الافتراضي لفك التشفير.

```python
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = VisionEncoderDecoderModel(config=config)
```

## تهيئة `VisionEncoderDecoderModel` من ترميز مُدرب مسبقًا وفك تشفير مُدرب مسبقًا.

يمكن تهيئة [`VisionEncoderDecoderModel`] من نقطة تفتيش ترميز مُدربة مسبقًا ونقطة تفتيش فك تشفير مُدربة مسبقًا. لاحظ أن أي نموذج رؤية قائم على محول، مثل [Swin](swin)، يمكن أن يعمل كترميز، ويمكن استخدام كل من نماذج الترميز التلقائي المُدربة مسبقًا، مثل BERT، ونماذج اللغة السببية المُدربة مسبقًا، مثل GPT2، وكذلك الجزء فك تشفير النماذج التسلسلية إلى تسلسلية، مثل فك تشفير BART، كفك تشفير.

اعتمادًا على البنية التي تختارها كفك تشفير، قد يتم تهيئة طبقات الاهتمام المتقاطع بشكل عشوائي.

تتطلب تهيئة [`VisionEncoderDecoderModel`] من نقطة تفتيش الترميز وفك التشفير المُدربة مسبقًا ضبط نموذج على مهمة أسفل النهر، كما هو موضح في [منشور المدونة *Warm-starting-encoder-decoder*](https://huggingface.co/blog/warm-starting-encoder-decoder).

للقيام بذلك، توفر فئة `VisionEncoderDecoderModel` طريقة [`VisionEncoderDecoderModel.from_encoder_decoder_pretrained`].

```python
>>> from transformers import VisionEncoderDecoderModel

>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "microsoft/swin-base-patch4-window7-224-in22k", "google-bert/bert-base-uncased"
... )
```

## تحميل نقطة تفتيش `VisionEncoderDecoderModel` الموجودة وإجراء الاستدلال.

لتحميل نقاط تفتيش مُدربة مُدربة مسبقًا لفئة `VisionEncoderDecoderModel`، توفر [`VisionEncoderDecoderModel`] طريقة `from_pretrained(...)` تمامًا مثل أي بنية نموذج أخرى في Transformers.

للاستدلال، يستخدم المرء طريقة [`generate`]، والتي تسمح بتوليد نص بشكل تلقائي. تدعم هذه الطريقة أشكالًا مختلفة من فك التشفير، مثل الجشع، والبحث الشعاعي، وأخذ العينات متعددة الحدود.

```python
>>> import requests
>>> from PIL import Image

>>> from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

>>> # تحميل نموذج وصف الصور المُدرب مسبقًا والمحلل اللغوي ومعالج الصور المقابلة
>>> model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> # دعونا نؤدي الاستدلال على صورة
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> # توليد التعليق التوضيحي بشكل تلقائي (يستخدم فك التشفير الجشع بشكل افتراضي)
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
قط يستلقي على بطانية بجانب قط يستلقي على سرير
```

## تحميل نقطة تفتيش PyTorch في `TFVisionEncoderDecoderModel`.

لا يدعم [`TFVisionEncoderDecoderModel.from_pretrained`] حاليًا تهيئة النموذج من
نقطة تفتيش PyTorch. سيؤدي تمرير `from_pt=True` إلى هذه الطريقة إلى إلقاء استثناء. إذا كانت هناك نقاط تفتيش PyTorch فقط
لنموذج ترميز فك تشفير رؤية معينة، فإن الحل البديل هو:

```python
>>> from transformers import VisionEncoderDecoderModel, TFVisionEncoderDecoderModel

>>> _model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # هذا فقط لنسخ بعض السمات المحددة لهذا النموذج المحدد.
>>> model.config = _model.config
```

## تدريب

بمجرد إنشاء النموذج، يمكن ضبطه بشكل دقيق مثل BART أو T5 أو أي نموذج ترميز وفك تشفير آخر على مجموعة بيانات من أزواج (الصورة، النص).

كما ترون، يلزم إدخالان فقط للنموذج من أجل حساب الخسارة: `pixel_values` (التي هي
الصور) و`labels` (التي هي `input_ids` للتسلسل المستهدف المشفر).

```python
>>> from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
>>> from datasets import load_dataset

>>> image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
... )

>>> model.config.decoder_start_token_id = tokenizer.cls_token_Multiplier
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> labels = tokenizer(
...     "an image of two cats chilling on a couch",
...     return_tensors="pt",
... ).input_ids

>>> # تقوم دالة التوجيه تلقائيًا بإنشاء decoder_input_ids الصحيحة
>>> loss = model(pixel_values=pixel_values, labels=labels).loss
```

تمت المساهمة بهذا النموذج بواسطة [nielsr](https://github.com/nielsrogge). تمت المساهمة في إصدارات TensorFlow وFlax من هذا النموذج
بواسطة [ydshieh](https://github.com/ydshieh).

## VisionEncoderDecoderConfig

[[autodoc]] VisionEncoderDecoderConfig

<frameworkcontent>
<pt>

## VisionEncoderDecoderModel

[[autodoc]] VisionEncoderDecoderModel

- forward
- from_encoder_decoder_pretrained

</pt>
<tf>

## TFVisionEncoderDecoderModel

[[autodoc]] TFVisionEncoderDecoderModel

- call
- from_encoder_decoder_pretrained

</tf>
<jax>

## FlaxVisionEncoderDecoderModel

[[autodoc]] FlaxVisionEncoderDecoderModel

- __call__
- from_encoder_decoder_pretrained

</jax>
</frameworkcontent>