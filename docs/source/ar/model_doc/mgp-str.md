# MGP-STR

## نظرة عامة
اقترح نموذج MGP-STR في [التعرف على نص المشهد متعدد الدقة التنبؤية](https://arxiv.org/abs/2209.03592) بواسطة Peng Wang، وCheng Da، وCong Yao. MGP-STR هو نموذج تعرف على نص المشهد (STR) **بسيط** من الناحية النظرية ولكنه **قوي**، مبني على [محول الرؤية (ViT)](vit). ولدمج المعرفة اللغوية، اقترحت استراتيجية التنبؤ متعدد الدقة (MGP) لحقن المعلومات من الطراز اللغوي في النموذج بطريقة ضمنية.

المستخلص من الورقة هو ما يلي:

> "كان التعرف على نص المشهد (STR) موضوع بحث نشط في رؤية الكمبيوتر لسنوات. ولمعالجة هذه المشكلة الصعبة، اقترحت العديد من الطرق المبتكرة بشكل متتالي وأصبح دمج المعرفة اللغوية في نماذج STR اتجاهًا بارزًا مؤخرًا. في هذا العمل، نستلهم أولاً من التقدم الأخير في محول الرؤية (ViT) لبناء نموذج STR بصري مفهومي بسيط ولكنه قوي، مبني على ViT ويتفوق على النماذج السابقة الرائدة في مجال التعرف على نص المشهد، بما في ذلك النماذج البصرية البحتة والأساليب المعززة باللغة. لدمج المعرفة اللغوية، نقترح أيضًا استراتيجية التنبؤ متعدد الدقة لحقن المعلومات من الطراز اللغوي في النموذج بطريقة ضمنية، أي أنه يتم تقديم تمثيلات الوحدات الفرعية (BPE وWordPiece) المستخدمة على نطاق واسع في NLP في مساحة الإخراج، بالإضافة إلى التمثيل التقليدي لمستوى الأحرف، في حين لم يتم اعتماد أي نموذج لغة مستقل. يمكن للخوارزمية الناتجة (يطلق عليها MGP-STR) دفع مظروف أداء STR إلى مستوى أعلى. على وجه التحديد، يحقق متوسط دقة التعرف بنسبة 93.35٪ في المعايير القياسية."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mgp_str_architecture.png" alt="drawing" width="600"/>

<small>هندسة MGP-STR. مأخوذة من <a href="https://arxiv.org/abs/2209.03592">الورقة الأصلية</a>.</small>

تم تدريب MGP-STR على مجموعتين من البيانات الاصطناعية [MJSynth]((http://www.robots.ox.ac.uk/~vgg/data/text/)) (MJ) و[SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) (ST) بدون ضبط دقيق على مجموعات بيانات أخرى. ويحقق نتائج رائدة على ستة معايير قياسية لنص المشهد اللاتيني، بما في ذلك 3 مجموعات بيانات نص عادية (IC13، SVT، IIIT) و3 مجموعات بيانات غير منتظمة (IC15، SVTP، CUTE).

تمت المساهمة بهذا النموذج بواسطة [yuekun](https://huggingface.co/yuekun). يمكن العثور على الكود الأصلي [هنا](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR).

## مثال الاستنتاج

يقبل [`MgpstrModel`] الصور كمدخلات وينشئ ثلاثة أنواع من التنبؤات، والتي تمثل معلومات نصية بمستويات دقة مختلفة.

يتم دمج الأنواع الثلاثة من التنبؤات لإعطاء نتيجة التنبؤ النهائية.

تكون فئة [`ViTImageProcessor`] مسؤولة عن معالجة الصورة المدخلة، ويقوم [`MgpstrTokenizer`] بفك رموز الرموز المميزة المولدة إلى السلسلة المستهدفة.

يغلف [`MgpstrProcessor`] [`ViTImageProcessor`] و [`MgpstrTokenizer`] في مثيل واحد لاستخراج ميزات الإدخال وفك تشفير معرّفات الرموز المميزة المتوقعة.

- التعرف الضوئي على الأحرف خطوة بخطوة (OCR)

```py
>>> from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
>>> import requests
>>> from PIL import Image

>>> processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
>>> model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

>>> # تحميل الصورة من مجموعة بيانات IIIT-5k
>>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> pixel_values = processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(pixel_values)

>>> generated_text = processor.batch_decode(outputs.logits)['generated_text']
```

## MgpstrConfig

[[autodoc]] MgpstrConfig

## MgpstrTokenizer

[[autodoc]] MgpstrTokenizer

- save_vocabulary

## MgpstrProcessor

[[autodoc]] MgpstrProcessor

- __call__

- batch_decode

## MgpstrModel

[[autodoc]] MgpstrModel

- forward

## MgpstrForSceneTextRecognition

[[autodoc]] MgpstrForSceneTextRecognition

- forward