# AltCLIP

## نظرة عامة
اقترح نموذج AltCLIP في "AltCLIP: تعديل مشفر اللغة في CLIP لقدرات لغوية موسعة" من قبل تشين زونغزهي، وليو جوان، وزانج بو-وين، واي فولونج، ويانج تشينجهونج، وو ليديل. AltCLIP (تعديل مشفر اللغة في CLIP) هو شبكة عصبية تم تدريبها على مجموعة متنوعة من أزواج الصور والنصوص والنصوص النصية. من خلال استبدال مشفر النص CLIP بمشفر نص متعدد اللغات XLM-R، يمكننا الحصول على أداء قريب جدًا من CLIP في جميع المهام تقريبًا، وتوسيع قدرات CLIP الأصلية مثل الفهم متعدد اللغات.

المقتطف من الورقة هو ما يلي:

في هذا العمل، نقدم طريقة بسيطة وفعالة من الناحية المفاهيمية لتدريب نموذج تمثيل متعدد الوسائط ثنائي اللغة قوي. بدءًا من نموذج التمثيل متعدد الوسائط المسبق التدريب CLIP الذي أصدرته OpenAI، قمنا بالتبديل بين مشفر النص الخاص به ومشفر نص متعدد اللغات مسبق التدريب XLM-R، ومواءمة تمثيلات اللغات والصور باستخدام مخطط تدريب مكون من مرحلتين يتكون من التعلم بواسطة المعلم والتعلم التمييزي. نحن نقيم طريقتنا من خلال تقييمات لمجموعة واسعة من المهام. نحدد أحدث أداء على مجموعة من المهام بما في ذلك ImageNet-CN، وFlicker30k- CN، وCOCO-CN. علاوة على ذلك، نحصل على أداء قريب جدًا من CLIP في جميع المهام تقريبًا، مما يشير إلى أنه يمكن للمرء ببساطة تغيير مشفر النص في CLIP لقدرات موسعة مثل الفهم متعدد اللغات.

تمت المساهمة بهذا النموذج من قبل [jongjyh](https://huggingface.co/jongjyh).

## نصائح الاستخدام ومثال

استخدام AltCLIP مشابه جدًا لـ CLIP. الفرق بين CLIP هو مشفر النص. لاحظ أننا نستخدم اهتمامًا ثنائي الاتجاه بدلاً من الاهتمام العرضي ونأخذ الرمز [CLS] في XLM-R لتمثيل تضمين النص.

AltCLIP هو نموذج متعدد الوسائط للرؤية واللغة. يمكن استخدامه لتشابه الصور والنص ولتصنيف الصور بدون إشراف. يستخدم AltCLIP محولًا مثل ViT لاستخراج الميزات المرئية ونموذج لغة ثنائي الاتجاه لاستخراج ميزات النص. ثم يتم إسقاط كل من ميزات النص والمرئيات إلى مساحة كامنة ذات أبعاد متطابقة. يتم بعد ذلك استخدام المنتج النقطي بين الصورة وميزات النص المسقطة كدرجة تشابه.

لتغذية الصور إلى محول المشفر، يتم تقسيم كل صورة إلى تسلسل من التصحيح الثابت الحجم غير المتداخل، والذي يتم تضمينه خطيًا بعد ذلك. تتم إضافة رمز [CLS] ليكون بمثابة تمثيل لصورة كاملة. يضيف المؤلفون أيضًا تضمينات موضعية مطلقة، ويقومون بإطعام تسلسل المتجهات الناتج إلى محول مشفر قياسي.

يمكن استخدام [`CLIPImageProcessor`] لإعادة تحجيم (أو إعادة تحجيم) الصور وتطبيعها للنموذج.

يغلف [`AltCLIPProcessor`] [`CLIPImageProcessor`] و [`XLMRobertaTokenizer`] في مثيل واحد لترميز النص وإعداد الصور. يوضح المثال التالي كيفية الحصول على درجات تشابه الصور والنص باستخدام [`AltCLIPProcessor`] و [`AltCLIPModel`].

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import AltCLIPModel, AltCLIPProcessor

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AltCLIPProcessor.from_pretrained("BAA0/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image # هذه هي درجة تشابه الصورة والنص
>>> probs = logits_per_image.softmax(dim=1) # يمكننا أخذ softmax للحصول على احتمالات التسمية
```

<Tip>

يستند هذا النموذج إلى `CLIPModel`، استخدمه كما تستخدم CLIP [الأصلي](clip).

</Tip>

## AltCLIPConfig

[[autodoc]] AltCLIPConfig

- from_text_vision_configs

## AltCLIPTextConfig

[[autodoc]] AltCLIPTextConfig

## AltCLIPVisionConfig

[[autodoc]] AltCLIPVisionConfig

## AltCLIPProcessor

[[autodoc]] AltCLIPProcessor

## AltCLIPModel

[[autodoc]] AltCLIPModel

- forward

- get_text_features

- get_image_features

## AltCLIPTextModel

[[autodoc]] AltCLIPTextModel

- forward

## AltCLIPVisionModel

[[autodoc]] AltCLIPVisionModel

- forward