# BridgeTower

## نظرة عامة

اقتُرح نموذج BridgeTower في ورقة "BridgeTower: بناء جسور بين المشفرين أحاديي النمط في التعلم التمثيلي للرؤية واللغة" بواسطة شياو شو، وتشنفي وو، وشاشار روزنمان، وفاسوديف لال، ووانكسيانغ تشي، ونان دوان. ويهدف هذا النموذج إلى بناء جسر بين كل مشفر أحادي النمط والمشفر متعدد الأنماط لتمكين التفاعل الشامل والمفصل في كل طبقة من المشفر متعدد الأنماط، وبالتالي تحقيق أداء ملحوظ في مختلف المهام دون زيادة تذكر في التكاليف الحسابية والأداء.

تم قبول هذه الورقة في مؤتمر AAAI'23.

وفيما يلي ملخص الورقة:

"سيطرت نماذج الرؤية واللغة (VL) ذات البنية ثنائية الأبراج على تعلم التمثيل البصري اللغوي في السنوات الأخيرة. تستخدم النماذج الحالية للرؤية واللغة إما مشفرات أحادية النمط خفيفة الوزن وتتعلم استخراج ومواءمة ودمج كلا النمطين بشكل متزامن في مشفر متعدد الأنماط عميق، أو تغذي التمثيلات أحادية النمط من الطبقة الأخيرة من المشفرات أحادية النمط مسبقة التدريب إلى المشفر متعدد الأنماط العلوي. يقيد كلا النهجين من تعلم التمثيل البصري اللغوي ويحد من أداء النموذج. في هذه الورقة، نقترح BRIDGETOWER، الذي يقدم طبقات جسر متعددة تربط بين الطبقات العليا من المشفرات أحادية النمط وكل طبقة من المشفر متعدد الأنماط. يسمح هذا بالمحاذاة والدمج الفعالين من الأسفل إلى الأعلى بين التمثيلات البصرية والنصية لمستويات دلالية مختلفة من المشفرات أحادية النمط مسبقة التدريب في المشفر متعدد الأنماط. حقق BRIDGETOWER، الذي تم تدريبه مسبقًا على 4 ملايين صورة فقط، أداءً متميزًا في مختلف مهام الرؤية واللغة. وعلى وجه التحديد، حقق BRIDGETOWER دقة تبلغ 78.73% على مجموعة اختبار VQAv2 test-std، متجاوزًا نموذج METER الأفضل في فئته بنسبة 1.09% باستخدام نفس بيانات التدريب المسبق وعدد إضافي ضئيل من المعلمات والتكاليف الحسابية. وعند زيادة حجم النموذج، حقق BRIDGETOWER دقة بلغت 81.15%، متفوقًا على النماذج التي تم تدريبها مسبقًا على مجموعات بيانات أكبر بعدة مرات."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/bridgetower_architecture%20.jpg"
alt="drawing" width="600"/>

<small> بنية BridgeTower. مأخوذة من <a href="https://arxiv.org/abs/2206.08657">الورقة الأصلية.</a> </small>

تمت المساهمة بهذا النموذج من قبل أناهيتا بهيوانديوالا، وتييب لي، وشاويين تسنغ. ويمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/BridgeTower).

## نصائح الاستخدام وأمثلة

يتكون BridgeTower من مشفر بصري ومشفر نصي ومشفر متعدد الأنماط مع طبقات جسر خفيفة متعددة. ويهدف هذا النهج إلى بناء جسر بين كل مشفر أحادي النمط والمشفر متعدد الأنماط لتمكين التفاعل الشامل والمفصل في كل طبقة من المشفر متعدد الأنماط.

من حيث المبدأ، يمكن تطبيق أي مشفر بصري أو نصي أو متعدد الأنماط في البنية المقترحة.

يغلف [`BridgeTowerProcessor`] كلاً من [`RobertaTokenizer`] و [`BridgeTowerImageProcessor`] في مثيل واحد لتشفير النص وإعداد الصور على التوالي.

يوضح المثال التالي كيفية تشغيل التعلم التبايني باستخدام [`BridgeTowerProcessor`] و [`BridgeTowerForContrastiveLearning`].

```python
>>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
>>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

>>> # forward pass
>>> scores = dict()
>>> for text in texts:
...     # prepare inputs
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs
```

يوضح المثال التالي كيفية تشغيل الاسترجاع الصوري النصي باستخدام [`BridgeTowerProcessor`] و [`BridgeTowerForImageAndTextRetrieval`].

```python
>>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> # forward pass
>>> scores = dict()
>>> for text in texts:
...     # prepare inputs
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs.logits[0, 1].item()
```

يوضح المثال التالي كيفية تشغيل نمذجة اللغة المقنعة باستخدام [`BridgeTowerProcessor`] و [`BridgeTowerForMaskedLM`].

```python
>>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> text = "a <mask> looking out of the window"

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> # prepare inputs
>>> encoding = processor(image, text, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**encoding)

>>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

>>> print(results)
.a cat looking out of the window.
```

## نصائح:

- يستخدم هذا التنفيذ من BridgeTower [`RobertaTokenizer`] لتوليد تضمينات نصية ونموذج CLIP/ViT من OpenAI لحساب التضمينات البصرية.
- تم إصدار نقاط تفتيش مسبقة التدريب لـ [bridgeTower-base](https://huggingface.co/BridgeTower/bridgetower-base) و [bridgetower masked language modeling and image text matching](https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm).
- يرجى الرجوع إلى [الجدول 5](https://arxiv.org/pdf/2206.08657.pdf) لأداء BridgeTower على استرجاع الصور ومهام أخرى.
- إصدار PyTorch من هذا النموذج متاح فقط في PyTorch 1.10 والإصدارات الأحدث.

## BridgeTowerConfig

[[autodoc]] BridgeTowerConfig

## BridgeTowerTextConfig

[[autodoc]] BridgeTowerTextConfig

## BridgeTowerVisionConfig

[[autodoc]] BridgeTowerVisionConfig

## BridgeTowerImageProcessor

[[autodoc]] BridgeTowerImageProcessor

- معالجة مسبقة

## BridgeTowerProcessor

[[autodoc]] BridgeTowerProcessor

- __call__

## BridgeTowerModel

[[autodoc]] BridgeTowerModel

- forward

## BridgeTowerForContrastiveLearning

[[autodoc]] BridgeTowerForContrastiveLearning

- forward

## BridgeTowerForMaskedLM

[[autodoc]] BridgeTowerForMaskedLM

- forward

## BridgeTowerForImageAndTextRetrieval

[[autodoc]] BridgeTowerForImageAndTextRetrieval

- forward