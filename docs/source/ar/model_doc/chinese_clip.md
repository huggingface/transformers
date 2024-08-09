# Chinese-CLIP

## نظرة عامة
تم اقتراح نموذج Chinese-CLIP في الورقة البحثية بعنوان "Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese" بواسطة An Yang و Junshu Pan و Junyang Lin و Rui Men و Yichang Zhang و Jingren Zhou و Chang Zhou.

نموذج Chinese-CLIP هو تطبيق لنموذج CLIP (Radford et al.، 2021) على مجموعة بيانات ضخمة من أزواج الصور والنصوص المكتوبة باللغة الصينية. يتميز النموذج بالقدرة على إجراء الاسترجاع عبر الوسائط، كما يمكن استخدامه كعمود فقري للمهام البصرية مثل تصنيف الصور بدون الإشراف والتعرف على الأشياء في النطاق المفتوح، وما إلى ذلك. تم إصدار الكود الأصلي لنموذج Chinese-CLIP [على هذا الرابط](https://github.com/OFA-Sys/Chinese-CLIP).

وفيما يلي الملخص المستخرج من الورقة البحثية:

*حقق نموذج CLIP (Radford et al.، 2021) نجاحًا هائلاً، مما عزز البحث والتطبيق في مجال التعلم التمييزي للتمثيل المسبق للرؤية واللغة. وفي هذا العمل، نقوم ببناء مجموعة بيانات ضخمة من أزواج الصور والنصوص باللغة الصينية، حيث تم استخراج معظم البيانات من مجموعات البيانات المتاحة للعموم، ثم نقوم بتدريب نماذج Chinese CLIP على مجموعة البيانات الجديدة. قمنا بتطوير 5 نماذج Chinese CLIP بأحجام متعددة، تتراوح من 77 إلى 958 مليون معامل. علاوة على ذلك، نقترح طريقة تدريب مكونة من مرحلتين، حيث يتم تدريب النموذج أولاً مع تثبيت مشفر الصور، ثم يتم تدريبه مع تحسين جميع المعلمات، وذلك لتحقيق أداء أفضل للنموذج. تُظهر تجاربنا الشاملة أن نموذج Chinese CLIP يمكن أن يحقق أداءً متميزًا على MUGE و Flickr30K-CN و COCO-CN في إعدادات التعلم بدون إشراف وتنقيح الدقة، كما أنه قادر على تحقيق أداء تنافسي في تصنيف الصور بدون إشراف بناءً على التقييم على معيار ELEVATER (Li et al.، 2022). وقد تم إصدار أكوادنا والنماذج التي تم تدريبها مسبقًا والعروض التوضيحية.*

تمت المساهمة في نموذج Chinese-CLIP بواسطة [OFA-Sys](https://huggingface.co/OFA-Sys).

## مثال على الاستخدام
يوضح مقتطف الكود التالي كيفية حساب ميزات الصور والنصوص ومستويات التشابه بينها:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import ChineseCLIPProcessor, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> # Squirtle, Bulbasaur, Charmander, Pikachu in English
>>> texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

>>> # compute image feature
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
>>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

>>> # compute text features
>>> inputs = processor(text=texts, padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
>>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

>>> # compute image-text similarity scores
>>> inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
```

حاليًا، تتوفر النماذج المُدربة مسبقًا التالية لنموذج Chinese-CLIP على منصة 🤗 Hub بالأحجام المختلفة:

- [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
- [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)
- [OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
- [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14)

## ChineseCLIPConfig

[[autodoc]] ChineseCLIPConfig

- from_text_vision_configs

## ChineseCLIPTextConfig

[[autodoc]] ChineseCLIPTextConfig

## ChineseCLIPVisionConfig

[[autodoc]] ChineseCLIPVisionConfig

## ChineseCLIPImageProcessor

[[autodoc]] ChineseCLIPImageProcessor

- preprocess

## ChineseCLIPFeatureExtractor

[[autodoc]] ChineseCLIPFeatureExtractor

## ChineseCLIPProcessor

[[autodoc]] ChineseCLIPProcessor

## ChineseCLIPModel

[[autodoc]] ChineseCLIPModel

- forward

- get_text_features

- get_image_features

## ChineseCLIPTextModel

[[autodoc]] ChineseCLIPTextModel

- forward

## ChineseCLIPVisionModel

[[autodoc]] ChineseCLIPVisionModel

- forward