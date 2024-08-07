# OWL-ViT

## نظرة عامة
OWL-ViT (اختصار لـ Vision Transformer for Open-World Localization) اقترحه Matthias Minderer, et al. في ورقتهم البحثية [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230). OWL-ViT هي شبكة كشف كائنات ذات مفردات مفتوحة تم تدريبها على مجموعة متنوعة من أزواج (الصورة، النص). يمكن استخدامه لاستعلام صورة واحدة أو أكثر باستخدام استعلامات نصية للبحث عن كائنات مستهدفة وكشفها موصوفة في النص.

مقتطف من الورقة البحثية على النحو التالي:

*لقد أدى الجمع بين البنى البسيطة والتعلم المسبق واسع النطاق إلى تحسينات هائلة في تصنيف الصور. بالنسبة لكشف الكائنات، فإن نهج التعلم المسبق والتوسع أقل رسوخًا، خاصة في الإعداد طويل الذيل والمفردات المفتوحة، حيث تكون بيانات التدريب نادرة نسبيًا. في هذه الورقة، نقترح وصفة قوية لنقل النماذج القائمة على الصور والنصوص إلى كشف الكائنات ذات المفردات المفتوحة. نستخدم بنية Vision Transformer القياسية مع تعديلات طفيفة، والتعلم المسبق التبايني للصور والنصوص، وضبط دقيق للكشف من النهاية إلى النهاية. يُظهر تحليلنا لخصائص التوسع في هذا الإعداد أن زيادة التعلم المسبق على مستوى الصورة وحجم النموذج ينتج عنه تحسينات ثابتة في مهمة الكشف النهائية. نقدم استراتيجيات التكيف والتنظيم المطلوبة لتحقيق أداء قوي جدًا في كشف الكائنات المشروط بالنص بدون بيانات تدريب وصفرية. الكود والنماذج متوفرة على GitHub.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/owlvit_architecture.jpg" alt="drawing" width="600"/>

<small>بنية OWL-ViT. مأخوذة من <a href="https://arxiv.org/abs/2205.06230">الورقة البحثية الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [adirik](https://huggingface.co/adirik). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit).

## نصائح الاستخدام
OWL-ViT هو نموذج كشف كائنات مشروط بالنص بدون بيانات تدريب. يستخدم OWL-ViT [CLIP](clip) كعمود فقري متعدد الوسائط، مع محول تشابهي مثل ViT للحصول على الميزات المرئية ونموذج لغة سببي للحصول على ميزات النص. لاستخدام CLIP للكشف، يزيل OWL-ViT طبقة تجميع الرموز النهائية من نموذج الرؤية ويرفق رأس تصنيف وخزان خفيف الوزن بكل رمز إخراج محول. يتم تمكين التصنيف مفتوح المفردات عن طريق استبدال أوزان طبقة التصنيف الثابتة برموز تضمين اسم الفئة التي يتم الحصول عليها من نموذج النص. يقوم المؤلفون أولاً بتدريب CLIP من الصفر وضبطها الدقيق من النهاية إلى النهاية مع رؤوس التصنيف والخزان على مجموعات بيانات الكشف القياسية باستخدام خسارة المطابقة الثنائية. يمكن استخدام استعلام نصي واحد أو أكثر لكل صورة لأداء الكشف عن الكائنات المشروط بالنص بدون بيانات تدريب.

يمكن استخدام [`OwlViTImageProcessor`] لإعادة تحديد حجم الصور (أو إعادة تحجيمها) وتطبيعها للنموذج، ويستخدم [`CLIPTokenizer`] لترميز النص. [`OwlViTProcessor`] يجمع [`OwlViTImageProcessor`] و [`CLIPTokenizer`] في مثيل واحد لترميز النص وإعداد الصور. يوضح المثال التالي كيفية تنفيذ الكشف عن الكائنات باستخدام [`OwlViTProcessor`] و [`OwlViTForObjectDetection`].

```python
>>> import requests
>>> from PIL import Image
>>> import torch

>>> from transformers import OwlViTProcessor, OwlViTForObjectDetection

>>> processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
>>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=texts, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.Tensor([image.size[::-1]])
>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
>>> i = 0  # Retrieve predictions for the first image for the corresponding text queries
>>> text = texts[i]
>>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
>>> for box, score, label in zip(boxes, scores, labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
```

## الموارد
يمكن العثور على دفتر ملاحظات توضيحي حول استخدام OWL-ViT للكشف عن الكائنات بدون بيانات تدريب وبدليل الصور [هنا](https://github.com/huggingface/notebooks/blob/main/examples/zeroshot_object_detection_with_owlvit.ipynb).

## OwlViTConfig

[[autodoc]] OwlViTConfig

- from_text_vision_configs

## OwlViTTextConfig

[[autodoc]] OwlViTTextConfig

## OwlViTVisionConfig

[[autodoc]] OwlViTVisionConfig

## OwlViTImageProcessor

[[autodoc]] OwlViTImageProcessor

- preprocess

- post_process_object_detection

- post_process_image_guided_detection

## OwlViTFeatureExtractor

[[autodoc]] OwlViTFeatureExtractor

- __call__

- post_process

- post_process_image_guided_detection

## OwlViTProcessor

[[autodoc]] OwlViTProcessor

## OwlViTModel

[[autodoc]] OwlViTModel

- forward

- get_text_features

- get_image_features

## OwlViTTextModel

[[autodoc]] OwlViTTextModel

- forward

## OwlViTVisionModel


[[autodoc]] OwlViTVisionModel

- forward

## OwlViTForObjectDetection

[[autodoc]] OwlViTForObjectDetection

- forward

- image_guided_detection