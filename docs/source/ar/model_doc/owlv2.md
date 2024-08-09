# OWLv2

## نظرة عامة
اقترح OWLv2 في "Scaling Open-Vocabulary Object Detection" من قبل Matthias Minderer وAlexey Gritsenko وNeil Houlsby. يوسع OWLv2 نطاق OWL-ViT باستخدام التدريب الذاتي، والذي يستخدم مكتشفًا موجودًا لتوليد تسميات توضيحية وهمية على أزواج الصور والنصوص. ويؤدي ذلك إلى مكاسب كبيرة على أحدث التقنيات للكشف عن الأشياء ذات الصفر.

ملخص الورقة هو كما يلي:

*استفاد الكشف عن الأشياء ذات المفردات المفتوحة بشكل كبير من النماذج البصرية اللغوية مسبقة التدريب، ولكنه لا يزال محدودًا بسبب كمية بيانات التدريب المتاحة للكشف. في حين يمكن توسيع بيانات التدريب على الكشف باستخدام أزواج الصور والنصوص على الويب كإشراف ضعيف، إلا أن ذلك لم يحدث على نطاقات قابلة للمقارنة مع التدريب المسبق على مستوى الصورة. هنا، نقوم بتوسيع نطاق بيانات الكشف باستخدام التدريب الذاتي، والذي يستخدم مكتشفًا موجودًا لتوليد تسميات توضيحية وهمية على أزواج الصور والنصوص. تتمثل التحديات الرئيسية في التدريب الذاتي على نطاق واسع في اختيار مساحة التسمية، وتصفية التسميات التوضيحية الوهمية، وكفاءة التدريب. نقدم نموذج OWLv2 ووصفة OWL-ST للتدريب الذاتي، والتي تتناول هذه التحديات. يتفوق OWLv2 على أداء أحدث الكاشفات ذات المفردات المفتوحة بالفعل على نطاقات تدريب مماثلة (~10 مليون مثال). ومع ذلك، باستخدام OWL-ST، يمكننا التوسع إلى أكثر من 1 مليار مثال، مما يؤدي إلى مزيد من التحسينات الكبيرة: باستخدام بنية L/14، يحسن OWL-ST AP على فئات LVIS النادرة، والتي لم ير النموذج لها أي تسميات توضيحية للصندوق البشري، من 31.2% إلى 44.6% (تحسين نسبته 43%). يفتح OWL-ST التدريب على نطاق الويب للتحديد الموضعي للعالم المفتوح، على غرار ما شوهد في تصنيف الصور ونمذجة اللغة.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/owlv2_overview.png"
alt="drawing" width="600"/>

<small>نظرة عامة عالية المستوى على OWLv2. مأخوذة من <a href="https://arxiv.org/abs/2306.09683">الورقة الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr).
يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit).

## مثال على الاستخدام
OWLv2 هو، مثل سابقه OWL-ViT، نموذج الكشف عن الأشياء المشروط بالنص بدون الصفر. يستخدم OWL-ViT CLIP كعمود فقري متعدد الوسائط، مع محول يشبه ViT للحصول على الميزات المرئية ونموذج اللغة السببية للحصول على الميزات النصية. لاستخدام CLIP للكشف، يزيل OWL-ViT طبقة تجميع الرموز النهائية لنموذج الرؤية ويرفق رأس تصنيف وخزان خفيفين بكل رمز إخراج محول. يتم تمكين التصنيف مفتوح المفردات عن طريق استبدال أوزان طبقة التصنيف الثابتة برموز تضمين اسم الفئة التي يتم الحصول عليها من نموذج النص. يقوم المؤلفون أولاً بتدريب CLIP من الصفر وتدريبه بشكل شامل مع رؤوس التصنيف والصندوق على مجموعات بيانات الكشف القياسية باستخدام خسارة المطابقة الثنائية. يمكن استخدام استعلام نصي واحد أو أكثر لكل صورة للكشف عن الأشياء المشروطة بالنص بدون الصفر.

يمكن استخدام [`Owlv2ImageProcessor`] لإعادة تحديد حجم الصور (أو إعادة تحجيمها) وتطبيعها للنموذج، ويستخدم [`CLIPTokenizer`] لترميز النص. [`Owlv2Processor`] يجمع [`Owlv2ImageProcessor`] و [`CLIPTokenizer`] في مثيل واحد لترميز النص وإعداد الصور. يوضح المثال التالي كيفية إجراء الكشف عن الأشياء باستخدام [`Owlv2Processor`] و [`Owlv2ForObjectDetection`].

```python
>>> import requests
>>> from PIL import Image
>>> import torch

>>> from transformers import Owlv2Processor, Owlv2ForObjectDetection

>>> processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
>>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = [["a photo of a cat", "a photo of a dog"]]
>>> inputs = processor(text=texts, images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
>>> target_sizes = torch.Tensor([image.size[::-1]])
>>> # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
>>> results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
>>> i = 0  # Retrieve predictions for the first image for the corresponding text queries
>>> text = texts[i]
>>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
>>> for box, score, label in zip(boxes, scores, labels):
...     box = [round(i, 2) for i in box.tolist()]
...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
Detected a photo of a cat with confidence 0.614 at location [341.67, 23.39, 642.32, 371.35]
Detected a photo of a cat with confidence 0.665 at location [6.75, 51.96, 326.62, 473.13]
```

## الموارد
- يمكن العثور على دفتر ملاحظات توضيحي حول استخدام OWLv2 للكشف عن الصفر واللقطة الواحدة (التي يوجهها الصور) [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OWLv2).
- [دليل مهام الكشف عن الأشياء ذات الصفر](../tasks/zero_shot_object_detection)

<Tip>
تطابق بنية OWLv2 تلك الموجودة في OWL-ViT، ولكن رأس الكشف عن الأشياء يشمل الآن أيضًا مصنفًا للكائنات، والذي يتنبأ باحتمالية وجود كائن (على عكس الخلفية) في صندوق متوقع. يمكن استخدام درجة الكائن لترتيب التوقعات أو تصفيتها بشكل مستقل عن استعلامات النص.
يتم استخدام OWLv2 بنفس طريقة استخدام [OWL-ViT](owlvit) مع معالج صورة جديد محدث ([`Owlv2ImageProcessor`]).
</Tip>

## Owlv2Config

[[autodoc]] Owlv2Config

- from_text_vision_configs

## Owlv2TextConfig

[[autodoc]] Owlv2TextConfig

## Owlv2VisionConfig

[[autodoc]] Owlv2VisionConfig

## Owlv2ImageProcessor

[[autodoc]] Owlv2ImageProcessor

- preprocess

- post_process_object_detection

- post_process_image_guided_detection

## Owlv2Processor

[[autodoc]] Owlv2Processor

## Owlv2Model

[[autodoc]] Owlv2Model

- forward

- get_text_features

- get_image_features

## Owlv2TextModel

[[autodoc]] Owlv2TextModel

- forward

## Owlv2VisionModel

[[autodoc]] Owlv2VisionModel

- forward

## Owlv2ForObjectDetection

[[autodoc]] Owlv2ForObjectDetection

- forward

- image_guided_detection