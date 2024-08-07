# SigLIP

## نظرة عامة
تم اقتراح نموذج SigLIP في [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) بواسطة Xiaohua Zhai وBasil Mustafa وAlexander Kolesnikov وLucas Beyer. ويقترح SigLIP استبدال دالة الخسارة المستخدمة في [CLIP](clip) بخسارة Sigmoid ثنائية بسيطة. ويؤدي ذلك إلى تحسين الأداء من حيث دقة التصنيف بدون الإشراف على ImageNet.

مقدمة الورقة البحثية هي كما يلي:

*نقترح خسارة Sigmoid ثنائية بسيطة للتعلم التمهيدي للغة والصورة (SigLIP). على عكس التعلم التمييزي القياسي مع التطبيع Softmax، تعمل خسارة Sigmoid فقط على أزواج الصور والنصوص ولا تتطلب رؤية شاملة للتشابهات الثنائية لأغراض التطبيع. تسمح خسارة Sigmoid أيضًا بزيادة حجم الدفعة في نفس الوقت، مع تحسين الأداء عند أحجام دفعات أصغر. وبالاقتران مع ضبط الصورة المقفلة، باستخدام أربع شرائح TPUv4 فقط، نقوم بتدريب نموذج SigLiT يحقق دقة 84.5٪ على ImageNet بدون إشراف في يومين. كما يسمح فصل حجم الدفعة عن الخسارة لدراسة تأثير الأمثلة مقابل الأزواج ونسبة السلبيات إلى الإيجابيات. وأخيرًا، نقوم بدفع حجم الدفعة إلى أقصى حد، حتى مليون، ونجد أن فوائد زيادة حجم الدفعة تتلاشى بسرعة، حيث يكون حجم الدفعة المعقول 32 ألفًا كافيًا.*

## نصائح الاستخدام

- استخدام SigLIP مشابه لـ [CLIP](clip). الفرق الرئيسي هو خسارة التدريب، والتي لا تتطلب رؤية شاملة لجميع التشابهات الثنائية للصور والنصوص داخل دفعة. يجب تطبيق دالة التنشيط Sigmoid على logits، بدلاً من Softmax.

- لا يتم دعم التدريب بعد. إذا كنت ترغب في ضبط نموذج SigLIP أو التدريب من الصفر، راجع دالة الخسارة من [OpenCLIP](https://github.com/mlfoundations/open_clip/blob/73ad04ae7fb93ede1c02dc9040a828634cb1edf1/src/open_clip/loss.py#L307)، والتي تستفيد من مختلف المرافق `torch.distributed`.

- عند استخدام [`SiglipTokenizer`] أو [`SiglipProcessor`] المستقلين، تأكد من تمرير `padding="max_length"` حيث تم تدريب النموذج بهذه الطريقة.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/siglip_table.jpeg"
alt="drawing" width="600"/>

<small> نتائج تقييم SigLIP مقارنة بـ CLIP. مأخوذة من <a href="https://arxiv.org/abs/2303.15343">الورقة البحثية الأصلية</a>.</small>

تمت المساهمة بهذا النموذج بواسطة [nielsr](https://huggingface.co/nielsr).
يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/big_vision/tree/main).

## مثال الاستخدام

هناك طريقتان رئيسيتان لاستخدام SigLIP: إما باستخدام واجهة برمجة التطبيقات الخاصة بالخطوط الأنابيب، والتي تُجرِّد كل التعقيد من أجلك، أو باستخدام فئة `SiglipModel` بنفسك.

### واجهة برمجة التطبيقات الخاصة بالخطوط الأنابيب

تسمح واجهة برمجة التطبيقات باستخدام النموذج في بضع سطور من الكود:

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> # تحميل الأنبوب
>>> image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-224")

>>> # تحميل الصورة
>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # الاستنتاج
>>> outputs = image_classifier(image, candidate_labels=["2 cats", "a plane", "a remote"])
>>> outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
>>> print(outputs)
[{'score': 0.1979, 'label': '2 cats'}, {'score': 0.0, 'label': 'a remote'}, {'score': 0.0, 'label': 'a plane'}]
```

### استخدام النموذج بنفسك

إذا كنت تريد القيام بالمعالجة المسبقة واللاحقة بنفسك، فهذا هو ما يجب فعله:

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, AutoModel
>>> import torch

>>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
>>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
>>> # من المهم: نمرر `padding=max_length` حيث تم تدريب النموذج بهذه الطريقة
>>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits_per_image = outputs.logits_per_image
>>> probs = torch.sigmoid(logits_per_image) # هذه هي الاحتمالات
>>> print(f"{probs[0][0]:.1%} احتمال أن تكون الصورة 0 هي '{texts[0]}'")
31.9% احتمال أن تكون الصورة 0 هي 'a photo of 2 cats'
```

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام SigLIP.

- [دليل مهام التصنيف الصوري بدون إشراف](../tasks/zero_shot_image_classification_md)

- يمكن العثور على دفاتر الملاحظات التجريبية لـ SigLIP [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SigLIP). 🌎

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب وسنراجعه! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

## SiglipConfig

[[autodoc]] SiglipConfig

- from_text_vision_configs

## SiglipTextConfig

[[autodoc]] SiglipTextConfig

## SiglipVisionConfig

[[autodoc]] SiglipVisionConfig

## SiglipTokenizer

[[autodoc]] SiglipTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## SiglipImageProcessor

[[autodoc]] SiglipImageProcessor

- preprocess

## SiglipProcessor

[[autodoc]] SiglipProcessor

## SiglipModel

[[autodoc]] SiglipModel

- forward
- get_text_features
- get_image_features

## SiglipTextModel

[[autodoc]] SiglipTextModel

- forward

## SiglipVisionModel

[[autodoc]] SiglipVisionModel

- forward

## SiglipForImageClassification

[[autodoc]] SiglipForImageClassification

- forward