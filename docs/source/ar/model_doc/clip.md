# CLIP

## نظرة عامة

اقترح نموذج CLIP في ورقة البحث [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) من قبل Alec Radford و Jong Wook Kim و Chris Hallacy و Aditya Ramesh و Gabriel Goh و Sandhini Agarwal و Girish Sastry و Amanda Askell و Pamela Mishkin و Jack Clark و Gretchen Krueger و Ilya Sutskever. CLIP (Contrastive Language-Image Pre-Training) هي شبكة عصبية مدربة على مجموعة متنوعة من أزواج (الصورة، النص). يمكن توجيهه باستخدام اللغة الطبيعية للتنبؤ بأكثر جزء من النص ملاءمة، بالنظر إلى صورة، دون التحسين المباشر للمهمة، على غرار القدرات ذات الصفر للتصوير المقطعي GPT-2 و 3.

الملخص من الورقة هو ما يلي:

> "تدرب أنظمة رؤية الكمبيوتر المتقدمة على التنبؤ بمجموعة ثابتة من فئات الكائنات المحددة مسبقًا. تحد هذه الصيغة المقيدة من الإشراف من عموميتها وقابليتها للاستخدام نظرًا للحاجة إلى بيانات موسومة إضافية لتحديد أي مفهوم بصري آخر. التعلم مباشرة من النص الخام حول الصور هو بديل واعد والذي يستفيد من مصدر أوسع بكثير من الإشراف. نحن نثبت أن مهمة ما قبل التدريب البسيطة المتمثلة في التنبؤ بالتعليق التوضيحي الذي يتوافق مع الصورة هي طريقة فعالة وقابلة للتطوير لتعلم تمثيلات SOTA للصور من الصفر على مجموعة بيانات تحتوي على 400 مليون زوج من (الصورة، النص) تم جمعها من الإنترنت. بعد ما قبل التدريب، يتم استخدام اللغة الطبيعية للإشارة إلى المفاهيم المرئية المكتسبة (أو وصف مفاهيم جديدة) لتمكين النقل الصفري للنموذج إلى مهام التدفق السفلي. نقوم بدراسة أداء هذا النهج من خلال وضع معايير لأكثر من 30 مجموعة بيانات رؤية حاسوبية مختلفة، تشمل مهام مثل التعرف الضوئي على الحروف، والتعرف على الإجراءات في مقاطع الفيديو، والجغرافيا، والعديد من أنواع التصنيف الدقيق للكائنات. ينتقل النموذج بشكل غير تافه إلى معظم المهام ويكون غالبًا قادرًا على المنافسة مع خط الأساس الخاضع للإشراف الكامل دون الحاجة إلى أي تدريب محدد لمجموعة البيانات. على سبيل المثال، نحن نطابق دقة ResNet-50 الأصلية على ImageNet Zero-shot دون الحاجة إلى استخدام أي من 1.28 مليون مثال تدريبي تم تدريبه عليها. نقوم بإصدار الكود الخاص بنا وأوزان النموذج المسبق التدريب على هذا الرابط https."

تمت المساهمة بهذا النموذج من قبل [valhalla](https://huggingface.co/valhalla). يمكن العثور على الكود الأصلي [هنا](https://github.com/openai/CLIP).

## نصائح الاستخدام ومثال

CLIP هو نموذج متعدد الوسائط للرؤية واللغة. يمكن استخدامه لتشابه الصور والنص ولتصنيف الصور ذات الصفر. يستخدم CLIP محولًا مثل ViT لاستخراج الميزات المرئية ونموذج لغة سببي لاستخراج ميزات النص. ثم يتم إسقاط كل من ميزات النص والمرئيات إلى مساحة كامنة ذات أبعاد متطابقة. يتم بعد ذلك استخدام المنتج النقطي بين الصورة وميزات النص المسقطة كنتيجة مماثلة.

لتغذية الصور في محول الترميز، يتم تقسيم كل صورة إلى تسلسل من رقع ثابتة الحجم غير المتداخلة، والتي يتم تضمينها خطيًا بعد ذلك. يتم إضافة رمز [CLS] ليعمل كتمثيل لصورة كاملة. يضيف المؤلفون أيضًا تضمينات الموضع المطلق، ويغذون تسلسل المتجهات الناتج في محول ترميز قياسي. يمكن استخدام [`CLIPImageProcessor`] لإعادة تحجيم (أو إعادة تحجيم) الصور وتطبيعها للنموذج.

يُستخدم [`CLIPTokenizer`] لتشفير النص. يغلف [`CLIPProcessor`] [`CLIPImageProcessor`] و [`CLIPTokenizer`] في مثيل واحد لتشفير النص وإعداد الصور. يوضح المثال التالي كيفية الحصول على درجات التشابه بين الصورة والنص باستخدام [`CLIPProcessor`] و [`CLIPModel`].

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image # هذه هي نتيجة التشابه بين الصورة والنص
>>> probs = logits_per_image.softmax(dim=1) # يمكننا أخذ softmax للحصول على احتمالات التصنيف
```

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها برمز 🌎) لمساعدتك في البدء باستخدام CLIP.

- [ضبط دقيق لـ CLIP باستخدام صور الاستشعار عن بعد (الصور الملتقطة عبر الأقمار الصناعية) والتعليقات التوضيحية](https://huggingface.co/blog/fine-tune-clip-rsicd)، وهي مشاركة مدونة حول كيفية ضبط دقة CLIP باستخدام [مجموعة بيانات RSICD](https://github.com/201528014227051/RSICD_optimal) ومقارنة التغييرات في الأداء بسبب زيادة البيانات.
- يوضح هذا [سكريبت المثال](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text) كيفية تدريب نموذج تشفير ثنائي للرؤية والنص يشبه CLIP باستخدام مشفر رؤية ونص مسبق التدريب باستخدام [مجموعة بيانات COCO](https://cocodataset.org/#home).
<PipelineTag pipeline="image-to-text"/>

- [دفتر الملاحظات](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing) حول كيفية استخدام CLIP مسبق التدريب للتنفيذ مع البحث الشعاعي لتوليد التعليقات التوضيحية للصور. 🌎

**استرجاع الصور**

- [دفتر ملاحظات](https://colab.research.google.com/drive/1bLVwVKpAndpEDHqjzxVPr_9nGrSbuOQd?usp=sharing) حول استرجاع الصور باستخدام CLIP مسبق التدريب وحساب MRR (متوسط المرتبة المتبادلة). 🌎
- [دفتر ملاحظات](https://colab.research.google.com/github/deep-diver/image_search_with_natural_language/blob/main/notebooks/Image_Search_CLIP.ipynb) حول استرجاع الصور وإظهار نتيجة التشابه. 🌎
- [دفتر الملاحظات](https://colab.research.google.com/drive/1xO-wC_m_GNzgjIBQ4a4znvQkvDoZJvH4?usp=sharing) حول كيفية رسم خرائط للصور والنصوص إلى نفس مساحة المتجه باستخدام CLIP متعددة اللغات. 🌎
- [دفتر الملاحظات](https://colab.research.google.com/github/vivien000/clip-demo/blob/master/clip.ipynb#scrollTo=uzdFhRGqiWkR) حول كيفية تشغيل CLIP على البحث عن الصور الدلالية باستخدام [Unsplash](https://unsplash.com) و [TMDB](https://www.themoviedb.org/) مجموعات البيانات. 🌎

**القابلية للتفسير**

- [دفتر ملاحظات](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb) حول كيفية تصور التشابه بين رمز الإدخال وقسم الصورة. 🌎

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه.

يجب أن يثبت المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

## CLIPConfig

[[autodoc]] CLIPConfig

- from_text_vision_configs

## CLIPTextConfig

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer

[[autodoc]] CLIPTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## CLIPTokenizerFast

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor

[[autodoc]] CLIPImageProcessor

- preprocess

## CLIPFeatureExtractor

[[autodoc]] CLIPFeatureExtractor

## CLIPProcessor

[[autodoc]] CLIPProcessor

<frameworkcontent>
<pt>

## CLIPModel

[[autodoc]] CLIPModel

- forward
- get_text_features
- get_image_features

## CLIPTextModel

[[autodoc]] CLIPTextModel

- forward

## CLIPTextModelWithProjection

[[autodoc]] CLIPTextModelWithProjection

- forward

## CLIPVisionModelWithProjection

[[autodoc]] CLIPVisionModelWithProjection

- forward

## CLIPVisionModel

[[autodoc]] CLIPVisionModel

- forward

## CLIPForImageClassification

[[autodoc]] CLIPForImageClassification

- forward

</pt>
<tf>

## TFCLIPModel

[[autodoc]] TFCLIPModel

- call
- get_text_features
- get_image_features


## TFCLIPTextModel

[[autodoc]] TFCLIPTextModel

- call

## TFCLIPVisionModel

[[autodoc]] TFCLIPVisionModel

- call

</tf>
<jax>

## FlaxCLIPModel

[[autodoc]] FlaxCLIPModel

- __call__
- get_text_features
- get_image_features

## FlaxCLIPTextModel

[[autodoc]] FlaxCLIPTextModel

- __call__

## FlaxCLIPTextModelWithProjection

[[autodoc]] FlaxCLIPTextModelWithProjection

- __call__

## FlaxCLIPVisionModel

[[autodoc]] FlaxCLIPVisionModel

- __call__

</jax>
</frameworkcontent>