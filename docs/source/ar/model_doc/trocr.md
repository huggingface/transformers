# TrOCR

## نظرة عامة

اقتُرح نموذج TrOCR في ورقة بحثية بعنوان "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models" من قبل Minghao Li وآخرون. يتكون نموذج TrOCR من محول صور (Image Transformer) وترميز نصي نصي تلقائي (autoregressive text Transformer decoder) لأداء التعرف البصري على الحروف (OCR).

وفيما يلي الملخص المستخلص من الورقة البحثية:

"يعد التعرف على النص مشكلة بحثية قائمة منذ فترة طويلة في مجال رقمنة المستندات. وتعتمد الطرق الحالية للتعرف على النص عادةً على شبكات CNN لفهم الصور وRNN لتوليد النص على مستوى الأحرف. بالإضافة إلى ذلك، تكون هناك حاجة عادةً إلى نموذج لغوي آخر لتحسين الدقة الإجمالية كخطوة ما بعد المعالجة. وفي هذه الورقة، نقترح نهجًا مباشرًا للتعرف على النص باستخدام محولات الصور والنصوص المسبقة التدريب، والتي يطلق عليها TrOCR، والتي تستفيد من بنية المحول لكل من فهم الصور وتوليد النص على مستوى wordpiece. ونموذج TrOCR بسيط وفعال، ويمكن تدريبه مسبقًا باستخدام بيانات ضخمة مصطنعة وضبط دقته باستخدام مجموعات بيانات موسومة يدويًا. وتظهر التجارب أن نموذج TrOCR يتفوق على النماذج الحالية المتقدمة في مهام التعرف على النص المطبوع والمكتوب بخط اليد."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trocr_architecture.jpg" alt="drawing" width="600"/>

<small> بنية نموذج TrOCR. مأخوذة من <a href="https://arxiv.org/abs/2109.10282">الورقة البحثية الأصلية</a>. </small>

يرجى الرجوع إلى فئة [`VisionEncoderDecoder`] لمعرفة كيفية استخدام هذا النموذج.

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). ويمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/unilm/tree/6f60612e7cc86a2a1ae85c47231507a587ab4e01/trocr).

## نصائح الاستخدام

- أسرع طريقة للبدء في استخدام TrOCR هي من خلال الاطلاع على [دفاتر الملاحظات التعليمية](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR)، والتي توضح كيفية استخدام النموذج في وقت الاستدلال، بالإضافة إلى الضبط الدقيق على بيانات مخصصة.

- يتم تدريب نموذج TrOCR مسبقًا على مرحلتين قبل ضبط دقته على مجموعات بيانات خاصة بمهمة معينة. ويحقق نتائج متميزة في كل من التعرف على النص المطبوع (مثل مجموعة بيانات SROIE) والمكتوب بخط اليد (مثل مجموعة بيانات IAM Handwriting dataset). لمزيد من المعلومات، يرجى الاطلاع على [النماذج الرسمية](https://huggingface.co/models?other=trocr>).

- يتم استخدام نموذج TrOCR دائمًا ضمن إطار عمل [VisionEncoderDecoder](vision-encoder-decoder).

## الموارد

فيما يلي قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء في استخدام TrOCR. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب (Pull Request) وسنقوم بمراجعته! ويفضل أن يظهر المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="text-classification"/>

- منشور مدونة حول [تسريع واجهة برمجة تطبيقات المستندات](https://huggingface.co/blog/document-ai) باستخدام TrOCR.

- منشور مدونة حول كيفية [واجهة برمجة تطبيقات المستندات](https://github.com/philschmid/document-ai-transformers) باستخدام TrOCR.

- دفتر ملاحظات حول كيفية [الضبط الدقيق لنموذج TrOCR على مجموعة بيانات IAM Handwriting Database باستخدام Seq2SeqTrainer](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb).

- دفتر ملاحظات حول [الاستدلال باستخدام TrOCR](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Inference_with_TrOCR_%2B_Gradio_demo.ipynb) وتجربة Gradio.

- دفتر ملاحظات حول [الضبط الدقيق لنموذج TrOCR على مجموعة بيانات IAM Handwriting Database](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_native_PyTorch.ipynb) باستخدام PyTorch الأصلي.

- دفتر ملاحظات حول [تقييم نموذج TrOCR على مجموعة اختبار IAM](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Evaluating_TrOCR_base_handwritten_on_the_IAM_test_set.ipynb).

<PipelineTag pipeline="text-generation"/>

- دليل مهمة [نمذجة اللغة العادية](https://huggingface.co/docs/transformers/tasks/language_modeling).

⚡️ الاستدلال

- عرض توضيحي تفاعلي حول [التعرف على الأحرف المكتوبة بخط اليد باستخدام TrOCR](https://huggingface.co/spaces/nielsr/TrOCR-handwritten).

## الاستدلال

يقبل نموذج [`VisionEncoderDecoder`] في TrOCR الصور كمدخلات ويستخدم [`~generation.GenerationMixin.generate`] لتوليد النص تلقائيًا بناءً على صورة المدخل.

تتولى فئة [`ViTImageProcessor`/`DeiTImageProcessor`] مسؤولية معالجة صورة المدخل، بينما تقوم فئة [`RobertaTokenizer`/`XLMRobertaTokenizer`] بفك تشفير الرموز المولدة إلى سلسلة الهدف. وتجمع فئة [`TrOCRProcessor`] بين [`ViTImageProcessor`/`DeiTImageProcessor`] و [`RobertaTokenizer`/`XLMRobertaTokenizer`] في مثيل واحد لاستخراج ميزات المدخلات وفك تشفير رموز الهدف المتوقعة.

- التعرف البصري على الحروف (OCR) خطوة بخطوة

``` py
>>> from transformers import TrOCRProcessor, VisionEncoderDecoderModel
>>> import requests
>>> from PIL import Image

>>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
>>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

>>> # تحميل صورة من مجموعة بيانات IAM
>>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> pixel_values = processor(image, return_tensors="pt").pixel_values
>>> generated_ids = model.generate(pixel_values)

>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

يمكنك الاطلاع على [مركز النماذج](https://huggingface.co/models?filter=trocr) للبحث عن نقاط تفتيش TrOCR.

## TrOCRConfig

[[autodoc]] TrOCRConfig

## TrOCRProcessor

[[autodoc]] TrOCRProcessor

- __call__
- from_pretrained
- save_pretrained
- batch_decode
- decode

## TrOCRForCausalLM

[[autodoc]] TrOCRForCausalLM

- forward