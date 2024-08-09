# LayoutLM

## نظرة عامة

تم اقتراح نموذج LayoutLM في الورقة البحثية [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) بواسطة Yiheng Xu و Minghao Li و Lei Cui و Shaohan Huang و Furu Wei و Ming Zhou. إنه أسلوب بسيط ولكنه فعال لتمثيل النص والتخطيط لفهم صورة المستند ومهام استخراج المعلومات، مثل فهم النماذج وفهم الإيصالات. يحقق نتائج متميزة في العديد من المهام الفرعية:

- فهم النماذج: مجموعة بيانات [FUNSD](https://guillaumejaume.github.io/FUNSD/) (مجموعة من 199 نموذجًا تمت معاينتها تضم أكثر من 30,000 كلمة).
- فهم الإيصالات: مجموعة بيانات [SROIE](https://rrc.cvc.uab.es/?ch=13) (مجموعة من 626 إيصالًا للتدريب و347 إيصالًا للاختبار).
- تصنيف صورة المستند: مجموعة بيانات [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) (مجموعة من 400,000 صورة تنتمي إلى واحدة من 16 فئة).

فيما يلي ملخص من الورقة البحثية:

*تم التحقق من تقنيات التمثيل المسبق بنجاح في مجموعة متنوعة من مهام معالجة اللغات الطبيعية في السنوات الأخيرة. على الرغم من الاستخدام الواسع لنماذج التمثيل المسبق لتطبيقات معالجة اللغات الطبيعية، فإنها تركز بشكل حصري تقريبًا على معالجة النص، بينما تتجاهل معلومات التخطيط والنمط التي تعد حيوية لفهم صورة المستند. في هذه الورقة، نقترح LayoutLM لنمذجة التفاعلات المشتركة بين نص ومعلومات التخطيط عبر صور المستندات الممسوحة ضوئيًا، والتي تفيد عددًا كبيرًا من مهام فهم صورة المستند في العالم الحقيقي مثل استخراج المعلومات من المستندات الممسوحة ضوئيًا. علاوة على ذلك، نستفيد أيضًا من ميزات الصور لإدراج المعلومات المرئية للكلمات في LayoutLM. حسب أفضل معرفتنا، هذه هي المرة الأولى التي يتم فيها تعلم النص والتخطيط معًا في إطار واحد للتمثيل المسبق على مستوى المستند. يحقق نتائج جديدة متميزة في العديد من المهام الفرعية، بما في ذلك فهم النماذج (من 70.72 إلى 79.27)، وفهم الإيصالات (من 94.02 إلى 95.24) وتصنيف صورة المستند (من 93.07 إلى 94.42).*

## نصائح الاستخدام

بالإضافة إلى *input_ids*، يتوقع [`~transformers.LayoutLMModel.forward`] أيضًا إدخال `bbox`، وهو عبارة عن صناديق محيطة (أي مواضع ثنائية الأبعاد) للرموز المميزة للإدخال. يمكن الحصول على هذه الصناديق باستخدام محرك OCR خارجي مثل [Tesseract](https://github.com/tesseract-ocr/tesseract) من Google (هناك [غلاف Python](https://pypi.org/project/pytesseract/) متاح). يجب أن يكون كل صندوق محيطي بتنسيق (x0, y0, x1, y1)، حيث يتوافق (x0, y0) مع موضع الركن العلوي الأيسر في الصندوق المحيطي، ويمثل (x1, y1) موضع الركن السفلي الأيمن. لاحظ أنه يجب عليك أولاً تطبيع الصناديق المحيطة لتكون على مقياس 0-1000. لتطبيع، يمكنك استخدام الدالة التالية:

```python
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

هنا، `width` و`height` يقابلان عرض وارتفاع المستند الأصلي الذي يحدث فيه الرمز المميز. يمكن الحصول على تلك باستخدام مكتبة Python Image Library (PIL)، على سبيل المثال، كما يلي:

```python
from PIL import Image

# يمكن أن تكون الوثيقة png أو jpg، إلخ. يجب تحويل ملفات PDF إلى صور.
image = Image.open(name_of_your_document).convert("RGB")

width, height = image.size
```

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام LayoutLM. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب Pull Request وسنقوم بمراجعته! يجب أن يُظهر المورد المثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- منشور مدونة حول [ضبط دقيق لـ LayoutLM لفهم المستندات باستخدام Keras وHugging Face Transformers](https://www.philschmid.de/fine-tuning-layoutlm-keras).
- منشور مدونة حول كيفية [ضبط دقيق لـ LayoutLM لفهم المستندات باستخدام Hugging Face Transformers فقط](https://www.philschmid.de/fine-tuning-layoutlm).
- دفتر ملاحظات حول كيفية [ضبط دقيق لـ LayoutLM على مجموعة بيانات FUNSD مع تضمين الصور](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb).
- راجع أيضًا: [دليل مهام الإجابة على الأسئلة الوثائقية](../tasks/document_question_answering)

- دفتر ملاحظات حول كيفية [ضبط دقيق لـ LayoutLM لتصنيف التسلسلات على مجموعة بيانات RVL-CDIP](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb).
- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)

- دفتر ملاحظات حول كيفية [ضبط دقيق لـ LayoutLM لتصنيف الرموز المميزة على مجموعة بيانات FUNSD](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb).
- [دليل مهام تصنيف الرموز المميزة](../tasks/token_classification)

**موارد أخرى**

- [دليل مهام نمذجة اللغة المُقنعة](../tasks/masked_language_modeling)

🚀 النشر

- منشور مدونة حول كيفية [نشر LayoutLM مع نقاط النهاية الاستدلالية لـ Hugging Face](https://www.philschmid.de/inference-endpoints-layoutlm).

## LayoutLMConfig

[[autodoc]] LayoutLMConfig

## LayoutLMTokenizer

[[autodoc]] LayoutLMTokenizer

## LayoutLMTokenizerFast

[[autodoc]] LayoutLMTokenizerFast

<frameworkcontent>
<pt>

## LayoutLMModel

[[autodoc]] LayoutLMModel

## LayoutLMForMaskedLM

[[autodoc]] LayoutLMForMaskedLM

## LayoutLMForSequenceClassification

[[autodoc]] LayoutLMForSequenceClassification

## LayoutLMForTokenClassification

[[autodoc]] LayoutLMForTokenClassification

## LayoutLMForQuestionAnswering

[[autodoc]] LayoutLMForQuestionAnswering

</pt>
<tf>

## TFLayoutLMModel

[[autodoc]] TFLayoutLMModel

## TFLayoutLMForMaskedLM

[[autodoc]] TFLayoutLMForMaskedLM

## TFLayoutLMForSequenceClassification

[[autodoc]] TFLayoutLMForSequenceClassification

## TFLayoutLMForTokenClassification

[[autodoc]] TFLayoutLMForTokenClassification

## TFLayoutLMForQuestionAnswering

[[autodoc]] TFLayoutLMForQuestionAnswering

</tf>
</frameworkcontent>