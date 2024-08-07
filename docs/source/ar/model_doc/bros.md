# BROS

## نظرة عامة
تم اقتراح نموذج BROS في [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539) بواسطة Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park.

BROS اختصار لـ *BERT Relying On Spatiality*. وهو نموذج محول يعتمد على الترميز فقط يأخذ تسلسل الرموز ونطاقاتهم كمدخلات وينتج عنها تسلسل للحالات المخفية. يقوم BROS بتشفير المعلومات المكانية النسبية بدلاً من استخدام المعلومات المكانية المطلقة.

تم تدريب النموذج مسبقًا باستخدام هدفين: هدف نمذجة اللغة المقنعة بالرموز (TMLM) المستخدم في BERT، وهدف نمذجة لغة جديدة تعتمد على المساحة (AMLM).

في TMLM، يتم إخفاء الرموز بشكل عشوائي، ويتوقع النموذج الرموز المخفية باستخدام المعلومات المكانية والرموز الأخرى غير المخفية.

أما AMLM فهو إصدار ثنائي الأبعاد من TMLM. فهو يقوم بإخفاء رموز النص بشكل عشوائي والتنبؤ بنفس المعلومات مثل TMLM، ولكنه يقوم بإخفاء كتل النص (المساحات).

يحتوي `BrosForTokenClassification` على طبقة خطية بسيطة أعلى `BrosModel`. فهو يتنبأ بعلامة كل رمز.

يتميز `BrosSpadeEEForTokenClassification` بـ `initial_token_classifier` و`subsequent_token_classifier` أعلى `BrosModel`. ويُستخدم `initial_token_classifier` للتنبؤ بالرمز الأول لكل كيان، بينما يُستخدم `subsequent_token_classifier` للتنبؤ بالرمز التالي ضمن الكيان. ويحتوي `BrosSpadeELForTokenClassification` على `entity_linker` أعلى `BrosModel`. ويُستخدم `entity_linker` للتنبؤ بالعلاقة بين كيانين.

يقوم كل من `BrosForTokenClassification` و`BrosSpadeEEForTokenClassification` بنفس المهمة بشكل أساسي. ومع ذلك، يفترض `BrosForTokenClassification` أن رموز الإدخال مسلسلة بشكل مثالي (وهي مهمة صعبة للغاية لأنها موجودة في مساحة ثنائية الأبعاد)، في حين يسمح `BrosSpadeEEForTokenClassification` بالمرونة في التعامل مع أخطاء التسلسل لأنه يتنبأ برموز الاتصال التالية من رمز واحد.

يقوم `BrosSpadeELForTokenClassification` بمهمة الربط داخل الكيان. فهو يتنبأ بالعلاقة من رمز واحد (لكيان واحد) إلى رمز آخر (لكيان آخر) إذا كان هذان الكيانان يتشاركان في بعض العلاقات.

يحقق BROS نتائج مماثلة أو أفضل في استخراج المعلومات الأساسية (KIE) مثل FUNSD، SROIE، CORD وSciTSR، دون الاعتماد على الميزات المرئية الصريحة.

ملخص الورقة البحثية هو كما يلي:

> تتطلب عملية استخراج المعلومات الأساسية (KIE) من صور المستندات فهم الدلالات السياقية والمكانية للنصوص في مساحة ثنائية الأبعاد. وتحاول العديد من الدراسات الحديثة حل المهمة من خلال تطوير نماذج اللغة التي تم تدريبها مسبقًا والتي تركز على دمج الميزات المرئية من صور المستندات مع النصوص وتخطيطها. من ناحية أخرى، تتناول هذه الورقة المشكلة من خلال العودة إلى الأساسيات: الدمج الفعال للنص والتخطيط. نقترح نموذج لغة تم تدريبه مسبقًا، يسمى BROS (BERT Relying On Spatiality)، والذي يقوم بتشفير المواضع النسبية للنصوص في مساحة ثنائية الأبعاد ويتعلم من المستندات غير المعنونة باستخدام إستراتيجية التعتيم على المناطق. مع هذا المخطط التدريبي المحسن لفهم النصوص في مساحة ثنائية الأبعاد، يظهر BROS أداءً مماثلاً أو أفضل مقارنة بالطرق السابقة في أربعة مقاييس لتقييم استخراج المعلومات الأساسية (FUNSD، SROIE*، CORD، وSciTSR) دون الاعتماد على الميزات المرئية. تكشف هذه الورقة أيضًا عن تحديين حقيقيين في مهام استخراج المعلومات الأساسية - (1) تقليل الخطأ الناتج عن ترتيب النص غير الصحيح و(2) التعلم الفعال من أمثلة المصب القليلة - وتوضح تفوق BROS على الطرق السابقة.

تمت المساهمة بهذا النموذج من قبل [jinho8345](https://huggingface.co/jinho8345). يمكن العثور على الكود الأصلي [هنا](https://github.com/clovaai/bros).

## نصائح الاستخدام والأمثلة

- يتطلب [`~transformers.BrosModel.forward`] `input_ids` و`bbox` (نطاق التصويب). يجب أن يكون كل نطاق تصويب بتنسيق (x0, y0, x1, y1) (الركن العلوي الأيسر، الركن السفلي الأيمن). يعتمد الحصول على نطاقات التصويب على نظام التعرف البصري على الحروف (OCR) الخارجي. يجب تطبيع إحداثية `x` عن طريق عرض صورة المستند، ويجب تطبيع إحداثية `y` عن طريق ارتفاع صورة المستند.

```python
def expand_and_normalize_bbox(bboxes, doc_width, doc_height):
# هنا، bboxes هي مصفوفة numpy

# تطبيع النطاق -> 0 ~ 1
bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / width
bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / height
```

- تتطلب [`~transformers.BrosForTokenClassification.forward`]، و[`~transformers.BrosSpadeEEForTokenClassification.forward`]، و[`~transformers.BrosSpadeEEForTokenClassification.forward`] ليس فقط `input_ids` و`bbox` ولكن أيضًا `box_first_token_mask` لحساب الخسارة. وهي قناع لتصفية الرموز غير الأولى لكل نطاق. يمكنك الحصول على هذا القناع عن طريق حفظ مؤشرات الرموز الأولى لنطاقات التصويب عند إنشاء `input_ids` من الكلمات. يمكنك إنشاء `box_first_token_mask` باستخدام الكود التالي:

```python
def make_box_first_token_mask(bboxes, words, tokenizer, max_seq_length=512):

    box_first_token_mask = np.zeros(max_seq_length, dtype=np.bool_)

    # تشفير (ترميز) كل كلمة من الكلمات (List[str])
    input_ids_list: List[List[int]] = [tokenizer.encode(e, add_special_tokens=False) for e in words]

    # الحصول على طول كل نطاق
    tokens_length_list: List[int] = [len(l) for l in input_ids_list]

    box_end_token_indices = np.array(list(itertools.accumulate(tokens_length_list)))
    box_start_token_indices = box_end_token_indices - np.array(tokens_length_list)

    # تصفية المؤشرات التي تتجاوز max_seq_length
    box_end_token_indices = box_end_token_indices[box_end_token_indices < max_seq_length - 1]
    if len(box_start_token_indices) > len(box_end_token_indices):
        box_start_token_indices = box_start_token_indices[: len(box_end_token_indices)]

    # تعيين box_start_token_indices إلى True
    box_first_token_mask[box_start_token_indices] = True

    return box_first_token_mask

```

## الموارد

- يمكن العثور على نصوص العرض التوضيحي [هنا](https://github.com/clovaai/bros).

## BrosConfig

[[autodoc]] BrosConfig

## BrosProcessor

[[autodoc]] BrosProcessor

- __call__

## BrosModel

[[autodoc]] BrosModel

- forward

## BrosForTokenClassification

[[autodoc]] BrosForTokenClassification

- forward

## BrosSpadeEEForTokenClassification

[[autodoc]] BrosSpadeEEForTokenClassification

- forward

## BrosSpadeELForTokenClassification

[[autodoc]] BrosSpadeELForTokenClassification

- forward