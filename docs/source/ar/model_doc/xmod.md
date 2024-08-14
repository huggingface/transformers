# X-MOD

## نظرة عامة

اقترح نموذج X-MOD في [رفع لعنة تعدد اللغات عن طريق تدريب المحولات النمطية](http://dx.doi.org/10.18653/v1/2022.naacl-main.255) بواسطة Jonas Pfeiffer، Naman Goyal، Xi Lin، Xian Li، James Cross، Sebastian Riedel، و Mikel Artetxe.
يوسع X-MOD نماذج اللغة المقنعة متعددة اللغات مثل [XLM-R](xlm-roberta) لتشمل مكونات نمطية خاصة باللغة (_مهايئات اللغة_) أثناء التدريب المسبق. بالنسبة للضبط الدقيق، يتم تجميد مهايئات اللغة في كل طبقة محول.

الملخص من الورقة هو ما يلي:

*من المعروف أن النماذج متعددة اللغات المُدربة مسبقًا تعاني من لعنة تعدد اللغات، والتي تتسبب في انخفاض الأداء لكل لغة مع زيادة عدد اللغات التي تغطيها. نحن نعالج هذه المشكلة عن طريق تقديم وحدات نمطية خاصة باللغة، مما يسمح لنا بزيادة السعة الإجمالية للنموذج، مع الحفاظ على إجمالي عدد المعلمات القابلة للتدريب لكل لغة ثابتة. على عكس العمل السابق الذي يتعلم المكونات الخاصة باللغة بعد التدريب، نقوم بتدريب وحدات نمطية لنموذجنا Cross-lingual Modular (X-MOD) من البداية. تُظهر تجاربنا على الاستدلال اللغوي الطبيعي، وتعريف الكيانات المسماة، والإجابة على الأسئلة أن نهجنا لا يخفف من التداخل السلبي بين اللغات فحسب، بل يمكّن أيضًا من النقل الإيجابي، مما يؤدي إلى تحسين الأداء أحادي اللغة والمشترك بين اللغات. علاوة على ذلك، يمكّن نهجنا من إضافة لغات بعد التدريب دون انخفاض ملحوظ في الأداء، مما لم يعد يقيد استخدام النموذج لمجموعة اللغات المدربة مسبقًا.*

تمت المساهمة بهذا النموذج بواسطة [jvamvas](https://huggingface.co/jvamvas).
يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/fairseq/models/xmod) والوثائق الأصلية موجودة [هنا](https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/examples/xmod).

## نصائح الاستخدام

نصائح:
- X-MOD مشابه لـ [XLM-R](xlm-roberta)، ولكن الاختلاف هو أن لغة الإدخال تحتاج إلى التحديد حتى يمكن تنشيط مهايئ اللغة الصحيح.
- تحتوي النماذج الرئيسية - الأساسية والكبيرة - على مهايئات لـ 81 لغة

## استخدام Adapter

### لغة الإدخال

هناك طريقتان لتحديد لغة الإدخال:

1. عن طريق تعيين لغة افتراضية قبل استخدام النموذج:

```python
from transformers import XmodModel

model = XmodModel.from_pretrained("facebook/xmod-base")
model.set_default_language("en_XX")
```

2. عن طريق تمرير مؤشر مهايئ اللغة لكل عينة بشكل صريح:

```python
import torch

input_ids = torch.tensor(
    [
        [0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2],
        [0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2],
    ]
)
lang_ids = torch.LongTensor(
    [
        0,  # en_XX
        8,  # de_DE
    ]
)
output = model(input_ids, lang_ids=lang_ids)
```

### التعديل الدقيق

توصي الورقة بتجميد طبقة التضمين ومهايئات اللغة أثناء التعديل الدقيق. يتم توفير طريقة للقيام بذلك:

```python
model.freeze_embeddings_and_language_adapters()
# Fine-tune the model ...
```

### النقل عبر اللغات

بعد التعديل الدقيق، يمكن اختبار النقل الصفري عبر اللغات عن طريق تنشيط مهايئ لغة الهدف:

```python
model.set_default_language("de_DE")
# Evaluate the model on German examples ...
```

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## XmodConfig

[[autodoc]] XmodConfig

## XmodModel

[[autodoc]] XmodModel

- forward

## XmodForCausalLM

[[autodoc]] XmodForCausalLM

- forward

## XmodForMaskedLM

[[autodoc]] XmodForMaskedLM

- forward

## XmodForSequenceClassification

[[autodoc]] XmodForSequenceClassification

- forward

## XmodForMultipleChoice

[[autodoc]] XmodForMultipleChoice

- forward

## XmodForTokenClassification

[[autodoc]] XmodForTokenClassification

- forward

## XmodForQuestionAnswering

[[autodoc]] XmodForQuestionAnswering

- forward
