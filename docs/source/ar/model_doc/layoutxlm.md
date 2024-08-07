# LayoutXLM

## نظرة عامة

اقترح LayoutXLM في [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836) بواسطة Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei. وهو امتداد متعدد اللغات لنموذج [LayoutLMv2](https://arxiv.org/abs/2012.14740) تم تدريبه على 53 لغة.

الملخص من الورقة هو ما يلي:

> حقق التعلم المتعدد الوسائط باستخدام النص والتخطيط والصورة أداءً متميزًا في مهام فهم المستندات الغنية بالمعلومات المرئية مؤخرًا، مما يدل على الإمكانات الكبيرة للتعلم المشترك عبر الوسائط المختلفة. في هذه الورقة، نقدم LayoutXLM، وهو نموذج متعدد الوسائط مُدرب مسبقًا لفهم المستندات متعددة اللغات، يهدف إلى سد الحواجز اللغوية لفهم المستندات الغنية بالمعلومات المرئية. ولتقييم LayoutXLM بدقة، نقدم أيضًا مجموعة بيانات مرجعية لفهم النماذج متعددة اللغات تسمى XFUN، والتي تتضمن نماذج لفهم النماذج باللغة الصينية واليابانية والإسبانية والفرنسية والإيطالية والألمانية والبرتغالية، وتم وضع علامات يدوية على أزواج القيم الرئيسية لكل لغة. وتظهر نتائج التجارب أن نموذج LayoutXLM قد تفوق بشكل كبير على النماذج المُدربة مسبقًا متعددة اللغات الموجودة حاليًا في مجموعة بيانات XFUN.

تمت المساهمة بهذا النموذج بواسطة [nielsr](https://huggingface.co/nielsr). يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/unilm).

## نصائح الاستخدام وأمثلة

يمكنك مباشرةً توصيل أوزان LayoutXLM في نموذج LayoutLMv2، مثل ما يلي:

```python
from transformers import LayoutLMv2Model

model = LayoutLMv2Model.from_pretrained("microsoft/layoutxlm-base")
```

لاحظ أن LayoutXLM لديه محدد خاص به، يعتمد على [`LayoutXLMTokenizer`]/[`LayoutXLMTokenizerFast`]. يمكنك تهيئته كما يلي:

```python
from transformers import LayoutXLMTokenizer

tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
```

مثل LayoutLMv2، يمكنك استخدام [`LayoutXLMProcessor`] (الذي يطبق داخليًا [`LayoutLMv2ImageProcessor`] و [`LayoutXLMTokenizer`]/[`LayoutXLMTokenizerFast`] بالتتابع) لإعداد جميع البيانات للنموذج.

<Tip>
نظرًا لأن بنية LayoutXLM تعادل بنية LayoutLMv2، فيمكن الرجوع إلى [صفحة توثيق LayoutLMv2](layoutlmv2) للحصول على جميع النصائح وأمثلة التعليمات البرمجية ومفكرات Jupyter Notebook.
</Tip>

## LayoutXLMTokenizer

[[autodoc]] LayoutXLMTokenizer
- __call__
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## LayoutXLMTokenizerFast

[[autodoc]] LayoutXLMTokenizerFast
- __call__

## LayoutXLMProcessor

[[autodoc]] LayoutXLMProcessor
- __call__