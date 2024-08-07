# NLLB

## سلوك محدث للمحلل الرمزي

**إخلاء المسؤولية:** تم إصلاح السلوك الافتراضي للمحلل الرمزي وبالتالي تغييره في أبريل 2023.

يضيف الإصدار السابق [self.eos_token_id, self.cur_lang_code] في نهاية تسلسل الرموز لكل من تحليل الرموز المصدر والهدف. وهذا خطأ لأن ورقة NLLB تذكر (الصفحة 48، 6.1.1. بنية النموذج):

> "لاحظ أننا نضيف بادئة إلى تسلسل المصدر بلغة المصدر، على عكس لغة الهدف كما تم القيام به سابقًا في عدة أعمال (Arivazhagan et al.، 2019؛ Johnson et al.، 2017). ويرجع ذلك في المقام الأول إلى أننا نعطي الأولوية لتحسين أداء الصفر للتصوير في أي زوج من 200 لغة بتكلفة بسيطة للأداء الخاضع للإشراف".

السلوك السابق:

```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> tokenizer("How was your day?").input_ids
[13374, 1398, 4260, 4039, 248130, 2, 256047]

>>> # 2: '</s>'
>>> # 256047 : 'eng_Latn'
```

السلوك الجديد:

```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> tokenizer("How was your day?").input_ids
[256047, 13374, 1398, 4260, 4039, 248130, 2]
```

يمكن تمكين السلوك القديم كما يلي:

```python
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
```

للحصول على مزيد من التفاصيل، لا تتردد في التحقق من PR و Issue المرتبطين.

## نظرة عامة

تم تقديم نموذج NLLB في [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) بواسطة Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang.

ملخص الورقة هو كما يلي:

> "انطلاقًا من هدف القضاء على الحواجز اللغوية على نطاق عالمي، رسخت الترجمة الآلية نفسها كمحور رئيسي لبحوث الذكاء الاصطناعي اليوم. ومع ذلك، فقد تجلت هذه الجهود في مجموعة فرعية صغيرة من اللغات، تاركة وراءها الغالبية العظمى من اللغات منخفضة الموارد في الغالب. ما الذي يتطلبه الأمر لاختراق حاجز 200 لغة مع ضمان نتائج آمنة وعالية الجودة، مع مراعاة الاعتبارات الأخلاقية في الوقت نفسه؟ في No Language Left Behind، تصدينا لهذا التحدي من خلال وضع سياق الحاجة إلى دعم الترجمة من اللغات منخفضة الموارد من خلال المقابلات الاستكشافية مع المتحدثين الأصليين. بعد ذلك، قمنا بإنشاء مجموعات بيانات ونماذج تهدف إلى تضييق الفجوة في الأداء بين اللغات منخفضة وذات الموارد العالية. على وجه التحديد، قمنا بتطوير نموذج حساب مشروط يعتمد على Sparse Gated Mixture of Experts تم تدريبه على بيانات تم الحصول عليها بتقنيات فعالة وجديدة لتعدين البيانات مصممة للغات منخفضة الموارد. نقترح العديد من التحسينات المعمارية والتدريبية لمكافحة الإفراط في التعلم أثناء التدريب على آلاف المهام. ومن الأهمية بمكان أننا قمنا بتقييم أداء أكثر من 40,000 اتجاه ترجمة مختلف باستخدام معيار مرجعي مترجم بشري، Flores-200، ودمج التقييم البشري مع معيار سمية جديد يغطي جميع اللغات في Flores-200 لتقييم سلامة الترجمة. يحقق نموذجنا تحسنًا بنسبة 44% في BLEU مقارنة بحالة الفنون السابقة، مما يضع أساسًا مهمًا نحو تحقيق نظام ترجمة شامل."

يحتوي هذا التنفيذ على النماذج الكثيفة المتاحة عند الإصدار.

**نموذج NLLB-MoE (خليط الخبراء) متاح الآن! مزيد من التفاصيل [هنا](nllb-moe)**

تمت المساهمة بهذا النموذج من قبل [Lysandre](https://huggingface.co/lysandre). يمكن العثور على كود المؤلفين [هنا](https://github.com/facebookresearch/fairseq/tree/nllb).

## التوليد باستخدام NLLB

عند توليد النص المستهدف، قم بتعيين `forced_bos_token_id` إلى معرف لغة الهدف. يوضح المثال التالي كيفية ترجمة الإنجليزية إلى الفرنسية باستخدام نموذج *facebook/nllb-200-distilled-600M*.

لاحظ أننا نستخدم رمز BCP-47 للفرنسية `fra_Latn`. راجع [هنا](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) للحصول على قائمة بجميع رموز BCP-47 في مجموعة بيانات Flores 200.

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

>>> article = "UN Chief says there is no military solution in Syria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
Le chef de l'ONU dit qu'il n'y a pas de solution militaire en Syrie
```

### التوليد من أي لغة أخرى غير الإنجليزية

تم تعيين الإنجليزية (`eng_Latn`) كلغة افتراضية للترجمة منها. لتحديد أنك تريد الترجمة من لغة مختلفة، يجب عليك تحديد رمز BCP-47 في وسيط `src_lang` من تهيئة المحلل الرمزي.

انظر المثال أدناه للترجمة من الرومانية إلى الألمانية:

```py
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained(
...     "facebook/nllb-200-distilled-600M", token=True, src_lang="ron_Latn"
... )
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token=True)

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
UN-Chef sagt, es gibt keine militärische Lösung in Syrien
```

## الموارد

- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)

## NllbTokenizer

[[autodoc]] NllbTokenizer

- build_inputs_with_special_tokens

## NllbTokenizerFast

[[autodoc]] NllbTokenizerFast

## استخدام Flash Attention 2

Flash Attention 2 هو إصدار أسرع وأكثر تحسينًا لحساب درجات الاهتمام والذي يعتمد على نوى `cuda`.

### التثبيت

أولاً، تحقق مما إذا كان عتادك متوافقًا مع Flash Attention 2. يمكن العثور على أحدث قائمة بالأجهزة المتوافقة في [الوثائق الرسمية](https://github.com/Dao-AILab/flash-attention#installation-and-features).

بعد ذلك، قم بتثبيت أحدث إصدار من Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

### الاستخدام

لتحميل نموذج باستخدام Flash Attention 2، يمكننا تمرير وسيط `attn_implementation="flash_attention_2"` إلى [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). يمكنك استخدام الدقة `torch.float16` أو `torch.bfloat16`.

```python
>>> import torch
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to("cuda").eval()
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt").to("cuda")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"UN-Chef sagt, es gibt keine militärische Lösung in Syrien"
```

### تسريع الأداء المتوقع

فيما يلي رسم بياني للتسريع المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي وFlash Attention 2.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/visheratin/documentation-images/resolve/main/nllb-speedup.webp">
</div>