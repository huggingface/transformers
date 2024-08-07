# NLLB-MOE

## نظرة عامة

تم تقديم نموذج NLLB في ورقة "No Language Left Behind: Scaling Human-Centered Machine Translation" بواسطة Marta R. Costa-jussà وآخرون. يهدف النموذج إلى كسر حاجز الـ 200 لغة مع ضمان نتائج آمنة وعالية الجودة، مع مراعاة الاعتبارات الأخلاقية. وقد قام المؤلفون بإجراء مقابلات استكشافية مع متحدثين أصليين لتوضيح الحاجة إلى دعم الترجمة للغات منخفضة الموارد، ثم قاموا بتطوير مجموعات بيانات ونماذج تهدف إلى تضييق الفجوة في الأداء بين اللغات منخفضة و عالية الموارد.

يقدم النموذج تحسينات معمارية وتدريبية لمكافحة الإفراط في التعلم أثناء التدريب على آلاف المهام. ومن المهم أن المؤلفين قاموا بتقييم أداء أكثر من 40,000 اتجاه ترجمة مختلف باستخدام معيار مرجعي مترجم بشريًا، وهو Flores-200، ودمجوا التقييم البشري مع معيار جديد للسمية يغطي جميع اللغات في Flores-200 لتقييم سلامة الترجمة.

يحقق النموذج تحسنًا بنسبة 44% في BLEU مقارنة بحالة التقنية السابقة، مما يضع أساسًا مهمًا نحو تحقيق نظام ترجمة شامل.

## نصائح الاستخدام

- M2M100ForConditionalGeneration هو النموذج الأساسي لكل من NLLB وNLLB MoE.
- يشبه NLLB-MoE نموذج NLLB إلى حد كبير، ولكنه يعتمد على تنفيذ SwitchTransformers في طبقة التغذية الأمامية الخاصة به.
- مُشَرِّح الرموز هو نفسه المستخدم في نماذج NLLB.

## الاختلافات في التنفيذ مع SwitchTransformers

أكبر اختلاف هو طريقة توجيه الرموز المميزة. يستخدم NLLB-MoE طريقة "top-2-gate"، والتي يتم من خلالها اختيار أفضل خبيرين فقط لكل إدخال بناءً على أعلى الاحتمالات المتوقعة من شبكة التوجيه، بينما يتم تجاهل الخبراء الآخرين. في SwitchTransformers، يتم حساب احتمالات أعلى مستوى فقط، مما يعني أن الرموز المميزة لديها احتمال أقل في التوجيه. علاوة على ذلك، إذا لم يتم توجيه رمز مميز إلى أي خبير، فإن SwitchTransformers لا تزال تضيف حالاته المخفية غير المعدلة (نوع من الاتصال المتبقي)، في حين يتم إخفاؤها في آلية التوجيه "top-2" الخاصة بـ NLLB.

## التوليد باستخدام NLLB-MoE

تتطلب نقاط التفتيش المتاحة حوالي 350 جيجابايت من مساحة التخزين. تأكد من استخدام "accelerate" إذا لم يكن لديك ذاكرة وصول عشوائي كافية على جهازك.

عند توليد النص المستهدف، قم بتعيين "forced_bos_token_id" إلى معرف اللغة المستهدفة. يوضح المثال التالي كيفية ترجمة الإنجليزية إلى الفرنسية باستخدام نموذج *facebook/nllb-200-distilled-600M*.

لاحظ أننا نستخدم الرمز BCP-47 للفرنسية `fra_Latn`. يمكنك الاطلاع على قائمة رموز BCP-47 الكاملة في مجموعة بيانات Flores 200 [هنا](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200).

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")

>>> article = "Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in his garage."
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=50
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Auparavant, le PDG de Ring, Jamie Siminoff, a fait remarquer que la société avait commencé lorsque sa sonnette n'était pas audible depuis sa boutique dans son garage."
```

### التوليد من أي لغة أخرى غير الإنجليزية

الإنجليزية (`eng_Latn`) هي اللغة الافتراضية للترجمة. لتحديد لغة مختلفة للترجمة منها، يجب تحديد رمز BCP-47 في وسيط `src_lang` أثناء تهيئة المُشَرِّح.

يوضح المثال التالي الترجمة من الرومانية إلى الألمانية:

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b", src_lang="ron_Latn")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
```

## الموارد

- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)

## NllbMoeConfig

[[autodoc]] NllbMoeConfig

## NllbMoeTop2Router

[[autodoc]] NllbMoeTop2Router
- route_tokens
- forward

## NllbMoeSparseMLP

[[autodoc]] NllbMoeSparseMLP
- forward

## NllbMoeModel

[[autodoc]] NllbMoeModel
- forward

## NllbMoeForConditionalGeneration

[[autodoc]] NllbMoeForConditionalGeneration
- forward