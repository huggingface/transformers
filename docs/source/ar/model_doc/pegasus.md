# Pegasus

## نظرة عامة

اقتُرح نموذج Pegasus في ورقة بحثية بعنوان: "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization" من قبل جينغكينغ جانغ، وياو تشاو، ومحمد صالح، وبيتر ج. ليو في 18 ديسمبر 2019.

وفقًا للملخص، فإن:

- مهمة التدريب المسبق لـ Pegasus متشابهة عن قصد مع الملخص: تتم إزالة/حجب جمل مهمة من وثيقة الإدخال، ثم يتم توليدها معًا كتسلسل إخراج واحد من الجمل المتبقية، على غرار الملخص الاستخلاصي.
- يحقق Pegasus أداءً متميزًا في مهام الملخص على جميع المهام الفرعية الـ 12، وفقًا لتقييم ROUGE والتقييم البشري.

تمت المساهمة بهذا النموذج من قبل [sshleifer](https://huggingface.co/sshleifer). ويمكن العثور على كود المؤلفين [هنا](https://github.com/google-research/pegasus).

## نصائح الاستخدام

- يعد Pegasus نموذجًا تسلسليًا تسلسليًا له نفس بنية الترميز فك الترميز مثل BART. تم تدريب Pegasus بشكل مسبق بشكل مشترك على دالتين للهدف ذاتي الإشراف: Masked Language Modeling (MLM) ودالة تدريب مسبق محددة للملخص تسمى Gap Sentence Generation (GSG).

* MLM: يتم استبدال رموز الإدخال للترميز بشكل عشوائي برموز قناع، ويجب التنبؤ بها بواسطة الترميز (مثل BERT)
* GSG: يتم استبدال جمل إدخال الترميز بالكامل برمز قناع ثانوي وإدخالها إلى فك الترميز، ولكن مع قناع سببي لإخفاء الكلمات المستقبلية مثل فك تشفير محول ذاتي الانحدار منتظم.

- لا يتم دعم FP16 (يتم تقدير المساعدة/الأفكار حول هذا الموضوع!).
- يوصى باستخدام محسن adafactor لضبط نموذج Pegasus بشكل دقيق.

## نقاط التفتيش

تم ضبط جميع [نقاط التفتيش](https://huggingface.co/models?search=pegasus) مسبقًا للتلخيص، باستثناء *pegasus-large*، حيث يتم ضبط نقاط التفتيش الأخرى:

- يحتل كل نقطة تفتيش مساحة 2.2 جيجابايت على القرص و568 مليون معلمة.
- لا يتم دعم FP16 (يتم تقدير المساعدة/الأفكار حول هذا الموضوع!).
- يستغرق تلخيص xsum في fp32 حوالي 400 مللي ثانية/عينة، مع معلمات افتراضية على وحدة معالجة الرسومات v100.
- يمكن العثور على نتائج الاستنساخ الكامل والبيانات المعالجة بشكل صحيح في هذه [القضية](https://github.com/huggingface/transformers/issues/6844#issue-689259666).
- يتم وصف [نقاط التفتيش المقطرة](https://huggingface.co/models?search=distill-pegasus) في هذه [الورقة](https://arxiv.org/abs/2010.13002).

## ملاحظات التنفيذ

- جميع النماذج هي محولات الترميز فك الترميز مع 16 طبقة في كل مكون.
- تم استلهام التنفيذ بالكامل من [`BartForConditionalGeneration`]
- بعض الاختلافات الرئيسية في التكوين:
- مواضع ثابتة، تضمينية جيبية
- يبدأ النموذج في التوليد باستخدام pad_token_id (الذي يحتوي على 0 token_embedding) كبادئة.
- يتم استخدام المزيد من الحزم (`num_beams=8`)
- جميع نقاط تفتيش Pegasus المسبقة التدريب متطابقة باستثناء ثلاث سمات: `tokenizer.model_max_length` (حجم الإدخال الأقصى)، و`max_length` (العدد الأقصى من الرموز المولدة)، و`length_penalty`.
- يمكن العثور على الكود لتحويل نقاط التفتيش المدربة في مستودع المؤلفين [هنا](https://github.com/google-research/pegasus) في `convert_pegasus_tf_to_pytorch.py`.

## مثال على الاستخدام

```python
>>> from transformers import PegasusForConditionalGeneration, PegasusTokenizer
>>> import torch

>>> src_text = [
...     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
... ]

... model_name = "google/pegasus-xsum"
... device = "cuda" if torch.cuda.is_available() else "cpu"
... tokenizer = PegasusTokenizer.from_pretrained(model_name)
... model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
... batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
... translated = model.generate(**batch)
... tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
... assert (
...     tgt_text[0]
...     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
... )
```

## الموارد

- [سكريبت](https://github.com/huggingface/transformers/tree/main/examples/research_projects/seq2seq-distillation/finetune_pegasus_xsum.sh) لضبط نموذج Pegasus بشكل دقيق على مجموعة بيانات XSUM. تعليمات تنزيل البيانات في [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).
- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهام الترجمة](../tasks/translation)
- [دليل مهام الملخص](../tasks/summarization)

## PegasusConfig

[[autodoc]] PegasusConfig

## PegasusTokenizer

تحذير: `add_tokens` لا يعمل في الوقت الحالي.

[[autodoc]] PegasusTokenizer

## PegasusTokenizerFast

[[autodoc]] PegasusTokenizerFast

<frameworkcontent>
<pt>

## PegasusModel

[[autodoc]] PegasusModel

- forward

## PegasusForConditionalGeneration

[[autodoc]] PegasusForConditionalGeneration

- forward

## PegasusForCausalLM

[[autodoc]] PegasusForCausalLM

- forward

</pt>
<tf>

## TFPegasusModel

[[autodoc]] TFPegasusModel

- call

## TFPegasusForConditionalGeneration

[[autodoc]] TFPegasusForConditionalGeneration

- call

</tf>
<jax>

## FlaxPegasusModel

[[autodoc]] FlaxPegasusModel

- __call__
- encode
- decode

## FlaxPegasusForConditionalGeneration

[[autodoc]] FlaxPegasusForConditionalGeneration

- __call__
- encode
- decode

</jax>
</frameworkcontent>