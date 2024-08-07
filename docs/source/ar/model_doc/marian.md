# MarianMT

## نظرة عامة
إطار عمل لنماذج الترجمة، باستخدام نفس النماذج مثل BART. يجب أن تكون الترجمات مشابهة، ولكنها ليست مطابقة للإخراج في مجموعة الاختبار المرتبطة بكل بطاقة نموذج.
تمت المساهمة بهذا النموذج من قبل [sshleifer](https://huggingface.co/sshleifer).

## ملاحظات التنفيذ
- يبلغ حجم كل نموذج حوالي 298 ميجابايت على القرص، وهناك أكثر من 1000 نموذج.
- يمكن العثور على قائمة أزواج اللغات المدعومة [هنا](https://huggingface.co/Helsinki-NLP).
- تم تدريب النماذج في الأصل من قبل [Jörg Tiedemann](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann) باستخدام مكتبة [Marian](https://marian-nmt.github.io/) C++، والتي تدعم التدريب والترجمة السريعين.
- جميع النماذج هي محولات مشفرة-فك تشفير مع 6 طبقات في كل مكون. يتم توثيق أداء كل نموذج
في بطاقة نموذج.
- لا يتم دعم 80 نموذجًا من نماذج Opus التي تتطلب معالجة BPE.
- رمز النمذجة هو نفسه [`BartForConditionalGeneration`] مع بعض التعديلات الطفيفة:
- المواضع الثابتة (ذات الشكل الجيبي) (`MarianConfig.static_position_embeddings=True`)
- لا يوجد layernorm_embedding (`MarianConfig.normalize_embedding=False`)
- يبدأ النموذج في التوليد باستخدام `pad_token_id` (الذي يحتوي على 0 كـ token_embedding) كبادئة (يستخدم Bart
`<s/>`)،
- يمكن العثور على التعليمات البرمجية لتحويل النماذج بالجملة في `convert_marian_to_pytorch.py`.

## التسمية
- تستخدم جميع أسماء النماذج التنسيق التالي: `Helsinki-NLP/opus-mt-{src}-{tgt}`
- رموز اللغة المستخدمة لتسمية النماذج غير متسقة. يمكن عادةً العثور على رموز مكونة من رقمين [هنا](https://developers.google.com/admin-sdk/directory/v1/languages)، بينما تتطلب الرموز المكونة من ثلاثة أرقام البحث عن "رمز اللغة {code}".
- الرموز التي تم تنسيقها مثل `es_AR` هي عادةً `code_{region}`. هذا باللغة الإسبانية من الأرجنتين.
- تم تحويل النماذج على مرحلتين. تستخدم أول 1000 نموذج رموز ISO-639-2 لتحديد اللغات، بينما تستخدم المجموعة الثانية مزيجًا من رموز ISO-639-5 وISO-639-2.

## أمثلة
- نظرًا لأن نماذج Marian أصغر من العديد من نماذج الترجمة الأخرى المتوفرة في المكتبة، فيمكن أن تكون مفيدة لتجارب الضبط الدقيق واختبارات الدمج.
- [الضبط الدقيق على GPU](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/train_distil_marian_enro.sh)

## النماذج متعددة اللغات
- تستخدم جميع أسماء النماذج التنسيق التالي: `Helsinki-NLP/opus-mt-{src}-{tgt}`:
- إذا كان بإمكان النموذج إخراج عدة لغات، فيجب عليك تحديد رمز لغة عن طريق إضافة رمز اللغة المرغوب إلى `src_text`.
- يمكنك الاطلاع على رموز اللغة المدعومة للنموذج في بطاقة النموذج الخاصة به، ضمن المكونات المستهدفة، كما هو الحال في [opus-mt-en-roa](https://huggingface.co/Helsinki-NLP/opus-mt-en-roa).
- لاحظ أنه إذا كان النموذج متعدد اللغات فقط على جانب المصدر، مثل `Helsinki-NLP/opus-mt-roa-en`، فلن تكون هناك حاجة إلى رموز اللغة.
تتطلب النماذج متعددة اللغات الجديدة من [مستودع Tatoeba-Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge) رموز لغة مكونة من 3 أحرف:

```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fra<< this is a sentence in english that we want to translate to french",
...     ">>por<< This should go to portuguese",
...     ">>esp<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-roa"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)
>>> print(tokenizer.supported_language_codes)
['>>zlm_Latn<<', '>>mfe<<', '>>hat<<', '>>pap<<', '>>ast<<', '>>cat<<', '>>ind<<', '>>glg<<', '>>wln<<', '>>spa<<', '>>fra<<', '>>ron<<', '>>por<<', '>>ita<<', '>>oci<<', '>>arg<<', '>>min<<']

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français",
'Isto deve ir para o português.',
'Y esto al español']
```

فيما يلي رمز لعرض جميع النماذج المسبقة التدريب المتوفرة في المركز:

```python
from huggingface_hub import list_models

model_list = list_models()
org = "Helsinki-NLP"
model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
suffix = [x.split("/")[1] for x in model_ids]
old_style_multi_models = [f"{org}/{s}" for s in suffix if s != s.lower()]
```

## النماذج متعددة اللغات بالأسلوب القديم
هذه هي النماذج متعددة اللغات بالأسلوب القديم المنقولة من مستودع OPUS-MT-Train: وأعضاء كل مجموعة لغات:

```python no-style
['Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU',
'Helsinki-NLP/opus-mt-ROMANCE-en',
'Helsinki-NLP/opus-mt-SCANDINAVIA-SCANDINAVIA',
'Helsinki-NLP/opus-mt-de-ZH',
'Helsinki-NLP/opus-mt-en-CELTIC',
'Helsinki-NLP/opus-mt-en-ROMANCE',
'Helsinki-NLP/opus-mt-es-NORWAY',
'Helsinki-NLP/opus-mt-fi-NORWAY',
'Helsinki-NLP/opus-mt-fi-ZH',
'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI',
'Helsinki-NLP/opus-mt-sv-NORWAY',
'Helsinki-NLP/opus-mt-sv-ZH']
GROUP_MEMBERS = {
    'ZH': ['cmn', 'cn', 'yue', 'ze_zh', 'zh_cn', 'zh_CN', 'zh_HK', 'zh_tw', 'zh_TW', 'zh_yue', 'zhs', 'zht', 'zh'],
    'ROMANCE': ['fr', 'fr_BE', 'fr_CA', 'fr_FR', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'es_AR', 'es_CL', 'es_CO', 'es_CR', 'es_DO', 'es_EC', 'es_ES', 'es_GT', 'es_HN', 'es_MX', 'es_NI', 'es_PA', 'es_PE', 'es_PR', 'es_SV', 'es_UY', 'es_VE', 'pt', 'pt_br', 'pt_BR', 'pt_PT', 'gl', 'lad', 'an', 'mwl', 'it', 'it_IT', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la'],
    'NORTH_EU': ['de', 'nl', 'fy', 'af', 'da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
    'SCANDINAVIA': ['da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
    'SAMI': ['se', 'sma', 'smj', 'smn', 'sms'],
    'NORWAY': ['nb_NO', 'nb', 'nn_NO', 'nn', 'nog', 'no_nb', 'no'],
    'CELTIC': ['ga', 'cy', 'br', 'gd', 'kw', 'gv']
}
```

مثال على ترجمة الإنجليزية إلى العديد من لغات الرومانسية، باستخدام رموز اللغة المكونة من حرفين بالأسلوب القديم

```python
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fr<< this is a sentence in english that we want to translate to french",
...     ">>pt<< This should go to portuguese",
...     ">>es<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français",
'Isto deve ir para o português.',
'Y esto al español']
```

## الموارد
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)
- [دليل مهمة النمذجة اللغوية السببية](../tasks/language_modeling)

## MarianConfig

[[autodoc]] MarianConfig

## MarianTokenizer

[[autodoc]] MarianTokenizer

- build_inputs_with_special_tokens

<frameworkcontent>
<pt>

## MarianModel

[[autodoc]] MarianModel

- forward

## MarianMTModel

[[autodoc]] MarianMTModel

- forward

## MarianForCausalLM

[[autodoc]] MarianForCausalLM

- forward

</pt>
<tf>

## TFMarianModel

[[autodoc]] TFMarianModel

- call

## TFMarianMTModel

[[autodoc]] TFMarianMTModel

- call

</tf>
<jax>

## FlaxMarianModel

[[autodoc]] FlaxMarianModel

- __call__

## FlaxMarianMTModel

[[autodoc]] FlaxMarianMTModel

- __call__

</jax>
</frameworkcontent>