# MBart و MBart-50

## نظرة عامة على MBart

تم تقديم نموذج MBart في [التدريب الأولي متعدد اللغات لإزالة التشويش من أجل الترجمة الآلية العصبية](https://arxiv.org/abs/2001.08210) بواسطة Yinhan Liu و Jiatao Gu و Naman Goyal و Xian Li و Sergey Edunov Marjan Ghazvininejad و Mike Lewis و Luke Zettlemoyer.

وفقًا للملخص، فإن MBART هو برنامج ترميز وإلغاء ترميز تسلسلي إلى تسلسلي تم تدريبه مسبقًا على نصوص أحادية اللغة واسعة النطاق بالعديد من اللغات باستخدام هدف BART. يعد mBART إحدى أولى الطرق لتدريب نموذج تسلسلي كامل عن طريق إزالة تشويش النصوص الكاملة بعدة لغات، في حين ركزت الأساليب السابقة فقط على الترميز أو فك الترميز أو إعادة بناء أجزاء من النص.

تمت المساهمة بهذا النموذج من قبل [valhalla](https://huggingface.co/valhalla). يمكن العثور على كود المؤلفين [هنا](https://github.com/pytorch/fairseq/tree/master/examples/mbart)

### تدريب MBart

MBart هو نموذج ترميز وفك ترميز متعدد اللغات (تسلسلي إلى تسلسلي) مصمم في المقام الأول لمهمة الترجمة. نظرًا لأن النموذج متعدد اللغات، فإنه يتوقع التسلسلات بتنسيق مختلف. تتم إضافة رمز معرف اللغة الخاص في كل من النص المصدر والنص المستهدف. يتكون تنسيق النص المصدر من `X [eos, src_lang_code]` حيث `X` هو نص المصدر. يتكون تنسيق النص المستهدف من `[tgt_lang_code] X [eos]`. لا يتم استخدام `bos` مطلقًا.

سيقوم [`~MBartTokenizer.__call__`] العادي بترميز تنسيق النص المصدر الذي تم تمريره كحجة أولى أو باستخدام كلمة `text` الرئيسية، وتنسيق النص المستهدف الذي تم تمريره باستخدام كلمة `text_label` الرئيسية.

- التدريب الخاضع للإشراف

```python
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
>>> example_english_phrase = "UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"

>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")

>>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
>>> # forward pass
>>> model(**inputs)
```

- التوليد Generate
عند إنشاء النص المستهدف، قم بتعيين `decoder_start_token_id` إلى معرف اللغة المستهدفة. يوضح المثال التالي كيفية ترجمة اللغة الإنجليزية إلى الرومانية باستخدام نموذج *facebook/mbart-large-en-ro*.

```python
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")
>>> article = "UN Chief Says There Is No Military Solution in Syria"
>>> inputs = tokenizer(article, return_tensors="pt")
>>> translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Şeful ONU declară că nu există o soluţie militară în Siria"
```

## نظرة عامة على MBart-50

تم تقديم MBart-50 في ورقة [الترجمة متعددة اللغات باستخدام التدريب الأولي متعدد اللغات القابل للتوسيع والتدريب الدقيق](https://arxiv.org/abs/2008.00401) بواسطة Yuqing Tang و Chau Tran و Xian Li و Peng-Jen Chen و Naman Goyal و Vishrav Chaudhary و Jiatao Gu و Angela Fan. تم إنشاء MBart-50 باستخدام نقطة تفتيش *mbart-large-cc25* الأصلية عن طريق تمديد طبقاتها المضمنة مع متجهات تم تهيئتها بشكل عشوائي لمجموعة إضافية من 25 رمزًا للغة ثم التدريب المسبق على 50 لغة.

وفقًا للملخص

*يمكن إنشاء نماذج الترجمة متعددة اللغات من خلال التدريب الدقيق متعدد اللغات. بدلاً من التدريب الدقيق في اتجاه واحد، يتم تدريب نموذج مسبق التدريب على العديد من الاتجاهات في نفس الوقت. فهو يثبت أن النماذج المسبقة التدريب يمكن توسيعها لتشمل لغات إضافية دون فقدان الأداء. يحسن التدريب الدقيق متعدد اللغات 1 BLEU في المتوسط فوق أقوى خطوط الأساس (سواء متعددة اللغات من الصفر أو التدريب الدقيق ثنائي اللغة) مع تحسين 9.3 BLEU في المتوسط فوق خطوط الأساس ثنائية اللغة من الصفر.*

### تدريب MBart-50

يختلف تنسيق النص لـ MBart-50 اختلافًا طفيفًا عن mBART. بالنسبة لـ MBart-50، يتم استخدام رمز معرف اللغة كبادئة لكل من النص المصدر والنص المستهدف أي أن تنسيق النص هو `[lang_code] X [eos]`، حيث `lang_code` هو معرف اللغة المصدر لنص المصدر ومعرف اللغة المستهدف للنص المستهدف، مع كون `X` هو نص المصدر أو النص المستهدف على التوالي.

لدى MBart-50 محول رموز خاص به [`MBart50Tokenizer`].

- التدريب الخاضع للإشراف

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")

src_text = " UN Chief Says There Is No Military Solution in Syria"
tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")

model(**model_inputs) # forward pass
```

- التوليد

لإنشاء نموذج الترجمة متعدد اللغات mBART-50، يتم استخدام `eos_token_id` كـ `decoder_start_token_id` ويتم فرض معرف اللغة المستهدفة كأول رمز يتم إنشاؤه. لفرض معرف اللغة المستهدفة كأول رمز يتم إنشاؤه، قم بتمرير معلمة *forced_bos_token_id* إلى طريقة *generate*. يوضح المثال التالي كيفية الترجمة بين الهندية والفرنسية والعربية والإنجليزية باستخدام نقطة تفتيش *facebook/mbart-50-large-many-to-many*.

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्ر के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Hindi to French
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire en Syrie."

# translate Arabic to English
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
```

## موارد التوثيق

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة تلخيص](../tasks/summarization)

## MBartConfig

[[autodoc]] MBartConfig

## MBartTokenizer

[[autodoc]] MBartTokenizer

- build_inputs_with_special_tokens

## MBartTokenizerFast

[[autodoc]] MBartTokenizerFast

## MBart50Tokenizer

[[autodoc]] MBart50Tokenizer

## MBart50TokenizerFast

[[autodoc]] MBart50TokenizerFast

<frameworkcontent>
<pt>

## MBartModel

[[autodoc]] MBartModel

## MBartForConditionalGeneration

[[autodoc]] MBartForConditionalGeneration

## MBartForQuestionAnswering

[[autodoc]] MBartForQuestionAnswering

## MBartForSequenceClassification

[[autodoc]] MBartForSequenceClassification

## MBartForCausalLM

[[autodoc]] MBartForCausalLM

- forward

</pt>
<tf>

## TFMBartModel

[[autodoc]] TFMBartModel

- call

## TFMBartForConditionalGeneration

[[autodoc]] TFMBartForConditionalGeneration

- call

</tf>
<jax>

## FlaxMBartModel

[[autodoc]] FlaxMBartModel

- __call__
- encode
- decode

## FlaxMBartForConditionalGeneration

[[autodoc]] FlaxMBartForConditionalGeneration

- __call__
- encode
- decode

## FlaxMBartForSequenceClassification

[[autodoc]] FlaxMBartForSequenceClassification

- __call__
- encode
- decode

## FlaxMBartForQuestionAnswering

[[autodoc]] FlaxMBartForQuestionAnswering

- __call__
- encode
- decode

</jax>
</frameworkcontent>