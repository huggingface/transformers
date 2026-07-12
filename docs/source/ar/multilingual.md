# النماذج متعددة اللغات للاستدلال

هناك العديد من النماذج متعددة اللغات في مكتبة 🤗 Transformers، وتختلف طريقة استخدامها للاستدلال عن النماذج أحادية اللغة. ولكن ليس كل استخدام النماذج متعددة اللغات مختلف. فبعض النماذج، مثل [google-bert/bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased)، يمكن استخدامها تمامًا مثل النموذج أحادي اللغة. سيوضح لك هذا الدليل كيفية استخدام النماذج متعددة اللغات التي تختلف طريقة استخدامها للاستدلال.

## XLM

يحتوي XLM على عشر نسخ مختلفة، واحدة منها فقط أحادية اللغة. ويمكن تقسيم نسخ النماذج التسع المتبقية إلى فئتين: نسخ التي تستخدم تضمينات اللغة (language embeddings)  وتلك التي لا تستخدمها.

### XLM مع تضمينات اللغة

تستخدم النماذج التالية من XLM تضمينات اللغة لتحديد اللغة المستخدمة أثناء الاستدلال:

- `FacebookAI/xlm-mlm-ende-1024` (نمذجة اللغة المقنعة، الإنجليزية-الألمانية)
- `FacebookAI/xlm-mlm-enfr-1024` (نمذجة اللغة المقنعة، الإنجليزية-الفرنسية)
- `FacebookAI/xlm-mlm-enro-1024` (نمذجة اللغة المقنعة، الإنجليزية-الرومانية)
- `FacebookAI/xlm-mlm-xnli15-1024` (نمذجة اللغة المقنعة، لغات XNLI)
- `FacebookAI/xlm-mlm-tlm-xnli15-1024` (نمذجة اللغة المقنعة + الترجمة، لغات XNLI)
- `FacebookAI/xlm-clm-enfr-1024` (نمذجة اللغة السببية، الإنجليزية-الفرنسية)
- `FacebookAI/xlm-clm-ende-1024` (نمذجة اللغة السببية، الإنجليزية-الألمانية)

تُمثل تضمينات اللغة على شكل مصفوفة بنفس شكل  `input_ids` التي يتم تمريره إلى النموذج. وتعتمد القيم في هذه المصفوفات على اللغة المستخدمة ويتم تحديدها بواسطة معاملى المجزىء `lang2id` و `id2lang`.

في هذا المثال، قم بتحميل نسخة `FacebookAI/xlm-clm-enfr-1024` ( نمذجة اللغة السببية، الإنجليزية-الفرنسية):

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
```

تُظهر خاصية `lang2id` في المجزىء اللغات وأرقام تعريفها في هذا النموذج:

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

بعد ذلك، قم بإنشاء مثال على المدخلات:

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1
```

قم بتعيين معرف اللغة إلى `"en"` واستخدمه لتحديد تضمين اللغة. وتضمين اللغة عبارة عن مصفوفة مملوءة بـ `0` لأن هذا هو معرف اللغة الإنجليزية. يجب أن تكون هذه المصفوفة بنفس حجم `input_ids`.

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # نقوم بإعادة تشكيلها لتكون بالحجم (batch_size، sequence_length)
>>> langs = langs.view(1, -1)  # الآن بالحجم [1، sequence_length] (لدينا batch size تساوي 1)
```

الآن يمكنك تمرير `input_ids` وتضمين اللغة إلى النموذج:

```py
>>> outputs = model(input_ids, langs=langs)
```

يمكن لنص البرنامج النصي [run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py) توليد النص باستخدام تضمينات اللغة مع نقاط تفتيش `xlm-clm`.

### XLM بدون تضمينات اللغة

النماذج التالية من XLM لا تتطلب تضمينات اللغة أثناء الاستنتاج:

- `FacebookAI/xlm-mlm-17-1280` (نمذجة اللغة المقنعة، 17 لغة)
- `FacebookAI/xlm-mlm-100-1280` (نمذجة اللغة المقنعة، 100 لغة)

تُستخدم هذه النماذج لتمثيل الجمل العامة، على عكس نسح XLM السابقة.

## BERT

يمكن استخدام النماذج التالية من BERT للمهام متعددة اللغات:

- `google-bert/bert-base-multilingual-uncased` (نمذجة اللغة المقنعة + التنبؤ بالجملة التالية، 102 لغة)
- `google-bert/bert-base-multilingual-cased` (نمذجة اللغة المقنعة + التنبؤ بالجملة التالية، 104 لغات)

لا تتطلب هذه النماذج تضمينات اللغة أثناء الاستدلال. يجب أن تُحدّد اللغة من السياق وتستنتج وفقاً لذلك.

## XLM-RoBERTa

يمكن استخدام النماذج التالية من XLM-RoBERTa للمهام متعددة اللغات:

- `FacebookAI/xlm-roberta-base` (نمذجة اللغة المقنعة، 100 لغة)
- `FacebookAI/xlm-roberta-large` (نمذجة اللغة المقنعة، 100 لغة)

تم تدريب XLM-RoBERTa على 2.5 تيرابايت من بيانات CommonCrawl الجديدة والمحسنة في 100 لغة. ويوفر مكاسب قوية على النماذج متعددة اللغات التي تم إصدارها سابقاً مثل mBERT أو XLM في مهام المصب مثل التصنيف، ووضع العلامات التسلسلية، والأسئلة والأجوبة.

## M2M100

يمكن استخدام النماذج التالية من M2M100 للترجمة متعددة اللغات:

- `facebook/m2m100_418M` (الترجمة)
- `facebook/m2m100_1.2B` (الترجمة)

في هذا المثال، قم بتحميل نسحة  `facebook/m2m100_418M` لترجمة النص من الصينية إلى الإنجليزية. يمكنك تعيين اللغة المصدر في المجزىء اللغوى:

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

تقسيم النّص إلى رموز:

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

يجبر M2M100 معرف اللغة الهدف كأول رمز مولد للترجمة إلى اللغة الهدف. قم بتعيين `forced_bos_token_id` إلى `en` في طريقة `generate` للترجمة إلى الإنجليزية:

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

يمكن استخدام النماذج التالية من MBart للترجمة متعددة اللغات:

- `facebook/mbart-large-50-one-to-many-mmt` (الترجمة الآلية متعددة اللغات من واحد إلى كثير، 50 لغة)
- `facebook/mbart-large-50-many-to-many-mmt` (الترجمة الآلية متعددة اللغات من كثير إلى كثير، 50 لغة)
- `facebook/mbart-large-50-many-to-one-mmt` (الترجمة الآلية متعددة اللغات من كثير إلى واحد، 50 لغة)
- `facebook/mbart-large-50` (الترجمة متعددة اللغات، 50 لغة)
- `facebook/mbart-large-cc25`

في هذا المثال، قم بتحميل نسخة `facebook/mbart-large-50-many-to-many-mmt` لترجمة النص من الفنلندية إلى الإنجليزية. يمكنك تعيين اللغة المصدر في المجزىء:

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

تقسيم النّص إلى رموز:

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

يجبر MBart معرف لغة الهدف كأول رمز مولد للترجمة إلى اللغة الهدف. قم بتعيين `forced_bos_token_id` إلى `en` في طريقة `generate` للترجمة إلى الإنجليزية:

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

إذا كنت تستخدم نسخة `facebook/mbart-large-50-many-to-one-mmt`، فلا تحتاج إلى إجبار معرف لغة الهدف كأول رمز مولد، وإلا فإن الاستخدام هو نفسه.
