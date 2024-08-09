# mT5

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=mt5">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-mt5-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/mt5-small-finetuned-arxiv-cs-finetuned-arxiv-cs-full">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## نظرة عامة

تم تقديم نموذج mT5 في ورقة "mT5: محول نصي مسبق التدريب متعدد اللغات بشكل كبير" من قبل لينتينغ شيوي ونوح كونستانت وآدم روبرتس وميهير كالي ورامي الرفو وأديتيا سيدانت وأديتيا باروا وكولين رافيل.

ملخص الورقة البحثية هو كما يلي:

"استفاد "محول النص إلى نص" (T5) الذي تم طرحه مؤخرًا من تنسيق موحد للنص إلى نص والمقياس لتحقيق نتائج متقدمة في العديد من مهام اللغة الإنجليزية. وفي هذه الورقة، نقدم mT5، وهو متغير متعدد اللغات من T5 تم تدريبه مسبقًا على مجموعة بيانات جديدة تعتمد على Common Crawl وتغطي 101 لغة. نتناول بالتفصيل تصميم mT5 والتدريب المعدل ونثبت أداءه المتقدم على العديد من المعايير المرجعية متعددة اللغات. كما نصف تقنية بسيطة لمنع "الترجمة العرضية" في الإعداد بدون بيانات تدريب، حيث يختار نموذج توليدي أن (يترجم جزئيًا) تنبؤه إلى اللغة الخاطئة. كل التعليمات البرمجية ونقاط التحقق من النموذج المستخدمة في هذا العمل متاحة للجمهور."

ملاحظة: تم تدريب mT5 مسبقًا فقط على [mC4](https://huggingface.co/datasets/mc4) مع استبعاد أي تدريب خاضع للإشراف. لذلك، يجب ضبط هذا النموذج الدقيق قبل استخدامه في مهمة تالية، على عكس نموذج T5 الأصلي. نظرًا لأن mT5 تم تدريبه بدون إشراف، فلا توجد ميزة حقيقية لاستخدام بادئة المهمة أثناء الضبط الدقيق لمهمة واحدة. إذا كنت تقوم بالضبط الدقيق متعدد المهام، فيجب عليك استخدام بادئة.

أطلقت Google المتغيرات التالية:

- [google/mt5-small](https://huggingface.co/google/mt5-small)
- [google/mt5-base](https://huggingface.co/google/mt5-base)
- [google/mt5-large](https://huggingface.co/google/mt5-large)
- [google/mt5-xl](https://huggingface.co/google/mt5-xl)
- [google/mt5-xxl](https://huggingface.co/google/mt5-xxl)

تمت المساهمة بهذا النموذج من قبل [patrickvonplaten](https://huggingface.co/patrickvonplaten). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/multilingual-t5).

## الموارد

- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)

## MT5Config

[[autodoc]] MT5Config

## MT5Tokenizer

[[autodoc]] MT5Tokenizer

انظر [`T5Tokenizer`] لمزيد من التفاصيل.

## MT5TokenizerFast

[[autodoc]] MT5TokenizerFast

انظر [`T5TokenizerFast`] لمزيد من التفاصيل.

<frameworkcontent>
<pt>

## MT5Model

[[autodoc]] MT5Model

## MT5ForConditionalGeneration

[[autodoc]] MT5ForConditionalGeneration

## MT5EncoderModel

[[autodoc]] MT5EncoderModel

## MT5ForSequenceClassification

[[autodoc]] MT5ForSequenceClassification

## MT5ForTokenClassification

[[autodoc]] MT5ForTokenClassification

## MT5ForQuestionAnswering

[[autodoc]] MT5ForQuestionAnswering

</pt>
<tf>

## TFMT5Model

[[autodoc]] TFMT5Model

## TFMT5ForConditionalGeneration

[[autodoc]] TFMT5ForConditionalGeneration

## TFMT5EncoderModel

[[autodoc]] TFMT5EncoderModel

</tf>
<jax>

## FlaxMT5Model

[[autodoc]] FlaxMT5Model

## FlaxMT5ForConditionalGeneration

[[autodoc]] FlaxMT5ForConditionalGeneration

## FlaxMT5EncoderModel

[[autodoc]] FlaxMT5EncoderModel

</jax>
</frameworkcontent>