# Gemma

## نظرة عامة
اقترح فريق Gemma من Google نموذج Gemma في [Gemma: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/gemma-open-models/) .

تم تدريب نماذج Gemma على 6T من الرموز، وتم إصدارها بإصدارين، 2b و 7b.

الملخص من الورقة هو ما يلي:

> يقدم هذا العمل Gemma، وهي عائلة جديدة من نماذج اللغة المفتوحة التي تظهر أداءً قوياً عبر المعايير الأكاديمية لفهم اللغة والتفكير والسلامة. نقوم بإصدار حجمين من النماذج (2 مليار و 7 مليار معلمة)، ونقدم نقاط تفتيش لكل من النماذج المسبقة التدريب والمعايرة الدقيقة. تتفوق Gemma على النماذج المفتوحة المماثلة في الحجم في 11 من أصل 18 مهمة قائمة على النص، ونقدم تقييمات شاملة لجوانب السلامة والمسؤولية للنماذج، إلى جانب وصف مفصل لتطوير نموذجنا. نعتقد أن الإصدار المسؤول لنماذج اللغة الكبيرة مهم لتحسين سلامة النماذج الرائدة، ولتمكين الموجة التالية من الابتكارات في نماذج اللغة الكبيرة.

نصائح:

- يمكن تحويل نقاط التفتيش الأصلية باستخدام نص البرنامج النصي للتحويل `src/transformers/models/gemma/convert_gemma_weights_to_hf.py`

تمت المساهمة بهذا النموذج من قبل [Arthur Zucker](https://huggingface.co/ArthurZ)، [Younes Belkada](https://huggingface.co/ybelkada)، [Sanchit Gandhi](https://huggingface.co/sanchit-gandhi)، [Pedro Cuenca](https://huggingface.co/pcuenq).

## GemmaConfig

[[autodoc]] GemmaConfig

## GemmaTokenizer

[[autodoc]] GemmaTokenizer

## GemmaTokenizerFast

[[autodoc]] GemmaTokenizerFast

## GemmaModel

[[autodoc]] GemmaModel

- forward

## GemmaForCausalLM

[[autodoc]] GemmaForCausalLM

- forward

## GemmaForSequenceClassification

[[autodoc]] GemmaForSequenceClassification

- forward

## GemmaForTokenClassification

[[autodoc]] GemmaForTokenClassification

- forward

## FlaxGemmaModel

[[autodoc]] FlaxGemmaModel

- __call__

## FlaxGemmaForCausalLM

[[autodoc]] FlaxGemmaForCausalLM

- __call__