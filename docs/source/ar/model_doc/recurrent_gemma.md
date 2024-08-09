# RecurrentGemma

## نظرة عامة

اقترح فريق Griffin و RLHF و Gemma من Google نموذج Recurrent Gemma في [RecurrentGemma: Moving Past Transformers for Efficient Open Language Models](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf).

مقدمة الورقة البحثية هي كما يلي:

*نحن نقدم RecurrentGemma، وهو نموذج لغة مفتوح يستخدم هندسة Griffin الجديدة من Google. يجمع Griffin بين التكرارات الخطية والاهتمام المحلي لتحقيق أداء ممتاز في اللغة. لديه حالة ثابتة الحجم، مما يقلل من استخدام الذاكرة ويمكّن الاستدلال الفعال على التسلسلات الطويلة. نقدم نموذجًا مُدربًا مسبقًا مع 2 بليون معلمة غير مضمنة، ومتغيرًا مُعدلًا للتعليمات. يحقق كلا النموذجين أداءً مماثلاً لـ Gemma-2B على الرغم من تدريبهما على عدد أقل من الرموز*.

نصائح:

- يمكن تحويل نقاط التفتيش الأصلية باستخدام نص البرنامج النصي للتحويل [`src/transformers/models/recurrent_gemma/convert_recurrent_gemma_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/recurrent_gemma/convert_recurrent_gemma_to_hf.py).

تمت المساهمة بهذا النموذج من قبل [Arthur Zucker](https://huggingface.co/ArthurZ). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-deepmind/recurrentgemma).

## RecurrentGemmaConfig

[[autodoc]] RecurrentGemmaConfig

## RecurrentGemmaModel

[[autodoc]] RecurrentGemmaModel

- forward

## RecurrentGemmaForCausalLM

[[autodoc]] RecurrentGemmaForCausalLM

- forward