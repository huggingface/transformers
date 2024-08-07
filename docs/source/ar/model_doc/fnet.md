# FNet

## نظرة عامة
اقترح نموذج FNet في [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) بواسطة James Lee-Thorp و Joshua Ainslie و Ilya Eckstein و Santiago Ontanon. يستبدل النموذج طبقة self-attention في نموذج BERT بتحويل فورييه الذي يعيد فقط الأجزاء الحقيقية من التحويل. النموذج أسرع بكثير من نموذج BERT لأنه يحتوي على عدد أقل من المعلمات وأكثر كفاءة في استخدام الذاكرة. يحقق النموذج حوالي 92-97% من دقة نظرائه BERT على معيار GLUE، ويتدرب بشكل أسرع بكثير من نموذج BERT. الملخص من الورقة هو كما يلي:

*نحن نبين أن هندسات ترميز المحول يمكن تسريعها، مع تكاليف دقة محدودة، عن طريق استبدال طبقات فرعية للاهتمام الذاتي بتحويلات خطية بسيطة "تمزج" رموز الإدخال. تثبت هذه الخلاطات الخطية، إلى جانب غير الخطية القياسية في الطبقات التغذوية الأمامية، كفاءتها في نمذجة العلاقات الدلالية في العديد من مهام تصنيف النصوص. والأكثر مفاجأة، أننا نجد أن استبدال الطبقة الفرعية للاهتمام الذاتي في ترميز المحول بتحويل فورييه قياسي وغير معلم يحقق 92-97% من دقة نظرائه BERT على معيار GLUE، ولكنه يتدرب بنسبة 80% أسرع على وحدات معالجة الرسوميات (GPUs) وبنسبة 70% أسرع على وحدات معالجة الدوال (TPUs) عند أطوال الإدخال القياسية 512. عند أطوال الإدخال الأطول، يكون نموذج FNet الخاص بنا أسرع بكثير: عند مقارنته بالمحولات "الفعالة" على معيار Long Range Arena، يطابق FNet دقة أكثر النماذج دقة، بينما يتفوق على أسرع النماذج عبر جميع أطوال التسلسل على وحدات معالجة الرسوميات (وعبر أطوال أقصر نسبيًا على وحدات معالجة الدوال). وأخيرًا، يتميز FNet ببصمة خفيفة الوزن في الذاكرة وكفاءة خاصة عند أحجام النماذج الصغيرة؛ بالنسبة لميزانية سرعة ودقة ثابتة، تفوق نماذج FNet الصغيرة نظرائها من المحولات.*

تمت المساهمة بهذا النموذج من قبل [gchhablani](https://huggingface.co/gchhablani). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/google-research/tree/master/f_net).

## نصائح الاستخدام
تم تدريب النموذج بدون قناع اهتمام لأنه يعتمد على تحويل فورييه. تم تدريب النموذج بطول تسلسل أقصاه 512 والذي يتضمن رموز الحشو. وبالتالي، يوصى بشدة باستخدام نفس الطول الأقصى للتسلسل للضبط الدقيق والاستنتاج.

## الموارد
- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## FNetConfig

[[autodoc]] FNetConfig

## FNetTokenizer

[[autodoc]] FNetTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## FNetTokenizerFast

[[autodoc]] FNetTokenizerFast

## FNetModel

[[autodoc]] FNetModel

- forward

## FNetForPreTraining

[[autodoc]] FNetForPreTraining

- forward

## FNetForMaskedLM

[[autodoc]] FNetForMaskedLM

- forward

## FNetForNextSentencePrediction

[[autodoc]] FNetForNextSentencePrediction

- forward

## FNetForSequenceClassification

[[autodoc]] FNetForSequenceClassification

- forward

## FNetForMultipleChoice

[[autodoc]] FNetForMultipleChoice

- forward

## FNetForTokenClassification

[[autodoc]] FNetForTokenClassification

- forward

## FNetForQuestionAnswering

[[autodoc]] FNetForQuestionAnswering

- forward