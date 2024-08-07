# DeBERTa-v2

## نظرة عامة

اقترح نموذج DeBERTa في [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) من قبل Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. وهو مبني على نموذج BERT من Google الذي صدر في عام 2018 ونموذج RoBERTa من Facebook الذي صدر في عام 2019.

يستند النموذج إلى RoBERTa مع اهتمام منفصل وتدريب محسن لترميز الأقنعة باستخدام نصف البيانات المستخدمة في RoBERTa.

الملخص من الورقة هو كما يلي:

*أدى التقدم الأخير في نماذج اللغة العصبية المسبقة التدريب إلى تحسين كبير في أداء العديد من مهام معالجة اللغة الطبيعية (NLP). في هذه الورقة، نقترح بنية نموذج جديدة تسمى DeBERTa (Decoding-enhanced BERT with disentangled attention) والتي تحسن نماذج BERT وRoBERTa باستخدام تقنيتين جديدتين. الأولى هي آلية الاهتمام المنفصل، حيث يتم تمثيل كل كلمة باستخدام متجهين يشفر كل منهما المحتوى والموضع على التوالي، ويتم حساب أوزان الاهتمام بين الكلمات باستخدام مصفوفات منفصلة لمحتوياتها ومواضعها النسبية. ثانياً، يتم استخدام ترميز قناع محسن لاستبدال طبقة softmax الإخراجية للتنبؤ بالرموز المميزة المقنعة لتدريب النموذج المسبق. نُظهر أن هاتين التقنيتين تحسنان بشكل كبير كفاءة التدريب المسبق للنموذج وأداء المهام اللاحقة. مقارنة بـ RoBERTa-Large، يؤدي نموذج DeBERTa الذي تم تدريبه على نصف بيانات التدريب إلى أداء أفضل باستمرار في مجموعة واسعة من مهام NLP، حيث يحقق تحسينات على MNLI بنسبة +0.9% (90.2% مقابل 91.1%)، وعلى SQuAD v2.0 بنسبة +2.3% (88.4% مقابل 90.7%) وعلى RACE بنسبة +3.6% (83.2% مقابل 86.8%). سيتم إتاحة كود DeBERTa والنماذج المُدربة مسبقًا للجمهور على https://github.com/microsoft/DeBERTa.*

يمكن العثور على المعلومات التالية مباشرة في [مستودع التنفيذ الأصلي](https://github.com/microsoft/DeBERTa). DeBERTa v2 هو الإصدار الثاني من نموذج DeBERTa. وهو يشمل نموذج 1.5B المستخدم في إرسال SuperGLUE single-model والذي حقق 89.9، مقابل خط الأساس البشري 89.8. يمكنك العثور على مزيد من التفاصيل حول هذا الإرسال في [مدونة](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/) المؤلفين.

جديد في v2:

- **المفردات** في v2، تم تغيير المحلل اللغوي لاستخدام مفردات جديدة بحجم 128K تم بناؤها من بيانات التدريب. بدلاً من المحلل اللغوي المستند إلى GPT2، أصبح المحلل اللغوي الآن [محللًا قائمًا على sentencepiece](https://github.com/google/sentencepiece).
- **nGiE(nGram Induced Input Encoding)** يستخدم نموذج DeBERTa-v2 طبقة تضمين إضافية إلى جانب طبقة المحول الأولى للتعرف بشكل أفضل على التبعية المحلية للرموز المميزة للإدخال.
- **مشاركة مصفوفة موضع الإسقاط مع مصفوفة إسقاط المحتوى في طبقة الاهتمام** بناءً على التجارب السابقة، يمكن أن يؤدي ذلك إلى توفير المعلمات دون التأثير على الأداء.
- **تطبيق bucket لتشفير المواضع النسبية** يستخدم نموذج DeBERTa-v2 log bucket لتشفير المواضع النسبية على غرار T5.
- **نموذج 900M & 1.5B** هناك حجمين إضافيين للنموذج متاحين: 900M و 1.5B، مما يحسن بشكل كبير أداء المهام اللاحقة.

تمت المساهمة بهذا النموذج من قبل [DeBERTa](https://huggingface.co/DeBERTa). تمت المساهمة في تنفيذ TF 2.0 من قبل [kamalkraj](https://huggingface.co/kamalkraj). يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/DeBERTa).

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز المميزة](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## DebertaV2Config

[[autodoc]] DebertaV2Config

## DebertaV2Tokenizer

[[autodoc]] DebertaV2Tokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## DebertaV2TokenizerFast

[[autodoc]] DebertaV2TokenizerFast

- build_inputs_with_special_tokens
- create_token_type_ids_from_sequences

<frameworkcontent>

<pt>

## DebertaV2Model

[[autodoc]] DebertaV2Model

- forward

## DebertaV2PreTrainedModel

[[autodoc]] DebertaV2PreTrainedModel

- forward

## DebertaV2ForMaskedLM

[[autodoc]] DebertaV2ForMaskedLM

- forward

## DebertaV2ForSequenceClassification

[[autodoc]] DebertaV2ForSequenceClassification

- forward

## DebertaV2ForTokenClassification

[[autodoc]] DebertaV2ForTokenClassification

- forward

## DebertaV2ForQuestionAnswering

[[autodoc]] DebertaV2ForQuestionAnswering

- forward

## DebertaV2ForMultipleChoice

[[autodoc]] DebertaV2ForMultipleChoice

- forward

</pt>

<tf>

## TFDebertaV2Model

[[autodoc]] TFDebertaV2Model

- call

## TFDebertaV2PreTrainedModel

[[autodoc]] TFDebertaV2PreTrainedModel

- call

## TFDebertaV2ForMaskedLM

[[autodoc]] TFDebertaV2ForMaskedLM

- call

## TFDebertaV2ForSequenceClassification

[[autodoc]] TFDebertaV2ForSequenceClassification

- call

## TFDebertaV2ForTokenClassification

[[autodoc]] TFDebertaV2ForTokenClassification

- call

## TFDebertaV2ForQuestionAnswering

[[autodoc]] TFDebertaV2ForQuestionAnswering

- call

## TFDebertaV2ForMultipleChoice

[[autodoc]] TFDebertaV2ForMultipleChoice

- call


</tf>

</frameworkcontent>