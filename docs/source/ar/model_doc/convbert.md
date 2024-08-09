# ConvBERT

## نظرة عامة

اقتُرح نموذج ConvBERT في [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496) بواسطة Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.

مقدمة الورقة البحثية هي التالية:

*حققت نماذج اللغة المُدربة مسبقًا مثل BERT ومتغيراته أداءً مثيرًا للإعجاب في العديد من مهام فهم اللغة الطبيعية. ومع ذلك، يعتمد BERT بشكل كبير على كتلة الاهتمام الذاتي العالمي، وبالتالي يعاني من آثار ذاكرة كبيرة وتكلفة حوسبة عالية. على الرغم من أن جميع رؤوس اهتمامه تستفسر عن التسلسل الكامل للإدخال لإنشاء خريطة الاهتمام من منظور عالمي، إلا أننا نلاحظ أن بعض الرؤوس تحتاج فقط إلى تعلم التبعيات المحلية، مما يعني وجود حوسبة زائدة. لذلك، نقترح حوسبة ديناميكية جديدة تعتمد على النطاق لاستبدال رؤوس الاهتمام الذاتي هذه لنمذجة التبعيات المحلية مباشرةً. تشكل رؤوس الحوسبة الجديدة، جنبًا إلى جنب مع بقية رؤوس الاهتمام الذاتي، كتلة اهتمام مختلطة أكثر كفاءة في تعلم السياق العالمي والمحلي. نقوم بتجهيز BERT بتصميم الاهتمام المختلط هذا وبناء نموذج ConvBERT. وقد أظهرت التجارب أن ConvBERT يتفوق بشكل كبير على BERT ومتغيراته في مهام مختلفة، بتكلفة تدريب أقل وعدد أقل من معلمات النموذج. ومن المثير للاهتمام أن نموذج ConvBERTbase يحقق نتيجة 86.4 في اختبار GLUE، أي أعلى بـ 0.7 من نموذج ELECTRAbase، مع استخدام أقل من ربع تكلفة التدريب. سيتم إطلاق الكود والنماذج المُدربة مسبقًا.*

تمت المساهمة بهذا النموذج من قبل [abhishek](https://huggingface.co/abhishek). ويمكن العثور على التنفيذ الأصلي هنا: https://github.com/yitu-opensource/ConvBert

## نصائح الاستخدام

نصائح تدريب ConvBERT مماثلة لتلك الخاصة بـ BERT. للاطلاع على نصائح الاستخدام، يرجى الرجوع إلى [توثيق BERT](bert).

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة المعقدة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## ConvBertConfig

[[autodoc]] ConvBertConfig

## ConvBertTokenizer

[[autodoc]] ConvBertTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## ConvBertTokenizerFast

[[autodoc]] ConvBertTokenizerFast

<frameworkcontent>
<pt>

## ConvBertModel

[[autodoc]] ConvBertModel

- forward

## ConvBertForMaskedLM

[[autodoc]] ConvBertForMaskedLM

- forward

## ConvBertForSequenceClassification

[[autodoc]] ConvBertForSequenceClassification

- forward

## ConvBertForMultipleChoice

[[autodoc]] ConvBertForMultipleChoice

- forward

## ConvBertForTokenClassification

[[autodoc]] ConvBertForTokenClassification

- forward

## ConvBertForQuestionAnswering

[[autodoc]] ConvBertForQuestionAnswering

- forward

</pt>
<tf>

## TFConvBertModel

[[autodoc]] TFConvBertModel

- call

## TFConvBertForMaskedLM

[[autodoc]] TFConvBertForMaskedLM

- call

## TFConvBertForSequenceClassification

[[autodoc]] TFConvBertForSequenceClassification

- call

## TFConvBertForMultipleChoice

[[autodoc]] TFConvBertForMultipleChoice

- call

## TFConvBertForTokenClassification

[[autodoc]] TFConvBertForTokenClassification

- call

## TFConvBertForQuestionAnswering

[[autodoc]] TFConvBertForQuestionAnswering

- call

</tf>
</frameworkcontent>