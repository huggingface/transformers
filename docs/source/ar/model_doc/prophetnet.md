# ProphetNet

## نظرة عامة
تم اقتراح نموذج ProphetNet في بحث بعنوان "ProphetNet: التنبؤ بـ N-gram المستقبلية للنمذجة المسبقة التسلسل إلى تسلسل" بواسطة Yu Yan وآخرون. في 13 يناير 2020.

ProphetNet هو نموذج ترميز وفك ترميز ويمكنه التنبؤ بـ n-tokens المستقبلية لـ "ngram" النمذجة اللغوية بدلاً من التنبؤ بالرمز التالي فقط.

الملخص من الورقة هو ما يلي:

في هذه الورقة، نقدم نموذجًا جديدًا للنمذجة المسبقة التسلسل إلى تسلسل يسمى ProphetNet، والذي يقدم هدفًا ذاتي الإشراف يسمى التنبؤ بـ n-gram المستقبلية وآلية الاهتمام الذاتي n-stream المقترحة. بدلاً من تحسين التنبؤ بخطوة واحدة إلى الأمام في النموذج التسلسلي التقليدي، يتم تحسين ProphetNet بواسطة التنبؤ بخطوات n إلى الأمام والتي تتنبأ بالـ n الرموز التالية في نفس الوقت بناءً على رموز السياق السابقة في كل خطوة زمنية. ويشجع التنبؤ بـ n-gram المستقبلية النموذج صراحةً على التخطيط لرموز المستقبل ويمنع الإفراط في التلاؤم مع الارتباطات المحلية القوية. نقوم بتعليم ProphetNet مسبقًا باستخدام مجموعة بيانات أساسية (16 جيجابايت) ومجموعة بيانات واسعة النطاق (160 جيجابايت) على التوالي. ثم نجري تجارب على CNN/DailyMail، وGigaword، وSQuAD 1.1 لمهام الملخص الاستخلاصي وتوليد الأسئلة. وتظهر النتائج التجريبية أن ProphetNet يحقق نتائج جديدة رائدة على جميع هذه المجموعات مقارنة بالنماذج التي تستخدم نفس حجم مجموعة بيانات التدريب.

يمكن العثور على شفرة المؤلفين [هنا](https://github.com/microsoft/ProphetNet).

## نصائح الاستخدام

- ProphetNet هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بتعبئة المدخلات على اليمين بدلاً من اليسار.

- تستند بنية النموذج إلى المحول الأصلي، ولكنه يستبدل آلية "الاهتمام الذاتي" القياسية في فك التشفير بآلية اهتمام ذاتي رئيسية وآلية اهتمام ذاتي وn-stream (التنبؤ).

## الموارد

- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

- [دليل مهام الترجمة](../tasks/translation)

- [دليل مهام الملخص](../tasks/summarization)

## ProphetNetConfig

[[autodoc]] ProphetNetConfig

## ProphetNetTokenizer

[[autodoc]] ProphetNetTokenizer

## مخرجات ProphetNet المحددة

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput

## ProphetNetModel

[[autodoc]] ProphetNetModel

- forward

## ProphetNetEncoder

[[autodoc]] ProphetNetEncoder

- forward

## ProphetNetDecoder

[[autodoc]] ProphetNetDecoder

- forward

## ProphetNetForConditionalGeneration

[[autodoc]] ProphetNetForConditionalGeneration

- forward

## ProphetNetForCausalLM

[[autodoc]] ProphetNetForCausalLM

- forward