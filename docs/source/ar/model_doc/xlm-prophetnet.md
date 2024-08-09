# XLM-ProphetNet

**تنصل:** إذا رأيت شيئًا غريبًا، فقم بتقديم [قضية على GitHub](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) وعيّنها إلى @patrickvonplaten

## نظرة عامة

تم اقتراح نموذج XLM-ProphetNet في [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063) بواسطة Yu Yan وWeizhen Qi وYeyun Gong وDayiheng Liu وNan Duan وJiusheng Chen وRuofei Zhang وMing Zhou في 13 يناير 2020.

XLM-ProphetNet هو نموذج ترميز وفك ترميز ويمكنه التنبؤ بـ n-tokens المستقبلية لـ "ngram" language modeling بدلاً من التنبؤ بالرمز التالي فقط. إنها مماثلة لمعمارية ProhpetNet، ولكن تم تدريب النموذج على إغراق Wikipedia متعدد اللغات "wiki100". تتشابه هندسة نموذج XLM-ProphetNet وهدفه من المعالجة المسبقة مع ProphetNet، ولكن تم تدريب XLM-ProphetNet على مجموعة البيانات متعددة اللغات XGLUE.

الملخص من الورقة هو ما يلي:

*في هذه الورقة، نقدم نموذجًا جديدًا للترميز إلى تسلسل يسمى ProphetNet، والذي يقدم هدفًا ذاتي الإشراف يسمى التنبؤ بـ n-gram المستقبلية وآلية الاهتمام الذاتي n-stream المقترحة. بدلاً من تحسين التنبؤ بخطوة واحدة مسبقًا في نموذج تسلسل إلى تسلسل التقليدي، يتم تحسين ProphetNet بواسطة التنبؤ بخطوات n مسبقًا والتي تتنبأ بالـ n الرموز التالية في نفس الوقت بناءً على رموز السياق السابقة في كل خطوة زمنية. يشجع التنبؤ بـ n-gram المستقبلي النموذج صراحة على التخطيط لرموز المستقبل ويمنع الإفراط في التلاؤم مع الارتباطات المحلية القوية. نقوم بتدريب ProphetNet باستخدام مجموعة بيانات أساسية الحجم (16 جيجابايت) ومجموعة بيانات واسعة النطاق (160 جيجابايت) على التوالي. ثم نجري تجارب على CNN/DailyMail وGigaword وSQuAD 1.1 لمهام الملخص والأسئلة التوليدية. تُظهر النتائج التجريبية أن ProphetNet يحقق نتائج جديدة رائدة على جميع هذه المجموعات مقارنة بالنماذج التي تستخدم نفس نطاق مجموعة بيانات المعالجة المسبقة.*

يمكن العثور على كود المؤلفين [هنا](https://github.com/microsoft/ProphetNet).

## الموارد

- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهام الترجمة](../tasks/translation)
- [دليل مهام الملخص](../tasks/summarization)

## XLMProphetNetConfig

[[autodoc]] XLMProphetNetConfig

## XLMProphetNetTokenizer

[[autodoc]] XLMProphetNetTokenizer

## XLMProphetNetModel

[[autodoc]] XLMProphetNetModel

## XLMProphetNetEncoder

[[autodoc]] XLMProphetNetEncoder

## XLMProphetNetDecoder

[[autodoc]] XLMProphetNetDecoder

## XLMProphetNetForConditionalGeneration

[[autodoc]] XLMProphetNetForConditionalGeneration

## XLMProphetNetForCausalLM

[[autodoc]] XLMProphetNetForCausalLM