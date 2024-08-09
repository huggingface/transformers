# CTRL

## نظرة عامة
اقترح نموذج CTRL في [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) بواسطة Nitish Shirish Keskar*، Bryan McCann*، Lav R. Varshney، Caiming Xiong و Richard Socher. وهو محول سببي (أحادي الاتجاه) تم تدريبه مسبقًا باستخدام نمذجة اللغة على مجموعة بيانات نصية كبيرة جدًا تبلغ حوالي 140 جيجابايت مع الاحتفاظ بالرمز الأول كرمز تحكم (مثل الروابط، والكتب، وويكيبيديا، وما إلى ذلك).

مقتطف من الورقة هو ما يلي:

*تظهر نماذج اللغة واسعة النطاق قدرات واعدة لتوليد النص، ولكن لا يمكن للمستخدمين التحكم بسهولة في جوانب معينة من النص المولد. نقدم CTRL، وهو نموذج لغة محول مشروط بحجم 1.63 مليار معلمة، تم تدريبه على رموز تحكم تحكم الأسلوب والمحتوى والسلوك الخاص بالمهمة. تم اشتقاق رموز التحكم من البنية التي تحدث بشكل طبيعي مع النص الخام، مما يحافظ على مزايا التعلم غير الخاضع للإشراف أثناء توفير تحكم أكثر صراحة في توليد النص. تسمح هذه الرموز أيضًا لـ CTRL بالتنبؤ بأجزاء بيانات التدريب التي من المحتمل أن تكون الأكثر احتمالًا تسلسل. توفر هذه الطريقة إمكانية تحليل كميات كبيرة من البيانات عبر عزو المصدر القائم على النموذج.*

تمت المساهمة بهذا النموذج من قبل [keskarnitishr](https://huggingface.co/keskarnitishr). يمكن العثور على الكود الأصلي [هنا](https://github.com/salesforce/ctrl).

## نصائح الاستخدام

- يستخدم CTRL رموز التحكم لتوليد النص: فهو يتطلب أن تبدأ الأجيال بكلمات أو جمل أو روابط معينة لتوليد نص متماسك. راجع [التنفيذ الأصلي](https://github.com/salesforce/ctrl) لمزيد من المعلومات.

- CTRL هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بتعبئة المدخلات على اليمين بدلاً من اليسار.

- تم تدريب CTRL باستخدام هدف نمذجة اللغة السببية (CLM) وهو لذلك قوي في التنبؤ بالرمز التالي في تسلسل. الاستفادة من هذه الميزة تسمح CTRL لتوليد النص متماسكة من الناحية التركيبية كما يمكن ملاحظته في مثال *run_generation.py* النصي.

- يمكن لنماذج PyTorch أن تأخذ `past_key_values` كإدخال، وهو أزواج الاهتمام بالقيمة الرئيسية التي تم حسابها مسبقًا. تقبل نماذج TensorFlow `past` كإدخال. يمنع استخدام قيمة `past_key_values` النموذج من إعادة حساب القيم المحسوبة مسبقًا في سياق توليد النص. راجع طريقة [`forward`](model_doc/ctrl#transformers.CTRLModel.forward) لمزيد من المعلومات حول استخدام هذا الحجة.

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)

- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

## CTRLConfig

[[autodoc]] CTRLConfig

## CTRLTokenizer

[[autodoc]] CTRLTokenizer

- save_vocabulary

<frameworkcontent>

<pt>

## CTRLModel

[[autodoc]] CTRLModel

- forward

## CTRLLMHeadModel

[[autodoc]] CTRLLMHeadModel

- forward

## CTRLForSequenceClassification

[[autodoc]] CTRLForSequenceClassification

- forward

</pt>

<tf>

## TFCTRLModel

[[autodoc]] TFCTRLModel

- call

## TFCTRLLMHeadModel

[[autodoc]] TFCTRLLMHeadModel

- call

## TFCTRLForSequenceClassification

[[autodoc]] TFCTRLForSequenceClassification

- call

</tf>

</frameworkcontent>