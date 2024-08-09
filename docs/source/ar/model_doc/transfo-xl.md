# Transformer XL

> ⚠️ هذا النموذج في وضع الصيانة فقط، لذلك لن نقبل أي PRs جديدة لتغيير شفرته. تم إيقاف استخدام هذا النموذج بسبب مشكلات أمنية مرتبطة بـ `pickle.load`.
> نوصي بالتحويل إلى النماذج الأحدث للحصول على أمان محسن.
> في حالة الرغبة في استخدام `TransfoXL` في تجاربك، نوصي باستخدام [نقطة تفتيش Hub](https://huggingface.co/transfo-xl/transfo-xl-wt103) مع مراجعة محددة لضمان تنزيل ملفات آمنة من Hub.
> ستحتاج إلى تعيين متغير البيئة `TRUST_REMOTE_CODE` إلى `True` للسماح باستخدام `pickle.load()`:

```python
import os
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

os.environ["TRUST_REMOTE_CODE"] = "True"

checkpoint = 'transfo-xl/transfo-xl-wt103'
revision = '40a186da79458c9f9de846edfaea79c412137f97'

tokenizer = TransfoXLTokenizer.from_pretrained(checkpoint, revision=revision)
model = TransfoXLLMHeadModel.from_pretrained(checkpoint, revision=revision)
```

> إذا واجهت أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.35.0.
> يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.35.0`.

## نظرة عامة

تم اقتراح نموذج Transformer-XL في [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) بواسطة Zihang Dai و Zhilin Yang و Yiming Yang و Jaime Carbonell و Quoc V. Le و Ruslan Salakhutdinov. إنه محول أحادي الاتجاه (اتجاهي) مع تضمين الموضع النسبي (التذبذب) الذي يمكنه إعادة استخدام الحالات المخفية المحسوبة مسبقًا للاهتمام بسياق أطول (ذاكرة). يستخدم هذا النموذج أيضًا إدخالات Softmax وإخراجها التكيفي (المتصل).

الملخص من الورقة هو كما يلي:

> "للتحويلات إمكانية تعلم الاعتماد طويل الأمد، ولكنها مقيدة بسياق ذي طول ثابت في إعداد نمذجة اللغة. نقترح بنية عصبية جديدة تسمى Transformer-XL التي تمكن من تعلم الاعتماد خارج طول ثابت دون تعطيل التماسك الزمني. يتكون من آلية تكرار على مستوى القطاع ومخطط ترميز موضعي جديد. لا تمكن طريقة عملنا من التقاط الاعتماد طويل الأمد فحسب، بل تحل أيضًا مشكلة تجزئة السياق. نتيجة لذلك، يتعلم Transformer-XL الاعتماد الذي يزيد بنسبة 80% عن شبكات RNN وبنسبة 450% عن المحولات الأساسية، ويحقق أداءً أفضل في كل من التسلسلات القصيرة والطويلة، وهو أسرع بمقدار 1800 مرة من المحولات الأساسية أثناء التقييم. وبشكل ملحوظ، نحسن نتائج الحالة المثلى من bpc/perplexity إلى 0.99 على enwiki8، و 1.08 على text8، و 18.3 على WikiText-103، و 21.8 على One Billion Word، و 54.5 على Penn Treebank (بدون ضبط دقيق). عندما يتم تدريب Transformer-XL فقط على WikiText-103، فإنه ينشئ مقالات نصية متماسكة ومعقولة مع آلاف الرموز."

تمت المساهمة بهذا النموذج من قبل [thomwolf](https://huggingface.co/thomwolf). يمكن العثور على الكود الأصلي [هنا](https://github.com/kimiyoung/transformer-xl).

## نصائح الاستخدام

- يستخدم Transformer-XL تضمين الموضع النسبي التذبذبي. يمكن إجراء التعبئة على اليسار أو اليمين. يستخدم التنفيذ الأصلي التدريب على SQuAD مع التعبئة على اليسار، لذلك يتم تعيين الوسائد الافتراضية إلى اليسار.
- Transformer-XL هو أحد النماذج القليلة التي لا يوجد بها حد لطول التسلسل.
- مثل نموذج GPT العادي، ولكنه يقدم آلية تكرار لقطاعين متتاليين (مشابهة لشبكات RNN العادية مع إدخالين متتاليين). في هذا السياق، القطاع هو عدد من الرموز المتتالية (على سبيل المثال 512) والتي قد تمتد عبر وثائق متعددة، ويتم تغذية القطاعات بترتيب إلى النموذج.
- بشكل أساسي، يتم دمج الحالات المخفية للقطاع السابق مع الإدخال الحالي لحساب درجات الاهتمام. يسمح هذا للنموذج بالاهتمام بالمعلومات التي كانت في القطاع السابق بالإضافة إلى الحالي. من خلال تكديس طبقات الاهتمام المتعددة، يمكن زيادة مجال الاستقبال إلى قطاعات متعددة سابقة.
- يغير هذا التضمين الموضعي إلى تضمين موضعي نسبي (نظرًا لأن التضمين الموضعي العادي من شأنه أن يعطي نفس النتائج في الإدخال الحالي والحالة المخفية الحالية في موضع معين) ويجب إجراء بعض التعديلات في طريقة حساب درجات الاهتمام.

> ⚠️ لا يعمل TransformerXL مع *torch.nn.DataParallel* بسبب وجود خلل في PyTorch، راجع [القضية #36035](https://github.com/pytorch/pytorch/issues/36035).

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)

## TransfoXLConfig

[[autodoc]] TransfoXLConfig

## TransfoXLTokenizer

[[autodoc]] TransfoXLTokenizer

- save_vocabulary

## المخرجات المحددة لـ TransfoXL

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput

[[autodoc]] models.deprecated.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput

[[autodoc]] models.deprecated.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLModelOutput

[[autodoc]] models.deprecated.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLLMHeadModelOutput

<frameworkcontent>
<pt>

## TransfoXLModel

[[autodoc]] TransfoXLModel

- forward

## TransfoXLLMHeadModel

[[autodoc]] TransfoXLLMHeadModel

- forward

## TransfoXLForSequenceClassification

[[autodoc]] TransfoXLForSequenceClassification

- forward

</pt>
<tf>

## TFTransfoXLModel

[[autodoc]] TFTransfoXLModel

- call

## TFTransfoXLLMHeadModel

[[autodoc]] TFTransfoXLLMHeadModel

- call

## TFTransfoXLForSequenceClassification

[[autodoc]] TFTransfoXLForSequenceClassification

- call

</tf>
</frameworkcontent>

## الطبقات الداخلية

[[autodoc]] AdaptiveEmbedding

[[autodoc]] TFAdaptiveEmbedding