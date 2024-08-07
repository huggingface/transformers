# PatchTST

## نظرة عامة
تم اقتراح نموذج PatchTST في ورقة بحثية بعنوان "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" بواسطة Yuqi Nie و Nam H. Nguyen و Phanwadee Sinthong و Jayant Kalagnanam.

بشكل عام، يقوم النموذج بتقطيع السلاسل الزمنية إلى رقع ذات حجم معين وترميز تسلسل المتجهات الناتج عبر محول (Transformer) يقوم بعد ذلك بإخراج توقع طول التنبؤ عبر رأس مناسب. يتم توضيح النموذج في الشكل التالي:

يقدم الملخص التالي من الورقة:

"نقترح تصميمًا فعالًا لنماذج المحولات القائمة على السلاسل الزمنية متعددة المتغيرات والتعلم التمثيلي الخاضع للإشراف الذاتي. يعتمد على مكونين رئيسيين: (1) تقسيم السلاسل الزمنية إلى رقع على مستوى السلاسل الفرعية والتي يتم استخدامها كرموز دخول إلى المحول؛ (2) الاستقلالية القناة حيث تحتوي كل قناة على سلسلة زمنية أحادية المتغيرات تشترك في نفس التضمين وأوزان المحول عبر جميع السلاسل. يحتوي التصميم التصحيح بشكل طبيعي على فائدة ثلاثية: يتم الاحتفاظ بالمعلومات الدلالية المحلية في التضمين؛ يتم تقليل استخدام الذاكرة والحساب لخرائط الاهتمام بشكل رباعي نظرًا لنفس نافذة النظر إلى الوراء؛ ويمكن للنموذج الاهتمام بتاريخ أطول. يمكن لمحولنا المستقل عن القناة لتصحيح السلاسل الزمنية (PatchTST) تحسين دقة التنبؤ على المدى الطويل بشكل كبير عند مقارنته بنماذج المحولات المستندة إلى SOTA. نطبق نموذجنا أيضًا على مهام التدريب المسبق الخاضعة للإشراف الذاتي ونحقق أداء ضبط دقيق ممتازًا، والذي يتفوق على التدريب الخاضع للإشراف على مجموعات البيانات الكبيرة. كما أن نقل التمثيل المقنع مسبقًا من مجموعة بيانات إلى أخرى ينتج أيضًا دقة تنبؤية على مستوى SOTA."

تمت المساهمة بهذا النموذج من قبل namctin و gsinthong و diepi و vijaye12 و wmgifford و kashif. يمكن العثور على الكود الأصلي [هنا](https://github.com/yuqinie98/PatchTST).

## نصائح الاستخدام
يمكن أيضًا استخدام النموذج لتصنيف السلاسل الزمنية والانحدار الزمني. راجع فئات [`PatchTSTForClassification`] و [`PatchTSTForRegression`] الخاصة بكل منهما.

## الموارد
- يمكن العثور على منشور المدونة الذي يشرح PatchTST بالتفصيل [هنا](https://huggingface.co/blog/patchtst). يمكن أيضًا فتح المدونة في Google Colab.

## PatchTSTConfig

[[autodoc]] PatchTSTConfig

## PatchTSTModel

[[autodoc]] PatchTSTModel

- forward

## PatchTSTForPrediction

[[autodoc]] PatchTSTForPrediction

- forward

## PatchTSTForClassification

[[autodoc]] PatchTSTForClassification

- forward

## PatchTSTForPretraining

[[autodoc]] PatchTSTForPretraining

- forward

## PatchTSTForRegression

[[autodoc]] PatchTSTForRegression

- forward