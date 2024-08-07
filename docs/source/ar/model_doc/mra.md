# MRA

## نظرة عامة
تم اقتراح نموذج MRA في [تحليل متعدد الدقة (MRA) للاهتمام الذاتي التقريبي](https://arxiv.org/abs/2207.10284) بواسطة Zhanpeng Zeng، وSourav Pal، وJeffery Kline، وGlenn M Fung، و Vikas Singh.

مقدمة الورقة البحثية هي كما يلي:

* برزت نماذج Transformers كنموذج مفضل للعديد من المهام في معالجة اللغات الطبيعية والرؤية. وقد حددت الجهود الأخيرة في تدريب ونشر Transformers بكفاءة أكبر العديد من الاستراتيجيات لتقريب مصفوفة الاهتمام الذاتي، وهي وحدة رئيسية في بنية Transformer. تشمل الأفكار الفعالة أنماط ندرة محددة مسبقًا مختلفة، وتوسعات الأساس منخفضة الرتبة، ومجموعات منها. في هذه الورقة، نعيد النظر في مفاهيم تحليل متعدد الدقة (MRA) الكلاسيكية مثل Wavelets، والتي لا تزال قيمتها المحتملة في هذا السياق غير مستغلة بالكامل حتى الآن. نحن نثبت أن التقريبات البسيطة القائمة على التعليقات الفعلية وخيارات التصميم التي تسترشد بتحديات الأجهزة والتنفيذ الحديثة، تؤدي في النهاية إلى نهج قائم على MRA للاهتمام الذاتي بأداء ممتاز عبر معظم المعايير المثيرة للاهتمام. نقوم بمجموعة واسعة من التجارب ونثبت أن هذا المخطط متعدد الدقة يتفوق على معظم مقترحات الاهتمام الذاتي الكفء ويفضل لكل من التسلسلات القصيرة والطويلة. الكود متاح على https://github.com/mlpen/mra-attention.*

تمت المساهمة بهذا النموذج من قبل [novice03](https://huggingface.co/novice03).
يمكن العثور على الكود الأصلي [هنا](https://github.com/mlpen/mra-attention).

## MraConfig

[[autodoc]] MraConfig

## MraModel

[[autodoc]] MraModel

- forward

## MraForMaskedLM

[[autodoc]] MraForMaskedLM

- forward

## MraForSequenceClassification

[[autodoc]] MraForSequenceClassification

- forward

## MraForMultipleChoice

[[autodoc]] MraForMultipleChoice

- forward

## MraForTokenClassification

[[autodoc]] MraForTokenClassification

- forward

## MraForQuestionAnswering

[[autodoc]] MraForQuestionAnswering

- forward