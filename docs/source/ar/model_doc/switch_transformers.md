# SwitchTransformers

## نظرة عامة

اقترح نموذج SwitchTransformers في "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" بواسطة William Fedus وBarret Zoph وNoam Shazeer.

يستخدم نموذج Switch Transformer بنية T5 encoder-decoder نادرة، حيث يتم استبدال MLP بمزيج من الخبراء (MoE). تقوم آلية التوجيه (أعلى 1 في هذه الحالة) بربط كل رمز بخبير، حيث يكون كل خبير عبارة عن MLP كثيف. على الرغم من أن محولات التبديل تحتوي على عدد أكبر بكثير من الأوزان مقارنة بالنماذج الكثيفة المكافئة، فإن النُدرة تسمح بزيادة النطاق بشكل أفضل وأداء ضبط أفضل عند النطاق.

أثناء التمرير للأمام، يتم استخدام جزء فقط من الأوزان. تسمح آلية التوجيه للنموذج باختيار الأوزان ذات الصلة أثناء التنقل، مما يزيد من سعة النموذج دون زيادة عدد العمليات.

الملخص من الورقة هو كما يلي:

> "في التعلم العميق، عادة ما تعيد النماذج استخدام نفس المعلمات لجميع المدخلات. تتحدى مجموعة الخبراء هذا وتختار بدلاً من ذلك معلمات مختلفة لكل مثال وارد. النتيجة هي نموذج مُنشط بشكل متقطع - بأعداد هائلة من المعلمات - ولكن بتكلفة حسابية ثابتة. ومع ذلك، على الرغم من العديد من النجاحات الملحوظة لـ MoE، فقد أعيق الاعتماد الواسع النطاق بسبب التعقيد وتكاليف الاتصال وعدم استقرار التدريب - نعالج هذه المشكلات باستخدام محول التبديل. نقوم بتبسيط خوارزمية توجيه MoE ونصمم نماذج محسنة بديهية بتكاليف اتصال وحساب مخفضة. تساعد تقنيات التدريب المقترحة في السيطرة على عدم الاستقرار ونظهر أنه يمكن تدريب النماذج النادرة الكبيرة، لأول مرة، بتنسيقات ذات دقة أقل (bfloat16). نقوم بتصميم نماذج بناءً على T5-Base وT5-Large للحصول على زيادات تصل إلى 7x في سرعة التدريب المسبق باستخدام نفس الموارد الحسابية. تمتد هذه التحسينات إلى الإعدادات متعددة اللغات حيث نقيس المكاسب على إصدار mT5-Base عبر جميع 101 لغة. وأخيراً، نقوم بزيادة الحجم الحالي لنماذج اللغة عن طريق التدريب المسبق على نماذج معلمات تصل إلى تريليون على "Colossal Clean Crawled Corpus" وتحقيق تسريع 4x على نموذج T5-XXL".

تمت المساهمة بهذا النموذج من قبل [Younes Belkada] و [Arthur Zucker].

يمكن العثور على الكود الأصلي [هنا] (https://github.com/google/flaxformer/tree/main/flaxformer/architectures/moe).

## نصائح الاستخدام

- يستخدم SwitchTransformers [`T5Tokenizer`]، والذي يمكن تحميله مباشرة من مستودع كل نموذج.
- تم تدريب الأوزان التي تم إصدارها مسبقًا على مهمة اللغة الإنجليزية [Masked Language Modeling]، ويجب ضبط دقتها.

## الموارد

- [دليل مهمة الترجمة] (../tasks/translation)
- [دليل مهمة الملخص] (../tasks/summarization)

## SwitchTransformersConfig

[[autodoc]] SwitchTransformersConfig

## SwitchTransformersTop1Router

[[autodoc]] SwitchTransformersTop1Router

- _compute_router_probabilities
- forward

## SwitchTransformersSparseMLP

[[autodoc]] SwitchTransformersSparseMLP

- forward

## SwitchTransformersModel

[[autodoc]] SwitchTransformersModel

- forward

## SwitchTransformersForConditionalGeneration

[[autodoc]] SwitchTransformersForConditionalGeneration

- forward

## SwitchTransformersEncoderModel

[[autodoc]] SwitchTransformersEncoderModel

- forward