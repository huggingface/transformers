# YOSO

## نظرة عامة
تم اقتراح نموذج YOSO في ورقة بحثية بعنوان "You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling" بواسطة Zhanpeng Zeng و Yunyang Xiong و Sathya N. Ravi و Shailesh Acharya و Glenn Fung و Vikas Singh. يقترح النموذج خطة أخذ عينات برنولي (Bernoulli) تعتمد على التجزئة الحساسة للمحلية (Locality Sensitive Hashing) لتقريب الانتباه القياسي للsoftmax. ومن حيث المبدأ، يمكن أخذ عينات من جميع المتغيرات العشوائية ذات الحدين باستخدام تجزئة واحدة.

ملخص الورقة البحثية على النحو التالي:

*تُستخدم النماذج القائمة على محول الترميز بشكل شائع في معالجة اللغات الطبيعية. ويعد الانتباه الذاتي عنصرًا أساسيًا في نموذج المحول، حيث يلتقط تفاعلات أزواج الرموز في تسلسلات الإدخال ويعتمد بشكل رباعي على طول التسلسل. إن تدريب مثل هذه النماذج على تسلسلات أطول أمر مكلف. في هذه الورقة، نُظهر أن آلية الانتباه القائمة على أخذ عينات برنولي (Bernoulli) والتجزئة الحساسة للمحلية (Locality Sensitive Hashing) تقلل من التعقيد الرباعي لهذه النماذج إلى خطي. نتجاوز التكلفة الرباعية من خلال اعتبار الانتباه الذاتي كمبلغ من الرموز الفردية المرتبطة بالمتغيرات العشوائية ذات الحدين التي يمكن، من حيث المبدأ، أخذ عينات منها مرة واحدة بواسطة تجزئة واحدة (على الرغم من أنه في الممارسة العملية، قد يكون هذا الرقم ثابتًا صغيرًا). يؤدي هذا إلى خطة أخذ عينات فعالة لتقدير الانتباه الذاتي والتي تعتمد على تعديلات محددة للتجزئة الحساسة للمحلية (LSH) (لتمكين النشر على بنيات وحدة معالجة الرسوميات). نقيم خوارزميتنا على معيار GLUE مع طول تسلسل قياسي يبلغ 512، حيث نرى أداءً أفضل مقارنة بالمحول القياسي المُدرب مسبقًا. على معيار Long Range Arena (LRA)، لتقييم الأداء على التسلسلات الطويلة، تحقق طريقتنا نتائج متسقة مع الانتباه الذاتي softmax ولكن مع تسريع كبير في السرعة ووفورات في الذاكرة، وغالباً ما تتفوق على طرق الانتباه الذاتي الأخرى الفعالة. يمكن العثور على شفرة البرنامج في هذا الرابط https.*

تمت المساهمة بهذا النموذج من قبل [novice03](https://huggingface.co/novice03). يمكن العثور على الشفرة الأصلية [هنا](https://github.com/mlpen/YOSO).

## نصائح الاستخدام

- يتم تنفيذ خوارزمية انتباه YOSO من خلال نوى CUDA مخصصة، وهي دالات مكتوبة بلغة CUDA C++ يمكن تنفيذها عدة مرات بشكل متواز على وحدة معالجة الرسوميات.

- توفر النواة دالة `fast_hash`، والتي تقارب الإسقاطات العشوائية للاستفسارات والمفاتيح باستخدام التحويل السريع لهادامارد. باستخدام رموز التجزئة هذه، تقارب دالة `lsh_cumulation` الانتباه الذاتي عبر أخذ عينات برنولي (Bernoulli) المستندة إلى التجزئة الحساسة للمحلية (LSH).

- لاستخدام النواة المخصصة، يجب على المستخدم تعيين `config.use_expectation = False`. لضمان تجميع النواة بنجاح، يجب على المستخدم تثبيت الإصدار الصحيح من PyTorch وcudatoolkit. بشكل افتراضي، `config.use_expectation = True`، والذي يستخدم YOSO-E ولا يتطلب تجميع نوى CUDA.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yoso_architecture.jpg" alt="drawing" width="600"/>

<small> خوارزمية انتباه YOSO. مأخوذة من <a href="https://arxiv.org/abs/2111.09714">الورقة البحثية الأصلية</a>.</small>

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)

- [دليل مهمة تصنيف الرموز](../tasks/token_classification)

- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)

- [دليل مهمة نمذجة اللغة المعقدة](../tasks/masked_language_modeling)

- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## YosoConfig

[[autodoc]] YosoConfig

## YosoModel

[[autodoc]] YosoModel

- forward

## YosoForMaskedLM

[[autodoc]] YosoForMaskedLM

- forward

## YosoForSequenceClassification

[[autodoc]] YosoForSequenceClassification

- forward

## YosoForMultipleChoice

[[autodoc]] YosoForMultipleChoice

- forward

## YosoForTokenClassification

[[autodoc]] YosoForTokenClassification

- forward

## YosoForQuestionAnswering

[[autodoc]] YosoForQuestionAnswering

- forward