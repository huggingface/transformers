# BigBirdPegasus

## نظرة عامة
تم اقتراح نموذج BigBird في ورقة "Big Bird: Transformers for Longer Sequences" بواسطة Zaheer وآخرون. BigBird هو محول قائم على الانتباه النادر الذي يوسع النماذج القائمة على المحول، مثل BERT، إلى تسلسلات أطول بكثير. بالإضافة إلى الانتباه النادر، يطبق BigBird أيضًا الانتباه العالمي والانتباه العشوائي على التسلسل المدخل. وقد ثبت نظريًا أن تطبيق الانتباه النادر والعالمي والعشوائي يقارب الانتباه الكامل، مع كونه أكثر كفاءة من الناحية الحسابية للتسلسلات الأطول. ونتيجة للقدرة على التعامل مع السياق الأطول، أظهر BigBird أداءً محسنًا في العديد من مهام معالجة اللغات الطبيعية للوثائق الطويلة، مثل الإجابة على الأسئلة والتلخيص، مقارنة بـ BERT أو RoBERTa.

ملخص الورقة هو كما يلي:

* تعد النماذج القائمة على المحولات، مثل BERT، أحد أكثر نماذج التعلم العميق نجاحًا في معالجة اللغات الطبيعية. للأسف، أحد قيودها الأساسية هو الاعتماد التربيعي (من حيث الذاكرة بشكل أساسي) على طول التسلسل بسبب آلية الانتباه الكامل الخاصة بها. ولعلاج ذلك، نقترح BigBird، وهي آلية انتباه نادرة تقلل هذا الاعتماد التربيعي إلى خطي. نُظهر أن BigBird هو محاكٍ تقريبي شامل لوظائف التسلسل وهو مكتمل من حيث تورينج، وبالتالي يحافظ على هذه الخصائص للنموذج الكامل للانتباه التربيعي. وفي أثناء ذلك، يكشف تحليلنا النظري عن بعض فوائد وجود رموز O(1) العالمية (مثل CLS)، والتي تحضر التسلسل بالكامل كجزء من آلية الانتباه النادرة. يمكن لانتباه النادر المقترح التعامل مع تسلسلات يصل طولها إلى 8x مما كان ممكنًا سابقًا باستخدام أجهزة مماثلة. ونتيجة للقدرة على التعامل مع السياق الأطول، يحسن BigBird الأداء بشكل كبير في العديد من مهام معالجة اللغات الطبيعية مثل الإجابة على الأسئلة والتلخيص. كما نقترح تطبيقات جديدة لبيانات الجينوم.*

يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/bigbird).

## نصائح الاستخدام

- للحصول على شرح تفصيلي لكيفية عمل انتباه BigBird، راجع [منشور المدونة هذا](https://huggingface.co/blog/big-bird).
- يأتي BigBird بتنفيذين: **original_full** و **block_sparse**. بالنسبة لطول التسلسل < 1024، يُنصح باستخدام **original_full** حيث لا توجد فائدة من استخدام انتباه **block_sparse**.
- يستخدم الكود حاليًا حجم نافذة يبلغ 3 كتل وكتلتين عالميتين.
- يجب أن يكون طول التسلسل قابلاً للقسمة على حجم الكتلة.
- يدعم التنفيذ الحالي **ITC** فقط.
- لا يدعم التنفيذ الحالي **num_random_blocks = 0**.
- يستخدم BigBirdPegasus [PegasusTokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pegasus/tokenization_pegasus.py).
- BigBird هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بتعبئة المدخلات من اليمين بدلاً من اليسار.

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة التلخيص](../tasks/summarization)

## BigBirdPegasusConfig

[[autodoc]] BigBirdPegasusConfig
- all

## BigBirdPegasusModel

[[autodoc]] BigBirdPegasusModel
- forword

## BigBirdPegasusForConditionalGeneration

[[autodoc]] BigBirdPegasusForConditionalGeneration
- forword

## BigBirdPegasusForSequenceClassification

[[autodoc]] BigBirdPegasusForSequenceClassification
- forword

## BigBirdPegasusForQuestionAnswering

[[autodoc]] BigBirdPegasusForQuestionAnswering
- forword

## BigBirdPegasusForCausalLM

[[autodoc]] BigBirdPegasusForCausalLM
- forword