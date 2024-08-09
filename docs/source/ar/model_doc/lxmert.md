# LXMERT

## نظرة عامة

تم اقتراح نموذج LXMERT في [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) بواسطة Hao Tan & Mohit Bansal. وهو عبارة عن سلسلة من مشفرات Transformer ثنائية الاتجاه (واحد لوضع الرؤية، وواحد لوضع اللغة، ثم واحد لدمج الوضعين) مُعلمة مسبقًا باستخدام مزيج من نمذجة اللغة المقنعة، ومواءمة النص المرئي، وانحدار ميزة ROI، ونمذجة السمات المرئية المقنعة، ونمذجة الأشياء المرئية المقنعة، وأهداف الإجابة على الأسئلة المرئية. ويتكون التعليم المُسبق من مجموعات بيانات متعددة الوسائط المتعددة: MSCOCO، وVisual-Genome + Visual-Genome Question Answering، وVQA 2.0، وGQA.

الملخص من الورقة هو ما يلي:

*يتطلب الاستدلال المرئي واللغوي فهم المفاهيم المرئية، ودلالة اللغة، والأهم من ذلك، محاذاة وعلاقات هاتين الطريقتين. لذلك نقترح إطار LXMERT (Learning Cross-Modality Encoder Representations from Transformers) لتعلم هذه الاتصالات المرئية واللغوية. في LXMERT، نقوم ببناء نموذج Transformer واسع النطاق يتكون من ثلاثة مشفرات: مشفر علاقات الكائنات، ومشفر اللغة، ومشفر متعدد الوسائط. بعد ذلك، لتمكين نموذجنا من القدرة على توصيل الدلالات المرئية واللغوية، نقوم بالتعليم المُسبق للنموذج باستخدام أزواج الصور والجمل الكبيرة، عبر خمس مهام تعليم مُسبق تمثيلية متنوعة: نمذجة اللغة المقنعة، والتنبؤ بالكائنات المقنعة (انحدار السمات وتصنيف العلامات)، والمطابقة متعددة الوسائط، والإجابة على أسئلة الصور. تساعد هذه المهام في تعلم العلاقات داخل الوسائط وعبر الوسائط. بعد الضبط الدقيق من معلماتنا المُعلمة مسبقًا، يحقق نموذجنا نتائج متقدمة على مجموعتين من بيانات الإجابة على الأسئلة المرئية (أي VQA وGQA). كما نوضح قابلية تعميم نموذجنا متعدد الوسائط المُعلم من خلال تكييفه مع مهمة استدلال مرئي صعبة، NLVR، وتحسين أفضل نتيجة سابقة بنسبة 22% مطلقة (من 54% إلى 76%). وأخيرًا، نقدم دراسات تفصيلية لإثبات أن كلًا من مكونات النموذج المبتكرة واستراتيجيات التعليم المُسبق تساهم بشكل كبير في نتائجنا القوية؛ ونقدم أيضًا العديد من تصورات الانتباه لمشفرات مختلفة.*

تمت المساهمة بهذا النموذج من قبل [eltoto1219](https://huggingface.co/eltoto1219). يمكن العثور على الكود الأصلي [هنا](https://github.com/airsplay/lxmert).

## نصائح الاستخدام

- لا يلزم استخدام صناديق الحدود في تضمين الميزات المرئية، حيث يمكن استخدام أي نوع من الميزات المرئية المكانية.
- يتم تمرير كل من الحالات المخفية اللغوية والحالات المخفية المرئية التي ينتجها LXMERT عبر طبقة متعددة الوسائط، لذا فهي تحتوي على معلومات من كلا الوضعين. للوصول إلى طريقة لا تنتبه إلا لنفسها، حدد الحالات المخفية المرئية/اللغوية من الإدخال الأول في الرباعي.
- لا تقوم طبقة الترميز ثنائية الاتجاه متعددة الوسائط إلا بإرجاع قيم الانتباه عند استخدام الوضع اللغوي كإدخال والوضع المرئي كمتجه سياق. علاوة على ذلك، بينما يحتوي المشفر متعدد الوسائط على الانتباه الذاتي لكل طريقة والانتباه المتبادل، يتم إرجاع الانتباه المتبادل فقط ويتم تجاهل كل من نتائج الانتباه الذاتي.

## الموارد

- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)

## LxmertConfig

[[autodoc]] LxmertConfig

## LxmertTokenizer

[[autodoc]] LxmertTokenizer

## LxmertTokenizerFast

[[autodoc]] LxmertTokenizerFast

## المخرجات الخاصة بـ Lxmert

[[autodoc]] models.lxmert.modeling_lxmert.LxmertModelOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput

[[autodoc]] models.lxmert.modeling_tf_lxmert.TFLxmertModelOutput

[[autodoc]] models.lxmert.modeling_tf_lxmert.TFLxmertForPreTrainingOutput

<frameworkcontent>

<pt>

## LxmertModel

[[autodoc]] LxmertModel

- forward

## LxmertForPreTraining

[[autodoc]] LxmertForPreTraining

- forward

## LxmertForQuestionAnswering

[[autodoc]] LxmertForQuestionAnswering

- forward

</pt>

<tf>

## TFLxmertModel

[[autodoc]] TFLxmertModel

- call

## TFLxmertForPreTraining

[[autodoc]] TFLxmertForPreTraining

- call

</tf>

</frameworkcontent>