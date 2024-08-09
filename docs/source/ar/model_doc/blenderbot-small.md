# Blenderbot Small

Note that [`BlenderbotSmallModel`] and [`BlenderbotSmallForConditionalGeneration`] are only used in combination with the checkpoint [facebook/blenderbot-90M](https://huggingface.co/facebook/blenderbot-90M). Larger Blenderbot checkpoints should instead be used with [`BlenderbotModel`] and [`BlenderbotForConditionalGeneration`].

## نظرة عامة

اقتُرح نموذج محادثة Blender في ورقة "وصفات لبناء محادث آلي مفتوح المجال" من تأليف ستيفن رولر، وإميلي دينان، ونامان جويال، ودا جو، وماري ويليامسون، ويينهان ليو، وجينغ شو، ومايل أوت، وكورت شاستر، وإريك م. سميث، وY-لان بوريو، وجيسون ويستون في 30 أبريل 2020.

وفيما يلي ملخص الورقة:

*إن بناء محادثات آلية مفتوحة المجال هو مجال صعب لأبحاث تعلم الآلة. في حين أظهرت الأعمال السابقة أن توسيع نطاق النماذج العصبية من حيث عدد المعلمات وحجم البيانات التي يتم تدريبها عليها يعطي نتائج محسنة، فإننا نُظهر أن هناك مكونات أخرى مهمة لأداء المحادث الآلي عالي الجودة. تتطلب المحادثة الجيدة عددًا من المهارات التي يمزجها المحاور الخبير بطريقة سلسة: تقديم نقاط محادثة جذابة والاستماع إلى الشركاء، وعرض المعرفة والتعاطف والشخصية بشكل مناسب، مع الحفاظ على شخصية متسقة. نُظهر أن النماذج واسعة النطاق يمكنها تعلم هذه المهارات عند تزويدها ببيانات التدريب المناسبة وخيار استراتيجية التوليد. نقوم ببناء متغيرات من هذه الوصفات باستخدام نماذج 90 مليون و2.7 مليار و9.4 مليار معلمة، ونقوم بإتاحة نماذجنا وشفرة البرنامج الخاصة بنا للعموم. تُظهر التقييمات البشرية أن أفضل نماذجنا متفوقة على النُهج الحالية في الحوار متعدد الأدوار من حيث قياسات الجاذبية والتشابه البشري. ثم نناقش حدود هذا العمل من خلال تحليل حالات الفشل في نماذجنا.*

ساهم بهذا النموذج [باتريك فون بلاتين](https://huggingface.co/patrickvonplaten). يمكن العثور على شفرة البرنامج الخاصة بالمؤلفين [هنا](https://github.com/facebookresearch/ParlAI).

## نصائح الاستخدام

Blenderbot Small هو نموذج يستخدم تضمين الموضع المطلق، لذلك يُنصح عادةً بإضافة مسافة بادئة إلى المدخلات من اليمين بدلاً من اليسار.

## الموارد

- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)

## BlenderbotSmallConfig

[[autodoc]] BlenderbotSmallConfig

## BlenderbotSmallTokenizer

[[autodoc]] BlenderbotSmallTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## BlenderbotSmallTokenizerFast

[[autodoc]] BlenderbotSmallTokenizerFast

<frameworkcontent>

<pt>

## BlenderbotSmallModel

[[autodoc]] BlenderbotSmallModel

- forward

## BlenderbotSmallForConditionalGeneration

[[autodoc]] BlenderbotSmallForConditionalGeneration

- forward

## BlenderbotSmallForCausalLM

[[autodoc]] BlenderbotSmallForCausalLM

- forward

</pt>

<tf>

## TFBlenderbotSmallModel

[[autodoc]] TFBlenderbotSmallModel

- call

## TFBlenderbotSmallForConditionalGeneration

[[autodoc]] TFBlenderbotSmallForConditionalGeneration

- call

</tf>

<jax>

## FlaxBlenderbotSmallModel

[[autodoc]] FlaxBlenderbotSmallModel

- __call__
- encode
- decode

## FlaxBlenderbotForConditionalGeneration

[[autodoc]] FlaxBlenderbotSmallForConditionalGeneration

- __call__
- encode
- decode

</jax>

</frameworkcontent>