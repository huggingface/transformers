# Blenderbot

## نظرة عامة

تم اقتراح نموذج دردشة Blenderbot في ورقة "وصفات لبناء دردشة مفتوحة المجال" من قبل ستيفن رولر، إيميلي دينان، نامان جويال، دا جو، ماري ويليامسون، يينهان ليو، جينغ شو، مايل أوت، كورت شاستر، إريك م. سميث، ي-لان بوريو، جيسون ويستون في 30 أبريل 2020.

ملخص الورقة هو كما يلي:

*بناء دردشات مفتوحة المجال هو مجال صعب لبحوث تعلم الآلة. في حين أن العمل السابق قد أظهر أن توسيع نطاق النماذج العصبية من حيث عدد المعلمات وحجم البيانات التي يتم تدريبها عليها يعطي نتائج محسنة، فإننا نوضح أن المكونات الأخرى مهمة لأداء الدردشة عالية الأداء. تتطلب المحادثة الجيدة عددًا من المهارات التي يمزجها المحاور الخبير بطريقة سلسة: تقديم نقاط حديث جذابة والاستماع إلى الشركاء، وعرض المعرفة والتعاطف والشخصية بشكل مناسب، مع الحفاظ على شخصية متسقة. نحن نوضح أن النماذج واسعة النطاق يمكن أن تتعلم هذه المهارات عند إعطائها بيانات التدريب المناسبة واختيار استراتيجية التوليد. نقوم ببناء متغيرات من هذه الوصفات باستخدام نماذج 90 مليون و2.7 مليار و9.4 مليار معلمة، ونقوم بإتاحة نماذجنا ورمزنا للجمهور. تُظهر التقييمات البشرية أن أفضل نماذجنا متفوقة على الأساليب الحالية في الحوار متعدد الأدوار من حيث قياسات الجاذبية والصفات البشرية. ثم نناقش قيود هذا العمل من خلال تحليل حالات الفشل في نماذجنا.*

تمت المساهمة بهذا النموذج من قبل [sshleifer](https://huggingface.co/sshleifer). يمكن العثور على كود المؤلفين [هنا](https://github.com/facebookresearch/ParlAI).

## نصائح الاستخدام ومثال

Blenderbot هو نموذج مع تضمين موضع مطلق، لذلك يُنصح عادةً بإضافة مسافات إلى المدخلات من اليمين بدلاً من اليسار.

مثال:

```python
>>> from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

>>> mname = "facebook/blenderbot-400M-distill"
>>> model = BlenderbotForConditionalGeneration.from_pretrained(mname)
>>> tokenizer = BlenderbotTokenizer.from_pretrained(mname)
>>> UTTERANCE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
>>> reply_ids = model.generate(**inputs)
>>> print(tokenizer.batch_decode(reply_ids))
["<s> That's unfortunate. Are they trying to lose weight or are they just trying to be healthier?</s>"]
```

## ملاحظات التنفيذ

- يستخدم Blenderbot بنية قياسية [نموذج seq2seq transformer](https://arxiv.org/pdf/1706.03762.pdf).
- يمكن العثور على نقاط التفتيش المتاحة في [مركز النماذج](https://huggingface.co/models?search=blenderbot).
- هذا هو فئة Blenderbot *الافتراضية*. ومع ذلك، فإن بعض نقاط التفتيش الأصغر، مثل `facebook/blenderbot_small_90M`، لها بنية مختلفة وبالتالي يجب استخدامها مع [BlenderbotSmall](blenderbot-small).

## الموارد

- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)

## BlenderbotConfig

[[autodoc]] BlenderbotConfig

## BlenderbotTokenizer

[[autodoc]] BlenderbotTokenizer

- build_inputs_with_special_tokens

## BlenderbotTokenizerFast

[[autodoc]] BlenderbotTokenizerFast

- build_inputs_with_special_tokens

<frameworkcontent>

<pt>

## BlenderbotModel

راجع [`~transformers.BartModel`] للحصول على الحجج إلى *forward* و*generate*

[[autodoc]] BlenderbotModel

- forward

## BlenderbotForConditionalGeneration

راجع [`~transformers.BartForConditionalGeneration`] للحصول على الحجج إلى *forward* و*generate*

[[autodoc]] BlenderbotForConditionalGeneration

- forward

## BlenderbotForCausalLM

[[autodoc]] BlenderbotForCausalLM

- forward

</pt>

<tf>

## TFBlenderbotModel

[[autodoc]] TFBlenderbotModel

- call

## TFBlenderbotForConditionalGeneration

[[autodoc]] TFBlenderbotForConditionalGeneration

- call

</tf>

<jax>

## FlaxBlenderbotModel

[[autodoc]] FlaxBlenderbotModel

- __call__
- encode
- decode

## FlaxBlenderbotForConditionalGeneration

[[autodoc]] FlaxBlenderbotForConditionalGeneration

- __call__
- encode
- decode

</jax>

</frameworkcontent>