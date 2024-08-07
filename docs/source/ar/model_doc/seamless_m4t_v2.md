# SeamlessM4T-v2

## نظرة عامة

اقترح فريق Seamless Communication من Meta AI نموذج SeamlessM4T-v2 في [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/).

SeamlessM4T-v2 عبارة عن مجموعة من النماذج المصممة لتوفير ترجمة عالية الجودة، مما يسمح للناس من مجتمعات لغوية مختلفة بالتواصل بسلاسة من خلال الكلام والنص. إنه تحسين على [الإصدار السابق](https://huggingface.co/docs/transformers/main/model_doc/seamless_m4t). لمزيد من التفاصيل حول الاختلافات بين الإصدارين v1 و v2، راجع القسم [الاختلاف مع SeamlessM4T-v1](#difference-with-seamlessm4t-v1).

يتيح SeamlessM4T-v2 مهام متعددة دون الاعتماد على نماذج منفصلة:

- الترجمة الكلامية النصية (S2ST)
- الترجمة الكلامية النصية (S2TT)
- الترجمة النصية الكلامية (T2ST)
- الترجمة النصية النصية (T2TT)
- التعرف التلقائي على الكلام (ASR)

يمكن لـ [`SeamlessM4Tv2Model`] تنفيذ جميع المهام المذكورة أعلاه، ولكن لكل مهمة أيضًا نموذج فرعي مخصص لها.

المقتطف من الورقة هو ما يلي:

*حققت التطورات الأخيرة في الترجمة الكلامية التلقائية توسعًا كبيرًا في تغطية اللغة، وتحسين القدرات متعددة الوسائط، وتمكين مجموعة واسعة من المهام والوظائف. ومع ذلك، تفتقر أنظمة الترجمة الكلامية التلقائية واسعة النطاق اليوم إلى ميزات رئيسية تساعد في جعل التواصل بوساطة الآلة سلسًا عند مقارنته بالحوار بين البشر. في هذا العمل، نقدم عائلة من النماذج التي تمكّن الترجمات التعبيرية والمتعددة اللغات بطريقة متواصلة. أولاً، نساهم في تقديم نسخة محسّنة من نموذج SeamlessM4T متعدد اللغات والوسائط بشكل كبير - SeamlessM4T v2. تم تدريب هذا النموذج الأحدث، الذي يتضمن إطار عمل UnitY2 المحدث، على المزيد من بيانات اللغات منخفضة الموارد. تضيف النسخة الموسعة من SeamlessAlign 114,800 ساعة من البيانات المحاذاة تلقائيًا لما مجموعه 76 لغة. يوفر SeamlessM4T v2 الأساس الذي تقوم عليه أحدث نماذجنا، SeamlessExpressive و SeamlessStreaming. يمكّن SeamlessExpressive الترجمة التي تحافظ على الأساليب الصوتية والتنغيم. مقارنة بالجهود السابقة في أبحاث الكلام التعبيري، يعالج عملنا بعض الجوانب الأقل استكشافًا للتنغيم، مثل معدل الكلام والتوقفات، مع الحفاظ على أسلوب صوت الشخص. أما بالنسبة لـ SeamlessStreaming، فإن نموذجنا يستفيد من آلية Efficient Monotonic Multihead Attention (EMMA) لتوليد ترجمات مستهدفة منخفضة الكمون دون انتظار نطق المصدر الكامل. يعد SeamlessStreaming الأول من نوعه، حيث يمكّن الترجمة الكلامية/النصية المتزامنة للغات المصدر والهدف المتعددة. لفهم أداء هذه النماذج، قمنا بدمج الإصدارات الجديدة والمعدلة من المقاييس التلقائية الموجودة لتقييم التنغيم والكمون والمتانة. وبالنسبة للتقييمات البشرية، قمنا بتكييف البروتوكولات الموجودة المصممة لقياس أكثر السمات ملاءمة في الحفاظ على المعنى والطبيعية والتعبير. ولضمان إمكانية استخدام نماذجنا بشكل آمن ومسؤول، قمنا بتنفيذ أول جهد معروف للفريق الأحمر للترجمة الآلية متعددة الوسائط، ونظام للكشف عن السمية المضافة والتخفيف منها، وتقييم منهجي للتحيز بين الجنسين، وآلية ترميز علامات مائية محلية غير مسموعة مصممة للتخفيف من تأثير مقاطع الفيديو المزيفة. وبالتالي، فإننا نجمع المكونات الرئيسية من SeamlessExpressive و SeamlessStreaming لتشكيل Seamless، وهو أول نظام متاح للجمهور يفتح التواصل التعبيري متعدد اللغات في الوقت الفعلي. في الختام، يمنحنا Seamless نظرة محورية على الأساس التقني اللازم لتحويل المترجم الشفوي العالمي من مفهوم الخيال العلمي إلى تكنولوجيا الواقع. وأخيرًا، يتم إطلاق المساهمات في هذا العمل - بما في ذلك النماذج والشفرة وكاشف العلامات المائية - علنًا ويمكن الوصول إليها من خلال الرابط أدناه.*

## الاستخدام

في المثال التالي، سنقوم بتحميل عينة صوت عربية وعينة نص إنجليزي وتحويلهما إلى كلام روسي ونص فرنسي.

أولاً، قم بتحميل المعالج ونقطة تفتيش للنموذج:

```python
>>> from transformers import AutoProcessor, SeamlessM4Tv2Model

>>> processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
>>> model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
```

يمكنك استخدام هذا النموذج بسلاسة على النص أو الصوت، لتوليد نص مترجم أو صوت مترجم.

هكذا تستخدم المعالج لمعالجة النص والصوت:

```python
>>> # دعنا نحمل عينة صوت من مجموعة بيانات كلام عربية
>>> from datasets import load_dataset
>>> dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
>>> audio_sample = next(iter(dataset))["audio"]

>>> # الآن، قم بمعالجته
>>> audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt")

>>> # الآن، قم بمعالجة بعض النصوص الإنجليزية أيضًا
>>> text_inputs = processor(text="Hello, my dog is cute", src_lang="eng", return_tensors="pt")
```

### الكلام

يمكن لـ [`SeamlessM4Tv2Model`] *بسلاسة* توليد نص أو كلام مع عدد قليل من التغييرات أو بدونها. دعنا نستهدف الترجمة الصوتية الروسية:

```python
>>> audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
>>> audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
```

بنفس الكود تقريبًا، قمت بترجمة نص إنجليزي وكلام عربي إلى عينات كلام روسية.

### النص

وبالمثل، يمكنك توليد نص مترجم من ملفات الصوت أو النص باستخدام نفس النموذج. عليك فقط تمرير `generate_speech=False` إلى [`SeamlessM4Tv2Model.generate`].

هذه المرة، دعنا نقوم بالترجمة إلى الفرنسية.

```python
>>> # من الصوت
>>> output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

>>> # من النص
>>> output_tokens = model.generate(**text_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
```

### نصائح

#### 1. استخدم النماذج المخصصة

[`SeamlessM4Tv2Model`] هو نموذج المستوى الأعلى في برنامج Transformers لتوليد الكلام والنص، ولكن يمكنك أيضًا استخدام النماذج المخصصة التي تؤدي المهمة دون مكونات إضافية، مما يقلل من بصمة الذاكرة.

على سبيل المثال، يمكنك استبدال جزء التعليمات البرمجية لتوليد الصوت بالنموذج المخصص لمهمة S2ST، والباقي هو نفس الكود بالضبط:

```python
>>> from transformers import SeamlessM4Tv2ForSpeechToSpeech
>>> model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
```

أو يمكنك استبدال جزء التعليمات البرمجية لتوليد النص بالنموذج المخصص لمهمة T2TT، عليك فقط إزالة `generate_speech=False`.

```python
>>> from transformers import SeamlessM4Tv2ForTextToText
>>> model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
```

لا تتردد في تجربة [`SeamlessM4Tv2ForSpeechToText`] و [`SeamlessM4Tv2ForTextToSpeech`] أيضًا.

#### 2. تغيير هوية المتحدث

لديك إمكانية تغيير المتحدث المستخدم للتخليق الصوتي باستخدام وسيط `speaker_id`. تعمل بعض قيم `speaker_id` بشكل أفضل من غيرها لبعض اللغات!

#### 3. تغيير استراتيجية التوليد

يمكنك استخدام استراتيجيات [توليد](../generation_strategies) مختلفة للنص، على سبيل المثال `.generate(input_ids=input_ids, text_num_beams=4, text_do_sample=True)` والتي ستؤدي فك تشفير شعاع متعدد الحدود على النموذج النصي. لاحظ أن توليد الكلام يدعم فقط الجشع - بشكل افتراضي - أو العينات متعددة الحدود، والتي يمكن استخدامها مع `.generate(..., speech_do_sample=True, speech_temperature=0.6)`.

#### 4. قم بتوليد الكلام والنص في نفس الوقت

استخدم `return_intermediate_token_ids=True` مع [`SeamlessM4Tv2Model`] لإرجاع كل من الكلام والنص!

## بنية النموذج

يتميز SeamlessM4T-v2 ببنية مرنة تتعامل بسلاسة مع التوليد التسلسلي للنص والكلام. يتضمن هذا الإعداد نموذجين تسلسليين (seq2seq). يقوم النموذج الأول بترجمة المدخلات إلى نص مترجم، بينما يقوم النموذج الثاني بتوليد رموز الكلام، المعروفة باسم "رموز الوحدة"، من النص المترجم.

لدى كل وسيطة مشفرها الخاص ببنية فريدة. بالإضافة إلى ذلك، بالنسبة لخرج الكلام، يوجد محول ترميز مستوحى من بنية [HiFi-GAN](https://arxiv.org/abs/2010.05646) أعلى نموذج seq2seq الثاني.

### الاختلاف مع SeamlessM4T-v1

تختلف بنية هذه النسخة الجديدة عن الإصدار الأول في بعض الجوانب:

#### التحسينات على نموذج المرور الثاني

نموذج seq2seq الثاني، المسمى نموذج النص إلى وحدة، غير تلقائي الآن، مما يعني أنه يحسب الوحدات في **تمريرة أمامية واحدة**. أصبح هذا الإنجاز ممكنًا من خلال:

- استخدام **تضمين مستوى الأحرف**، مما يعني أن لكل حرف من النص المترجم المتنبأ به تضمينه الخاص، والذي يتم استخدامه بعد ذلك للتنبؤ برموز الوحدة.
- استخدام مُنبئ المدة الوسيطة، الذي يتنبأ بمدة الكلام على مستوى **الشخصية** في النص المترجم المتنبأ به.
- استخدام فك تشفير النص إلى وحدة جديد يمزج بين التعرّف على التعرّف والاهتمام الذاتي للتعامل مع السياق الأطول.

#### الاختلاف في مشفر الكلام

يختلف مشفر الكلام، الذي يتم استخدامه أثناء عملية توليد المرور الأول للتنبؤ بالنص المترجم، بشكل أساسي عن مشفر الكلام السابق من خلال هذه الآليات:

- استخدام قناع اهتمام مجزأ لمنع الاهتمام عبر المقاطع، وضمان أن يحضر كل موضع فقط إلى المواضع الموجودة داخل مقطعته وعدد ثابت من المقاطع السابقة.
- استخدام التضمينات الموضعية النسبية التي لا تأخذ في الاعتبار سوى المسافة بين عناصر التسلسل بدلاً من المواضع المطلقة. يرجى الرجوع إلى [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155) لمزيد من التفاصيل.
- استخدام التعرّف المتعمق السببي بدلاً من غير السببي.

### عملية التوليد

هكذا تعمل عملية التوليد:

- تتم معالجة النص أو الكلام المدخل من خلال مشفر محدد.
- يقوم فك التشفير بإنشاء رموز نصية باللغة المطلوبة.
- إذا كانت هناك حاجة إلى توليد الكلام، فإن نموذج seq2seq الثاني يقوم بتوليد رموز الوحدة بطريقة غير تلقائية.
- يتم بعد ذلك تمرير رموز الوحدة هذه عبر محول الترميز النهائي لإنتاج الكلام الفعلي.

تمت المساهمة بهذا النموذج من قبل [ylacombe](https://huggingface.co/ylacombe). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/seamless_communication).

## SeamlessM4Tv2Model

[[autodoc]] SeamlessM4Tv2Model

- generate

## SeamlessM4Tv2ForTextToSpeech

[[autodoc]] SeamlessM4Tv2ForTextToSpeech

- generate

## SeamlessM4Tv2ForSpeechToSpeech

[[autodoc]] SeamlessM4Tv2ForSpeechToSpeech

- generate

## SeamlessM4Tv2ForTextToText

[[autodoc]] transformers.SeamlessM4Tv2ForTextToText

- forward
- generate

## SeamlessM4Tv2ForSpeechToText

[[autodoc]] transformers.SeamlessM4Tv2ForSpeechToText

- forward
- generate

## SeamlessM4Tv2Config

[[autodoc]] SeamlessM4Tv2Config