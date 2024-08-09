# SeamlessM4T

## نظرة عامة

اقترح نموذج SeamlessM4T في [SeamlessM4T — Massively Multilingual & Multimodal Machine Translation](https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf) بواسطة فريق Seamless Communication من Meta AI.

هذا هو إصدار **الإصدار 1** من النموذج. للإصدار المحدث **الإصدار 2**، راجع [Seamless M4T v2 docs](https://huggingface.co/docs/transformers/main/model_doc/seamless_m4t_v2).

SeamlessM4T عبارة عن مجموعة من النماذج المصممة لتوفير ترجمة عالية الجودة، مما يسمح للأفراد من مجتمعات لغوية مختلفة بالتواصل بسلاسة من خلال الكلام والنص.

تمكّن SeamlessM4T من تنفيذ مهام متعددة دون الاعتماد على نماذج منفصلة:

- الترجمة من الكلام إلى الكلام (S2ST)
- الترجمة من الكلام إلى النص (S2TT)
- الترجمة من النص إلى الكلام (T2ST)
- الترجمة من نص إلى نص (T2TT)
- التعرف التلقائي على الكلام (ASR)

يمكن لـ [`SeamlessM4TModel`] تنفيذ جميع المهام المذكورة أعلاه، ولكن لكل مهمة أيضًا نموذجها الفرعي المخصص.

المقتطف من الورقة هو ما يلي:

*ما الذي يتطلبه الأمر لخلق سمكة بابل، وهي أداة يمكن أن تساعد الأفراد على ترجمة الكلام بين أي لغتين؟ في حين أن الاختراقات الأخيرة في النماذج القائمة على النص دفعت بتغطية الترجمة الآلية إلى ما بعد 200 لغة، إلا أن نماذج الترجمة الموحدة من الكلام إلى الكلام لم تحقق بعد قفزات مماثلة. على وجه التحديد، تعتمد أنظمة الترجمة التقليدية من الكلام إلى الكلام على الأنظمة المتتالية التي تؤدي الترجمة تدريجيًا، مما يجعل الأنظمة الموحدة عالية الأداء بعيدة المنال. لمعالجة هذه الفجوات، نقدم SeamlessM4T، وهو نموذج واحد يدعم الترجمة من الكلام إلى الكلام، والترجمة من الكلام إلى النص، والترجمة من النص إلى الكلام، والترجمة من نص إلى نص، والتعرف التلقائي على الكلام لما يصل إلى 100 لغة. لبناء هذا، استخدمنا مليون ساعة من بيانات الصوت المفتوحة لتعلم التمثيلات الصوتية ذاتية الإشراف باستخدام w2v-BERT 2.0. بعد ذلك، قمنا بإنشاء مجموعة بيانات متعددة الوسائط من ترجمات الكلام المحاذاة تلقائيًا. بعد التصفية والدمج مع البيانات التي تحمل علامات بشرية وبيانات ذات علامات زائفة، قمنا بتطوير أول نظام متعدد اللغات قادر على الترجمة من وإلى اللغة الإنجليزية لكل من الكلام والنص. في FLEURS، يحدد SeamlessM4T معيارًا جديدًا للترجمات إلى لغات متعددة مستهدفة، محققًا تحسنًا بنسبة 20% في BLEU مقارنة بـ SOTA السابق في الترجمة المباشرة من الكلام إلى النص. مقارنة بالنماذج المتتالية القوية، يحسن SeamlessM4T جودة الترجمة إلى اللغة الإنجليزية بمقدار 1.3 نقطة BLEU في الترجمة من الكلام إلى النص وبمقدار 2.6 نقطة ASR-BLEU في الترجمة من الكلام إلى الكلام. تم اختبار نظامنا للمتانة، ويؤدي أداءً أفضل ضد الضوضاء الخلفية وتغيرات المتحدث في مهام الترجمة من الكلام إلى النص مقارنة بنموذج SOTA الحالي. من الناحية الحرجة، قمنا بتقييم SeamlessM4T على التحيز بين الجنسين والسمية المضافة لتقييم سلامة الترجمة. أخيرًا، جميع المساهمات في هذا العمل مفتوحة المصدر ومتاحة على https://github.com/facebookresearch/seamless_communication*

## الاستخدام

أولاً، قم بتحميل المعالج ونقطة تفتيش للنموذج:

```python
>>> from transformers import AutoProcessor, SeamlessM4TModel

>>> processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
>>> model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
```

يمكنك استخدام هذا النموذج بسلاسة على النص أو الصوت، لإنشاء نص مترجم أو صوت مترجم.

فيما يلي كيفية استخدام المعالج لمعالجة النص والصوت:

```python
>>> # دعنا نحمل عينة صوتية من مجموعة بيانات خطاب عربي
>>> from datasets import load_dataset
>>> dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
>>> audio_sample = next(iter(dataset))["audio"]

>>> # الآن، قم بمعالجته
>>> audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt")

>>> # الآن، قم بمعالجة بعض النصوص الإنجليزية أيضًا
>>> text_inputs = processor(text="Hello, my dog is cute", src_lang="eng", return_tensors="pt")
```

### الكلام

يمكن لـ [`SeamlessM4TModel`] *بسلاسة* إنشاء نص أو كلام مع عدد قليل من التغييرات أو بدونها. دعنا نستهدف الترجمة الصوتية الروسية:

```python
>>> audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
>>> audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
```

بنفس الرمز تقريبًا، قمت بترجمة نص إنجليزي وكلام عربي إلى عينات كلام روسية.

### نص

وبالمثل، يمكنك إنشاء نص مترجم من ملفات الصوت أو من النص باستخدام نفس النموذج. كل ما عليك فعله هو تمرير `generate_speech=False` إلى [`SeamlessM4TModel.generate`].

هذه المرة، دعنا نترجم إلى الفرنسية.

```python
>>> # من الصوت
>>> output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

>>> # من النص
>>> output_tokens = model.generate(**text_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_partum_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
```

### نصائح

#### 1. استخدم النماذج المخصصة

[`SeamlessM4TModel`] هو نموذج المستوى الأعلى في المحولات لإنشاء الكلام والنص، ولكن يمكنك أيضًا استخدام النماذج المخصصة التي تؤدي المهمة دون مكونات إضافية، مما يقلل من بصمة الذاكرة.

على سبيل المثال، يمكنك استبدال جزء الرمز الخاص بإنشاء الصوت باستخدام النموذج المخصص لمهمة S2ST، وبقية الرمز هو نفسه بالضبط:

```python
>>> from transformers import SeamlessM4TForSpeechToSpeech
>>> model = SeamlessM4TForSpeechToSpeech.from_pretrained("facebook/hf-seamless-m4t-medium")
```

أو يمكنك استبدال جزء الرمز الخاص بإنشاء النص باستخدام النموذج المخصص لمهمة T2TT، وكل ما عليك فعله هو إزالة `generate_speech=False`.

```python
>>> from transformers import SeamlessM4TForTextToText
>>> model = SeamlessM4TForTextToText.from_pretrained("facebook/hf-seamless-m4t-medium")
```

لا تتردد في تجربة [`SeamlessM4TForSpeechToText`] و [`SeamlessM4TForTextToSpeech`] أيضًا.

#### 2. تغيير هوية المتحدث

لديك إمكانية تغيير المتحدث المستخدم للتخليق الصوتي باستخدام حجة `spkr_id`. تعمل بعض قيم `spkr_id` بشكل أفضل من غيرها لبعض اللغات!

#### 3. تغيير استراتيجية التوليد

يمكنك استخدام استراتيجيات [توليد](./generation_strategies) مختلفة للكلام وتوليد النص، على سبيل المثال `.generate(input_ids=input_ids، text_num_beams=4، speech_do_sample=True)` والذي سيؤدي فك تشفير شعاع على النموذج النصي، وعينة متعددة الحدود على النموذج الصوتي.

#### 4. قم بتوليد الكلام والنص في نفس الوقت

استخدم `return_intermediate_token_ids=True` مع [`SeamlessM4TModel`] لإرجاع كل من الكلام والنص!

## بنية النموذج

تتميز SeamlessM4T ببنية مرنة تتعامل بسلاسة مع التوليد التسلسلي للنص والكلام. يتضمن هذا الإعداد نموذجين تسلسليين (seq2seq). يترجم النموذج الأول الطريقة المدخلة إلى نص مترجم، بينما يقوم النموذج الثاني بتوليد رموز الكلام، والمعروفة باسم "رموز الوحدة"، من النص المترجم.

لدى كل طريقة مشفر خاص بها ببنية فريدة. بالإضافة إلى ذلك، بالنسبة لإخراج الكلام، يوجد جهاز فك ترميز مستوحى من بنية [HiFi-GAN](https://arxiv.org/abs/2010.05646) أعلى نموذج seq2seq الثاني.

فيما يلي كيفية عمل عملية التوليد:

- تتم معالجة النص أو الكلام المدخل من خلال مشفر محدد.
- يقوم فك التشفير بإنشاء رموز نصية باللغة المطلوبة.
- إذا كانت هناك حاجة إلى توليد الكلام، يقوم نموذج seq2seq الثاني، باتباع بنية مشفر-فك تشفير قياسية، بتوليد رموز الوحدة.
- يتم بعد ذلك تمرير رموز الوحدة هذه عبر فك التشفير النهائي لإنتاج الكلام الفعلي.

تمت المساهمة بهذا النموذج بواسطة [ylacombe](https://huggingface.co/ylacombe). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/seamless_communication).

## SeamlessM4TModel

[[autodoc]] SeamlessM4TModel

- generate

## SeamlessM4TForTextToSpeech

[[autodoc]] SeamlessM4TForTextToSpeech

- generate

## SeamlessM4TForSpeechToSpeech

[[autodoc]] SeamlessM4TForSpeechToSpeech

- generate

## SeamlessM4TForTextToText

[[autodoc]] transformers.SeamlessM4TForTextToText

- forward
- generate

## SeamlessM4TForSpeechToText

[[autodoc]] transformers.SeamlessM4TForSpeechToText

- forward
- generate

## SeamlessM4TConfig

[[autodoc]] SeamlessM4TConfig

## SeamlessM4TTokenizer

[[autodoc]] SeamlessM4TTokenizer

- __call__
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## SeamlessM4TTokenizerFast

[[autodoc]] SeamlessM4TTokenizerFast

- __call__

## SeamlessM4TFeatureExtractor

[[autodoc]] SeamlessM4TFeatureExtractor

- __call__

## SeamlessM4TProcessor

[[autodoc]] SeamlessM4TProcessor

- __call__

## SeamlessM4TCodeHifiGan

[[autodoc]] SeamlessM4TCodeHifiGan

## SeamlessM4THifiGan

[[autodoc]] SeamlessM4THifiGan

## SeamlessM4TTextToUnitModel

[[autodoc]] SeamlessM4TTextToUnitModel

## SeamlessM4TTextToUnitForConditionalGeneration

[[autodoc]] SeamlessM4TTextToUnitForConditionalGeneration