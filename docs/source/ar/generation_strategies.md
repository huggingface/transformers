# استراتيجيات توليد النص

يعد توليد النص أمرًا أساسيًا للعديد من مهام معالجة اللغة الطبيعية، مثل توليد النص المفتوح، والتلخيص، والترجمة، وأكثر من ذلك. كما يلعب دورًا في مجموعة متنوعة من تطبيقات الطرائق المختلطة التي يكون النص فيها كإخراج مثل تحويل الكلام إلى نص، والتحويل من رؤية إلى نص. بعض النماذج التي يمكنها توليد النص تشمل GPT2، وXLNet، وOpenAI GPT، وCTRL، وTransformerXL، وXLM، وBart، وT5، وGIT، وWhisper.

تفقد بعض الأمثلة التي تستخدم طريقة [~generation.GenerationMixin.generate] لإنتاج مخرجات نصية لمهام مختلفة:

- [تلخيص النص](./tasks/summarization#inference)
- [وضع عنوان للصورة](./model_doc/git#transformers.GitForCausalLM.forward.example)
- [نسخ الصوت](./model_doc/whisper#transformers.WhisperForConditionalGeneration.forward.example)

لاحظ أن المدخلات لطريقة التوليد تعتمد على طريقة النموذج. يتم إرجاعها بواسطة فئة المعالجة المسبقة للنموذج، مثل AutoTokenizer أو AutoProcessor. إذا أنشأت معالجة مسبقة للنموذج أكثر من نوع واحد من الإدخال، فقم بتمرير جميع الإدخالات إلى generate(). يمكنك معرفة المزيد حول معالجة مسبقة فردية للنموذج في وثائق النموذج المقابلة.

تُعرف عملية اختيار الرموز المميزة للإخراج لتوليد النص باسم فك التشفير، ويمكنك تخصيص استراتيجية فك التشفير التي ستستخدمها طريقة `generate()`. لا يؤدي تعديل استراتيجية فك التشفير إلى تغيير قيم أي معلمات قابلة للتدريب. ومع ذلك، يمكن أن يكون له تأثير ملحوظ على جودة الإخراج المولد. يمكن أن يساعد في تقليل التكرار في النص وجعله أكثر تماسكًا.

يصف هذا الدليل ما يلي:

- تكوين التوليد الافتراضي
- استراتيجيات فك التشفير الشائعة وبارامتراتها الرئيسية
- حفظ ومشاركة تكوينات التوليد المخصصة مع نموذج التدريب الدقيق الخاص بك على 🤗 Hub

## تكوين التوليد الافتراضي للنص

تتم تعريف استراتيجية فك التشفير لنموذج في تكوين التوليد الخاص به. عند استخدام النماذج المُدربة مسبقًا للاستنتاج داخل [`pipeline`]، تقوم النماذج باستدعاء طريقة `PreTrainedModel.generate()` التي تطبق تكوين التوليد الافتراضي تحت الغطاء. يتم أيضًا استخدام التكوين الافتراضي عندما لا يتم حفظ أي تكوين مخصص مع النموذج.

عندما تقوم بتحميل نموذج بشكل صريح، يمكنك فحص تكوين التوليد الذي يأتي معه من خلال `model.generation_config`:

```python
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> model.generation_config
GenerationConfig {
  "bos_token_id": 50256,
  "eos_token_id": 50256
}
<BLANKLINE>
```

يكشف طباعة `model.generation_config` فقط عن القيم التي تختلف عن تكوين التوليد الافتراضي، ولا يدرج أيًا من القيم الافتراضية.

يقتصر تكوين التوليد الافتراضي على حجم الإخراج المدمج مع موجه الإدخال إلى حد أقصى 20 رمزًا لتجنب مواجهة قيود الموارد. استراتيجية فك التشفير الافتراضية هي البحث الجشع، والتي تعد أبسط استراتيجية فك تشفير تختار رمزًا مميزًا به أعلى احتمال كرمز مميز التالي. بالنسبة للعديد من المهام وأحجام الإخراج الصغيرة، يعمل هذا بشكل جيد. ومع ذلك، عندما يتم استخدامه لتوليد مخرجات أطول، يمكن أن يبدأ البحث الجشع في إنتاج نتائج متكررة للغاية.

## تخصيص توليد النص

يمكنك تجاوز أي `generation_config` عن طريق تمرير البارامترات وقيمها مباشرةً إلى طريقة [`generate`]:

```python
>>> my_model.generate(**inputs, num_beams=4, do_sample=True)  # doctest: +SKIP
```

حتى إذا كانت استراتيجية فك التشفير الافتراضية تعمل بشكل أساسي لمهمتك، فلا يزال بإمكانك ضبط بعض الأشياء. بعض البارامترات التي يتم ضبطها بشكل شائع تشمل:

- `max_new_tokens`: العدد الأقصى من الرموز المميزة التي سيتم توليدها. وبعبارة أخرى، حجم تسلسل الإخراج، وليس بما في ذلك الرموز المميزة في الموجه. كبديل لاستخدام طول الإخراج كمعيار إيقاف، يمكنك اختيار إيقاف التوليد في أي وقت يتجاوز فيه التوليد الكامل مقدارًا معينًا من الوقت. لمعرفة المزيد، تحقق من [`StoppingCriteria`].
- `num_beams`: من خلال تحديد عدد الحزم أكبر من 1، فأنت تقوم بشكل فعال بالتبديل من البحث الجشع إلى البحث الشعاعي. تقيّم هذه الاستراتيجية العديد من الفرضيات في كل خطوة زمنية وتختار في النهاية الفرضية التي لها أعلى احتمال إجمالي للتسلسل بأكمله. تتمثل ميزة هذه الاستراتيجية في تحديد تسلسلات عالية الاحتمال تبدأ برموز مميزة أولية منخفضة الاحتمال والتي ستتجاهلها البحث الجشع. قم بتصور كيفية عمله [هنا](https://huggingface.co/spaces/m-ric/beam_search_visualizer).
- `do_sample`: إذا تم تعيينه على `True`، فإن هذا البارامتر يمكّن استراتيجيات فك التشفير مثل أخذ العينات متعددة الحدود، والبحث الشعاعي متعدد الحدود، وأخذ العينات الأعلى K، وأخذ العينات الأعلى p. تقوم جميع هذه الاستراتيجيات بتحديد الرمز المميز التالي من توزيع الاحتمالية عبر المفردات بأكملها مع تعديلات محددة للاستراتيجية.
- `num_return_sequences`: عدد تسلسلات المرشحين التي سيتم إرجاعها لكل إدخال. هذا الخيار متاح فقط لاستراتيجيات فك التشفير التي تدعم عدة تسلسلات مرشحة، على سبيل المثال، اختلافات البحث الشعاعي وأخذ العينات. تعيد استراتيجيات فك التشفير مثل البحث الجشع والبحث التبايني تسلسل إخراج واحد.

## حفظ استراتيجية فك تشفير مخصصة مع نموذج

إذا كنت ترغب في مشاركة نموذج التدريب الدقيق الخاص بك بتكوين توليد محدد، فيمكنك:

- إنشاء مثيل لفئة [`GenerationConfig`]
- تحديد بارامترات استراتيجية فك التشفير
- حفظ تكوين التوليد الخاص بك باستخدام [`GenerationConfig.save_pretrained`]، والتأكد من ترك حجته `config_file_name` فارغة
- قم بتعيين `push_to_hub` إلى `True` لتحميل تكوينك إلى مستودع النموذج

```python
>>> from transformers import AutoModelForCausalLM, GenerationConfig

>>> model = AutoModelForCausalLM.from_pretrained("my_account/my_model")  # doctest: +SKIP
>>> generation_config = GenerationConfig(
...     max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
... )
>>> generation_config.save_pretrained("my_account/my_model", push_to_hub=True)  # doctest: +SKIP
```

يمكنك أيضًا تخزين العديد من تكوينات التوليد في دليل واحد، باستخدام حجة `config_file_name` في [`GenerationConfig.save_pretrained`]. يمكنك لاحقًا استدعاء مثيل لها باستخدام [`GenerationConfig.from_pretrained`]. هذا مفيد إذا كنت تريد تخزين العديد من تكوينات التوليد لنموذج واحد (على سبيل المثال، واحد لتوليد نص إبداعي مع أخذ العينات، وواحد للتلخيص باستخدام البحث الشعاعي). يجب أن يكون لديك الأذونات الصحيحة على Hub لإضافة ملفات تكوين إلى نموذج.

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

>>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

>>> translation_generation_config = GenerationConfig(
...     num_beams=4,
...     early_stopping=True,
...     decoder_start_token_id=0,
...     eos_token_id=model.config.eos_token_id,
...     pad_token=model.config.pad_token_id,
... )

>>> # Tip: add `push_to_hub=True` to push to the Hub
>>> translation_generation_config.save_pretrained("/tmp", "translation_generation_config.json")
>>> # Tip: add `push_to_hub=True` to push to the Hub
>>> translation_generation_config.save_pretrained("/tmp", "translation_generation_config.json")

>>> # You could then use the named generation config file to parameterize generation
>>> generation_config = GenerationConfig.from_pretrained("/tmp", "translation_generation_config.json")
>>> inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
>>> outputs = model.generate(**inputs, generation_config=generation_config)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Les fichiers de configuration sont faciles à utiliser!']
```

## البث

تدعم طريقة `generate()` البث، من خلال إدخالها `streamer`. يتوافق إدخال `streamer` مع أي مثيل من فئة بها الطرق التالية: `put()` و`end()`. داخليًا، يتم استخدام `put()` لدفع الرموز المميزة الجديدة و`end()` للإشارة إلى نهاية توليد النص.

<Tip warning={true}>

لا يزال API لفئات البث قيد التطوير وقد يتغير في المستقبل.

</Tip>

من الناحية العملية، يمكنك إنشاء فئة بث مخصصة لجميع أنواع الأغراض! لدينا أيضًا فئات بث أساسية جاهزة للاستخدام. على سبيل المثال، يمكنك استخدام فئة [`TextStreamer`] لبث إخراج `generate()` إلى شاشتك، كلمة واحدة في كل مرة:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

## تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم

تدعم طريقة `generate()` تخزين مفاتيح وقيم المفاتيح والقيم المؤقتة لتعزيز الكفاءة وتجنب إعادة الحسابات. ومع ذلك، يمكن أن تشغل ذاكرة التخزين المؤقت للمفاتيح والقيم جزءًا كبيرًا من الذاكرة، مما يصبح عنق زجاجة لتوليد السياق الطويل، خاصة بالنسبة للنماذج اللغوية كبيرة الحجم.

يمكن أن يؤدي تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم عند استخدام `generate()` إلى تقليل متطلبات الذاكرة بشكل كبير على حساب السرعة.

يستلهم تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم في `transformers` إلى حد كبير من الورقة [KIVI: Quantization Asymmetric 2bit Quantization for KV Cache] (https://arxiv.org/abs/2402.02750) ويدعم حاليًا `quanto` و`HQQ` كخلفيات. لمزيد من المعلومات حول طريقة العمل الداخلية، راجع الورقة.

لتمكين تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم، يجب الإشارة إلى `cache_implementation="quantized"` في `generation_config`. يجب تمرير الحجج المتعلقة بالتكميم إلى `generation_config` إما كـ `dict` أو كمثيل لفئة [`QuantizedCacheConfig`]. يجب الإشارة إلى خلفية التكميم التي سيتم استخدامها في [`QuantizedCacheConfig`]، والافتراضي هو `quanto`.

<Tip warning={true}>

يمكن أن يكون تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم ضارًا إذا كان طول السياق قصيرًا وهناك ذاكرة وصول عشوائي GPU كافية متوفرة لتشغيلها بدون تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم.

</Tip>

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"nbits": 4, "backend": "quanto"})
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. I like to listen to it when I'm feeling
```

## نقل ذاكرة التخزين المؤقت للمفاتيح والقيم خارج الذاكرة

على غرار تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم، تهدف هذه الاستراتيجية إلى تقليل استخدام ذاكرة وصول عشوائي GPU.
فهي تقوم بذلك عن طريق نقل ذاكرة التخزين المؤقت للمفاتيح والقيم لمعظم الطبقات إلى وحدة المعالجة المركزية.
مع تقدم طريقة `forward()` للنموذج عبر الطبقات، تحافظ هذه الاستراتيجية على ذاكرة التخزين المؤقت للمفاتيح والقيم للطبقة الحالية على GPU.
في الوقت نفسه، يقوم باسترداد ذاكرة التخزين المؤقت للمفاتيح والقيم للطبقة التالية بشكل غير متزامن وإرسال ذاكرة التخزين المؤقت للمفاتيح والقيم للطبقة السابقة مرة أخرى إلى وحدة المعالجة المركزية.
على عكس تكميم ذاكرة التخزين المؤقت للمفاتيح والقيم، تنتج هذه الاستراتيجية دائمًا نفس النتيجة مثل تنفيذ ذاكرة التخزين المؤقت للمفاتيح والقيم الافتراضية.
لذلك، يمكن استخدامه كبديل أو كخطة احتياطية له.

اعتمادًا على نموذجك وخصائص مهمة التوليد الخاصة بك (حجم السياق، وعدد الرموز المميزة المولدة، وعدد الحزم، وما إلى ذلك)
قد تلاحظ انخفاضًا طفيفًا في إنتاجية التوليد مقارنة بتنفيذ ذاكرة التخزين المؤقت للمفاتيح والقيم الافتراضية.

لتمكين نقل ذاكرة التخزين المؤقت للمفاتيح والقيم خارج الذاكرة، قم بتمرير `cache_implementation="offloaded"` في `generation_config`.

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> ckpt = "microsoft/Phi-3-mini-4k-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(ckpt)
>>> model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("Fun fact: The shortest", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=23, cache_implementation="offloaded")
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=23)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.
```

<Tip warning={true}>

يت
### البحث التبايني

اقترحت ورقة عام 2022 [إطار عمل تبايني لتوليد النصوص العصبية](https://arxiv.org/abs/2202.06417) استراتيجية فك تشفير البحث التبايني.
وهو يظهر نتائج متفوقة لتوليد مخرجات طويلة متماسكة وغير مكررة. لمعرفة كيفية عمل البحث التبايني، تحقق من [هذه التدوينة](https://huggingface.co/blog/introducing-csearch).

هناك معياران رئيسيان يمكنان من التحكم في سلوك البحث التبايني وهما `penalty_alpha` و`top_k`:

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> checkpoint = "openai-community/gpt2-large"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> prompt = "Hugging Face Company is"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Hugging Face Company is a family owned and operated business. We pride ourselves on being the best
in the business and our customer service is second to none.\n\nIf you have any questions about our
products or services, feel free to contact us at any time. We look forward to hearing from you!']
```

### المعاينة متعددة الحدود

على عكس البحث الشره الذي يختار دائمًا رمزًا له أعلى احتمال كونه الرمز التالي، فإن المعاينة متعددة الحدود (يطلق عليها أيضًا المعاينة السلفية) تختار الرمز التالي بشكل عشوائي بناءً على توزيع الاحتمالية عبر المفردات بالكامل التي يمنحها النموذج. كل رمز له احتمال غير صفري لديه فرصة أن يتم اختياره، مما يقلل من

خطر التكرار.

لتمكين المعاينة متعددة الحدود، قم بتعيين `do_sample=True` و`num_beams=1`.

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
>>> set_seed(0) # من أجل إمكانية إعادة الإنتاج

>>> checkpoint = "openai-community/gpt2-large"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> prompt = "Today was an amazing day because"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
["Today was an amazing day because we received these wonderful items by the way of a gift shop. The box arrived on a Thursday and I opened it on Monday afternoon to receive the gifts. Both bags featured pieces from all the previous years!\n\nThe box had lots of surprises in it, including some sweet little mini chocolate chips! I don't think I'd eat all of these. This was definitely one of the most expensive presents I have ever got, I actually got most of them for free!\n\nThe first package came"]
```

### فك تشفير البحث الشعاعي

على عكس البحث الشره، يحتفظ فك تشفير البحث الشعاعي بعدة فرضيات في كل خطوة زمنية ويختار في النهاية
الفرضية التي لها أعلى احتمال إجمالي للتسلسل بأكمله. تتمثل ميزة ذلك في تحديد تسلسلات عالية الاحتمال
التي تبدأ برموز أولية ذات احتمالية أقل والتي ستتجاهلها عملية البحث الشره.

<a href="https://huggingface.co/spaces/m-ric/beam_search_visualizer" class="flex flex-col justify-center">
    <img style="max-width: 90%; margin: auto;" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beam_search.png"/>
</a>

يمكنك تصور كيفية عمل فك تشفير البحث الشعاعي في [هذا العرض التوضيحي التفاعلي](https://huggingface.co/spaces/m-ric/beam_search_visualizer): اكتب جملتك المدخلة، ولعب مع المعلمات لمشاهدة كيفية تغيير حزم فك التشفير.

لتمكين استراتيجية فك التشفير هذه، حدد `num_beams` (أي عدد الفرضيات التي يجب تتبعها) أكبر من 1.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "It is astonishing how one can"
>>> checkpoint = "openai-community/gpt2-medium"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['It is astonishing how one can have such a profound impact on the lives of so many people in such a short period of
time."\n\nHe added: "I am very proud of the work I have been able to do in the last few years.\n\n"I have']
```

### معاينة شعاع متعددة الحدود

كما يوحي الاسم، تجمع استراتيجية فك التشفير هذه بين البحث الشعاعي والمعاينة متعددة الحدود. يجب عليك تحديد
`num_beams` أكبر من 1، وتعيين `do_sample=True` لاستخدام استراتيجية فك التشفير هذه.

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
>>> set_seed(0) # من أجل إمكانية إعادة الإنتاج

>>> prompt = "translate English to German: The house is wonderful."
>>> checkpoint = "google-t5/t5-small"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, do_sample=True)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Das Haus ist wunderbar.'
```

### فك تشفير البحث الشعاعي المتنوع

استراتيجية فك تشفير البحث الشعاعي المتنوع هي امتداد لاستراتيجية البحث الشعاعي التي تتيح توليد مجموعة أكثر تنوعًا
من تسلسلات الشعاع للاختيار من بينها. لمعرفة كيفية عمله، راجع [بحث شعاعي متنوع: فك تشفير حلول متنوعة من نماذج التسلسل العصبي](https://arxiv.org/pdf/1610.02424.pdf).

لدى هذا النهج ثلاثة معلمات رئيسية: `num_beams`، `num_beam_groups`، و`diversity_penalty`.
تضمن عقوبة التنوع تميز الإخراج عبر المجموعات، ويتم استخدام البحث الشعاعي داخل كل مجموعة.


```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> checkpoint = "google/pegasus-xsum"
>>> prompt = (
...     "The Permaculture Design Principles are a set of universal design principles "
...     "that can be applied to any location, climate and culture, and they allow us to design "
...     "the most efficient and sustainable human habitation and food production systems. "
...     "Permaculture is a design system that encompasses a wide variety of disciplines, such "
...     "as ecology, landscape design, environmental science and energy conservation, and the "
...     "Permaculture design principles are drawn from these various disciplines. Each individual "
...     "design principle itself embodies a complete conceptual framework based on sound "
...     "scientific principles. When we bring all these separate  principles together, we can "
...     "create a design system that both looks at whole systems, the parts that these systems "
...     "consist of, and how those parts interact with each other to create a complex, dynamic, "
...     "living system. Each design principle serves as a tool that allows us to integrate all "
...     "the separate parts of a design, referred to as elements, into a functional, synergistic, "
...     "whole system, where the elements harmoniously interact and work together in the most "
...     "efficient way possible."
... )

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, num_beam_groups=5, max_new_tokens=30, diversity_penalty=1.0)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'The Design Principles are a set of universal design principles that can be applied to any location, climate and
culture, and they allow us to design the'
```
يوضح هذا الدليل المعلمات الرئيسية التي تمكن استراتيجيات فك التشفير المختلفة. هناك معلمات أكثر تقدمًا لـ
طريقة [`generate`]، والتي تمنحك مزيدًا من التحكم في سلوك طريقة [`generate`].

للاطلاع على القائمة الكاملة للمعلمات المتاحة، راجع [توثيق API](./main_classes/text_generation.md).

### فك التشفير التخميني

فك التشفير التخميني (المعروف أيضًا باسم فك التشفير بمساعدة) هو تعديل لاستراتيجيات فك التشفير المذكورة أعلاه، والذي يستخدم
نموذج مساعد (يفضل أن يكون أصغر بكثير) بنفس المعالج اللغوي، لتوليد بعض الرموز المرشحة. ثم يقوم النموذج الرئيسي
بتحقق من الرموز المرشحة في تمرير توجيهي واحد، والذي يسرع عملية فك التشفير. إذا
`do_sample=True`، يتم استخدام التحقق من الرمز مع إعادة المعاينة المقدمة في
[ورقة فك التشفير التخميني](https://arxiv.org/pdf/2211.17192.pdf).

حاليًا، يتم دعم البحث الشره والمعاينة فقط مع فك التشفير بمساعدة، ولا يدعم فك التشفير بمساعدة الإدخالات المجمعة.
لمعرفة المزيد حول فك التشفير بمساعدة، تحقق من [هذه التدوينة](https://huggingface.co/blog/assisted-generation).

لتمكين فك التشفير بمساعدة، قم بتعيين وسيط `assistant_model` بنموذج.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```

عند استخدام فك التشفير بمساعدة مع طرق المعاينة، يمكنك استخدام وسيط `temperature` للتحكم في العشوائية،
تمامًا كما هو الحال في المعاينة متعددة الحدود. ومع ذلك، في فك التشفير بمساعدة، قد يساعد تقليل درجة الحرارة في تحسين الكمون.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> set_seed(42) # من أجل إمكانية إعادة الإنتاج

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob, a couple of friends of mine, who are both in the same office as']
```

بدلاً من ذلك، يمكنك أيضًا تعيين `prompt_lookup_num_tokens` لتشغيل فك التشفير بمساعدة n-gram، بدلاً من
فك التشفير بمساعدة النماذج. يمكنك قراءة المزيد عنه [هنا](https://twitter.com/joao_gante/status/1747322413006643259).
### فك تشفير DoLa

**D** فك التشفير عن طريق تباين **La** فك تشفير الطبقات (DoLa) هو استراتيجية فك تشفير تبايني لتحسين الواقعية والحد من
الهلوسة في LLMs، كما هو موضح في هذه الورقة ICLR 2024 [DoLa: فك تشفير الطبقات التبايني يحسن الواقعية في نماذج اللغة الكبيرة](https://arxiv.org/abs/2309.03883).

يتم تحقيق DoLa من خلال تضخيم الاختلافات في logits التي تم الحصول عليها من الطبقات النهائية
مقابل الطبقات السابقة، وبالتالي تضخيم المعرفة الواقعية الموضعية في جزء معين من طبقات المحول.
يتم تحقيق DoLa من خلال تضخيم الاختلافات في logits التي تم الحصول عليها من الطبقات النهائية
مقابل الطبقات السابقة، وبالتالي تضخيم المعرفة الواقعية الموضعية في جزء معين من طبقات المحول.

اتبع الخطوتين التاليتين لتنشيط فك تشفير DoLa عند استدعاء وظيفة `model.generate`:

1. قم بتعيين وسيط `dola_layers`، والذي يمكن أن يكون إما سلسلة أو قائمة من الأعداد الصحيحة.
    - إذا تم تعيينه على سلسلة، فيمكن أن يكون أحد `low`، `high`.
    - إذا تم تعيينه على قائمة من الأعداد الصحيحة، فيجب أن يكون قائمة بمؤشرات الطبقات بين 0 والعدد الإجمالي للطبقات في النموذج. طبقة 0 هي طبقة تضمين الكلمات، والطبقة 1 هي أول طبقة محول، وهكذا.
2. يُقترح تعيين `repetition_penalty = 1.2` لتقليل التكرار في فك تشفير DoLa.

راجع الأمثلة التالية لفك تشفير DoLa باستخدام نموذج LLaMA-7B المكون من 32 طبقة.

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
>>> model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16)
>>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
>>> model.to(device)
>>> set_seed(42)

>>> text = "On what date was the Declaration of Independence officially signed?"
>>> inputs = tokenizer(text, return_tensors="pt").to(device)

# Vanilla greddy decoding
>>> vanilla_output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
>>> tokenizer.batch_decode(vanilla_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
['\nThe Declaration of Independence was signed on July 4, 1776.\nWhat was the date of the signing of the Declaration of Independence?\nThe Declaration of Independence was signed on July 4,']

# DoLa decoding with contrasting higher part of layers (layers 16,18,...,30)
>>> dola_high_output = model.generate(**inputs, do_sample=False, max_new_tokens=50, dola_layers='high')
>>> tokenizer.batch_decode(dola_high_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
['\nJuly 4, 1776, when the Continental Congress voted to separate from Great Britain. The 56 delegates to the Continental Congress signed the Declaration on August 2, 1776.']

# DoLa decoding with contrasting specific layers (layers 28 and 30)
>>> dola_custom_output = model.generate(**inputs, do_sample=False, max_new_tokens=50, dola_layers=[28,30], repetition_penalty=1.2)
>>> tokenizer.batch_decode(dola_custom_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
['\nIt was officially signed on 2 August 1776, when 56 members of the Second Continental Congress, representing the original 13 American colonies, voted unanimously for the resolution for independence. The 2']
```

#### فهم معاملات 'dola_layers'

يمثل 'dola_layers' طبقات المرشح في الاختيار المبكر للطبقة، كما هو موضح في ورقة DoLa. ستتم مقارنة الطبقة المبكرة المحددة بالطبقة النهائية.

يؤدي تعيين 'dola_layers' إلى 'low' أو 'high' إلى تحديد الجزء السفلي أو العلوي من الطبقات للمقارنة، على التوالي.

- بالنسبة لنماذج 'N-layer' مع 'N <= 40' layer، يتم استخدام الطبقات من 'range(0، N // 2، 2)' و'range(N // 2، N، 2)' لـ 'low' و 'high' layers، على التوالي.

- بالنسبة للنماذج التي تحتوي على 'N > 40' layer، يتم استخدام الطبقات من 'range(0، 20، 2)' و'range(N - 20، N، 2)' لـ 'low' و 'high' layers، على التوالي.

- إذا كان للنموذج تعليقات توضيحية مرتبطة بالكلمات، فإننا نتخطى طبقة التعليقات التوضيحية للكلمات (الطبقة 0) ونبدأ من الطبقة الثانية، نظرًا لأن الخروج المبكر من التعليقات التوضيحية للكلمات سيصبح دالة الهوية.

- قم بتعيين 'dola_layers' إلى قائمة من الأعداد الصحيحة لفهرسة الطبقات لمقارنة الطبقات المحددة يدويًا. على سبيل المثال، يؤدي تعيين 'dola_layers=[28،30]' إلى مقارنة الطبقة النهائية (الطبقة 32) بالطبقات 28 و30.

اقترحت الورقة أن مقارنة الطبقات 'العالية' لتحسين مهام الإجابات القصيرة مثل TruthfulQA، ومقارنة الطبقات 'المنخفضة' لتحسين جميع مهام الاستدلال بالإجابات الطويلة الأخرى، مثل GSM8K وStrategyQA وFACTOR وVicunaQA. لا يوصى بتطبيق DoLa على النماذج الأصغر مثل GPT-2، كما هو موضح في الملحق N من الورقة.