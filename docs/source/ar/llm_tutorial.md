# التوليد باستخدام نماذج اللغات الكبيرة (LLMs)

[[open-in-colab]]

تعد LLMs، أو نماذج اللغة الكبيرة، المكون الرئيسي وراء توليد النصوص. وباختصار، تتكون من نماذج محول كبيرة مسبقة التدريب تم تدريبها للتنبؤ بالكلمة التالية (أو، بشكل أكثر دقة، الرمز اللغوي) بالنظر إلى نص معين. نظرًا لأنها تتنبأ برمز واحد في كل مرة، يجب عليك القيام بشيء أكثر تعقيدًا لتوليد جمل جديدة بخلاف مجرد استدعاء النموذج - يجب عليك إجراء التوليد التلقائي.

التوليد التلقائي هو إجراء وقت الاستدلال الذي يتضمن استدعاء النموذج بشكل متكرر باستخدام مخرجاته الخاصة، بالنظر إلى بعض المدخلات الأولية. في 🤗 Transformers، يتم التعامل مع هذا بواسطة دالة [`~generation.GenerationMixin.generate`]، والتي تتوفر لجميع النماذج ذات القدرات التوليدية.

سيوضح هذا البرنامج التعليمي كيفية:

* تتوليد نص باستخدام نموذج اللغات الكبيرة (LLM)
* تجنب الوقوع في الأخطاء الشائعة
* الخطوات التالية لمساعدتك في الاستفادة القصوى من LLM الخاص بك

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```

## توليد النص

يأخذ نموذج اللغة المدرب لـ [نمذجة اللغة السببية](tasks/language_modeling) يأخذ تسلسلًا من رموز نصية كمدخل ويعيد توزيع الاحتمالية للرمز التالي.

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
    <figcaption>"التنبؤ بالكلمة التالية لنموذج اللغة (LLM)"</figcaption>
</figure>

هناك جانب بالغ الأهمية في التوليد التلقائي باستخدام LLMs وهو كيفية اختيار الرمز التالي من توزيع الاحتمالية هذا. كل شيء مسموح به في هذه الخطوة طالما أنك تنتهي برمز للتكرار التالي. وهذا يعني أنه يمكن أن يكون بسيطًا مثل اختيار الرمز الأكثر احتمالًا من توزيع الاحتمالية أو معقدًا مثل تطبيق عشرات التحولات قبل أخذ العينات من التوزيع الناتج.

<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
    <figcaption>"التوليد التلقائي المتسلسل"</figcaption>
</figure>

تتكرر العملية الموضحة أعلاه بشكل تكراري حتى يتم الوصول إلى شرط التوقف. في الوضع المثالي، يحدد النموذج شرط التوقف، والذي يجب أن يتعلم عند إخراج رمز نهاية التسلسل (`EOS`). إذا لم يكن الأمر كذلك، يتوقف التوليد عند الوصول إلى طول أقصى محدد مسبقًا.

من الضروري إعداد خطوة اختيار الرمز وشرط التوقف بشكل صحيح لجعل نموذجك يتصرف كما تتوقع في مهمتك. ولهذا السبب لدينا [`~generation.GenerationConfig`] ملف مرتبط بكل نموذج، والذي يحتوي على معلمة توليدية افتراضية جيدة ويتم تحميله جنبًا إلى جنب مع نموذجك.

دعنا نتحدث عن الكود!


<Tip>

إذا كنت مهتمًا بالاستخدام الأساسي لـ LLM، فإن واجهة [`Pipeline`](pipeline_tutorial) عالية المستوى هي نقطة انطلاق رائعة. ومع ذلك، غالبًا ما تتطلب LLMs ميزات متقدمة مثل التكميم والتحكم الدقيق في خطوة اختيار الرمز، والتي يتم تنفيذها بشكل أفضل من خلال [`~generation.GenerationMixin.generate`]. التوليد التلقائي باستخدام LLMs  يستهلك الكثير من المواردد ويجب تنفيذه على وحدة معالجة الرسومات للحصول على أداء كافٍ.

</Tip>

أولاً، تحتاج إلى تحميل النموذج.

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

ستلاحظ وجود معاملين في الاستدعاء `from_pretrained`:

 - `device_map` يضمن انتقال النموذج إلى وحدة معالجة الرسومات (GPU) الخاصة بك
 - `load_in_4bit` يطبق [4-bit dynamic quantization](main_classes/quantization) لخفض متطلبات الموارد بشكل كبير

هناك طرق أخرى لتهيئة نموذج، ولكن هذا خط أساس جيد للبدء باستخدام LLM.

بعد ذلك، تحتاج إلى معالجة إدخال النص الخاص بك باستخدام [مُجزّئ اللغوي](tokenizer_summary).

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

يحتوي متغير `model_inputs` على النص المدخل بعد تقسيمه إلى وحدات لغوية (tokens)، بالإضافة إلى قناع الانتباه. في حين أن [`~generation.GenerationMixin.generate`] تبذل قصارى جهدها لاستنتاج قناع الانتباه عندما لا يتم تمريره، نوصي بتمريره كلما أمكن ذلك للحصول على نتائج مثالية.

بعد تقسيم المدخلات إلى وحدات لغوية، يمكنك استدعاء الدالة [`~generation.GenerationMixin.generate`] لإرجاع الوحدات اللغوية الناتجة. يجب بعد ذلك تحويل الوحدات المولدة إلى نص قبل طباعته.

```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, orange, purple, pink,'
```

أخيرًا، ليس عليك معالجة المتتاليات الواحدة تلو الأخرى! يمكنك معالجة مجموعة من المدخلات دفعة واحدة، والتي ستعمل على تحسين الإنتاجية بشكل كبير بتكلفة صغيرة في زمن الاستجابة واستهلاك الذاكر. كل ما عليك التأكد منه هو  تعبئة المدخلات بشكل صحيح (المزيد حول ذلك أدناه).

```py
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model_inputs = tokenizer(
...     ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['A list of colors: red, blue, green, yellow, orange, purple, pink,',
'Portugal is a country in southwestern Europe, on the Iber']
```

وهذا كل شيء! في بضع سطور من التعليمات البرمجية، يمكنك تسخير قوة LLM.

## الأخطاء الشائعة

هناك العديد من [استراتيجيات التوليد](generation_strategies)، وفي بعض الأحيان قد لا تكون القيم الافتراضية مناسبة لحالتك الاستخدام. إذا لم تكن الإخراج الخاصة بك متوافقة مع ما تتوقعه، فقد قمنا بإنشاء قائمة بأكثر الأخطاء الشائعة وكيفية تجنبها.

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

### الإخراج المولد قصير جدًا/طويل جدًا

إذا لم يتم تحديد العدد الأقصى للرموز في [`~generation.GenerationConfig`] الملف، `generate` يعيد ما يصل إلى 20 رمزًا بشكل افتراضي. نوصي بشدة بتعيين `max_new_tokens` يدويًا في مكالمة `generate` للتحكم في العدد الأقصى من الرموز الجديدة التي يمكن أن يعيدها. ضع في اعتبارك أن LLMs (بشكل أكثر دقة، [نماذج فك التشفير فقط](https://huggingface.co/learn/nlp-course/chapter1/6؟fw=pt)) تعيد أيضًا المدخلات الأصلية كجزء من الناتج.
```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### وضع التوليد الافتراضي

بشكل افتراضي، وما لم يتم تحديده في [`~generation.GenerationConfig`] الملف، `generate` يحدد الكلمة الأكثر احتمالًا فى كل خطوة من خطوات عملية التوليد (وهذا يُعرف بالتشفير الجشع). اعتمادًا على مهمتك، قد يكون هذا غير مرغوب فيه؛ تستفيد المهام الإبداعية مثل برامج الدردشة أو كتابة مقال ستفيد من أسلوب العينة العشوائية في اختيار الكلمات، تمن ناحية أخرى، فإن المهام التي تعتمد على مدخلات محددة  مثل تحويل الصوت إلى نص أو الترجم من فك التشفير الجشع. قم بتفعيل أسلوب العينات العشوائية باستخدام `do_sample=True`، ويمكنك معرفة المزيد حول هذا الموضوع في [تدوينة المدونة](https://huggingface.co/blog/how-to-generate).

```py
>>> # Set seed or reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(42)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.  Specifically, I am an indoor-only cat.  I'
```

### مشكلة حشو المدخلات فى الاتجاة الخطأ

LLMs هي [معماريات فك التشفير فقط](https://huggingface.co/learn/nlp-course/chapter1/6؟fw=pt)، مما يعني أنها تستمر في التكرار على موجه الإدخال الخاص بك. فإن جميع المدخلات يجب أن تكون بنفس الطول. لحل هذه المسألة، يتم إضافة رموز حشو إلى المدخلات الأقصر. نظرًا لأن LLMs  لا تولي اهتمامًا لرموز الحشو هذه، ذلك، يجب تحديد الجزء المهم من المدخل الذي يجب أن يركز عليه النموذج، وهذا يتم عن طريق ما يسمى بـ "قناع الانتباه". يجب أن يكون الحشو في بداية المدخل (الحشو من اليسار)، وليس في نهايته.

```py
>>> # The tokenizer initialized above has right-padding active by default: the 1st sequence,
>>> # which is shorter, has padding on the right side. Generation fails to capture the logic.
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 33333333333'

>>> # With left-padding, it works as expected!
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
>>> tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

### موجه غير صحيح

تتوقع بعض نماذج اللغات الكبيرة على صيغة محددة للمدخلات  للعمل بشكل صحيح. إذا لم يتم اتباع هذه الصيغة، فإن أداء النموذج يتأثر سلبًا: لكن هذا التدهور قد لا يكون واضحًا للعيان. تتوفر معلومات إضافية حول التوجيه، بما في ذلك النماذج والمهام التي تحتاج إلى توخي الحذر، في [الدليل](tasks/prompting). دعنا نرى مثالاً باستخدام LLM للدردشة، والذي يستخدم [قالب الدردشة](chat_templating):
```python
>>> tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
>>> model = AutoModelForCausalLM.from_pretrained(
...     "HuggingFaceH4/zephyr-7b-alpha", device_map="auto", load_in_4bit=True
... )
>>> set_seed(0)
>>> prompt = """How many helicopters can a human eat in one sitting? Reply as a thug."""
>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> input_length = model_inputs.input_ids.shape[1]
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=20)
>>> print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"I'm not a thug, but i can tell you that a human cannot eat"
>>> # Oh no, it did not follow our instruction to reply as a thug! Let's see what happens when we write
>>> # a better prompt and use the right template for this model (through `tokenizer.apply_chat_template`)

>>> set_seed(0)
>>> messages = [
...     {
...         "role": "system",
...         "content": "You are a friendly chatbot who always responds in the style of a thug",
...     },
...     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
... ]
>>> model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
>>> input_length = model_inputs.shape[1]
>>> generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=20)
>>> print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
'None, you thug. How bout you try to focus on more useful questions?'
>>> # As we can see, it followed a proper thug style 😎
```

## موارد إضافية

في حين أن عملية التوليد التلقائي بسيطة نسبيًا، فإن الاستفادة القصوى من LLM الخاص بك يمكن أن تكون مهمة صعبة لأن هناك العديد من الأجزاء المتحركة. للخطوات التالية لمساعدتك في الغوص بشكل أعمق في استخدام LLM وفهمه:

### استخدامات متقدمة للتوليد في نماذج اللغات الكبيرة

1. دليل حول كيفية [التحكم في طرق التوليد المختلفة](generation_strategies)، وكيفية إعداد ملف تكوين التوليد، وكيفية بث الناتج؛
2. [تسريع توليد النص](llm_optims)؛
3.[قوالب موجهات للدردشة LLMs](chat_
4. [دليل تصميم الموجه](tasks/prompting);
5. مرجع واجهة برمجة التطبيقات (API)  [`~generation.GenerationConfig`], [`~generation.GenerationMixin.generate`], و  [generate-related classes](internal/generation_utils). والعديد من الفئات الأخرى المرتبطة بعملية التوليد.!

### لوحات صدارة نماذج اللغات الكبيرة
1. لوحة صدارة نماذج اللغات الكبيرة المفتوحة المصدر (Open LLM Leaderboard): تركز على جودة النماذج مفتوحة المصدر [رابط لوحة الصدارة](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
2. لوحة صدارة أداء نماذج اللغات الكبيرة المفتوحة المصدر (Open LLM-Perf Leaderboard): تركز على إنتاجية نماذج اللغات الكبيرة [رابط لوحة الصدارة](https://huggingface.co/spaces/optimum/llm-perf-leaderboard).

### زمن الاستجابة والإنتاجية واستهلاك الذاكرة
1. دليل تحسين نماذج اللغات الكبيرة من حيث السرعة والذاكرة: دليل تحسين نماذج اللغات الكبيرة.
2. التكميم (Quantization): دليل حول تقنية التكميم التكميم مثل تقنيتي bitsandbytes و autogptq، والتي توضح كيفية تقليل متطلبات الذاكرة بشكل كبير.

### مكتبات مرتبطة
1. [`optimum`](https://github.com/huggingface/optimum), امتداد لمكتبة Transformers يعمل على تحسين الأداء لأجهزة معينة.
2. [`outlines`](https://github.com/outlines-dev/outlines), مكتبة للتحكم في توليد النصوص (على سبيل المثال، لتوليد ملفات JSON).
3. [`SynCode`](https://github.com/uiuc-focal-lab/syncode), مكتبة للتوليد الموجه بقواعد اللغة الخالية من السياق (على سبيل المثال، JSON، SQL، Python).
4. [`text-generation-inference`](https://github.com/huggingface/text-generation-inference), خادم جاهز للإنتاج لنماذج اللغات الكبيرة.
5. [`text-generation-webui`](https://github.com/oobabooga/text-generation-webui), واجهة مستخدم لتوليد النصوص.   
