# المهام المتعلقة بالصور باستخدام IDEFICS

[[open-in-colab]]

في حين يمكن معالجة المهام الفردية عن طريق ضبط نماذج متخصصة، هناك نهج بديل ظهر مؤخرًا واكتسب شعبية وهو استخدام النماذج الكبيرة لمجموعة متنوعة من المهام دون ضبط دقيق. على سبيل المثال، يمكن لنماذج اللغة الكبيرة التعامل مع مهام NLP مثل الملخص والترجمة والتصنيف وأكثر من ذلك. لم يعد هذا النهج يقتصر على طريقة واحدة، مثل النص، وفي هذا الدليل، سنوضح كيف يمكنك حل مهام الصور والنصوص باستخدام نموذج متعدد الوسائط كبير يسمى IDEFICS.

[IDEFICS](../model_doc/idefics) هو نموذج مفتوح الوصول للرؤية واللغة يعتمد على [Flamingo](https://huggingface.co/papers/2204.14198)، وهو نموذج لغة بصرية متطور طورته DeepMind في الأصل. يقبل النموذج تسلسلات عشوائية من إدخالات الصور والنصوص وينتج نصًا متماسكًا كإخراج. يمكنه الإجابة على الأسئلة حول الصور، ووصف المحتوى المرئي، وخلق قصص قائمة على صور متعددة، وهلم جرا. يأتي IDEFICS في متغيرين - [80 مليار معلمة](https://huggingface.co/HuggingFaceM4/idefics-80b) و [9 مليار معلمة](https://huggingface.co/HuggingFaceM4/idefics-9b)، وكلاهما متاح على Hub 🤗. وبالنسبة لكل متغير، يمكنك أيضًا العثور على إصدارات مُعلَّمة من النموذج المُعدَّل للاستخدامات المحادثية.

هذا النموذج متعدد الاستخدامات بشكل استثنائي ويمكن استخدامه لمجموعة واسعة من المهام المتعلقة بالصور والوسائط المتعددة. ومع ذلك، فإن كونها نموذجًا كبيرًا يعني أنها تتطلب موارد حوسبة وهياكل أساسية كبيرة. الأمر متروك لك لتقرر ما إذا كان هذا النهج يناسب حالتك الاستخدامية بشكل أفضل من ضبط النماذج المتخصصة لكل مهمة فردية.

في هذا الدليل، ستتعلم كيفية:
- [تحميل IDEFICS](#تحميل-النموذج) و [تحميل الإصدار الكمي من النموذج](#النموذج-الكمي)
- استخدام IDEFICS لما يلي:
  - [وضع عنوان للصورة](#وضع-عنوان-للصورة)
  - [وضع عنوان للصورة بناءً على موجهات](#وضع-عنوان-للصورة-بناءً-على-موجهات)
  - [التوجيه القليل](#التوجيه-القليل)
  - [الإجابة على الأسئلة البصرية](#الإجابة-على-الأسئلة-البصرية)
  - [تصنيف الصور](#تصنيف-الصور)
  - [توليد النص الموجه بالصورة](#توليد-النص-الموجه-بالصورة)
- [تشغيل الاستدلال في وضع الدفعة](#تشغيل-الاستدلال-في-وضع-الدفعة)
- [تشغيل IDEFICS instruct للاستخدامات المحادثية](#تشغيل-idefics-instruct-للاستخدامات-المحادثية)

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية.

```bash
pip install -q bitsandbytes sentencepiece accelerate transformers
```

<Tip>
لتشغيل الأمثلة التالية باستخدام إصدار غير كمي من نقطة تفتيش النموذج، ستحتاج إلى ذاكرة GPU لا تقل عن 20 جيجابايت.
</Tip>

## تحميل النموذج

دعونا نبدأ بتحميل نقطة تفتيش معلمات النموذج البالغة 9 مليارات:

```py
>>> checkpoint = "HuggingFaceM4/idefics-9b"
```

تمامًا مثل النماذج الأخرى لـ Transformers، يلزمك تحميل معالج والنموذج نفسه من نقطة التفتيش. يجمع معالج IDEFICS بين [`LlamaTokenizer`] ومعالج الصور IDEFICS في معالج واحد للاهتمام بإعداد إدخالات النص والصورة للنموذج.

```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)
```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

يؤدي تعيين `device_map` إلى `"auto"` إلى تحديد كيفية تحميل وتخزين أوزان النموذج بالطريقة الأكثر تحسينًا بالنظر إلى الأجهزة الموجودة.

### النموذج الكمي

إذا كانت ذاكرة GPU عالية السعة تمثل مشكلة، فيمكنك تحميل الإصدار الكمي من النموذج. لتحميل النموذج والمعالج بدقة 4 بت، قم بتمرير `BitsAndBytesConfig` إلى طريقة `from_pretrained` وسيتم ضغط النموذج أثناء التحميل.

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

>>> quantization_config = BitsAndBytesConfig(
...     load_in_4bit=True,
...     bnb_4bit_compute_dtype=torch.float16,
... )

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(
...     checkpoint,
...     quantization_config=quantization_config,
...     device_map="auto"
... )
```

الآن بعد أن قمت بتحميل النموذج بإحدى الطرق المقترحة، دعنا ننتقل إلى استكشاف المهام التي يمكنك استخدام IDEFICS لها.

## وضع عنوان للصورة
وضع عنوان للصورة هي مهمة التنبؤ بعنوان لصورة معينة. أحد التطبيقات الشائعة هو مساعدة الأشخاص ضعاف البصر على التنقل في مختلف المواقف، على سبيل المثال، استكشاف محتوى الصورة عبر الإنترنت.

لتوضيح المهمة، احصل على صورة لوضع عنوان لها، على سبيل المثال:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-im-captioning.jpg" alt="صورة لجرو في سرير من الزهور"/>
</div>

الصورة بواسطة [Hendo Wang](https://unsplash.com/@hendoo).

يقبل IDEFICS موجهات النص والصورة. ومع ذلك، لوضع عنوان لصورة، لا يلزم تقديم موجه نصي إلى النموذج، فقط الصورة المدخلة المعالجة مسبقًا. بدون موجه نصي، سيبدأ النموذج في إنشاء نص من رمز BOS (بداية التسلسل) وبالتالي إنشاء عنوان.

كإدخال صورة للنموذج، يمكنك استخدام كائن صورة (`PIL.Image`) أو عنوان URL يمكن استرداد الصورة منه.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDBzjixsfHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
A puppy in a flower bed
```

<Tip>

من الجيد تضمين `bad_words_ids` في المكالمة إلى `generate` لتجنب الأخطاء الناشئة عند زيادة `max_new_tokens`: سيحاول النموذج إنشاء رمز `<image>` أو `<fake_token_around_image>` جديد عندما لا تكون هناك صورة ينشئها النموذج.
يمكنك تعيينه أثناء التنقل كما هو موضح في هذا الدليل، أو تخزينه في `GenerationConfig` كما هو موضح في دليل [استراتيجيات إنشاء النص](../generation_strategies).
</Tip>

## وضع عنوان للصورة بناءً على موجهات

يمكنك توسيع نطاق وضع عنوان الصورة عن طريق توفير موجه نصي، والذي سيواصل النموذج تنفيذه بناءً على الصورة. دعنا نأخذ صورة أخرى لتوضيح ذلك:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg" alt="صورة لبرج إيفل ليلاً"/>
</div>

الصورة بواسطة [Denys Nevozhai](https://unsplash.com/@dnevozhai).

يمكن تمرير موجهات النص والصورة إلى معالج النموذج كقائمة واحدة لإنشاء الإدخالات المناسبة.
الصورة بواسطة [Denys Nevozhai](https://unsplash.com/@dnevozhai).

يمكن تمرير موجهات النص والصورة إلى معالج النموذج كقائمة واحدة لإنشاء الإدخالات المناسبة.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...     "This is an image of ",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
This is an image of the Eiffel Tower in Paris, France.
```

## التوجيه القليل

في حين أن IDEFICS يظهر نتائج ممتازة بدون توجيه، فقد تتطلب مهمتك تنسيقًا معينًا للعنوان، أو قد تأتي بقيود أو متطلبات أخرى تزيد من تعقيد المهمة. يمكن استخدام التوجيه القليل لتمكين التعلم في السياق. من خلال توفير أمثلة في الموجه، يمكنك توجيه النموذج لتوليد نتائج تحاكي تنسيق الأمثلة المعطاة.

دعونا نستخدم صورة برج إيفل السابقة كمثال للنموذج ونبني موجهًا يوضح للنموذج أنه بالإضافة إلى تعلم ما هو الكائن في الصورة، نود أيضًا الحصول على بعض المعلومات المثيرة للاهتمام عنه. ثم دعونا نرى، إذا كان بإمكاننا الحصول على نفس تنسيق الاستجابة لصورة تمثال الحرية:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg" alt="صورة لتمثال الحرية"/>
</div>

الصورة بواسطة [Juan Mayobre](https://unsplash.com/@jmayobres).

```py
>>> prompt = ["User:",
...            "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...            "Describe this image.\nAssistant: An image of the Eiffel Tower at night. Fun fact: the Eiffel Tower is the same height as an 81-storey building.\n",
...            "User:",
...            "https://images.unsplash.com/photo-1524099163253-32b7f0256868?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3387&q=80",
...            "Describe this image.\nAssistant:"
...            ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
User: Describe this image.
Assistant: An image of the Eiffel Tower at night. Fun fact: the Eiffel Tower is the same height as an 81-storey building. 
User: Describe this image.
Assistant: An image of the Statue of Liberty. Fun fact: the Statue of Liberty is 151 feet tall.
```

لاحظ أنه من مجرد مثال واحد (أي، 1-shot)، تعلم النموذج كيفية أداء المهمة. بالنسبة للمهام الأكثر تعقيدًا، لا تتردد في التجربة بعدد أكبر من الأمثلة (على سبيل المثال، 3-shot، 5-shot، إلخ).

## الإجابة على الأسئلة البصرية

الإجابة على الأسئلة المرئية (VQA) هي مهمة الإجابة على الأسئلة المفتوحة بناءً على صورة. على غرار وضع عنوان الصورة، يمكن استخدامه في تطبيقات إمكانية الوصول، ولكن أيضًا في التعليم (الاستدلال حول المواد المرئية)، وخدمة العملاء (الأسئلة حول المنتجات بناءً على الصور)، واسترجاع الصور.

دعنا نحصل على صورة جديدة لهذه المهمة:
دعنا نحصل على صورة جديدة لهذه المهمة:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-vqa.jpg" alt="صورة لزوجين يتناولان نزهة"/>
</div>

الصورة بواسطة [Jarritos Mexican Soda](https://unsplash.com/@jarritos).

يمكنك توجيه النموذج من وضع عنوان الصورة إلى الإجابة على الأسئلة المرئية عن طريق توجيهه بتعليمات مناسبة:

```py
>>> prompt = [
...     "Instruction: Provide an answer to the question. Use the image to answer.\n",
...     "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...     "Question: Where are these people and what's the weather like? Answer:"
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Provide an answer to the question. Use the image to answer.
 Question: Where are these people and what's the weather like? Answer: They're in a park in New York City, and it's a beautiful day.
```

## تصنيف الصور

IDEFICS قادر على تصنيف الصور إلى فئات مختلفة دون تدريب صريح على البيانات التي تحتوي على أمثلة موسومة من تلك الفئات المحددة. بالنظر إلى قائمة الفئات واستخدام قدراته في فهم الصور والنصوص، يمكن للنموذج استنتاج الفئة التي تنتمي إليها الصورة على الأرجح.

لنفترض أن لدينا هذه الصورة لطاولة الخضار:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-classification.jpg" alt="صورة لطاولة الخضار"/>
</div>

تصوير فوتوغرافي بواسطة [بيتر ويندت](https://unsplash.com/@peterwendt).

يمكننا توجيه النموذج لتصنيف الصورة إلى إحدى الفئات التي لدينا:

```py
>>> categories = ['animals','vegetables', 'city landscape', 'cars', 'office']
>>> prompt = [f"Instruction: Classify the following image into a single category from the following list: {categories}.\n",
...     "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",    
...     "Category: "
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=6, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Classify the following image into a single category from the following list: ['animals', 'vegetables', 'city landscape', 'cars', 'office'].
Category: Vegetables
```

في المثال أعلاه، نوجه النموذج لتصنيف الصورة إلى فئة واحدة، ومع ذلك، يمكنك أيضًا توجيه النموذج للقيام بالتصنيف الترتيبي.

## توليد النص الموجه بالصور

لتطبيقات أكثر إبداعًا، يمكنك استخدام توليد النص الموجه بالصور لتوليد نص بناءً على صورة. يمكن أن يكون هذا
مفيدًا لإنشاء أوصاف للمنتجات والإعلانات ووصف المشهد، وما إلى ذلك.

دعونا نوجه IDEFICS لكتابة قصة بناءً على صورة بسيطة لباب أحمر:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-story-generation.jpg" alt="صورة لباب أحمر مع قرع على الدرجات"/>
</div>

تصوير فوتوغرافي بواسطة [كريج تيدبول](https://unsplash.com/@devonshiremedia).

```py
>>> prompt = ["Instruction: Use the image to write a story. \n",
...     "https://images.unsplash.com/photo-1517086822157-2b0358e7684a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2203&q=80",
...     "Story: \n"]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, num_beams=2, max_new_tokens=200, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0]) 
Instruction: Use the image to write a story. 
 Story: 
Once upon a time, there was a little girl who lived in a house with a red door.  She loved her red door.  It was the prettiest door in the whole world.

One day, the little girl was playing in her yard when she noticed a man standing on her doorstep.  He was wearing a long black coat and a top hat.

The little girl ran inside and told her mother about the man.

Her mother said, “Don’t worry, honey.  He’s just a friendly ghost.”

The little girl wasn’t sure if she believed her mother, but she went outside anyway.

When she got to the door, the man was gone.

The next day, the little girl was playing in her yard again when she noticed the man standing on her doorstep.

He was wearing a long black coat and a top hat.

The little girl ran
```

يبدو أن IDEFICS لاحظ القرع على عتبة الباب وذهب إلى قصة مخيفة في عيد الهالوين حول شبح.

<Tip>

بالنسبة للنواتج الأطول مثل هذا، ستستفيد بشكل كبير من ضبط استراتيجية توليد النص. يمكن أن يساعدك هذا بشكل كبير
يمكنك تحسين جودة الإخراج المولد بشكل كبير. تحقق من [استراتيجيات توليد النص](../generation_strategies)
لمعرفة المزيد.
</Tip>

## تشغيل الاستنتاج في وضع الدفعة

توضح جميع الأقسام السابقة IDEFICS لمثال واحد. بطريقة مشابهة جدًا، يمكنك تشغيل الاستنتاج
بالنسبة لدفعة من الأمثلة عن طريق تمرير قائمة من المطالبات:

```py
>>> prompts = [
...     [   "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
... ]

>>> inputs = processor(prompts, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i,t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n") 
0:
This is an image of the Eiffel Tower in Paris, France.

1:
This is an image of a couple on a picnic blanket.

2:
This is an image of a vegetable stand.
```

## IDEFICS instruct للاستخدام المحادثي

بالنسبة لحالات الاستخدام المحادثية، يمكنك العثور على إصدارات مضبوطة الدقة من النموذج على 🤗 Hub:
`HuggingFaceM4/idefics-80b-instruct` و `HuggingFaceM4/idefics-9b-instruct`.

هذه نقاط تفتيش ناتجة عن ضبط دقيق لنماذج القاعدة الخاصة بها على مزيج من مجموعات البيانات الخاضعة للإشراف والتعليمات
ضبط دقيق، مما يعزز الأداء في اتجاه مجرى النهر أثناء جعل النماذج أكثر قابلية للاستخدام في إعدادات المحادثة.

الاستخدام والتشغيل للاستخدام المحادثي مشابه جدًا لاستخدام نماذج القاعدة:

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> checkpoint = "HuggingFaceM4/idefics-9b-instruct"
>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> prompts = [
...     [
...         "User: What is in this image?",
...         "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
...         "<end_of_utterance>",

...         "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

...         "\nUser:",
...         "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
...         "And who is that?<end_of_utterance>",

...         "\nAssistant:",
...     ],
... ]

>>> # --batched mode
>>> inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
>>> # --single sample mode
>>> # inputs = processor(prompts[0], return_tensors="pt").to(device)

>>> # Generation args
>>> exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i, t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n")
```