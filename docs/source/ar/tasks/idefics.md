<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# مهام الصور باستخدام IDEFICS

[[open-in-colab]]

بينما يمكن التعامل مع المهام الفردية عبر تحسين (Fine-tuning) نماذج متخصصة، ظهرت مقاربة بديلة مؤخرًا واكتسبت شعبية، وهي استخدام نماذج ضخمة لمجموعة متنوعة من المهام دون تحسين. على سبيل المثال، يمكن لنماذج اللغة الكبيرة التعامل مع مهام المعالجة اللغوية الطبيعية مثل التلخيص والترجمة والتصنيف والمزيد. لم يعد هذا النهج مقتصرًا على أسلوب واحد مثل النص فقط، وفي هذا الدليل سنوضح كيف يمكنك حل مهام الصورة-نص باستخدام نموذج متعدد الوسائط كبير يُدعى IDEFICS.

[IDEFICS](../model_doc/idefics) هو نموذج رؤية ولغة مفتوح الوصول يعتمد على [Flamingo](https://huggingface.co/papers/2204.14198)، وهو نموذج رائد للغة المرئية طُوّر مبدئيًا بواسطة DeepMind. يقبل النموذج تسلسلات عشوائية من مدخلات الصور والنصوص ويولّد نصًا متماسكًا كمخرج. يمكنه الإجابة عن أسئلة حول الصور، ووصف المحتوى البصري، وابتكار قصص مستندة إلى عدة صور، وغير ذلك. يأتي IDEFICS بنسختين: [80 مليار معلمة](https://huggingface.co/HuggingFaceM4/idefics-80b) و[9 مليارات معلمة](https://huggingface.co/HuggingFaceM4/idefics-9b)، وكلاهما متاح على 🤗 Hub. ولكل نسخة، ستجد أيضًا إصدارات محسّنة بالتعليمات مكيّفة لحالات الاستخدام الحوارية.

هذا النموذج متعدد الاستخدامات بشكل استثنائي ويمكن استخدامه لمجموعة واسعة من مهام الصور والمهام متعددة الوسائط. لكن كونه نموذجًا كبيرًا يعني أنه يتطلب موارد حوسبة وبنية تحتية كبيرة. يعود لك تحديد ما إذا كان هذا النهج مناسبًا لحالتك أكثر من تحسين نماذج متخصصة لكل مهمة فردية.

في هذا الدليل، ستتعلم كيفية:
- [تحميل IDEFICS](#loading-the-model) و[تحميل النسخة المُكمَّمة من النموذج](#quantized-model)
- استخدام IDEFICS في: 
  - [وصف الصور (Image captioning)](#image-captioning)
  - [وصف الصور مع تحفيز نصي (Prompted image captioning)](#prompted-image-captioning)
  - [التحفيز بعدة أمثلة (Few-shot prompting)](#few-shot-prompting)
  - [الإجابة البصرية عن الأسئلة (VQA)](#visual-question-answering)
  - [تصنيف الصور](#image-classification)
  - [توليد نص موجّه بالصور](#image-guided-text-generation)
- [تشغيل الاستدلال على دفعات](#running-inference-in-batch-mode)
- [تشغيل IDEFICS instruct للاستخدام الحواري](#idefics-instruct-for-conversational-use)

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات اللازمة.

```bash
pip install -q bitsandbytes sentencepiece accelerate transformers
```

<Tip>
لتشغيل الأمثلة التالية باستخدام نسخة غير مُكمَّمة من نقطة تحقق النموذج، ستحتاج إلى 20GB على الأقل من ذاكرة GPU.
</Tip>

## Loading the model

لنبدأ بتحميل نقطة تحقق النسخة ذات 9 مليارات معلمة:

```py
>>> checkpoint = "HuggingFaceM4/idefics-9b"
```

كما هو الحال مع نماذج Transformers الأخرى، تحتاج إلى تحميل "معالج" (processor) والنموذج نفسه من نقطة التحقق.
يلف معالج IDEFICS كلًا من [`LlamaTokenizer`] ومعالج الصور الخاص بـ IDEFICS في معالج واحد ليتولى تجهيز مدخلات النص والصورة للنموذج.

```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

تعيين `device_map` إلى `"auto"` سيحدد تلقائيًا كيفية تحميل وتخزين أوزان النموذج بأكثر الطرق ملاءمةً للأجهزة المتاحة.

### Quantized model

إذا كانت ذاكرة GPU عالية السعة تمثل مشكلة، يمكنك تحميل نسخة مُكمَّمة من النموذج. لتحميل النموذج والمعالج بدقة 4-بت، مرّر `BitsAndBytesConfig` إلى دالة `from_pretrained` وسيتم ضغط النموذج أثناء التحميل.

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

بعد أن قمت بتحميل النموذج بإحدى الطرق المقترحة، لننتقل لاستكشاف المهام التي يمكنك استخدام IDEFICS فيها.

## Image captioning
وصف الصور هو مهمة توقع تسمية توضيحية لصورة معينة. يُعد ذلك تطبيقًا شائعًا لمساعدة ذوي الإعاقة البصرية على التنقل في مواقف مختلفة، مثل استكشاف محتوى الصور عبر الإنترنت.

لتوضيح المهمة، احصل على صورة لوصفها، مثلًا:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-im-captioning.jpg" alt="Image of a puppy in a flower bed"/>
</div>

Photo by [Hendo Wang](https://unsplash.com/@hendoo).

يقبل IDEFICS محثّات نصية وصورية. ولكن من أجل وصف صورة، لست مضطرًا لتقديم محثّ نصي للنموذج، يكفي فقط الصورة المُحضّرة. بدون محثّ نصي، سيبدأ النموذج توليد النص من رمز بداية التسلسل (BOS) لتكوين التسمية التوضيحية.

كمدخل صورة للنموذج، يمكنك استخدام كائن صورة (`PIL.Image`) أو رابط URL يمكن منه جلب الصورة.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
A puppy in a flower bed
```

<Tip>
من الجيد تضمين `bad_words_ids` في استدعاء `generate` لتجنب أخطاء قد تنشأ عند زيادة `max_new_tokens`: سيحاول النموذج توليد رمز `<image>` أو `<fake_token_around_image>` جديد عندما لا توجد صورة يجري توليدها. يمكنك تعيينه أثناء التشغيل كما في هذا الدليل، أو تخزينه في `GenerationConfig` كما هو موضح في دليل [استراتيجيات توليد النص](../generation_strategies).
</Tip>

## Prompted image captioning

يمكنك توسيع وصف الصور عبر تقديم محثّ نصي سيُكملُه النموذج بالنظر إلى الصورة. لنأخذ صورة أخرى للتوضيح:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg" alt="Image of the Eiffel Tower at night"/>
</div>

Photo by [Denys Nevozhai](https://unsplash.com/@dnevozhai).

يمكن تمرير المحثّات النصية والصورية إلى معالج النموذج كقائمة واحدة لإنشاء المدخلات المناسبة.

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

## Few-shot prompting

على الرغم من أن IDEFICS يقدّم نتائج جيدة دون أمثلة (zero-shot)، إلا أن مهمتك قد تتطلب صيغة معينة للتسمية التوضيحية، أو تأتي بقيود أو متطلبات تزيد من تعقيد المهمة. يمكن استخدام التحفيز بعدة أمثلة (few-shot) لتمكين التعلم داخل السياق. عبر تقديم أمثلة في المحثّ، يمكنك توجيه النموذج لتوليد نتائج تحاكي صيغة الأمثلة المعطاة.

لنستخدم صورة برج إيفل السابقة كمثال للنموذج ونبني محثًا يوضح للنموذج أننا، بالإضافة إلى تعرّف الكائن في الصورة، نريد أيضًا الحصول على معلومة ممتعة عنه. ثم لنرَ إن كان بإمكاننا الحصول على نفس صيغة الاستجابة لصورة تمثال الحرية:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg" alt="Image of the Statue of Liberty"/>
</div>

Photo by [Juan Mayobre](https://unsplash.com/@jmayobres).

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

لاحظ أنه من مجرد مثال واحد (أي 1-shot) تعلّم النموذج كيفية أداء المهمة. للمهام الأكثر تعقيدًا، لا تتردد في تجربة عدد أكبر من الأمثلة (3-shot، 5-shot، إلخ).

## Visual question answering

الإجابة البصرية عن الأسئلة (VQA) هي مهمة الإجابة عن أسئلة مفتوحة بالاعتماد على صورة. وبشكل مماثل لوصف الصور، يمكن استخدامها لتطبيقات إمكانية الوصول، وأيضًا في التعليم (الاستدلال على مواد بصرية)، وخدمة العملاء (أسئلة حول المنتجات بناءً على الصور)، واسترجاع الصور.

لنحصل على صورة جديدة لهذه المهمة:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-vqa.jpg" alt="Image of a couple having a picnic"/>
</div>

Photo by [Jarritos Mexican Soda](https://unsplash.com/@jarritos).

يمكنك توجيه النموذج من وصف الصور إلى الإجابة البصرية عن الأسئلة عبر محثّات مناسبة:

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

## Image classification

يستطيع IDEFICS تصنيف الصور إلى فئات مختلفة دون أن يكون قد دُرّب صراحةً على بيانات تحتوي أمثلة معنونة من تلك الفئات المحددة. بالنظر إلى قائمة بالفئات ومعتمداً على فهمه للصورة والنص، يمكن للنموذج استنتاج الفئة الأكثر احتمالًا لانتماء الصورة إليها.

لنفترض أن لدينا هذه الصورة لكشك خضروات:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-classification.jpg" alt="Image of a vegetable stand"/>
</div>

Photo by [Peter Wendt](https://unsplash.com/@peterwendt).

يمكننا توجيه النموذج لتصنيف الصورة ضمن إحدى الفئات الموجودة لدينا:

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

في المثال أعلاه وجّهنا النموذج لتصنيف الصورة ضمن فئة واحدة، لكن يمكنك أيضًا تحفيزه لإجراء تصنيف ترتيبي (rank classification).

## Image-guided text generation

للتطبيقات الأكثر إبداعًا، يمكنك استخدام توليد النص الموجّه بالصور لإنتاج نص بالاعتماد على صورة. يمكن أن يكون ذلك مفيدًا لإنشاء أوصاف للمنتجات والإعلانات ووصف مشاهد، وما إلى ذلك.

لنحفّز IDEFICS على كتابة قصة بناءً على صورة بسيطة لباب أحمر:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-story-generation.jpg" alt="Image of a red door with a pumpkin on the steps"/>
</div>

Photo by [Craig Tidball](https://unsplash.com/@devonshiremedia).

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

يبدو أن IDEFICS لاحظ اليقطينة على العتبة وذهب نحو قصة هالووين مخيفة عن شبح.

<Tip>
للنواتج الأطول كهذا المثال، ستستفيد كثيرًا من ضبط استراتيجية توليد النص. فهذا يساعد بشكل ملحوظ على تحسين جودة المخرجات. اطّلع على [استراتيجيات توليد النص](../generation_strategies) لمعرفة المزيد.
</Tip>

## Running inference in batch mode

عرضت الأقسام السابقة IDEFICS على مثال واحد. وبطريقة مشابهة جدًا، يمكنك تشغيل الاستدلال لدفعة من الأمثلة عبر تمرير قائمة من المحثّات:

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

## IDEFICS instruct for conversational use

لحالات الاستخدام الحوارية، ستجد إصدارات مُحسّنة بالتعليمات للنموذج على 🤗 Hub: 
`HuggingFaceM4/idefics-80b-instruct` و`HuggingFaceM4/idefics-9b-instruct`.

هذه النقاط هي نتيجة تحسين النماذج الأساسية المماثلة على مزيج من مجموعات بيانات الإشراف والتعليمات، ما يعزّز الأداء على المهام اللاحقة ويجعل النماذج أكثر قابلية للاستخدام في الإعدادات الحوارية.

استخدام النموذج وتوليد المحثّات في السيناريو الحواري مشابه جدًا لاستخدام النماذج الأساسية:

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor
>>> from accelerate.test_utils.testing import get_backend

>>> device, _, _ = get_backend() # يكتشف تلقائيًا نوع الجهاز الأساسي (CUDA, CPU, XPU, MPS, etc.)
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
