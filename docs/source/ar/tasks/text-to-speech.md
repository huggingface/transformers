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

# تحويل النص إلى كلام

[[open-in-colab]]

تحويل النص إلى كلام (Text-to-Speech, TTS) هو مهمة إنشاء كلام طبيعي من النص، حيث يمكن توليد الكلام بعدة لغات ولعدة متحدثين. هناك العديد من نماذج تحويل النص إلى كلام المتاحة حاليًا في 🤗 Transformers مثل
[Bark](../model_doc/bark)، و[MMS](../model_doc/mms)، و[VITS](../model_doc/vits)، و[SpeechT5](../model_doc/speecht5).

يمكنك بسهولة توليد صوت باستخدام "text-to-audio" عبر خط المعالجة "pipeline" (أو الاسم المستعار له - "text-to-speech"). بعض النماذج، مثل Bark،
يمكن أيضًا تهيئتها لتوليد تعبيرات غير لفظية مثل الضحك والتنهد والبكاء، أو حتى إضافة موسيقى.
إليك مثالًا على كيفية استخدام خط المعالجة "text-to-speech" مع Bark:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="suno/bark-small")
>>> text = "[clears throat] This is a test ... and I just took a long pause."
>>> output = pipe(text)
```

فيما يلي مقتطف كود يمكنك استخدامه للاستماع إلى الصوت الناتج داخل الدفتر (Notebook):

```python
>>> from IPython.display import Audio
>>> Audio(output["audio"], rate=output["sampling_rate"])
```

لمزيد من الأمثلة حول ما يمكن أن يقدمه Bark ونماذج TTS المُدرَّبة مسبقًا الأخرى، راجع
[دورة الصوت](https://huggingface.co/learn/audio-course/chapter6/pre-trained_models).

إذا كنت تبحث عن تحسين (Fine-tune) نموذج TTS، فإن نماذج تحويل النص إلى كلام المتاحة حاليًا في 🤗 Transformers هي
[SpeechT5](model_doc/speecht5) و[FastSpeech2Conformer](model_doc/fastspeech2_conformer)، مع إضافة المزيد مستقبلًا. تم تدريب SpeechT5 مسبقًا على مزيج من بيانات تحويل الكلام إلى نص (ASR) وتحويل النص إلى كلام، مما يتيح له تعلم فضاء موحد من التمثيلات المخفية المشتركة بين النص والصوت. هذا يعني أنه يمكن تحسين نفس النموذج المُدرّب مسبقًا لمهام مختلفة. علاوة على ذلك، يدعم SpeechT5 تعدد المتحدثين عبر تضمينات المتحدث (x-vector).

يوضّح ما تبقى من هذا الدليل كيفية:

1. تحسين [SpeechT5](../model_doc/speecht5) الذي تم تدريبه أصلاً على الكلام الإنجليزي على اللغة الهولندية (`nl`) من مجموعة بيانات [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli).
2. استخدام النموذج المُحسَّن للاستدلال بطريقتين: عبر خط المعالجة (pipeline) أو يدويًا مباشرةً.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات اللازمة:

```bash
pip install datasets soundfile speechbrain accelerate
```

ثبّت 🤗Transformers من المصدر لأن جميع ميزات SpeechT5 لم تُدمج بعد في إصدار رسمي:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

لمتابعة هذا الدليل ستحتاج إلى وحدة معالجة رسومات (GPU). إذا كنت تعمل داخل Notebook، نفّذ السطر التالي للتحقق من توفر GPU:

```bash
!nvidia-smi
```

أو بديلًا لـ AMD GPUs:

```bash
!rocm-smi
```

</Tip>

نوصيك بتسجيل الدخول إلى حسابك على Hugging Face لرفع النموذج ومشاركته مع المجتمع. عند المطالبة، أدخل رمزك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة البيانات

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) هي مجموعة ضخمة متعددة اللغات من بيانات الكلام، تم جمعها من تسجيلات فعاليات البرلمان الأوروبي بين عامي 2009-2020. تحتوي على بيانات صوتية-نصية معنونة لـ 15 لغة أوروبية. في هذا الدليل، نستخدم مجموعة اللغة الهولندية، ويمكنك اختيار مجموعة أخرى إذا رغبت.

لاحظ أن VoxPopuli أو أي مجموعة بيانات للتعرف التلقائي على الكلام (ASR) قد لا تكون الأنسب لتدريب نماذج TTS. الميزات المفيدة لـ ASR، مثل الضوضاء الخلفية الزائدة، غالبًا ما تكون غير مرغوبة في TTS. مع ذلك، قد يكون من الصعب العثور على مجموعات بيانات TTS عالية الجودة ومتعددة اللغات ومتعددة المتحدثين.

لنبدأ بتحميل البيانات:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

يجب أن تكون 20968 عينة كافية للتحسين. يتوقع SpeechT5 أن تكون البيانات الصوتية بمعدل أخذ عينات 16 كيلوهرتز، لذا تأكد من أن أمثلة المجموعة تلبي هذا الشرط:

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## معالجة البيانات مسبقًا

لنبدأ بتحديد نقطة تحقق النموذج (checkpoint) المراد استخدامها وتحميل المعالج المناسب:

```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### تنظيف النص لملاءمة ترميز SpeechT5

ابدأ بتنظيف بيانات النص. ستحتاج إلى جزء المُجزِّئ (tokenizer) من المعالج لمعالجة النص:

```py
>>> tokenizer = processor.tokenizer
```

تحتوي أمثلة مجموعة البيانات على الميزتين `raw_text` و`normalized_text`. عند اختيار أيهما لاستخدامه كمدخل نصي، ضع في اعتبارك أن مُجزِّئ SpeechT5 لا يحتوي على رموز للأرقام. في `normalized_text` تُكتب الأرقام ككلمات، ولذلك فهو أنسب، وننصح باستخدام `normalized_text` كنص إدخال.

نظرًا لأن SpeechT5 تم تدريبه على اللغة الإنجليزية، فقد لا يتعرف على بعض الأحرف في المجموعة الهولندية. إذا تُركت كما هي، فسيتم تحويل هذه الأحرف إلى رموز `<unk>`. لكن في الهولندية، تُستخدم بعض الأحرف مثل `à` للتأكيد على المقاطع. للحفاظ على معنى النص، يمكننا استبدال هذا الحرف بحرف `a` عادي.

لتحديد الرموز غير المدعومة، استخرج جميع الأحرف الفريدة في مجموعة البيانات باستخدام `SpeechT5Tokenizer` الذي يعمل على مستوى الأحرف كرموز. للقيام بذلك، اكتب دالة `extract_all_chars` التي تقوم بضم نصوص جميع الأمثلة في سلسلة واحدة وتحويلها إلى مجموعة أحرف.
تأكد من تعيين `batched=True` و`batch_size=-1` في `dataset.map()` بحيث تكون جميع النصوص متاحة دفعة واحدة لدالة التحويل.

```py
>>> def extract_all_chars(batch):
...     all_text = " ".join(batch["normalized_text"])
...     vocab = list(set(all_text))
...     return {"vocab": [vocab], "all_text": [all_text]}


>>> vocabs = dataset.map(
...     extract_all_chars,
...     batched=True,
...     batch_size=-1,
...     keep_in_memory=True,
...     remove_columns=dataset.column_names,
... )

>>> dataset_vocab = set(vocabs["vocab"][0])
>>> tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```

الآن لديك مجموعتان من الأحرف: واحدة من مفردات مجموعة البيانات، وأخرى من مفردات المُجزِّئ. لتحديد أي أحرف غير مدعومة في مجموعة البيانات، خذ الفرق بين المجموعتين. ستحتوي المجموعة الناتجة على الأحرف الموجودة في البيانات ولكن ليست في المُجزِّئ.

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

للتعامل مع الأحرف غير المدعومة المحددة في الخطوة السابقة، عرّف دالة تقوم باستبدال هذه الأحرف برموز صالحة. لاحظ أن المسافات يتم استبدالها بالفعل بـ `▁` في المُجزِّئ ولا تحتاج إلى معالجة منفصلة.

```py
>>> replacements = [
...     ("à", "a"),
...     ("ç", "c"),
...     ("è", "e"),
...     ("ë", "e"),
...     ("í", "i"),
...     ("ï", "i"),
...     ("ö", "o"),
...     ("ü", "u"),
... ]


>>> def cleanup_text(inputs):
...     for src, dst in replacements:
...         inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
...     return inputs


>>> dataset = dataset.map(cleanup_text)
```

الآن بعد أن تعاملت مع الأحرف الخاصة في النص، حان الوقت للتركيز على البيانات الصوتية.

### المتحدثون

تتضمن مجموعة VoxPopuli كلامًا لعدة متحدثين، لكن كم عدد المتحدثين المُمثلين في المجموعة؟ لتحديد ذلك، يمكننا عدّ عدد المتحدثين الفريدين وعدد الأمثلة التي يقدّمها كل متحدث في المجموعة.
مع إجمالي 20,968 مثالًا في المجموعة، ستمنحنا هذه المعلومات فهمًا أفضل لتوزيع المتحدثين والأمثلة في البيانات.

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

من خلال رسم مخطط هيستوغرام، يمكنك الحصول على فكرة عن مقدار البيانات لكل متحدث.

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.hist(speaker_counts.values(), bins=20)
>>> plt.ylabel("Speakers")
>>> plt.xlabel("Examples")
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_speakers_histogram.png" alt="Speakers histogram"/>
</div>

يكشف المخطط أن حوالي ثلث المتحدثين في المجموعة لديهم أقل من 100 مثال، بينما لدى حوالي عشرة متحدثين أكثر من 500 مثال. لتحسين كفاءة التدريب وتحقيق توازن أفضل، يمكننا قصر البيانات على المتحدثين الذين لديهم بين 100 و400 مثال.

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

لنرَ كم عدد المتحدثين المتبقين:

```py
>>> len(set(dataset["speaker_id"]))
42
```

ولنرَ كم عدد الأمثلة المتبقية:

```py
>>> len(dataset)
9973
```

تبقى لديك أقل بقليل من 10,000 مثال من حوالي 40 متحدثًا فريدًا، وهو عدد ينبغي أن يكون كافيًا.

لاحظ أن بعض المتحدثين ذوي الأمثلة القليلة قد يكون لديهم في الواقع صوت أكثر إذا كانت الأمثلة طويلة. لكن تحديد إجمالي مدة الصوت لكل متحدث يتطلب مسح المجموعة كاملة، وهو إجراء يستغرق وقتًا ويشمل تحميل وفك ترميز كل ملف صوتي، لذا نتجاوزه هنا.

### تضمينات المتحدث (Speaker embeddings)

لتمكين نموذج TTS من التمييز بين المتحدثين المتعددين، ستحتاج إلى إنشاء تضمين للمتحدث لكل مثال. يعد تضمين المتحدث إدخالًا إضافيًا إلى النموذج يلتقط خصائص صوت متحدث معين.
لإنشاء هذه التضمينات، استخدم نموذج [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) المُدرّب مسبقًا من SpeechBrain.

أنشئ الدالة `create_speaker_embedding()` التي تستقبل موجة صوتية كمدخل وتُخرج متجهًا من 512 عنصرًا يحتوي على تضمين المتحدث المقابل.

```py
>>> import os
>>> import torch
>>> from speechbrain.inference.classifiers import EncoderClassifier
>>> from accelerate.test_utils.testing import get_backend

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
>>> device, _, _ = get_backend() # يكتشف تلقائيًا نوع الجهاز الأساسي (CUDA, CPU, XPU, MPS, etc.)
>>> speaker_model = EncoderClassifier.from_hparams(
...     source=spk_model_name,
...     run_opts={"device": device},
...     savedir=os.path.join("/tmp", spk_model_name),
... )


>>> def create_speaker_embedding(waveform):
...     with torch.no_grad():
...         speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
...         speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
...         speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
...     return speaker_embeddings
```

من المهم ملاحظة أن نموذج `speechbrain/spkrec-xvect-voxceleb` تم تدريبه على كلام باللغة الإنجليزية من مجموعة VoxCeleb، بينما أمثلة التدريب في هذا الدليل باللغة الهولندية. وعلى الرغم من اعتقادنا أن هذا النموذج سيولّد تضمينات متحدث معقولة لمجموعتنا الهولندية، إلا أن هذا الافتراض قد لا يكون صحيحًا دائمًا.

لأفضل النتائج، نوصي بتدريب نموذج X-vector على لغة الهدف أولًا. هذا يضمن قدرة أفضل على التقاط خصائص الصوت الفريدة الموجودة في اللغة الهولندية.

### معالجة مجموعة البيانات

أخيرًا، لنعالج البيانات إلى الصيغة التي يتوقعها النموذج. أنشئ دالة `prepare_dataset` التي تستقبل مثالًا واحدًا وتستخدم كائن `SpeechT5Processor` لتجزئة النص المدخل وتحميل الصوت الهدف إلى مخطط طيفي لوغاريتمي (log-mel spectrogram).
ويجب أيضًا إضافة تضمينات المتحدث كمدخل إضافي.

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example = processor(
...         text=example["normalized_text"],
...         audio_target=audio["array"],
...         sampling_rate=audio["sampling_rate"],
...         return_attention_mask=False,
...     )

...     # إزالة بُعد الدفعة
...     example["labels"] = example["labels"][0]

...     # استخدام SpeechBrain للحصول على x-vector
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

تحقّق من صحة المعالجة بالنظر إلى مثال واحد:

```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

يجب أن تكون تضمينات المتحدث متجهًا بطول 512 عنصرًا:

```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

يجب أن تكون التسميات (labels) مخططًا طيفيًا لوغاريتميًا بعدد 80 قناة mel.

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.imshow(processed_example["labels"].T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_1.png" alt="Log-mel spectrogram with 80 mel bins"/>
</div>

ملاحظة: إذا وجدت هذا المخطط الطيفي مربكًا، فقد يكون ذلك بسبب اعتيادك على العُرف القائل بوضع الترددات المنخفضة في الأسفل والمرتفعة في الأعلى عند الرسم. ومع ذلك، عند رسم المخططات الطيفية كصورة باستخدام مكتبة matplotlib، يُقلب محور y وتظهر المخططات الطيفية رأسًا على عقب.

الآن طبّق دالة المعالجة على المجموعة كاملة. سيستغرق هذا بين 5 و10 دقائق.

```py
>>> dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

سترى تحذيرًا يفيد بأن بعض الأمثلة في المجموعة أطول من الحد الأقصى لطول المدخل الذي يمكن للنموذج التعامل معه (600 رمز). أزل هذه الأمثلة من المجموعة. هنا نذهب أبعد قليلًا وللسماح بأحجام دفعات أكبر نحذف أي شيء يزيد عن 200 رمز.

```py
>>> def is_not_too_long(input_ids):
...     input_length = len(input_ids)
...     return input_length < 200


>>> dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
>>> len(dataset)
8259
```

بعد ذلك، أنشئ تقسيمًا بسيطًا للتدريب/الاختبار:

```py
>>> dataset = dataset.train_test_split(test_size=0.1)
```

### مجمّع البيانات (Data collator)

لدمج عدة أمثلة في دفعة واحدة، تحتاج إلى تعريف مجمّع بيانات مخصص. سيقوم هذا المجمّع بملء التسلسلات الأقصر برموز الحشو، ما يضمن أن جميع الأمثلة لها نفس الطول. بالنسبة لتسميات المخططات الطيفية، تُستبدل الأجزاء المملوءة بالقيمة الخاصة `-100`. هذه القيمة الخاصة تُعلِم النموذج بتجاهل ذلك الجزء من المخطط الطيفي عند حساب خسارة المخطط (spectrogram loss).

```py
>>> from dataclasses import dataclass
>>> from typing import Any, Dict, List, Union


>>> @dataclass
... class TTSDataCollatorWithPadding:
...     processor: Any

...     def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
...         input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
...         label_features = [{"input_values": feature["labels"]} for feature in features]
...         speaker_features = [feature["speaker_embeddings"] for feature in features]

...         # تجميع المدخلات والأهداف في دفعة واحدة
...         batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

...         # استبدال الحشو بـ -100 لتجاهله بشكل صحيح أثناء حساب الخسارة
...         batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

...         # غير مستخدمة أثناء التحسين
...         del batch["decoder_attention_mask"]

...         # تقليل أطوال الأهداف إلى مضاعفات عامل الاختزال
...         if model.config.reduction_factor > 1:
...             target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
...             target_lengths = target_lengths.new(
...                 [length - length % model.config.reduction_factor for length in target_lengths]
...             )
...             max_length = max(target_lengths)
...             batch["labels"] = batch["labels"][:, :max_length]

...         # إضافة تضمينات المتحدث أيضًا
...         batch["speaker_embeddings"] = torch.tensor(speaker_features)

...         return batch
```

في SpeechT5، يتم تقليل مدخلات جزء المُفكِّك (decoder) من النموذج بعامل 2. بمعنى آخر، يتم حذف كل خطوة زمنية ثانية من تسلسل الهدف. ثم يتنبأ المُفكِّك بتسلسل بطول ضعف الطول. نظرًا لأن طول تسلسل الهدف الأصلي قد يكون فرديًا، يتأكد مجمّع البيانات من تقريب الطول الأقصى للدفعة إلى أسفل ليكون مضاعفًا للعدد 2.

```py
>>> data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## تدريب النموذج

حمّل النموذج المُدرّب مسبقًا من نفس نقطة التحقق التي استخدمتها لتحميل المعالج:

```py
>>> from transformers import SpeechT5ForTextToSpeech

>>> model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

خيار `use_cache=True` غير متوافق مع التحقق المرحلي للتدرج (gradient checkpointing). عطّله أثناء التدريب.

```py
>>> model.config.use_cache = False
```

عرّف معاملات التدريب. هنا لن نحسب أي مقاييس تقييم أثناء عملية التدريب. بدلًا من ذلك، سننظر فقط إلى الخسارة:

```python
>>> from transformers import Seq2SeqTrainingArguments

>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="speecht5_finetuned_voxpopuli_nl",  # غيّرها إلى اسم المستودع الذي تريده
...     per_device_train_batch_size=4,
...     gradient_accumulation_steps=8,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=4000,
...     gradient_checkpointing=True,
...     fp16=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=2,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     report_to=["tensorboard"],
...     load_best_model_at_end=True,
...     greater_is_better=False,
...     label_names=["labels"],
...     push_to_hub=True,
... )
```

أنشئ كائن `Trainer` ومرر إليه النموذج ومجموعة البيانات ومجمّع البيانات.

```py
>>> from transformers import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
...     args=training_args,
...     model=model,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     data_collator=data_collator,
...     processing_class=processor,
... )
```

وبهذا، أنت جاهز لبدء التدريب! سيستغرق التدريب عدة ساعات. اعتمادًا على بطاقة الرسومات لديك، قد تواجه خطأ "نفاد الذاكرة" (CUDA out-of-memory). في هذه الحالة، يمكنك تقليل قيمة `per_device_train_batch_size` تدريجيًا بعوامل 2 وزيادة `gradient_accumulation_steps` بمقدار 2× للتعويض.

```py
>>> trainer.train()
```

لكي تتمكن من استخدام نقطة التحقق مع خط المعالجة (pipeline)، تأكد من حفظ المعالج مع نقطة التحقق:

```py
>>> processor.save_pretrained("YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

ادفع (Push) النموذج النهائي إلى 🤗 Hub:

```py
>>> trainer.push_to_hub()
```

## الاستدلال

### الاستدلال عبر خط المعالجة (Pipeline)

رائع، الآن بعد أن قمت بتحسين النموذج، يمكنك استخدامه للاستدلال!
أولًا، لنرَ كيف يمكنك استخدامه مع خط المعالجة المناسب. أنشئ خط معالجة "text-to-speech" باستخدام نقطة التحقق الخاصة بك:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

اختر مقطع نصي باللغة الهولندية ترغب في إلقائه، مثلًا:

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

لاستخدام SpeechT5 مع خط المعالجة، ستحتاج إلى تضمين متحدث. لنأخذه من مثال في مجموعة الاختبار:

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

الآن يمكنك تمرير النص وتضمينات المتحدث إلى خط المعالجة، وسيتولى الباقي:

```py
>>> forward_params = {"speaker_embeddings": speaker_embeddings}
>>> output = pipe(text, forward_params=forward_params)
>>> output
{'audio': array([-6.82714235e-05, -4.26525949e-04,  1.06134125e-04, ...,
        -1.22392643e-03, -7.76011671e-04,  3.29112721e-04], dtype=float32),
 'sampling_rate': 16000}
```

ثم يمكنك الاستماع إلى النتيجة:

```py
>>> from IPython.display import Audio
>>> Audio(output['audio'], rate=output['sampling_rate'])
```

### تشغيل الاستدلال يدويًا

يمكنك الوصول إلى نفس نتائج الاستدلال دون استخدام خط المعالجة، لكن سيُطلب عدد أكبر من الخطوات.

حمّل النموذج من 🤗 Hub:

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

اختر مثالًا من مجموعة الاختبار واحصل على تضمين متحدث.

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

عرّف النص المدخل وقم بتجزئته.

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
>>> inputs = processor(text=text, return_tensors="pt")
```

أنشئ مخططًا طيفيًا باستخدام نموذجك:

```py
>>> spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

يمكنك عرض المخطط الطيفي إن رغبت:

```py
>>> plt.figure()
>>> plt.imshow(spectrogram.T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_2.png" alt="Generated log-mel spectrogram"/>
</div>

أخيرًا، استخدم المُفكِّر الصوتي (vocoder) لتحويل المخطط الطيفي إلى صوت.

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

وفق خبرتنا، قد يكون من الصعب الحصول على نتائج مُرضية من هذا النموذج. يبدو أن جودة تضمينات المتحدث عامل مهم. بما أن SpeechT5 تم تدريبه مسبقًا باستخدام x-vectors إنجليزية، فهو يعمل بأفضل شكل عند استخدام تضمينات متحدثين باللغة الإنجليزية. إذا كان الصوت المُركَّب ضعيف الجودة، جرّب تضمين متحدث مختلف.

زيادة مدة التدريب تُحسّن غالبًا جودة النتائج أيضًا. ومع ذلك، يكون الكلام بوضوح باللغة الهولندية بدل الإنجليزية، ويجسّد خصائص صوت المتحدث (قارن مع الصوت الأصلي في المثال).
شيء آخر للتجربة هو ضبط إعدادات النموذج. على سبيل المثال، جرّب استخدام `config.reduction_factor = 1` لترى ما إذا كان ذلك يُحسّن النتائج.

أخيرًا، من الضروري أخذ الجوانب الأخلاقية بالحسبان. على الرغم من أن تقنية TTS لها العديد من التطبيقات المفيدة، إلا أنه قد يُساء استخدامها، مثل انتحال صوت شخص ما دون علمه أو موافقته. يرجى استخدام TTS بحكمة ومسؤولية.
