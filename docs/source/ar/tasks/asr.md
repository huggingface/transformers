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

# التعرّف التلقائي على الكلام (Automatic Speech Recognition - ASR)

[[open-in-colab]]

<Youtube id="TksaY_FDgnk"/>

يحوّل التعرّف التلقائي على الكلام (ASR) الإشارة الصوتية إلى نص، وذلك عبر مواءمة تسلسل من مُدخلات الصوت مع مُخرجات نصية. تستخدم المساعدات الصوتية مثل Siri وAlexa نماذج ASR لمساعدة المستخدمين يوميًا، وهناك العديد من تطبيقات المستخدم النهائي المفيدة الأخرى مثل الترجمة النصية الحية وتدوين الملاحظات أثناء الاجتماعات.

سيُرشدك هذا الدليل إلى كيفية:

1. إجراء ضبط دقيق (fine-tuning) لـ [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) على مجموعة البيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) لتحويل الصوت إلى نص.
2. استخدام نموذجك المضبوط للاستخلاص (inference).

<Tip>

لمعرفة جميع البُنى ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالاطلاع على [صفحة المهمة](https://huggingface.co/tasks/automatic-speech-recognition).

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate jiwer
```

نوصيك بتسجيل الدخول إلى حسابك على Hugging Face حتى تتمكن من رفع نموذجك ومشاركته مع المجتمع. عند المطالبة، أدخل الرمز المميز لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة البيانات MInDS-14

ابدأ بتحميل جزء أصغر من مجموعة البيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) من مكتبة 🤗 Datasets. سيمنحك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

قسّم جزء `train` من مجموعة البيانات إلى مجموعة تدريب واختبار باستخدام الطريقة [`~Dataset.train_test_split`]:

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

ثم ألقِ نظرة على مجموعة البيانات:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

بينما تحتوي مجموعة البيانات على الكثير من المعلومات المفيدة مثل `lang_id` و`english_transcription`، يركّز هذا الدليل على حقلي `audio` و`transcription`. أزِل الأعمدة الأخرى باستخدام الطريقة [`~datasets.Dataset.remove_columns`]:

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

راجع المثال مرة أخرى:

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

هناك حقلان:

- `audio`: مصفوفة أحادية البُعد `array` للإشارة الصوتية يجب استدعاؤها لتحميل ملف الصوت وإعادة تشكيل معدل العيّنة.
- `transcription`: النص الهدف.

## المعالجة المسبقة (Preprocess)

الخطوة التالية هي تحميل معالج (processor) Wav2Vec2 لمعالجة الإشارة الصوتية:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

تملك مجموعة بيانات MInDS-14 معدل عيّنة 8000 هرتز (يمكنك العثور على هذه المعلومة في [بطاقة مجموعة البيانات](https://huggingface.co/datasets/PolyAI/minds14))، ما يعني أنك ستحتاج إلى إعادة تشكيلها إلى 16000 هرتز لاستخدام نموذج Wav2Vec2 المُدرّب مسبقًا:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

كما ترى في الحقل `transcription` أعلاه، يحتوي النص على مزيج من الأحرف الكبيرة والصغيرة. تم تدريب المُقسِّم (tokenizer) الخاص بـ Wav2Vec2 على أحرف كبيرة فقط، لذا عليك التأكد من أن النص يطابق مفردات المُقسِّم:

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

أنشئ الآن دالة للمعالجة المسبقة تقوم بما يلي:

1. تستدعي عمود `audio` لتحميل ملف الصوت وإعادة تشكيل معدل العيّنة.
2. تستخرج `input_values` من ملف الصوت وتُجزّئ عمود `transcription` باستخدام المعالج (processor).

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"]) 
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات كاملة، استخدم دالة [`~datasets.Dataset.map`] في 🤗 Datasets. يمكنك تسريع `map` بزيادة عدد العمليات عبر المعامل `num_proc`. أزِل الأعمدة التي لا تحتاجها باستخدام الطريقة [`~datasets.Dataset.remove_columns`]:

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

لا توفّر 🤗 Transformers مُجمِّع بيانات (data collator) مخصّصًا لـ ASR، لذا ستحتاج إلى تكييف [`DataCollatorWithPadding`] لإنشاء دفعة أمثلة (batch). كما سيقوم هذا المجمِّع بحشو (padding) النصوص والتسميات ديناميكيًا إلى طول أطول عنصر داخل الدفعة (بدلًا من مجموعة البيانات كلها) لتكون بطول موحّد. رغم إمكانية تنفيذ الحشو في دالة `tokenizer` عبر `padding=True`، إلا أن الحشو الديناميكي أكثر كفاءة.

على عكس مُجمِّعات البيانات الأخرى، يحتاج هذا المجمِّع تحديدًا إلى تطبيق طريقة حشو مختلفة على `input_values` و`labels`:

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

الآن قم بتهيئة `DataCollatorCTCWithPadding`:

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## التقييم (Evaluate)

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك بسرعة تحميل طريقة تقييم عبر مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، حمّل مقياس [معدل الخطأ في الكلمات (Word Error Rate - WER)](https://huggingface.co/spaces/evaluate-metric/wer) (راجع [الجولة السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) في 🤗 Evaluate لمعرفة المزيد حول التحميل والحساب):

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

ثم أنشئ دالة تمُرّر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب WER:

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

أصبحت دالة `compute_metrics` جاهزة الآن، وسنعود إليها عند إعداد التدريب.

## التدريب (Train)

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن معتادًا على إجراء الضبط الدقيق لنموذج باستخدام [`Trainer`]، ألقِ نظرة على الدليل الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت الآن جاهز لبدء تدريب نموذجك! قم بتحميل Wav2Vec2 باستخدام [`AutoModelForCTC`]. حدّد طريقة الاختزال (reduction) عبر المعامل `ctc_loss_reduction`. غالبًا ما يكون استخدام المتوسط أفضل من الجمع الافتراضي:

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```

في هذه المرحلة، تبقّت ثلاث خطوات فقط:

1. عرّف فرط-معاملات التدريب (hyperparameters) في [`TrainingArguments`]. المعامل الوحيد المطلوب هو `output_dir` الذي يحدد مكان حفظ نموذجك. ستدفع هذا النموذج إلى Hub بتعيين `push_to_hub=True` (تحتاج إلى تسجيل الدخول إلى Hugging Face لرفع نموذجك). في نهاية كل حقبة (epoch)، سيُقيّم [`Trainer`] قيمة WER ويحفظ نقطة التحقق التدريبية.
2. مرّر معاملات التدريب إلى [`Trainer`] مع النموذج ومجموعة البيانات والمعالج (tokenizer/processor) ومجمّع البيانات ودالة `compute_metrics`.
3. استدعِ [`~Trainer.train`] لإجراء الضبط الدقيق لنموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     group_by_length=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=processor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام الطريقة [`~transformers.Trainer.push_to_hub`] ليكون متاحًا للجميع:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

للحصول على مثال أكثر تفصيلًا حول كيفية إجراء الضبط الدقيق لنموذج للتعرّف التلقائي على الكلام، اطّلع على [هذه التدوينة](https://huggingface.co/blog/fine-tune-wav2vec2-english) لـ ASR باللغة الإنجليزية وعلى [هذه التدوينة](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) الخاصة بـ ASR متعدد اللغات.

</Tip>

## الاستدلال (Inference)

رائع! بعد أن أجريت الضبط الدقيق لنموذجك، يمكنك استخدامه الآن للاستدلال.

حمّل ملفًا صوتيًا ترغب بتشغيل الاستدلال عليه. تذكّر إعادة تشكيل معدل العيّنة لملف الصوت ليتوافق مع معدل العيّنة الخاص بالنموذج إذا احتجت لذلك!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

أسهل طريقة لتجربة نموذجك المضبوط من أجل الاستدلال هي استخدامه ضمن [`pipeline`]. قم بإنشاء `pipeline` خاص بالتعرّف التلقائي على الكلام باستخدام نموذجك، ثم مرّر له ملف الصوت:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

<Tip>

الناتج النصي جيد، لكنه قد يكون أفضل! حاول إجراء ضبط دقيق لنموذجك على عدد أكبر من الأمثلة لتحصل على نتائج أفضل.

</Tip>

يمكنك أيضًا إعادة تنفيذ نتائج `pipeline` يدويًا إذا رغبت بذلك:

<frameworkcontent>
<pt>
حمّل معالجًا (processor) لتهيئة ملف الصوت والنص وإرجاع `input` على شكل موترات PyTorch:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

مرّر المُدخلات إلى النموذج واسترجع القيم اللوغارية (logits):

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

احصل على `input_ids` المتنبأ بها ذات الاحتمالية الأعلى، ثم استخدم المعالج (processor) لفك تشفير `input_ids` المتنبأ بها إلى نص:

```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```
</pt>
</frameworkcontent>
