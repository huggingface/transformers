# Speech2Text2

> **ملاحظة:** هذا النموذج في وضع الصيانة فقط، ولا نقبل أي طلبات سحب (PRs) جديدة لتغيير شفرته. إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2. يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.

## نظرة عامة

يُستخدم نموذج Speech2Text2 مع [Wav2Vec2](wav2vec2) لنماذج ترجمة الكلام المقترحة في ورقة "Large-Scale Self- and Semi-Supervised Learning for Speech Translation" من تأليف Changhan Wang وAnne Wu وJuan Pino وAlexei Baevski وMichael Auli وAlexis Conneau.

Speech2Text2 هو نموذج محول *فقط للترميز* يمكن استخدامه مع أي ترميز صوتي *فقط*، مثل [Wav2Vec2](wav2vec2) أو [HuBERT](hubert) لمهام تحويل الكلام إلى نص. يرجى الرجوع إلى فئة [SpeechEncoderDecoder](speech-encoder-decoder) لمعرفة كيفية دمج Speech2Text2 مع أي نموذج ترميز صوتي *فقط*.

تمت المساهمة بهذا النموذج من قبل [Patrick von Platen](https://huggingface.co/patrickvonplaten).

يمكن العثور على الكود الأصلي [هنا](https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/models/wav2vec/wav2vec2_asr.py#L266).

## نصائح الاستخدام

- يحقق Speech2Text2 نتائج متقدمة على مجموعة بيانات CoVoST Speech Translation dataset. لمزيد من المعلومات، راجع [النماذج الرسمية](https://huggingface.co/models?other=speech2text2).
- يتم استخدام Speech2Text2 دائمًا ضمن إطار عمل [SpeechEncoderDecoder](speech-encoder-decoder).
- يعتمد مُرمز Speech2Text2 على [fastBPE](https://github.com/glample/fastBPE).

## الاستنتاج

يقبل نموذج [`SpeechEncoderDecoderModel`] في Speech2Text2 قيم الإدخال الموجي الخام من الكلام ويستخدم [`~generation.GenerationMixin.generate`] لترجمة الكلام المدخل بشكل تلقائي إلى اللغة المستهدفة.

تكون فئة [`Wav2Vec2FeatureExtractor`] مسؤولة عن معالجة إدخال الكلام، ويقوم [`Speech2Text2Tokenizer`] بفك تشفير الرموز المولدة المستهدفة إلى سلسلة مستهدفة. وتعمل فئة [`Speech2Text2Processor`] على لف [`Wav2Vec2FeatureExtractor`] و [`Speech2Text2Tokenizer`] في مثيل واحد لاستخراج ميزات الإدخال وفك تشفير معرّفات الرموز المتوقعة.

- ترجمة الكلام خطوة بخطوة

```python
>>> import torch
>>> from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
>>> processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
>>> generated_ids = model.generate(inputs=inputs["input_values"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids)
```

- ترجمة الكلام باستخدام خطوط الأنابيب

يمكن أيضًا استخدام خط أنابيب التعرف التلقائي على الكلام لترجمة الكلام في بضع أسطر من التعليمات البرمجية

```python
>>> from datasets import load_dataset
>>> from transformers import pipeline

>>> librispeech_en = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> asr = pipeline(
...     "automatic-speech-recognition",
...     model="facebook/s2t-wav2vec2-large-en-de",
...     feature_extractor="facebook/s2t-wav2vec2-large-en-de",
... )

>>> translation_de = asr(librispeech_en[0]["file"])
```

راجع [Model Hub](https://huggingface.co/models?filter=speech2text2) للبحث عن نقاط تفتيش Speech2Text2.

## الموارد

- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)

## Speech2Text2Config

[[autodoc]] Speech2Text2Config

## Speech2TextTokenizer

[[autodoc]] Speech2Text2Tokenizer

- batch_decode
- decode
- save_vocabulary

## Speech2Text2Processor

[[autodoc]] Speech2Text2Processor

- __call__
- from_pretrained
- save_pretrained
- batch_decode
- decode

## Speech2Text2ForCausalLM

[[autodoc]] Speech2Text2ForCausalLM

- forward