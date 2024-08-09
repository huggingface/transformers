# Speech2Text

## نظرة عامة
تم اقتراح نموذج Speech2Text في [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) بواسطة Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. وهو عبارة عن نموذج تسلسل إلى تسلسل (ترميز فك الترميز) يعتمد على المحول المصمم للتعرف التلقائي على الكلام (ASR) وترجمة الكلام (ST). يستخدم مخفضًا للتعرّف على الشكل لتقصير طول إدخالات الكلام بمقدار 3/4 قبل إدخالها في الترميز. يتم تدريب النموذج باستخدام خسارة ذاتي الارتباط عبر الإنتروبيا القياسية ويولد النسخ/الترجمات ذاتيًا. تم ضبط نموذج Speech2Text بشكل دقيق على العديد من مجموعات البيانات لـ ASR و ST: [LibriSpeech](http://www.openslr.org/12)، [CoVoST 2](https://github.com/facebookresearch/covost)، [MuST-C](https://ict.fbk.eu/must-c/).

تمت المساهمة بهذا النموذج من قبل [valhalla](https://huggingface.co/valhalla). يمكن العثور على الكود الأصلي [هنا](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text).

## الاستنتاج
Speech2Text هو نموذج كلام يقبل Tensor عائمًا لميزات بنك التصفية اللوغاريتمية المستخرجة من إشارة الكلام. إنه نموذج تسلسل إلى تسلسل يعتمد على المحول، لذلك يتم إنشاء النسخ/الترجمات ذاتيًا. يمكن استخدام طريقة `generate()` للاستدلال.

تعد فئة [`Speech2TextFeatureExtractor`] مسؤولة عن استخراج ميزات بنك التصفية اللوغاريتمية. يجمع [`Speech2TextProcessor`] [`Speech2TextFeatureExtractor`] و [`Speech2TextTokenizer`] في مثيل واحد لاستخراج ميزات الإدخال وفك تشفير معرّفات الرموز المتوقعة.

يعتمد المستخرج المميز على `torchaudio` ويعتمد الرمز المميز على `sentencepiece`، لذا تأكد من تثبيت هذه الحزم قبل تشغيل الأمثلة. يمكنك تثبيت تلك الحزم كاعتمادات كلام إضافية مع `pip install transformers"[speech, sentencepiece]"` أو تثبيت الحزم بشكل منفصل باستخدام `pip install torchaudio sentencepiece`. يتطلب `torchaudio` أيضًا الإصدار التنموي من حزمة [libsndfile](http://www.mega-nerd.com/libsndfile/) والتي يمكن تثبيتها عبر مدير حزم النظام. على Ubuntu، يمكن تثبيته على النحو التالي: `apt install libsndfile1-dev`

- التعرف على الكلام وترجمة الكلام

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription
['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
```

- ترجمة الكلام متعددة اللغات

بالنسبة لنماذج ترجمة الكلام متعددة اللغات، يتم استخدام `eos_token_id` كـ `decoder_start_token_id`
ويتم فرض معرف لغة الهدف كرابع رمز مولد. لفرض معرف لغة الهدف كرابع رمز مولد، قم بتمرير معلمة `forced_bos_token_id` إلى طريقة `generate()`. يوضح المثال التالي كيفية ترجمة الكلام باللغة الإنجليزية إلى نص باللغة الفرنسية باستخدام نقطة تفتيش *facebook/s2t-medium-mustc-multilingual-st*.

```python
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(
...     inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
... )

>>> translation = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> translation
["(Vidéo) Si M. Kilder est l'apossible des classes moyennes, et nous sommes heureux d'être accueillis dans son évangile."]
```

راجع [نموذج المركز](https://huggingface.co/models؟filter=speech_to_text) للبحث عن نقاط تفتيش Speech2Text.

## Speech2TextConfig

[[autodoc]] Speech2TextConfig

## Speech2TextTokenizer

[[autodoc]] Speech2TextTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## Speech2TextFeatureExtractor

[[autodoc]] Speech2TextFeatureExtractor

- __call__

## Speech2TextProcessor

[[autodoc]] Speech2TextProcessor

- __call__
- from_pretrained
- save_pretrained
- batch_decode
- decode

<frameworkcontent>
<pt>

## Speech2TextModel

[[autodoc]] Speech2TextModel

- forward

## Speech2TextForConditionalGeneration

[[autodoc]] Speech2TextForConditionalGeneration

- forward

</pt>
<tf>

## TFSpeech2TextModel

[[autodoc]] TFSpeech2TextModel

- call

## TFSpeech2TextForConditionalGeneration

[[autodoc]] TFSpeech2TextForConditionalGeneration

- call

</tf>

</frameworkcontent>