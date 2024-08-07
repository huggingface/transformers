# Pop2Piano

## نظرة عامة

اقتُرح نموذج Pop2Piano في ورقة "Pop2Piano: توليد عزف بيانو قائم على موسيقى البوب" بواسطة جونغهو تشوي وكيوغو لي.

يستمتع الكثيرون بعزف البيانو لموسيقى البوب، ولكن توليدها من الموسيقى ليست مهمة سهلة. يتطلب الأمر خبرة كبيرة في العزف على البيانو، بالإضافة إلى معرفة الخصائص والألحان المختلفة للأغنية. مع Pop2Piano، يمكنك توليد عزف بيانو مباشرةً من الموجة الصوتية للأغنية. وهو أول نموذج يقوم بتوليد عزف بيانو مباشرةً من موسيقى البوب دون الحاجة إلى وحدات استخراج اللحن والنغمات.

Pop2Piano هو نموذج محول Encoder-Decoder يعتمد على T5. يتم تحويل الصوت المدخل إلى موجته ثم تمريره إلى المشفر، الذي يحوله إلى تمثيل كامن. يستخدم فك المشفر هذه التمثيلات لتوليد رموز الرموز بطريقة تدريجية. يرتبط كل رمز من رموز الرموز بأحد الأنواع الرمزية الأربعة: الوقت، والسرعة، والنوتة، و"خاص". ثم يتم فك ترميز رموز الرموز إلى ملف MIDI المكافئ.

ملخص الورقة هو كما يلي:

> "يستمتع الكثيرون بعزف البيانو لموسيقى البوب. ومع ذلك، لا تزال مهمة توليد عزف بيانو لموسيقى البوب تلقائيًا غير مدروسة بشكل كافٍ. ويرجع ذلك جزئيًا إلى نقص أزواج البيانات المتزامنة {Pop, Piano Cover}، والتي جعلت من الصعب تطبيق أحدث الطرق القائمة على التعلم العميق المكثف للبيانات. وللاستفادة من قوة النهج القائم على البيانات، نقوم بإنشاء كمية كبيرة من أزواج البيانات المتزامنة {Pop, Piano Cover} باستخدام خط أنابيب مؤتمت. في هذه الورقة، نقدم Pop2Piano، وهي شبكة محول تقوم بتوليد عزف بيانو بناءً على موجات صوتية لموسيقى البوب. على حد علمنا، هذا هو أول نموذج يقوم بتوليد عزف بيانو مباشرةً من صوت البوب دون استخدام وحدات استخراج اللحن والنغمات. ونظهر أن Pop2Piano، الذي تم تدريبه على مجموعتنا من البيانات، قادر على إنتاج عزف بيانو معقول."

تمت المساهمة بهذا النموذج بواسطة سوسناتو دار. يمكن العثور على الكود الأصلي هنا.

## نصائح الاستخدام

- لاستخدام Pop2Piano، ستحتاج إلى تثبيت مكتبة 🤗 Transformers، بالإضافة إلى وحدات الطرف الثالث التالية:

```bash
pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy
```

يرجى ملاحظة أنك قد تحتاج إلى إعادة تشغيل وقت التشغيل بعد التثبيت.

- Pop2Piano هو نموذج Encoder-Decoder مثل T5.

- يمكن استخدام Pop2Piano لتوليد ملفات صوتية MIDI لتسلسل صوتي معين.

- يمكن أن يؤدي اختيار ملحنين مختلفين في `Pop2PianoForConditionalGeneration.generate()` إلى نتائج متنوعة.

- يمكن أن يؤدي تعيين معدل أخذ العينات إلى 44.1 كيلو هرتز عند تحميل ملف الصوت إلى أداء جيد.

- على الرغم من أن Pop2Piano تم تدريبه بشكل أساسي على موسيقى البوب الكورية، إلا أنه يعمل بشكل جيد أيضًا على أغاني البوب أو الهيب هوب الغربية.

## أمثلة

- مثال باستخدام مجموعة بيانات HuggingFace:

```python
>>> from datasets import load_dataset
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
>>> ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

>>> inputs = processor(
...     audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
... )
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- مثال باستخدام ملف الصوت الخاص بك:

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> audio, sr = librosa.load("<your_audio_file_here>", sr=44100)  # يمكنك تغيير sr إلى قيمة مناسبة.
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
>>> model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"][0]
>>> tokenizer_output.write("./Outputs/midi_output.mid")
```

- مثال على معالجة ملفات صوتية متعددة في دفعة واحدة:

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

>>> # يمكنك تغيير sr إلى قيمة مناسبة.
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = processor(audio=[audio1, audio2], sampling_rate=[sr1, sr2], return_attention_mask=True, return_tensors="pt")
>>> # نظرًا لأننا نقوم الآن بتوليد دفعة (ملفين صوتيين)، يجب علينا تمرير attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = processor.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # نظرًا لأن لدينا الآن ملفي MIDI تم توليدهما
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

- مثال على معالجة ملفات صوتية متعددة في دفعة واحدة (باستخدام `Pop2PianoFeatureExtractor` و`Pop2PianoTokenizer`):

```python
>>> import librosa
>>> from transformers import Pop2PianoForConditionalGeneration, Pop2PianoFeatureExtractor, Pop2PianoTokenizer

>>> # يمكنك تغيير sr إلى قيمة مناسبة.
>>> audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)
>>> audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
>>> feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
>>> tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

>>> inputs = feature_extractor(
...     audio=[audio1, audio2],
...     sampling_rate=[sr1, sr2],
...     return_attention_mask=True,
...     return_tensors="pt",
... )
>>> # نظرًا لأننا نقوم الآن بتوليد دفعة (ملفين صوتيين)، يجب علينا تمرير attention_mask
>>> model_output = model.generate(
...     input_features=inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     composer="composer1",
... )
>>> tokenizer_output = tokenizer.batch_decode(
...     token_ids=model_output, feature_extractor_output=inputs
... )["pretty_midi_objects"]

>>> # نظرًا لأن لدينا الآن ملفي MIDI تم توليدهما
>>> tokenizer_output[0].write("./Outputs/midi_output1.mid")
>>> tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

## Pop2PianoConfig

[[autodoc]] Pop2PianoConfig

## Pop2PianoFeatureExtractor

[[autodoc]] Pop2PianoFeatureExtractor

- __call__

## Pop2PianoForConditionalGeneration

[[autodoc]] Pop2PianoForConditionalGeneration

- forward

- generate

## Pop2PianoTokenizer

[[autodoc]] Pop2PianoTokenizer

- __call__

## Pop2PianoProcessor

[[autodoc]] Pop2PianoProcessor

- __call__