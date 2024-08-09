# Wav2Vec2

## نظرة عامة
تم اقتراح نموذج Wav2Vec2 في [wav2vec 2.0: إطار للتعلم الذاتي لتمثيلات الكلام](https://arxiv.org/abs/2006.11477) بواسطة Alexei Baevski وHenry Zhou وAbdelrahman Mohamed وMichael Auli.

الملخص من الورقة هو ما يلي:

> نُظهر للمرة الأولى أن تعلم التمثيلات القوية من صوت الكلام وحده، ثم الضبط الدقيق على الكلام المنقول، يمكن أن يتفوق على أفضل الطرق شبه المُشرف عليها مع كونها أبسط من الناحية المفاهيمية. يقوم Wav2Vec 2.0 بقناع إدخال الكلام في الفضاء الكامن ويحل مهمة التباين المحددة على كمونة من التمثيلات الكامنة التي يتم تعلمها بشكل مشترك. تحقق التجارب التي تستخدم جميع البيانات المُوسومة من Librispeech نسبة خطأ كلمة تبلغ 1.8/3.3% على مجموعات الاختبار النظيفة/الأخرى. عندما يتم تقليل كمية البيانات الموسومة إلى ساعة واحدة، يتفوق Wav2Vec 2.0 على الحالة السابقة للفن في المجموعة الفرعية التي تبلغ 100 ساعة مع استخدام 100 ضعف البيانات الموسومة. لا يزال استخدام 10 دقائق فقط من البيانات الموسومة والتعلم المسبق على 53000 ساعة من البيانات غير الموسومة يحقق نسبة 4.8/8.2% من خطأ الكلمة. يثبت هذا جدوى التعرف على الكلام بكميات محدودة من البيانات الموسومة.

تمت المساهمة بهذا النموذج من قبل [patrickvonplaten](https://huggingface.co/patrickvonplaten).

ملاحظة: أصدرت Meta (FAIR) إصدارًا جديدًا من [Wav2Vec2-BERT 2.0](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) - فهو مُعلم مسبقًا على 4.5 مليون ساعة من الصوت. نوصي بشكل خاص باستخدامه لمهام الضبط الدقيق، على سبيل المثال كما هو موضح في [هذا الدليل](https://huggingface.co/blog/fine-tune-w2v2-bert).

## نصائح الاستخدام

- Wav2Vec2 هو نموذج كلام يقبل مصفوفة عائمة تتوافق مع الشكل الموجي الخام لإشارة الكلام.

- تم تدريب نموذج Wav2Vec2 باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].

## استخدام Flash Attention 2

Flash Attention 2 هو إصدار أسرع وأكثر تحسينًا من النموذج.

### التثبيت

أولاً، تحقق مما إذا كان الأجهزة الخاصة بك متوافقة مع Flash Attention 2. يمكن العثور على أحدث قائمة من الأجهزة المتوافقة في [الوثائق الرسمية](https://github.com/Dao-AILab/flash-attention#installation-and-features). إذا لم يكن الأجهزة الخاص بك متوافقًا مع Flash Attention 2، فيمكنك الاستفادة من تحسينات نواة الاهتمام من خلال دعم Transformer الأفضل المشمولة [أعلاه](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

بعد ذلك، قم بتثبيت أحدث إصدار من Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

### الاستخدام

لتحميل نموذج باستخدام Flash Attention 2، يمكننا تمرير الحجة `attn_implementation="flash_attention_2"` إلى [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). سنقوم أيضًا بتحميل النموذج في نصف الدقة (على سبيل المثال `torch.float16`)، حيث يؤدي ذلك إلى تقليل استخدام الذاكرة وسرعة الاستدلال بشكل كبير مع عدم وجود تدهور تقريبًا في جودة الصوت:

```python
>>> from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
...
```

### تسريع الأداء المتوقع

فيما يلي رسم بياني لتسريع الأداء المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات لنموذج `facebook/wav2vec2-large-960h-lv60-self` وإصدارات flash-attention-2 وsdpa (scale-dot-product-attention). . نعرض متوسط التسريع الذي تم الحصول عليه على تقسيم التحقق من `librispeech_asr` `clean`:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/kamilakesbi/transformers_image_doc/resolve/main/data/Wav2Vec2_speedup.png">
</div>

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها برمز 🌎) لمساعدتك في البدء باستخدام Wav2Vec2. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يثبت المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="audio-classification"/>

- دفتر ملاحظات حول كيفية [الاستفادة من نموذج Wav2Vec2 المعلم مسبقًا لتصنيف المشاعر](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb). 🌎

- [`Wav2Vec2ForCTC`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).

- [دليل مهمة تصنيف الصوت](../tasks/audio_classification)

<PipelineTag pipeline="automatic-speech-recognition"/>

- منشور مدونة حول [تعزيز Wav2Vec2 مع n-grams في 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram).

- منشور مدونة حول كيفية [الضبط الدقيق لـ Wav2Vec2 للتعرف التلقائي على الكلام باللغة الإنجليزية باستخدام 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english).

- منشور مدونة حول [الضبط الدقيق لـ XLS-R للتعرف التلقائي على الكلام متعدد اللغات باستخدام 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2).

- دفتر ملاحظات حول كيفية [إنشاء تعليقات توضيحية من YouTube لأي فيديو عن طريق نسخ صوت الفيديو باستخدام Wav2Vec2](https://colab.research.google.com/github/Muennighoff/ytclipcc/blob/main/wav2vec_youtube_captions.ipynb). 🌎

- [`Wav2Vec2ForCTC`] مدعوم من قبل دفتر ملاحظات حول كيفية [الضبط الدقيق لنموذج التعرف على الكلام باللغة الإنجليزية](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb)، و [كيفية الضبط الدقيق لنموذج التعرف على الكلام بأي لغة](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).

- [دليل مهمة التعرف التلقائي على الكلام](../tasks/asr)

🚀 النشر

- منشور مدونة حول كيفية نشر Wav2Vec2 لـ [التعرف التلقائي على الكلام باستخدام محولات Hugging Face وAmazon SageMaker](https://www.philschmid.de/automatic-speech-recognition-sagemaker).

## Wav2Vec2Config

[[autodoc]] Wav2Vec2Config

## Wav2Vec2CTCTokenizer

[[autodoc]] Wav2Vec2CTCTokenizer

- __call__
- save_vocabulary
- decode
- batch_decode
- set_target_lang

## Wav2Vec2FeatureExtractor

[[autodoc]] Wav2Vec2FeatureExtractor

- __call__

## Wav2Vec2Processor

[[autodoc]] Wav2Vec2Processor

- __call__
- pad
- from_pretrained
- save_pretrained
- batch_decode
- decode

## Wav2Vec2ProcessorWithLM

[[autodoc]] Wav2Vec2ProcessorWithLM

- __call__
- pad
- from_pretrained
- save_pretrained
- batch_decode
- decode

### فك تشفير مقاطع صوتية متعددة

إذا كنت تخطط لفك تشفير دفعات متعددة من المقاطع الصوتية، فيجب عليك النظر في استخدام [`~Wav2Vec2ProcessorWithLM.batch_decode`] وتمرير `multiprocessing.Pool` مثبت.

في حالة عدم ذلك، ستكون سرعة أداء [`~Wav2Vec2ProcessorWithLM.batch_decode`] أبطأ من استدعاء [`~Wav2Vec2ProcessorWithLM.decode`] لكل مقطع صوتي بشكل فردي، حيث يقوم بإنشاء مثيل فرعي جديد لـ `Pool` لكل استدعاء. راجع المثال أدناه:

```python
>>> # دعونا نرى كيفية استخدام بركة يديرها المستخدم للترميز الدفعي للمقاطع الصوتية المتعددة
>>> from multiprocessing import get_context
>>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> # استيراد النموذج ومستخرج الميزات ومحلل الرموز
>>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm").to("cuda")
>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

>>> # تحميل مجموعة بيانات المثال
>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))


>>> def map_to_array(batch):
...     batch["speech"] = batch["audio"]["array"]
...     return batch


>>> # إعداد بيانات الكلام للاستدلال الدفعي
>>> dataset = dataset.map(map_to_array, remove_columns=["audio"])


>>> def map_to_pred(batch, pool):
...     inputs = processor(batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt")
...     inputs = {k: v.to("cuda") for k, v in inputs.items()}

...     with torch.no_grad():
...         logits = model(**inputs).logits

...     transcription = processor.batch_decode(logits.cpu().numpy(), pool).text
...     batch["transcription"] = transcription
...     return batch


>>> # ملاحظة: يجب إنشاء البركة *بعد* `Wav2Vec2ProcessorWithLM`.
>>> # في حالة عدم ذلك، لن تكون لغة النمذجة متاحة للعمليات الفرعية للبركة.
>>> # حدد عدد العمليات وحجم الدفعة بناءً على عدد وحدات المعالجة المركزية المتوفرة وحجم مجموعة البيانات
>>> with get_context("fork").Pool(processes=2) as pool:
...     result = dataset.map(
...         map_to_pred, batched=True, batch_size=2, fn_kwargs={"pool": pool}, remove_columns=["speech"]
...     )

>>> result["transcription"][:2]
['MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL', "NOR IS MISTER COULTER'S MANNER LESS INTERESTING THAN HIS MATTER"]
```

## مخرجات Wav2Vec2 المحددة

[[autodoc]] models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput

[[autodoc]] models.wav2vec2.modeling_wav2vec2.Wav2Vec2BaseModelOutput

[[autodoc]] models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput

[[autodoc]] models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput

[[autodoc]] models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput

<frameworkcontent>

<pt>

## Wav2Vec2Model

[[autodoc]] Wav2Vec2Model

- forward

## Wav2Vec2ForCTC

[[autodoc]] Wav2Vec2ForCTC

- forward
- load_adapter

## Wav2Vec2ForSequenceClassification

[[autodoc]] Wav2Vec2ForSequenceClassification

- forward

## Wav2Vec2ForAudioFrameClassification

[[autodoc]] Wav2Vec2ForAudioFrameClassification

- forward

## Wav2Vec2ForXVector

[[autodoc]] Wav2Vec2ForXVector

- forward

## Wav2Vec2ForPreTraining

[[autodoc]] Wav2Vec2ForPreTraining

- forward

</pt>

<tf>

## TFWav2Vec2Model

[[autodoc]] TFWav2Vec2Model

- call

## TFWav2Vec2ForSequenceClassification

[[autodoc]] TFWav2Vec2ForSequenceClassification

- call

## TFWav2Vec2ForCTC

[[autodoc]] TFWav2Vec2ForCTC

- call

</tf>

<jax>

## FlaxWav2Vec2Model

[[autodoc]] FlaxWav2Vec2Model

- __call__

## FlaxWav2Vec2ForCTC

[[autodoc]] FlaxWav2Vec2ForCTC

- __call__

## FlaxWav2Vec2ForPreTraining

[[autodoc]] FlaxWav2Vec2ForPreTraining

- __call__

</jax>

</frameworkcontent>