# MusicGen

## نظرة عامة

اقتُرح نموذج MusicGen في الورقة البحثية [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) من قبل Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi و Alexandre Défossez.

MusicGen هو نموذج تحويل أحادي المرحلة ذاتي الرجعية قادر على توليد عينات موسيقية عالية الجودة مشروطة بوصف نصي أو ملف صوتي. يتم تمرير الأوصاف النصية عبر نموذج مشفر نص مجمد للحصول على تسلسل من تمثيلات الحالة المخفية. ثم يتم تدريب MusicGen للتنبؤ برموز صوتية منفصلة، أو "رموز صوتية"، مشروطة بهذه الحالات المخفية. بعد ذلك، يتم فك تشفير هذه الرموز الصوتية باستخدام نموذج ضغط صوتي، مثل EnCodec، لاستعادة الموجة الصوتية.

من خلال نمط تشابك الرموز بكفاءة، لا يتطلب MusicGen تمثيلًا دلاليًا ذاتي الإشراف للنصوص/الملفات الصوتية، مما يؤدي إلى القضاء على الحاجة إلى تعشيق عدة نماذج للتنبؤ بمجموعة من كتب الرموز (على سبيل المثال، بشكل هرمي أو عن طريق تكبير الحجم). بدلاً من ذلك، فهو قادر على توليد جميع كتب الرموز في عملية تكرار واحدة.

الملخص من الورقة هو كما يلي:

*نحن نتناول مهمة توليد الموسيقى الشرطية. نقدم MusicGen، وهو نموذج لغة واحد يعمل على عدة تيارات من التمثيل الموسيقي المنفصل المضغوط، أي الرموز. على عكس الأعمال السابقة، يتكون MusicGen من نموذج لغة محول أحادي المرحلة جنبًا إلى جنب مع أنماط تشابك الرموز الفعالة، والتي تقضي على الحاجة إلى تعشيق عدة نماذج، على سبيل المثال، بشكل هرمي أو عن طريق تكبير الحجم. وفقًا لهذا النهج، نوضح كيف يمكن لـ MusicGen توليد عينات عالية الجودة، مع الشرطية على الوصف النصي أو الميزات اللحنية، مما يسمح بمزيد من التحكم في الإخراج المولد. نجري تقييمًا تجريبيًا واسع النطاق، مع مراعاة الدراسات التلقائية والبشرية على حد سواء، مما يُظهر أن النهج المقترح متفوق على خطوط الأساس المقيمة على معيار قياسي للنص إلى الموسيقى. من خلال دراسات التحليل التلوي، نسلط الضوء على أهمية كل مكون من المكونات التي يتكون منها MusicGen.*

تمت المساهمة بهذا النموذج من قبل [sanchit-gandhi](https://huggingface.co/sanchit-gandhi). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/audiocraft). يمكن العثور على نقاط التحقق المسبقة على [Hugging Face Hub](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen-).

## نصائح الاستخدام

- بعد تنزيل نقاط التحقق الأصلية من [هنا](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#importing--exporting-models)، يمكنك تحويلها باستخدام **سكريبت التحويل** المتاح في `src/transformers/models/musicgen/convert_musicgen_transformers.py` باستخدام الأمر التالي:

```bash
python src/transformers/models/musicgen/convert_musicgen_transformers.py \
--checkpoint small --pytorch_dump_folder /output/path --safe_serialization
```

## التوليد

MusicGen متوافق مع وضعي التوليد: الجشع والعينات. في الممارسة العملية، يؤدي أخذ العينات إلى نتائج أفضل بكثير من الجشع، لذلك نشجع على استخدام وضع العينات حيثما أمكن. يتم تمكين العينات بشكل افتراضي، ويمكن تحديدها بشكل صريح عن طريق تعيين `do_sample=True` في المكالمة إلى [`MusicgenForConditionalGeneration.generate`], أو عن طريق تجاوز تكوين التوليد للنموذج (انظر أدناه).

يقتصر التوليد بواسطة الترميز الموضعي الجيبي على مدخلات 30 ثانية. وهذا يعني أن MusicGen لا يمكنه توليد أكثر من 30 ثانية من الصوت (1503 رموز)، وتساهم ميزة التوليد الصوتي المدفوعة بالصوت في هذا الحد لذا، نظرًا لمدخل صوتي مدته 20 ثانية، لا يمكن لـ MusicGen توليد أكثر من 10 ثوانٍ إضافية من الصوت.

تدعم Transformers كلا الإصدارات الأحادية (قناة واحدة) والستيريو (قناتان) من MusicGen. تقوم إصدارات القناة الفردية بتوليد مجموعة واحدة من كتب الرموز. تقوم الإصدارات الستيريو بتوليد مجموعتين من كتب الرموز، واحدة لكل قناة (يسار/يمين)، ويتم فك تشفير كل مجموعة من كتب الرموز بشكل مستقل من خلال نموذج ضغط الصوت. يتم دمج تيارات الصوت لكل قناة لإعطاء الإخراج الستيريو النهائي.

### التوليد غير المشروط

يمكن الحصول على المدخلات للتوليد غير المشروط (أو 'null') من خلال طريقة [`MusicgenForConditionalGeneration.get_unconditional_inputs`]:

```python
>>> from transformers import MusicgenForConditionalGeneration

>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
>>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)

>>> audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```

الإخراج الصوتي عبارة عن مصفوفة Torch ثلاثية الأبعاد ذات شكل `(batch_size، num_channels، sequence_length)`. للاستماع إلى عينات الصوت المولدة، يمكنك تشغيلها في دفتر ملاحظات ipynb:

```python
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```

أو حفظها كملف `.wav` باستخدام مكتبة تابعة لجهة خارجية، على سبيل المثال `scipy`:

```python
>>> import scipy

>>> sampling_rate = model.config.audio_encoder.sampling_rate
>>> scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

### التوليد النصي المشروط

يمكن للنموذج توليد عينة صوتية مشروطة بملف نصي من خلال استخدام [`MusicgenProcessor`] لمعالجة المدخلات:

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> inputs = processor(
...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

يتم استخدام `guidance_scale` في التوجيه المجاني للتصنيف (CFG)، حيث يحدد وزن اللوغاريتمات الشرطية (التي يتم التنبؤ بها من موجهات النص) واللوغاريتمات غير المشروطة (التي يتم التنبؤ بها من موجه أو 'null'). يؤدي ارتفاع مقياس التوجيه إلى تشجيع النموذج على توليد عينات مرتبطة ارتباطًا وثيقًا بملف الإدخال المطروح، عادةً على حساب جودة الصوت الرديئة. يتم تمكين CFG عن طريق تعيين `guidance_scale > 1`. للحصول على أفضل النتائج، استخدم `guidance_scale=3` (افتراضي).

### التوليد المدفوع بالصوت

يمكن استخدام نفس [`MusicgenProcessor`] لمعالجة ملف صوتي يستخدم كملف صوتي للاستمرار. في المثال التالي، نقوم بتحميل ملف صوتي باستخدام مكتبة مجموعات البيانات 🤗، والتي يمكن تثبيتها عبر الأمر أدناه:

```bash
pip install --upgrade pip
pip install datasets[audio]
```

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]

>>> # take the first half of the audio sample
>>> sample["array"] = sample["array"][: len(sample["array"]) // 2]

>>> inputs = processor(
...     audio=sample["array"],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```

بالنسبة للتوليد الصوتي المدفوع بالصوت، يمكن معالجة القيم الصوتية المولدة لإزالة الحشو باستخدام فئة [`MusicgenProcessor`]:

```python
>>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]

>>> # take the first quarter of the audio sample
>>> sample_1 = sample["array"][: len(sample["array"]) // 4]

>>> # take the first half of the audio sample
>>> sample_2 = sample["array"][: len(sample["array"]) // 2]

>>> inputs = processor(
...     audio=[sample_1, sample_2],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)

>>> # post-process to remove padding from the batched audio
>>> audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
```

### تكوين التوليد

يمكن العثور على المعلمات الافتراضية التي تتحكم في عملية التوليد، مثل العينات ومقياس التوجيه وعدد الرموز المولدة، في تكوين التوليد للنموذج، وتحديثها حسب الرغبة:

```python
>>> from transformers import MusicgenForConditionalGeneration

>>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

>>> # inspect the default generation config
>>> model.generation_config

>>> # increase the guidance scale to 4.0
>>> model.generation_config.guidance_scale = 4.0

>>> # decrease the max length to 256 tokens
>>> model.generation_config.max_length = 256
```

لاحظ أن أي حجج يتم تمريرها إلى طريقة التوليد سوف **تحل محل** تلك الموجودة في تكوين التوليد، لذا فإن تعيين `do_sample=False` في مكالمة التوليد سوف يحل محل إعداد `model.generation_config.do_sample` في تكوين التوليد.

## هيكل النموذج

يمكن تقسيم نموذج MusicGen إلى ثلاث مراحل مميزة:

1. مشفر النص: يقوم بتعيين إدخالات النص إلى تسلسل من تمثيلات الحالة المخفية. تستخدم نماذج MusicGen المسبقة التدريب مشفر نص مجمد إما من T5 أو Flan-T5
2. فك تشفير MusicGen: نموذج لغة (LM) يقوم بتوليد رموز صوتية تلقائيًا مشروطًا بتمثيلات الحالة المخفية للمشفر
3. مشفر/فك تشفير الصوت: يستخدم لتشفير ملف صوتي لاستخدامه كرموز موجهة، واستعادة الموجة الصوتية من الرموز الصوتية التي يتنبأ بها فك التشفير  
وهكذا، يمكن استخدام نموذج MusicGen كنموذج فك تشفير مستقل، وهو ما يقابله فئة [`MusicgenForCausalLM`], أو كنموذج مركب يتضمن مشفر النص ومشفر/فك تشفير الصوت، وهو ما يقابله فئة [`MusicgenForConditionalGeneration`]. إذا كانت هناك حاجة فقط لتحميل فك التشفير من نقطة التحقق المسبقة التدريب، فيمكن تحميله أولاً عن طريق تحديد التكوين الصحيح، أو يمكن الوصول إليه من خلال السمة `.decoder` للنموذج المركب:

```python
>>> from transformers import AutoConfig, MusicgenForCausalLM, MusicgenForConditionalGeneration

>>> # Option 1: get decoder config and pass to `.from_pretrained`
>>> decoder_config = AutoConfig.from_pretrained("facebook/musicgen-small").decoder
>>> decoder = MusicgenForCausalLM.from_pretrained("facebook/musicgen-small", **decoder_config)

>>> # Option 2: load the entire composite model, but only return the decoder
>>> decoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").decoder
```

نظرًا لأن مشفر النص ومشفر/فك تشفير الصوت مجمدان أثناء التدريب، يمكن تدريب فك تشفير MusicGen [`MusicgenForCausalLM`] بشكل مستقل على مجموعة بيانات من حالات المشفر المخفية ورموز الصوت. للاستدلال، يمكن دمج فك التشفير المدرب مع مشفر النص ومشفر/فك تشفير الصوت المجمدين لاسترداد نموذج [`MusicgenForConditionalGeneration`] المركب.

نصائح:

* تم تدريب MusicGen على نقطة التحقق 32 كيلو هرتز من Encodec. يجب التأكد من استخدام إصدار متوافق من نموذج Encodec.
* يميل وضع العينات إلى تقديم نتائج أفضل من الجشع - يمكنك التبديل بين العينات باستخدام المتغير `do_sample` في المكالمة إلى [`MusicgenForConditionalGeneration.generate`]

## MusicgenDecoderConfig

[[autodoc]] MusicgenDecoderConfig

## MusicgenConfig

[[autodoc]] MusicgenConfig

## MusicgenProcessor

[[autodoc]] MusicgenProcessor

## MusicgenModel

[[autodoc]] MusicgenModel

- forward

## MusicgenForCausalLM

[[autodoc]] MusicgenForCausalLM

- forward
## MusicgenForConditionalGeneration 

<!--
The MusicgenForConditionalGeneration module provides a simple interface for generating music using conditional generation. It allows users to specify conditions such as the style, mood, or length of the music they want to create, and then generates music that satisfies those conditions.
-->

تقدم وحدة MusicgenForConditionalGeneration واجهة بسيطة لإنشاء الموسيقى باستخدام الإنشاء الشرطي. تتيح للمستخدمين تحديد شروط مثل الأسلوب أو الحالة المزاجية أو طول الموسيقى التي يريدون إنشاءها، ثم تقوم بإنشاء موسيقى تلبي هذه الشروط.

```python
from musicgen import MusicgenForConditionalGeneration

# إنشاء كائن MusicgenForConditionalGeneration
musicgen = MusicgenForConditionalGeneration()

# إنشاء موسيقى شرطية
conditions = {"style": "كلاسيكي", "mood": "سعيد", "length": "قصير"}
music = musicgen.generate(**conditions)

# عرض الموسيقى المولدة
music.show()
```

في الكود أعلاه، نقوم باستيراد الفئة `MusicgenForConditionalGeneration` من وحدة `musicgen`. بعد ذلك، نقوم بإنشاء كائن من هذه الفئة يسمى `musicgen`.

نحدد الشروط التي نريد أن تتوافق معها الموسيقى التي سنقوم بإنشائها. في هذا المثال، نريد موسيقى كلاسيكية سعيدة وقصيرة. نقوم بتمرير هذه الشروط على شكل وسائط كلمات رئيسية إلى دالة `generate` للكائن `musicgen`، والتي تقوم بإرجاع كائن `Music` الذي يمثل الموسيقى المولدة.

أخيرًا، نستخدم دالة `show` لعرض الموسيقى المولدة. يمكن أيضًا حفظ الموسيقى كملف أو تشغيلها مباشرةً اعتمادًا على المكتبات الصوتية المثبتة.

تتيح فئة `MusicgenForConditionalGeneration` إمكانات قوية لإنشاء الموسيقى بناءً على شروط محددة. يمكن للمستخدمين تجربة شروط مختلفة ودمجها لإنشاء قطع موسيقية فريدة ومثيرة.