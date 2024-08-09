# MusicGen Melody  

## نظرة عامة  

اقترح نموذج MusicGen Melody في ورقة "Simple and Controllable Music Generation" من قبل جيد كوبيت، وفيليكس كروك، وإيتاي جات، وتال ريميز، وديفيد كانت، وغابرييل سينيف، ويوسي عدي، وألكسندر ديفوسيه.  

MusicGen Melody هو نموذج تحويلي أحادي المرحلة يعتمد على التوليد الذاتي، وقادر على توليد عينات موسيقية عالية الجودة بناءً على أوصاف نصية أو إشارات صوتية. تمر الأوصاف النصية عبر نموذج مشفر نصي مجمد للحصول على تسلسل من التمثيلات المخفية. ثم يتم تدريب MusicGen للتنبؤ بالرموز الصوتية المنفصلة، أو "رموز الصوت"، المشروطة بهذه الحالات المخفية. بعد ذلك، يتم فك تشفير هذه الرموز الصوتية باستخدام نموذج ضغط صوتي مثل EnCodec لاستعادة الموجة الصوتية.  

من خلال نمط تشابك الرموز بكفاءة، لا يحتاج MusicGen إلى تمثيل دلالي ذاتي الإشراف للإشارات النصية/الصوتية، مما يلغي الحاجة إلى تسلسل نماذج متعددة للتنبؤ بمجموعة من كتب الرموز (على سبيل المثال، تسلسليًا أو عن طريق زيادة العينات). بدلاً من ذلك، فهو قادر على توليد جميع كتب الرموز في عملية توجيه واحدة.  

الملخص من الورقة هو ما يلي:  

*نحن نتناول مهمة توليد الموسيقى الشرطي. نقدم MusicGen، وهو نموذج لغة واحد يعمل على عدة تيارات من التمثيل الموسيقي المنفصل المضغوط، أي الرموز. على عكس العمل السابق، يتكون MusicGen من نموذج لغة محول أحادي المرحلة جنبًا إلى جنب مع أنماط تشابك الرموز الفعالة، والتي تُلغي الحاجة إلى تسلسل عدة نماذج، على سبيل المثال، تسلسليًا أو عن طريق زيادة العينات. من خلال هذا النهج، نوضح كيف يمكن لـ MusicGen توليد عينات عالية الجودة، مع الشرط على الوصف النصي أو الميزات اللحنية، مما يسمح بضوابط أفضل على الإخراج المولد. نجري تقييمًا تجريبيًا واسع النطاق، مع مراعاة الدراسات التلقائية والبشرية على حد سواء، مما يُظهر أن النهج المقترح متفوق على خطوط الأساس المقيمة على معيار قياسي للنص إلى الموسيقى. من خلال دراسات التحليل التلوي، نسلط الضوء على أهمية كل مكون من المكونات التي يتكون منها MusicGen.*  

تمت المساهمة بهذا النموذج من قبل [ylacombe](https://huggingface.co/ylacombe). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/audiocraft). يمكن العثور على نقاط التحقق المسبقة التدريب على [Hugging Face Hub](https://huggingface.co/models?sort=downloads&search=facebook%2Fmusicgen).  

## الفرق مع [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)  

هناك اختلافان رئيسيان مع MusicGen:  
1. يتم استخدام الإشارة الصوتية هنا كإشارة شرطية للعينة الصوتية المولدة، بينما يتم استخدامها لاستمرار الصوت في [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen).  
2. يتم دمج الإشارات النصية والصوتية الشرطية مع حالات فك التشفير المخفية بدلاً من استخدامها كإشارة اهتمام متقاطع، كما هو الحال في MusicGen.  

## التوليد  

MusicGen Melody متوافق مع وضعي التوليد: الجشع والنمذجة. من الناحية العملية، يؤدي النمذجة إلى نتائج أفضل بكثير من الجشع، لذلك نشجع على استخدام وضع النمذجة حيثما أمكن ذلك. يتم تمكين النمذجة بشكل افتراضي، ويمكن تحديدها بشكل صريح عن طريق تعيين `do_sample=True` في المكالمة إلى [`MusicgenMelodyForConditionalGeneration.generate`]` `، أو عن طريق تجاوز تكوين التوليد للنموذج (انظر أدناه).  

تدعم Transformers كلا الإصدارين أحادي القناة (1-channel) و ستيريو (2-channel) من MusicGen Melody. تقوم إصدارات القناة أحادية القناة بتوليد مجموعة واحدة من كتب الرموز. تقوم الإصدارات الستيريو بتوليد مجموعتين من كتب الرموز، واحدة لكل قناة (يسار/يمين)، ويتم فك تشفير كل مجموعة من كتب الرموز بشكل مستقل من خلال نموذج ضغط الصوت. يتم دمج تيارات الصوت لكل قناة لإعطاء الإخراج الستيريو النهائي.  

#### التوليد الشرطي الصوتي  

يمكن للنموذج توليد عينة صوتية مشروطة بنص وإشارة صوتية باستخدام [`MusicgenMelodyProcessor`] لمعالجة الإدخالات.  

في الأمثلة التالية، نقوم بتحميل ملف صوتي باستخدام مكتبة Datasets 🤗، والتي يمكن تثبيتها عبر الأمر أدناه:  

```
pip install --upgrade pip
pip install datasets[audio]
```  

يتم تحميل ملف الصوت الذي سنستخدمه على النحو التالي:  

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
>>> sample = next(iter(dataset))["audio"]
```  

من الناحية المثالية، يجب أن تكون الإشارة الصوتية خالية من الإشارات ذات التردد المنخفض التي تنتجها عادةً آلات مثل الطبول والغيتار الباص. يمكن استخدام نموذج [Demucs](https://github.com/adefossez/demucs/tree/main) لفصل الأصوات والضوضاء الأخرى عن مكونات الطبول والغيتار الباص.  

إذا كنت ترغب في استخدام Demucs، فيجب عليك أولاً اتباع خطوات التثبيت [هنا](https://github.com/adefossez/demucs/tree/main?tab=readme-ov-file#for-musicians) قبل استخدام المقتطف التالي:  

```python
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
import torch


wav = torch.tensor(sample["array"]).to(torch.float32)

demucs = pretrained.get_model('htdemucs')

wav = convert_audio(wav[None], sample["sampling_rate"], demucs.samplerate, demucs.audio_channels)
wav = apply_model(demucs, wav[None])
```  

بعد ذلك، يمكنك استخدام المقتطف التالي لتوليد الموسيقى:  

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> inputs = processor(
...     audio=wav,
...     sampling_rate=demucs.samplerate,
...     text=["80s blues track with groovy saxophone"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```  

يمكنك أيضًا تمرير الإشارة الصوتية مباشرة دون استخدام Demucs، على الرغم من أن جودة التوليد قد تتدهور:  

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> inputs = processor(
...     audio=sample["array"],
...     sampling_rate=sample["sampling_rate"],
...     text=["80s blues track with groovy saxophone"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```  

الإخراج الصوتي عبارة عن مصفوفة ثلاثية الأبعاد من Tensor ذات الشكل `(batch_size، num_channels، sequence_length)`. للاستماع إلى عينات الصوت المولدة، يمكنك تشغيلها في دفتر ملاحظات ipynb:  

```python
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```  

أو حفظها كملف `.wav` باستخدام مكتبة خارجية، على سبيل المثال `soundfile`:  

```python
>>> import soundfile as sf

>>> sampling_rate = model.config.audio_encoder.sampling_rate
>>> sf.write("musicgen_out.wav", audio_values[0].T.numpy(), sampling_rate)
```  

### التوليد الشرطي النصي فقط  

يمكن استخدام نفس [`MusicgenMelodyProcessor`] لمعالجة إشارة نصية فقط.  

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> inputs = processor(
...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
...     padding=True,
...     return_tensors="pt",
... )
>>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
```  

يتم استخدام `guidance_scale` في التوجيه الحر للتصنيف (CFG)، والذي يحدد وزن اللوغاريتمات الشرطية (التي يتم التنبؤ بها من مpts النصية) واللوغاريتمات غير الشرطية (التي يتم التنبؤ بها من مpt غير مشروط أو 'null'). يؤدي ارتفاع مقياس التوجيه إلى تشجيع النموذج على توليد عينات مرتبطة ارتباطًا أوثق بالإشارة الشرطية، وعادة ما يكون ذلك على حساب جودة الصوت الرديئة. يتم تمكين CFG عن طريق تعيين `guidance_scale > 1`. للحصول على أفضل النتائج، استخدم `guidance_scale=3` (افتراضي).  

يمكنك أيضًا التوليد بالدفعات:  

```python
>>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

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
```  

### التوليد غير الشرطي  

يمكن الحصول على إدخالات التوليد غير الشرطي (أو 'null') من خلال طريقة [`MusicgenMelodyProcessor.get_unconditional_inputs`]:  

```python
>>> from transformers import MusicgenMelodyForConditionalGeneration, MusicgenMelodyProcessor

>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
>>> unconditional_inputs = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody").get_unconditional_inputs(num_samples=1)

>>> audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
```  

### تكوين التوليد  

يمكن العثور على المعلمات الافتراضية التي تتحكم في عملية التوليد، مثل النمذجة، ومقياس التوجيه، وعدد الرموز المولدة، في تكوين التوليد للنموذج، وتحديثها حسب الرغبة:  

```python
>>> from transformers import MusicgenMelodyForConditionalGeneration

>>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

>>> # فحص تكوين التوليد الافتراضي
>>> model.generation_config

>>> # زيادة مقياس التوجيه إلى 4.0
>>> model.generation_config.guidance_scale = 4.0

>>> # تقليل الطول الأقصى إلى 256 رمزًا
>>> model.generation_config.max_length = 256
```  

لاحظ أن أي حجج يتم تمريرها إلى طريقة التوليد ستكون لها الأسبقية على تلك الموجودة في تكوين التوليد، لذا فإن تعيين `do_sample=False` في مكالمة التوليد ستكون لها الأسبقية على إعداد `model.generation_config.do_sample` في تكوين التوليد.
## بنية النموذج

يمكن تفكيك نموذج MusicGen إلى ثلاث مراحل مميزة:

1. **ترميز النص**: يقوم بترميز مدخلات النص إلى تسلسل من التمثيلات المخفية. تستخدم النماذج المُدربة مسبقًا من MusicGen ترميز نص مجمدًا إما من T5 أو Flan-T5.

2. **فك تشفير MusicGen Melody**: نموذج لغة يقوم بتوليد رموز صوتية (أو أكواد) بشكل تلقائي تنازلي مشروط بتمثيلات الحالة المخفية للترميز.

3. **فك تشفير الصوت**: يستخدم لاستعادة الموجات الصوتية من الرموز الصوتية التي يتنبأ بها فك التشفير.

وبالتالي، يمكن استخدام نموذج MusicGen إما كنموذج فك تشفير مستقل، والذي يتوافق مع الفئة [`MusicgenMelodyForCausalLM`]، أو كنموذج مركب يشمل ترميز النص وفك تشفير الصوت، والذي يتوافق مع الفئة [`MusicgenMelodyForConditionalGeneration`]. إذا كان الأمر يتطلب فقط تحميل فك التشفير من نقطة التفتيش المُدربة مسبقًا، فيمكن تحميله أولاً عن طريق تحديد التكوين الصحيح، أو الوصول إليه من خلال سمة `.decoder` للنموذج المركب:

```python
>>> from transformers import AutoConfig, MusicgenMelodyForCausalLM, MusicgenMelodyForConditionalGeneration

>>> # الخيار 1: الحصول على تكوين فك التشفير وإمراره إلى `.from_pretrained`
>>> decoder_config = AutoConfig.from_pretrained("facebook/musicgen-melody").decoder
>>> decoder = MusicgenMelodyForCausalLM.from_pretrained("facebook/musicgen-melody", **decoder_config.to_dict())

>>> # الخيار 2: تحميل النموذج المركب بالكامل، ولكن إرجاع فك التشفير فقط
>>> decoder = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody").decoder
```

نظرًا لأن نماذج ترميز النص وفك تشفير الصوت تكون مجمدة أثناء التدريب، فيمكن تدريب فك تشفير MusicGen [`MusicgenMelodyForCausalLM`] بشكل مستقل على مجموعة بيانات من حالات الترميز المخفية ورموز الصوت. بالنسبة للاستدلال، يمكن دمج فك التشفير المدرب مع ترميز النص المجمد وفك تشفير الصوت لاستعادة النموذج المركب [`MusicgenMelodyForConditionalGeneration`].

## تحويل نقطة التفتيش

بعد تنزيل نقاط التفتيش الأصلية من [هنا](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#importing--exporting-models)، يمكنك تحويلها باستخدام **سكريبت التحويل** المتاح في `src/transformers/models/musicgen_melody/convert_musicgen_melody_transformers.py` باستخدام الأمر التالي:

```bash
python src/transformers/models/musicgen_melody/convert_musicgen_melody_transformers.py \
--checkpoint="facebook/musicgen-melody" --pytorch_dump_folder /output/path
```

نصائح:

- تم تدريب MusicGen على نقطة تفتيش Encodec 32 كيلو هرتز. يجب التأكد من استخدام إصدار متوافق من نموذج Encodec.
- يميل وضع أخذ العينات إلى تقديم نتائج أفضل من الوضع الجشع - يمكنك التبديل بين وضعي أخذ العينات باستخدام المتغير `do_sample` في المكالمة إلى [`MusicgenMelodyForConditionalGeneration.generate`].

## MusicgenMelodyDecoderConfig

[[autodoc]] MusicgenMelodyDecoderConfig

## MusicgenMelodyProcessor

[[autodoc]] MusicgenMelodyProcessor

- get_unconditional_inputs

## MusicgenMelodyFeatureExtractor

[[autodoc]] MusicgenMelodyFeatureExtractor

- _extract_stem_indices

## MusicgenMelodyConfig

[[autodoc]] MusicgenMelodyConfig

## MusicgenMelodyModel

[[autodoc]] MusicgenMelodyModel

- forward

## MusicgenMelodyForCausalLM

[[autodoc]] MusicgenMelodyForCausalLM

- forward

## MusicgenMelodyForConditionalGeneration

[[autodoc]] MusicgenMelodyForConditionalGeneration

- forward