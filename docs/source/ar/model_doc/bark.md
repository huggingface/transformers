# Bark

## نظرة عامة

Bark عبارة عن نموذج تحويل نص إلى كلام يعتمد على النص الذي اقترحته Suno AI في [suno-ai/bark](https://github.com/suno-ai/bark).
يتكون Bark من 4 نماذج رئيسية:

- [`BarkSemanticModel`] (يُشار إليه أيضًا باسم نموذج "النص"): نموذج محول ذاتي الانحدار السببي الذي يأخذ كمدخلات نصًا مميزًا، ويتوقع رموز نصية دلالية تلتقط معنى النص.
- [`BarkCoarseModel`] (يُشار إليه أيضًا باسم نموذج "الصوتيات الخشنة"): محول ذاتي الانحدار السببي، يأخذ كمدخلات نتائج نموذج [`BarkSemanticModel`]. ويهدف إلى التنبؤ بأول كتابين صوتيين ضروريين لـ EnCodec.
- [`BarkFineModel`] (نموذج "الصوتيات الدقيقة")، هذه المرة محول تشفير ذاتي غير سببي، والذي يتنبأ بشكل تكراري بكتب التعليمات البرمجية الأخيرة بناءً على مجموع تضمينات كتب التعليمات البرمجية السابقة.
- بعد التنبؤ بجميع قنوات كتاب التعليمات البرمجية من [`EncodecModel`]، يستخدم Bark لترميز صفيف الإخراج الصوتي.

تجدر الإشارة إلى أن كل وحدة من الوحدات النمطية الثلاث الأولى يمكن أن تدعم تضمينات المتحدث الشرطية لشرط إخراج الصوت وفقًا لصوت محدد مسبقًا.

تمت المساهمة بهذا النموذج من قبل [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe) و [Sanchit Gandhi (sanchit-gandhi)](https://github.com/sanchit-gandhi).
يمكن العثور على الكود الأصلي [هنا](https://github.com/suno-ai/bark).

### تحسين Bark

يمكن تحسين Bark ببضع أسطر إضافية من التعليمات البرمجية، والتي **تخفض بشكل كبير من بصمة ذاكرته** و**تسرع الاستدلال**.

#### استخدام نصف الدقة

يمكنك تسريع الاستدلال وتقليل استخدام الذاكرة بنسبة 50% ببساطة عن طريق تحميل النموذج بنصف الدقة.

```python
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
```

#### استخدام تفريغ وحدة المعالجة المركزية

كما ذكرنا سابقًا، يتكون Bark من 4 نماذج فرعية، يتم استدعاؤها بالتتابع أثناء إنشاء الصوت. وبعبارة أخرى، أثناء استخدام نموذج فرعي واحد، تكون النماذج الفرعية الأخرى خاملة.

إذا كنت تستخدم جهاز CUDA، فإن الحل البسيط للاستفادة من انخفاض بنسبة 80% في بصمة الذاكرة هو تفريغ النماذج الفرعية من وحدة معالجة الرسومات (GPU) إلى وحدة المعالجة المركزية (CPU) عندما تكون خاملة. يُطلق على هذه العملية اسم *تفريغ وحدة المعالجة المركزية*. يمكنك استخدامه بأسطر برمجية واحدة على النحو التالي:

```python
model.enable_cpu_offload()
```

لاحظ أنه يجب تثبيت 🤗 Accelerate قبل استخدام هذه الميزة. [هنا كيفية تثبيته.](Https://huggingface.co/docs/accelerate/basic_tutorials/install)

#### استخدام محول أفضل

محول أفضل هو ميزة 🤗 Optimum التي تقوم بدمج النواة تحت الغطاء. يمكنك تحقيق مكاسب تتراوح بين 20% و 30% في السرعة دون أي تدهور في الأداء. فهو يتطلب سطر برمجية واحد فقط لتصدير النموذج إلى محول أفضل:

```python
model = model.to_bettertransformer()
```

لاحظ أنه يجب تثبيت 🤗 Optimum قبل استخدام هذه الميزة. [هنا كيفية تثبيته.](Https://huggingface.co/docs/optimum/installation)

#### استخدام Flash Attention 2

Flash Attention 2 هو إصدار محسّن أسرع من التحسين السابق.

##### التثبيت

أولاً، تحقق مما إذا كان الأجهزة الخاصة بك متوافقة مع Flash Attention 2. يمكن العثور على أحدث قائمة من الأجهزة المتوافقة في [الوثائق الرسمية](https://github.com/Dao-AILab/flash-attention#installation-and-features). إذا لم يكن الأجهزة الخاص بك متوافقًا مع Flash Attention 2، فيمكنك الاستفادة من تحسين نواة الاهتمام من خلال دعم محول أفضل المشمول [أعلاه](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

بعد ذلك، قم بتثبيت أحدث إصدار من Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

##### الاستخدام

لتحميل نموذج باستخدام Flash Attention 2، يمكننا تمرير `attn_implementation="flash_attention_2"` العلم إلى [`.from_pretrained`](Https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). سنقوم أيضًا بتحميل النموذج بنصف الدقة (على سبيل المثال `torch.float16`)، حيث يؤدي ذلك إلى تدهور جودة الصوت بشكل ضئيل جدًا ولكنه يقلل بشكل كبير من استخدام الذاكرة ويُسرع الاستدلال:

```python
model = BarkModel.from_pretrained("suno/bark-small"، torch_dtype=torch.float16، attn_implementation="flash_attention_2").to(device)
```

##### مقارنة الأداء

يوضح الرسم البياني التالي الكمون لتنفيذ الاهتمام الأصلي (بدون تحسين) مقابل محول أفضل وFlash Attention 2. في جميع الحالات، نقوم بتوليد 400 رمزًا دلاليًا على GPU A100 بسعة 40 جيجابايت باستخدام PyTorch 2.1. Flash Attention 2 أسرع أيضًا من محول أفضل، ويتحسن أداؤه بشكل أكبر مع زيادة أحجام الدُفعات:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/Bark%20Optimization%20Benchmark.png">
</div>

لوضع ذلك في المنظور، على NVIDIA A100 وعند إنشاء 400 رمزًا دلاليًا بحجم دفعة يبلغ 16، يمكنك الحصول على 17 ضعف [السرعة](https://huggingface.co/blog/optimizing-bark#throughput) وما زلت أسرع من إنشاء الجمل واحدة تلو الأخرى باستخدام تنفيذ النموذج الأصلي. وبعبارة أخرى، سيتم إنشاء جميع العينات أسرع 17 مرة.

بحجم دفعة يبلغ 8، على NVIDIA A100، Flash Attention 2 أسرع بنسبة 10% من محول أفضل، وبحجم دفعة يبلغ 16، أسرع بنسبة 25%.

#### الجمع بين تقنيات التحسين

يمكنك الجمع بين تقنيات التحسين، واستخدام تفريغ وحدة المعالجة المركزية ونصف الدقة وFlash Attention 2 (أو 🤗 محول أفضل) في وقت واحد.

```python
from transformers import BarkModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# تحميل في fp16 واستخدام Flash Attention 2
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)

# تمكين تفريغ وحدة المعالجة المركزية
model.enable_cpu_offload()
```

اعرف المزيد حول تقنيات تحسين الاستدلال [هنا](https://huggingface.co/docs/transformers/perf_infer_gpu_one).

### نصائح الاستخدام

تقدم Suno مكتبة من إعدادات الصوت بعدد من اللغات [هنا](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c).
تم أيضًا تحميل هذه الإعدادات المسبقة في المركز [هنا](https://huggingface.co/suno/bark-small/tree/main/speaker_embeddings) أو [هنا](https://huggingface.co/suno/bark/tree/main/speaker_embeddings).

```python
>>> from transformers import AutoProcessor, BarkModel

>>> processor = AutoProcessor.from_pretrained("suno/bark")
>>> model = BarkModel.from_pretrained("suno/bark")

>>> voice_preset = "v2/en_speaker_6"

>>> inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

يمكن لـ Bark إنشاء كلام **متعدد اللغات** واقعي للغاية بالإضافة إلى أصوات أخرى - بما في ذلك الموسيقى وضجيج الخلفية والمؤثرات الصوتية البسيطة.

```py
>>> # Multilingual speech - simplified Chinese
>>> inputs = processor("惊人的！我会说中文")

>>> # Multilingual speech - French - let's use a voice_preset as well
>>> inputs = processor("Incroyable! Je peux générer du son.", voice_preset="fr_speaker_5")

>>> # Bark can also generate music. You can help it out by adding music notes around your lyrics.
>>> inputs = processor("♪ Hello, my dog is cute ♪")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

يمكن للنموذج أيضًا إنتاج اتصالات **غير لفظية** مثل الضحك والتنهد والبكاء.

```python
>>> # Adding non-speech cues to the input text
>>> inputs = processor("Hello uh ... [clears throat], my dog is cute [laughter]")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

لحفظ الصوت، ما عليك سوى أخذ معدل العينة من تكوين النموذج وبعض برامج مساعدة SciPy:

```python
>>> from scipy.io.wavfile import write as write_wav

>>> # حفظ الصوت على القرص، ولكن أولاً خذ معدل العينة من تكوين النموذج
>>> sample_rate = model.generation_config.sample_rate
>>> write_wav("bark_generation.wav", sample_rate, audio_array)
```

## BarkConfig

[[autodoc]] BarkConfig
- all

## BarkProcessor

[[autodoc]] BarkProcessor
- all
- __call__

## BarkModel

[[autodoc]] BarkModel
- generate
- enable_cpu_offload

## BarkSemanticModel

[[autodoc]] BarkSemanticModel
- forward

## BarkCoarseModel

[[autodoc]] BarkCoarseModel
- forward

## BarkFineModel

[[autodoc]] BarkFineModel
- forward

## BarkCausalModel

[[autodoc]] BarkCausalModel
- forward

## BarkCoarseConfig

[[autodoc]] BarkCoarseConfig
- all

## BarkFineConfig

[[autodoc]] BarkFineConfig
- all

## BarkSemanticConfig

[[autodoc]] BarkSemanticConfig
- all