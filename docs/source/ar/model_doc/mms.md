# MMS

## نظرة عامة

اقترح نموذج MMS في [توسيع نطاق تقنية الكلام إلى 1000+ لغة](https://arxiv.org/abs/2305.13516) من قبل فينيل براتاب، وأندروز تيجاندرا، وبوين شي، وبادن توماسيلو، وأرون بابو، وسايان كوندو، وعلي الكحكي، وزهاوينج ني، وأبورف فياس، ومريم فاضل زراندي، وأليكسي بايفسكي، ويوسي عدي، وشياوهوي تشانج، ووي-نينج هسو، وأليكسيس كونو، ومايكل أولي.

الملخص من الورقة هو ما يلي:

*إن توسيع نطاق تغطية اللغة لتكنولوجيا الكلام يحمل إمكانية تحسين الوصول إلى المعلومات للعديد من الأشخاص. ومع ذلك، تقتصر تقنية الكلام الحالية على حوالي مائة لغة، وهو جزء صغير من أكثر من 7000 لغة منطوقة في جميع أنحاء العالم. يزيد مشروع الكلام متعدد اللغات بشكل كبير من عدد اللغات المدعومة بمقدار 10-40 مرة، حسب المهمة. المكونات الرئيسية هي مجموعة بيانات جديدة تستند إلى قراءات النصوص الدينية المتاحة للجمهور والاستفادة الفعالة من التعلم الذاتي. قمنا ببناء نماذج Wav2Vec 2.0 مسبقة التدريب تغطي 1406 لغة، ونموذج واحد متعدد اللغات للتعرف التلقائي على الكلام لـ 1107 لغة، ونماذج تركيب الكلام لنفس عدد اللغات، بالإضافة إلى نموذج لتحديد اللغة لـ 4017 لغة. تُظهر التجارب أن نموذج التعرف على الكلام متعدد اللغات لدينا يقلل من معدل خطأ الكلمة في Whisper إلى النصف لأكثر من 54 لغة من معيار FLEURS بينما يتم تدريبه على جزء صغير من البيانات المُوسمة.*

فيما يلي النماذج المختلفة مفتوحة المصدر في مشروع MMS. تم إصدار النماذج والرمز الأصلي [هنا](https://github.com/facebookresearch/fairseq/tree/main/examples/mms). لقد أضفناها إلى إطار عمل المحولات، مما يجعلها أسهل في الاستخدام.

### التعرف التلقائي على الكلام (ASR)

يمكن العثور على نقاط تفتيش نموذج ASR هنا: [mms-1b-fl102](https://huggingface.co/facebook/mms-1b-fl102)، [mms-1b-l1107](https://huggingface.co/facebook/mms-1b-l1107)، [mms-1b-all](https://huggingface.co/facebook/mms-1b-all). للحصول على أفضل دقة، استخدم نموذج `mms-1b-all`.

نصائح:

- تقبل جميع نماذج ASR مصفوفة عائمة تتوافق مع الموجة الصوتية للإشارة الصوتية. يجب معالجة الموجة الصوتية الخام مسبقًا باستخدام [`Wav2Vec2FeatureExtractor`].
- تم تدريب النماذج باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].
- يمكنك تحميل أوزان محول لغة مختلفة للغات مختلفة عبر [`~Wav2Vec2PreTrainedModel.load_adapter`]. تتكون محولات اللغة من حوالي مليوني معلمة فقط ويمكن تحميلها بكفاءة أثناء التنقل عند الحاجة.

#### التحميل

يحمل MMS بشكل افتراضي أوزان المحول للغة الإنجليزية. إذا كنت تريد تحميل أوزان محول لغة أخرى، فتأكد من تحديد `target_lang=<your-chosen-target-lang>` وكذلك `"ignore_mismatched_sizes=True`.

يجب تمرير الكلمة الأساسية `ignore_mismatched_sizes=True` للسماح برأس نموذج اللغة لإعادة الحجم وفقًا لمعجم اللغة المحددة.

وبالمثل، يجب تحميل المعالج بنفس اللغة المستهدفة:

```py
from transformers import Wav2Vec2ForCTC, AutoProcessor

model_id = "facebook/mms-1b-all"
target_lang = "fra"

processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)
model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=target_lang, ignore_mismatched_sizes=True)
```

<Tip>

يمكنك تجاهل تحذير مثل:

```text
لم يتم تهيئة بعض أوزان Wav2Vec2ForCTC من نقطة تفتيش النموذج في facebook/mms-1b-all وهي مهيأة حديثًا لأن الأشكال لم تتطابق:
- lm_head.bias: تم العثور على الشكل torch.Size([154]) في نقطة التفتيش والشكل torch.Size([314]) في النموذج الذي تم تثبيته
- lm_head.weight: تم العثور على الشكل torch.Size([154، 1280]) في نقطة التفتيش والشكل torch.Size([314، 1280]) في النموذج الذي تم تثبيته
يجب عليك على الأرجح تدريب هذا النموذج على مهمة لأسفل لتتمكن من استخدامه للتنبؤات والاستدلال.
```

</Tip>

إذا كنت تريد استخدام خط أنابيب ASR، فيمكنك تحميل لغة مستهدفة التي اخترتها على النحو التالي:

```py
from transformers import pipeline

model_id = "facebook/mms-1b-all"
target_lang = "fra"

pipe = pipeline(model=model_id, model_kwargs={"target_lang": "fra", "ignore_mismatched_sizes": True})
```

#### الاستدلال

بعد ذلك، دعونا نلقي نظرة على كيفية تشغيل MMS في الاستدلال وتغيير طبقات المحول بعد استدعاء [`~PretrainedModel.from_pretrained`]

أولاً، نقوم بتحميل بيانات الصوت بلغات مختلفة باستخدام [Datasets](https://github.com/huggingface/datasets).

```py
from datasets import load_dataset, Audio

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# French
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "fr", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
fr_sample = next(iter(stream_data))["audio"]["array"]
```

بعد ذلك، نقوم بتحميل النموذج والمعالج:

```py
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
```

الآن نقوم بمعالجة بيانات الصوت، وإرسال بيانات الصوت المعالجة إلى النموذج، ونقوم بنقل إخراج النموذج، تمامًا مثلما نفعل عادةً لـ [`Wav2Vec2ForCTC`].

```py
inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# 'joe keton disapproved of films and buster also had reservations about the media'
```

الآن يمكننا الاحتفاظ بنفس النموذج في الذاكرة والتبديل ببساطة بين محولات اللغة عن طريق

استدعاء وظيفة [`~Wav2Vec2ForCTC.load_adapter`] الملائمة للنموذج و [`~Wav2Vec2CTCTokenizer.set_target_lang`] للمعالج.

نمرر اللغة المستهدفة كإدخال - `"fra"` للفرنسية.

```py
processor.tokenizer.set_target_lang("fra")
model.load_adapter("fra")

inputs = processor(fr_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# "ce dernier est volé tout au long de l'histoire romaine"
```

وبنفس الطريقة، يمكن استبدال اللغة لجميع اللغات الأخرى المدعومة. يرجى الاطلاع على:

```py
processor.tokenizer.vocab.keys()
```

لمعرفة جميع اللغات المدعومة.

للمزيد من تحسين الأداء من نماذج ASR، يمكن استخدام فك تشفير نموذج اللغة. راجع الوثائق [هنا](https://huggingface.co/facebook/mms-1b-all) لمزيد من التفاصيل.

### تركيب الكلام (TTS)

يستخدم MMS-TTS نفس بنية النموذج مثل VITS، والتي تم إضافتها إلى 🤗 المحولات في v4.33. يقوم MMS بتدريب نقطة تفتيش نموذج منفصلة لكل من 1100+ لغة في المشروع. يمكن العثور على جميع نقاط التفتيش المتاحة على Hugging Face Hub: [facebook/mms-tts](https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts)، ووثائق الاستدلال تحت [VITS](https://huggingface.co/docs/transformers/main/en/model_doc/vits).

#### الاستدلال

لاستخدام نموذج MMS، قم أولاً بتحديث أحدث إصدار من مكتبة المحولات:

```bash
pip install --upgrade transformers accelerate
```

نظرًا لأن النموذج القائم على التدفق في VITS غير محدد، فمن الجيد تحديد البذور لضمان إمكانية إعادة إنتاج الإخراج.

- بالنسبة للغات ذات الأبجدية الرومانية، مثل الإنجليزية أو الفرنسية، يمكن استخدام المعالج مباشرةً لمعالجة إدخالات النص. يقوم مثال الكود التالي بتشغيل تمرير إلى الأمام باستخدام نقطة تفتيش MMS-TTS الإنجليزية:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  # make deterministic

with torch.no_grad():
    outputs = model(**inputs)

waveform = outputs.waveform[0]
```

يمكن حفظ الموجة الناتجة كملف `.wav`:

```python
import scipy

scipy.io.wavfile.write("synthesized_speech.wav", rate=model.config.sampling_rate, data=waveform)
```

أو عرضها في Jupyter Notebook / Google Colab:

```python
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

بالنسبة للغات معينة ذات أبجديات غير رومانية، مثل العربية أو الماندرين أو الهندية، تكون حزمة [`uroman`](https://github.com/isi-nlp/uroman)

مطلوب perl لمعالجة إدخالات النص إلى الأبجدية الرومانية.

يمكنك التحقق مما إذا كنت بحاجة إلى حزمة `uroman` للغة الخاصة بك عن طريق فحص سمة `is_uroman`

من المعالج المُدرب مسبقًا:

```python
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```

إذا لزم الأمر، فيجب تطبيق حزمة uroman على إدخالات النص **قبل** تمريرها إلى `VitsTokenizer`،

حيث أن المعالج لا يدعم حاليًا إجراء المعالجة المسبقة بنفسه.

للقيام بذلك، قم أولاً باستنساخ مستودع uroman على جهازك المحلي وقم بتعيين متغير bash `UROMAN` إلى المسار المحلي:

```bash
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

بعد ذلك، يمكنك معالجة إدخال النص باستخدام مقتطف الكود التالي. يمكنك إما الاعتماد على استخدام متغير bash

`UROMAN` للإشارة إلى مستودع uroman، أو يمكنك تمرير دليل uroman كحجة إلى وظيفة `uromaize`:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import subprocess

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor")

def uromanize(input_string, uroman_path):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Execute the perl command
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # Return the output as a string and skip the new-line character at the end
    return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromaized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromaized_text, return_tensors="pt")

set_seed(555)  # make deterministic
with torch.no_grad():
   outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```

**نصائح:**

* تم تدريب نقاط تفتيش MMS-TTS على نص مكتوب بأحرف صغيرة بدون علامات ترقيم. بشكل افتراضي، يقوم `VitsTokenizer` *بتطبيع* الإدخالات عن طريق إزالة أي حالة وعلامات ترقيم، لتجنب تمرير الأحرف غير الموجودة في القاموس إلى النموذج. وبالتالي، فإن النموذج لا يميز بين الحالة وعلامات الترقيم، لذا يجب تجنبها في موجه النص. يمكنك تعطيل التطبيع عن طريق تعيين `normalize=False` في مكالمة المعالج، ولكن هذا سيؤدي إلى سلوك غير متوقع ولا يوصى به.
* يمكن تغيير معدل الكلام عن طريق تعيين سمة `model.speaking_rate` إلى قيمة مختارة. وبالمثل، يتم التحكم في عشوائية الضوضاء بواسطة `model.noise_scale`:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

# make deterministic
set_seed(555)

# make speech faster and more noisy
model.speaking_rate = 1.5
model.noise_scale = 0.8

with torch.no_grad():
    outputs = model(**inputs)
```
لم يتم ترجمة الأجزاء التي طلبت تركها كما هي.

### التعرف على اللغة (LID)
تتوفر نماذج LID مختلفة بناءً على عدد اللغات التي يمكنها التعرف عليها - [126](https://huggingface.co/facebook/mms-lid-126)، [256](https://huggingface.co/facebook/mms-lid-256)، [512](https://huggingface.co/facebook/mms-lid-512)، [1024](https://huggingface.co/facebook/mms-lid-1024)، [2048](https://huggingface.co/facebook/mms-lid-2048)، [4017](https://huggingface.co/facebook/mms-lid-4017).

#### الاستنتاج
أولاً، نقوم بتثبيت مكتبة المحولات وبعض المكتبات الأخرى:

```bash
pip install torch accelerate datasets[audio]
pip install --upgrade transformers
```

بعد ذلك، نقوم بتحميل بعض عينات الصوت عبر `datasets`. تأكد من أن بيانات الصوت تم أخذ عينات منها بمعدل 16000 كيلو هرتز.

```py
from datasets import load_dataset, Audio

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# Arabic
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "ar", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
ar_sample = next(iter(stream_data))["audio"]["array"]
```

بعد ذلك، نقوم بتحميل النموذج والمعالج:

```py
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

model_id = "facebook/mms-lid-126"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
```

الآن نقوم بمعالجة بيانات الصوت، وإرسال بيانات الصوت المعالجة إلى النموذج لتصنيفها إلى لغة، تمامًا كما نفعل عادةً مع نماذج تصنيف الصوت Wav2Vec2 مثل [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition)

```py
# English
inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'eng'

# Arabic
inputs = processor(ar_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

lang_id = torch.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'ara'
```

لعرض جميع اللغات المدعومة لنقطة تفتيش، يمكنك طباعة معرفات اللغة كما يلي:

```py
processor.id2label.values()
```

### النماذج المسبقة التدريب للصوت
تتوفر نماذج مسبقة التدريب لحجمين مختلفين - [300M](https://huggingface.co/facebook/mms-300m)، [1Bil](https://huggingface.co/facebook/mms-1b).

<Tip>

تعتمد بنية MMS for ASR على نموذج Wav2Vec2، راجع [صفحة وثائق Wav2Vec2](wav2vec2) لمزيد من التفاصيل حول كيفية الضبط الدقيق مع النماذج لمختلف المهام الفرعية.

يستخدم MMS-TTS نفس بنية النموذج مثل VITS، راجع [صفحة وثائق VITS](vits) للرجوع إلى API.

</Tip>