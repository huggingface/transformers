هذا هو نص الترجمة وفقاً للتعليمات المحددة: 

# UnivNet

## نظرة عامة

اقتُرح نموذج UnivNet في ورقة بحثية بعنوان "UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation" من قبل Won Jang و Dan Lim و Jaesam Yoon و Bongwan Kin و Juntae Kim.

نموذج UnivNet هو شبكة تنافسية توليدية (GAN) مدربة على تركيب موجات صوتية عالية الدقة. نموذج UnivNet المتوفر في مكتبة `transformers` هو الـ *مولد*، والذي يقوم برسم خريطة لطيف mel الشرطي وسلسلة الضوضاء الاختيارية إلى موجة صوتية (مثل vocoder). المولد فقط هو المطلوب للاستنتاج. الـ *المميز* المستخدم لتدريب الـ `generator` غير منفذ.

الملخص من الورقة هو كما يلي:

*تستخدم معظم برامج تركيب الكلام العصبية طيف mel المحدود نطاقًا لتركيب الموجات الصوتية. إذا تم استخدام ميزات الطيف الكامل كمدخلات، فيمكن تزويد برنامج تركيب الكلام بأكبر قدر ممكن من المعلومات الصوتية. ومع ذلك، في بعض النماذج التي تستخدم طيف mel الكامل، تحدث مشكلة الإفراط في التسطيح كجزء منها يتم توليد طيف غير حاد. لمعالجة هذه المشكلة، نقترح UnivNet، وهو برنامج تركيب الكلام العصبي الذي يقوم بتركيب الموجات الصوتية عالية الدقة في الوقت الفعلي. واستلهامًا من الأعمال في مجال اكتشاف نشاط الصوت، أضفنا مميز طيف متعدد الدقة يستخدم عدة قيم مطلقة للطيف الخطي محسوبة باستخدام مجموعات معلمات مختلفة. باستخدام طيف mel الكامل كإدخال، نتوقع توليد إشارات عالية الدقة عن طريق إضافة مميز يستخدم طيفًا بدقات متعددة كإدخال. في تقييم لمجموعة بيانات تحتوي على معلومات حول مئات المتحدثين، حصل UnivNet على أفضل النتائج الموضوعية والذاتية بين النماذج المتنافسة لكل من المتحدثين المعروفين وغير المعروفين. توضح هذه النتائج، بما في ذلك أفضل درجة ذاتية لتركيب الكلام، إمكانية التكيف السريع مع المتحدثين الجدد دون الحاجة إلى التدريب من الصفر.*

نصائح:

- يجب أن يكون وسيط `noise_sequence` في [`UnivNetModel.forward`] ضوضاء غاوسية قياسية (مثل تلك التي تم الحصول عليها من `torch.randn`) بشكل `([batch_size], noise_length, model.config.model_in_channels)`، حيث يجب أن يتطابق `noise_length` مع البعد الطولي (البعد 1) لوسيط `input_features`. إذا لم يتم توفيره، فسيتم توليده بشكل عشوائي؛ يمكن تزويد مولد `torch` إلى وسيط `generator` بحيث يمكن إعادة إنتاج تمرير الإرسال. (ملاحظة: سيقوم [`UnivNetFeatureExtractor`] بشكل افتراضي بإرجاع ضوضاء تم إنشاؤها، لذا لا ينبغي أن يكون من الضروري توليد `noise_sequence` يدويًا.)

- يمكن إزالة الحشو الذي أضافه [`UnivNetFeatureExtractor`] من إخراج [`UnivNetModel`] من خلال طريقة [`UnivNetFeatureExtractor.batch_decode`]، كما هو موضح في مثال الاستخدام أدناه.

- يمكن أن يؤدي حشو نهاية كل موجة صوتية بالصمت إلى تقليل التشوهات في نهاية عينة الصوت المولدة. يمكن القيام بذلك عن طريق تزويد `pad_end = True` إلى [`UnivNetFeatureExtractor.__call__`]. راجع [هذه المشكلة](https://github.com/seungwonpark/melgan/issues/8) لمزيد من التفاصيل.

مثال الاستخدام:

```python
import torch
from scipy.io.wavfile import write
from datasets import Audio, load_dataset

from transformers import UnivNetFeatureExtractor, UnivNetModel

model_id_or_path = "dg845/univnet-dev"
model = UnivNetModel.from_pretrained(model_id_or_path)
feature_extractor = UnivNetFeatureExtractor.from_pretrained(model_id_or_path)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# Resample the audio to the model and feature extractor's sampling rate.
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
# Pad the end of the converted waveforms to reduce artifacts at the end of the output audio samples.
inputs = feature_extractor(
ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], pad_end=True, return_tensors="pt"
)

with torch.no_grad():
audio = model(**inputs)

# Remove the extra padding at the end of the output.
audio = feature_extractor.batch_decode(**audio)[0]
# Convert to wav file
write("sample_audio.wav", feature_extractor.sampling_rate, audio)
```

تمت المساهمة بهذا النموذج من قبل [dg845](https://huggingface.co/dg845).

على حد علمي، لا يوجد إصدار رسمي للرمز، ولكن يمكن العثور على تنفيذ غير رسمي في [maum-ai/univnet](https://github.com/maum-ai/univnet) مع نقاط تفتيش مسبقة التدريب [هنا](https://github.com/maum-ai/univnet#pre-trained-model).

## UnivNetConfig

[[autodoc]] UnivNetConfig

## UnivNetFeatureExtractor

[[autodoc]] UnivNetFeatureExtractor

- __call__

## UnivNetModel

[[autodoc]] UnivNetModel

- forward