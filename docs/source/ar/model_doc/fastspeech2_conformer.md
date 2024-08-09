# FastSpeech2Conformer

## نظرة عامة
تم اقتراح نموذج FastSpeech2Conformer في الورقة البحثية بعنوان "التطورات الحديثة في حزمة Espnet المدعومة بواسطة Conformer" بواسطة Pengcheng Guo وآخرون. ويهدف النموذج إلى تحسين جودة الصوت في أنظمة تركيب الكلام غير المعتمدة على النماذج الذاتية (non-autoregressive text-to-speech)، وذلك من خلال معالجة بعض العيوب الموجودة في النموذج السابق FastSpeech.

يوفر الملخص من الورقة الأصلية لـ FastSpeech2 السياق التالي:

> تواجه نماذج تركيب الكلام غير المعتمدة على النماذج الذاتية مشكلة الخريطة من واحد إلى متعدد، حيث يمكن أن تتوافق عدة متغيرات من الكلام مع نفس النص. وعلى الرغم من أن نموذج FastSpeech يعالج هذه المشكلة من خلال استخدام نموذج معلم ذاتي لتوفير معلومات إضافية، إلا أنه يعاني من بعض العيوب مثل تعقيد وطول عملية التدريب، وعدم دقة كافية في استخراج المدة، وفقدان المعلومات في الميلات المقطعية المقطرة. يقدم FastSpeech 2 تحسينات من خلال التدريب المباشر باستخدام الهدف الحقيقي بدلاً من الإخراج المبسط من المعلم، وإدخال معلومات متغيرة إضافية مثل الطبقة والنطاق ومدة أكثر دقة.

تمت المساهمة في هذا النموذج من قبل كونور هندرسون، ويمكن العثور على الكود الأصلي على مستودع Espnet على GitHub.

## 🤗 تصميم النموذج
تم تنفيذ البنية العامة لـ FastSpeech2 مع فك تشفير Mel-spectrogram، وتم استبدال كتل المحول التقليدية بكتل Conformer كما هو الحال في مكتبة ESPnet.

#### تصميم نموذج FastSpeech2
![تصميم نموذج FastSpeech2](https://www.microsoft.com/en-us/research/uploads/prod/2021/04/fastspeech2-1.png)

#### كتل Conformer
![كتل Conformer](https://www.researchgate.net/profile/Hirofumi-Inaguma-2/publication/344911155/figure/fig2/AS:951455406108673@1603856054097/An-overview-of-Conformer-block.png)

#### وحدة التجزئة
![وحدة التجزئة](https://d3i71xaburhd42.cloudfront.net/8809d0732f6147d4ad9218c8f9b20227c837a746/2-Figure1-1.png)

## 🤗 استخدام مكتبة Transformers
يمكنك تشغيل FastSpeech2Conformer محليًا باستخدام مكتبة 🤗 Transformers. فيما يلي الخطوات:

1. قم بتثبيت مكتبة 🤗 Transformers ومكتبة g2p-en:
```bash
pip install --upgrade pip
pip install --upgrade transformers g2p-en
```

2. قم بتشغيل الاستنتاج باستخدام كود النمذجة في المكتبة مع النموذج وHiFiGan بشكل منفصل:
```python
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import soundfile as sf

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")
input_ids = inputs["input_ids"]

model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
output_dict = model(input_ids, return_dict=True)
spectrogram = output_dict["spectrogram"]

hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
waveform = hifigan(spectrogram)

sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)
```

3. قم بتشغيل الاستنتاج باستخدام كود النمذجة في المكتبة مع النموذج وHiFiGan مجتمعين:
```python
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan
import soundfile as sf

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")
input_ids = inputs["input_ids"]

model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")
output_dict = model(input_ids, return_dict=True)
waveform = output_dict["waveform"]

sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)
```

4. قم بتشغيل الاستنتاج باستخدام خط أنابيب وتحديد برنامج ترميز الصوت الذي تريد استخدامه:
```python
from transformers import pipeline, FastSpeech2ConformerHifiGan
import soundfile as sf

vocoder = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
synthesiser = pipeline(model="espnet/fastspeech2_conformer", vocoder=vocoder)

speech = synthesiser("Hello, my dog is cooler than you!")

sf.write("speech.wav", speech["audio"].squeeze(), samplerate=speech["sampling_rate"])
```

## FastSpeech2ConformerConfig

## FastSpeech2ConformerHifiGanConfig

## FastSpeech2ConformerWithHifiGanConfig

## FastSpeech2ConformerTokenizer

- __call__
- save_vocabulary
- decode
- batch_decode

## FastSpeech2ConformerModel

- forward

## FastSpeech2ConformerHifiGan

- forward

## FastSpeech2ConformerWithHifiGan

- forward