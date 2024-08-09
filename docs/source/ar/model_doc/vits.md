# VITS

## نظرة عامة

اقتُرح نموذج VITS في الورقة البحثية بعنوان "التعلم التمييزي للشبكة التوليدية التباينية المشروطة من أجل تحويل النص إلى كلام من النهاية إلى النهاية" من قبل جايهيون كيم، وجونجيل كونغ، وجوهي سون.

VITS (**V**ariational **I**nference with adversarial learning for end-to-end **T**ext-to-**S**peech) هو نموذج تركيب كلام من النهاية إلى النهاية يتنبأ بموجة صوتية مشروطة بتسلسل نصي مدخل. إنه عبارة عن شبكة توليدية تباينية مشروطة (VAE) تتكون من مشفر لاحق، وفك تشفير، ومتوقع مشروط.

تتم التنبؤ بمجموعة من الميزات الصوتية المستندة إلى المخطط الطيفي بواسطة الوحدة المستندة إلى التدفق، والتي تتكون من مشفر نصي قائم على Transformer وعدة طبقات اقتران. يتم فك تشفير المخطط الطيفي باستخدام مكدس من طبقات التحويل الترانزستور، بنفس أسلوب مشفر HiFi-GAN. بدافع من طبيعة مشكلة تحويل النص إلى كلام من واحد إلى كثير، حيث يمكن نطق نفس الإدخال النصي بطرق متعددة، يتضمن النموذج أيضًا متوقعًا عشوائيًا للمدة، مما يسمح للنموذج بتركيب الكلام بإيقاعات مختلفة من نفس إدخال النص.

يتم تدريب النموذج من النهاية إلى النهاية باستخدام مزيج من الخسائر المستمدة من الحد الأدنى التبايني والتدريب التمييزي. لتحسين تعبيرية النموذج، يتم تطبيق التدفقات العادية على التوزيع السابق المشروط. أثناء الاستدلال، يتم زيادة حجم رموز النص بناءً على وحدة التنبؤ بالمدة، ثم يتم رسمها إلى الموجة باستخدام شلال من وحدة التدفق وفك تشفير HiFi-GAN. نظرًا لطبيعة متوقع المدة العشوائي، فإن النموذج غير محدد، وبالتالي يتطلب بذرة ثابتة لتوليد نفس الموجة الصوتية.

الملخص من الورقة هو كما يلي:

*اقتُرحت مؤخرًا عدة نماذج لتحويل النص إلى كلام من النهاية إلى النهاية والتي تمكّن التدريب أحادي المرحلة والنمذجة المتوازية، ولكن جودة عيناتها لا تضاهي أنظمة تحويل النص إلى كلام ثنائية المراحل. في هذا العمل، نقدم طريقة موازية لتحويل النص إلى كلام من النهاية إلى النهاية تولد صوتًا أكثر طبيعية من النماذج ثنائية المراحل الحالية. تعتمد طريقتنا على الاستدلال التبايني المعزز بالتدفقات العادية وعملية التدريب التمييزي، والتي تحسن القوة التعبيرية للنموذج التوليدي. نقترح أيضًا متوقعًا عشوائيًا للمدة لتركيب الكلام بإيقاعات متنوعة من النص المدخل. مع نمذجة عدم اليقين على المتغيرات الكامنة ومتوقع المدة العشوائي، تعبر طريقتنا عن العلاقة الطبيعية من واحد إلى كثير حيث يمكن نطق إدخال النص بطرق متعددة بإيقاعات ونبرات مختلفة. يظهر التقييم البشري الذاتي (درجة رأي متوسطة، أو MOS) على خطاب LJ، وهو مجموعة بيانات متحدث واحد، أن طريقتنا تتفوق على أفضل أنظمة تحويل النص إلى كلام المتاحة للجمهور وتحقق MOS مماثلة للحقيقة الأرضية.*

يمكن أيضًا استخدام هذا النموذج مع نقاط تفتيش تحويل النص إلى كلام من [الكلام متعدد اللغات بشكل هائل (MMS)](https://arxiv.org/abs/2305.13516) حيث تستخدم هذه النقاط المعمارية نفسها ومشفرًا مماثلًا قليلًا.

تمت المساهمة بهذا النموذج من قبل [ماتثيس](https://huggingface.co/Matthijs) و[سانشيت غاندي](https://huggingface.co/sanchit-gandhi). يمكن العثور على الكود الأصلي [هنا](https://github.com/jaywalnut310/vits).

## أمثلة الاستخدام

يمكن استخدام كل من نقاط تفتيش VITS وMMS-TTS مع نفس واجهة برمجة التطبيقات. نظرًا لأن النموذج المستند إلى التدفق غير محدد، فمن الجيد تحديد بذرة لضمان إمكانية إعادة إنتاج المخرجات. بالنسبة للغات التي تستخدم الأبجدية الرومانية، مثل الإنجليزية أو الفرنسية، يمكن استخدام المشفر مباشرة لمعالجة النص المدخلات. يقوم مثال الكود التالي بتشغيل تمرير إلى الأمام باستخدام نقطة تفتيش الإنجليزية MMS-TTS:

```python
import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555) # make deterministic

with torch.no_grad():
    outputs = model(**inputs)

waveform = outputs.waveform[0]
```

يمكن حفظ الموجة الناتجة كملف `.wav`:

```python
import scipy

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=waveform)
```

أو عرضها في دفتر Jupyter / Google Colab:

```python
from IPython.display import Audio

Audio(waveform, rate=model.config.sampling_rate)
```

بالنسبة للغات معينة ذات أبجدية غير رومانية، مثل العربية أو الماندرين أو الهندية، تكون حزمة Perl [`uroman`](https://github.com/isi-nlp/uroman) مطلوبة لمعالجة النص المدخلات إلى الأبجدية الرومانية.

يمكنك التحقق مما إذا كنت بحاجة إلى حزمة `uroman` للغة الخاصة بك عن طريق فحص سمة `is_uroman` للمشفر المُدرب مسبقًا:

```python
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```

إذا لزم الأمر، فيجب تطبيق حزمة uroman على النص المدخلات **قبل** تمريرها إلى `VitsTokenizer`، حيث لا يدعم المشفر حاليًا إجراء المعالجة المسبقة بنفسه.

للقيام بذلك، قم أولاً باستنساخ مستودع uroman إلى جهازك المحلي وقم بتعيين متغير bash `UROMAN` إلى المسار المحلي:

```bash
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```

بعد ذلك، يمكنك معالجة النص المدخل باستخدام مقتطف الكود التالي. يمكنك إما الاعتماد على استخدام متغير bash `UROMAN` للإشارة إلى مستودع uroman، أو يمكنك تمرير دليل uroman كحجة إلى وظيفة `uromaize`:

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

set_seed(555) # make deterministic
with torch.no_grad():
    outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```

## VitsConfig

[[autodoc]] VitsConfig

## VitsTokenizer

[[autodoc]] VitsTokenizer

- __call__
- save_vocabulary

## VitsModel

[[autodoc]] VitsModel

- forward