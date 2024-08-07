# DePlot

## نظرة عامة

تم اقتراح DePlot في الورقة البحثية [DePlot: One-shot visual language reasoning by plot-to-table translation](https://arxiv.org/abs/2212.10505) من قبل Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun.

يوضح ملخص الورقة ما يلي:

*يعد اللغة المرئية مثل الرسوم البيانية والمخططات منتشرة في العالم البشري. يتطلب فهم المخططات والرسوم البيانية مهارات استدلالية قوية. تتطلب النماذج السابقة المتقدمة في هذا المجال عشرات الآلاف من أمثلة التدريب على الأقل، ولا تزال قدراتها الاستدلالية محدودة، خاصة عند التعامل مع الاستفسارات المعقدة التي يكتبها البشر. تقدم هذه الورقة أول حل قائم على التعلم لمرة واحدة للاستدلال اللغوي المرئي. نقوم بتفصيل تحدي الاستدلال اللغوي المرئي إلى خطوتين: (1) الترجمة من المخطط إلى النص، و (2) الاستدلال على النص المترجم. وتكمن الفكرة الأساسية لهذه الطريقة في وحدة تحويل الطراز، والتي يُطلق عليها DePlot، والتي تقوم بترجمة صورة المخطط أو الرسم البياني إلى جدول خطي. بعد ذلك، يمكن استخدام إخراج DePlot مباشرة لتوجيه نموذج اللغة الضخم (LLM) مسبق التدريب، مما يستغل قدرات الاستدلال القليلة لنموذج اللغة الضخم. وللحصول على DePlot، نقوم بتوحيد مهمة الترجمة من المخطط إلى الجدول من خلال إنشاء تنسيقات ومقاييس موحدة للمهمة، ثم تدريب DePlot بطريقة متكاملة على هذه المهمة. يمكن بعد ذلك استخدام DePlot مباشرة مع نماذج اللغة الضخمة بطريقة قابلة للتوصيل والتشغيل. مقارنة بنموذج متقدم في هذا المجال تم ضبط دقته على أكثر من 28000 نقطة بيانات، حقق DePlot+LLM مع توجيه التعلم لمرة واحدة تحسنًا بنسبة 24.0% مقارنة بالنماذج المتقدمة في هذا المجال عند استخدامه على استفسارات مكتوبة من قبل البشر من مهمة الأسئلة والأجوبة حول الرسوم البيانية.*

DePlot هو نموذج تم تدريبه باستخدام بنية `Pix2Struct`. يمكنك العثور على مزيد من المعلومات حول `Pix2Struct` في [توثيق Pix2Struct](https://huggingface.co/docs/transformers/main/en/model_doc/pix2struct).

DePlot هو مجموعة فرعية من بنية "Visual Question Answering" الخاصة بـ `Pix2Struct`. حيث يقوم برسم السؤال المدخل على الصورة والتنبؤ بالإجابة.

## مثال على الاستخدام

حاليًا، تتوفر نقطة تفتيش واحدة لـ DePlot:

- `google/deplot`: DePlot تمت معايرته بدقة على مجموعة بيانات ChartQA

```python
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

## الضبط الدقيق

لضبط دقة DePlot، راجع دفتر ملاحظات الضبط الدقيق [pix2struct](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb). بالنسبة لنماذج `Pix2Struct`، وجدنا أن ضبط دقة النموذج باستخدام Adafactor ومخطط معدل التعلم المتناقص يؤدي إلى تقارب أسرع:

```python
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=0.01, weight_decay=1e-05)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=40000)
```

<Tip>

DePlot هو نموذج تم تدريبه باستخدام بنية `Pix2Struct`. للرجوع إلى واجهة برمجة التطبيقات (API)، راجع وثائق [`Pix2Struct`](pix2struct).

</Tip>