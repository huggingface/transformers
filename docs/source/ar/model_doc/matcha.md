# MatCha

## نظرة عامة

اقتُرح MatCha في الورقة البحثية [MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering](https://arxiv.org/abs/2212.09662)، من تأليف Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, Julian Martin Eisenschlos.

يوضح ملخص الورقة البحثية ما يلي:

> تنتشر البيانات المرئية اللغوية مثل الرسوم البيانية والمخططات والرسوم المعلوماتية انتشاراً واسعاً في العالم البشري. ومع ذلك، لا تؤدي النماذج اللغوية المرئية المتقدمة حالياً أداءً جيداً على هذه البيانات. نقترح MatCha (التدريب المسبق على الاستدلال الرياضي وإلغاء عرض المخططات) لتعزيز قدرات النماذج اللغوية المرئية في النمذجة المشتركة للمخططات/الرسوم البيانية والبيانات اللغوية. وعلى وجه التحديد، نقترح عدة مهام للتدريب المسبق تغطي تفكيك المخطط والاستدلال العددي، والتي تعد قدرات رئيسية في النمذجة اللغوية المرئية. ونقوم بالتدريب المسبق لـ MatCha انطلاقاً من Pix2Struct، وهو نموذج لغوي مرئي للصور النصية تم اقتراحه مؤخراً. وعلى المعايير القياسية مثل PlotQA وChartQA، يتفوق نموذج MatCha على الطرق الحالية بأداء أفضل يصل إلى 20%. كما نقوم بفحص مدى جودة نقل التدريب المسبق لـ MatCha إلى مجالات مثل لقطات الشاشة، ومخططات الكتب المدرسية، ورسوم الوثائق، ونلاحظ تحسناً عاماً، مما يؤكد فائدة التدريب المسبق لـ MatCha على المهام اللغوية المرئية الأوسع نطاقاً.

## وصف النموذج

MatCha هو نموذج تم تدريبه باستخدام بنية `Pix2Struct`. يمكنك العثور على مزيد من المعلومات حول `Pix2Struct` في [وثائق Pix2Struct](https://huggingface.co/docs/transformers/main/en/model_doc/pix2struct).

MatCha هو مجموعة فرعية من الأسئلة البصرية والإجابة عليها في بنية `Pix2Struct`. فهو يقوم برسم السؤال المدخل على الصورة والتنبؤ بالإجابة.

## الاستخدام

تتوفر حالياً 6 نقاط تفتيش لـ MatCha:

- `google/matcha`: نموذج MatCha الأساسي، المستخدم لتعديل نموذج MatCha لمهام تالية
- `google/matcha-chartqa`: نموذج MatCha المعدل لبيانات ChartQA. يمكن استخدامه للإجابة على الأسئلة حول المخططات.
- `google/matcha-plotqa-v1`: نموذج MatCha المعدل لبيانات PlotQA. يمكن استخدامه للإجابة على الأسئلة حول الرسوم البيانية.
- `google/matcha-plotqa-v2`: نموذج MatCha المعدل لبيانات PlotQA. يمكن استخدامه للإجابة على الأسئلة حول الرسوم البيانية.
- `google/matcha-chart2text-statista`: نموذج MatCha المعدل لمجموعة بيانات Statista.
- `google/matcha-chart2text-pew`: نموذج MatCha المعدل لمجموعة بيانات Pew.

تكون النماذج المعدلة على `chart2text-pew` و`chart2text-statista` أكثر ملاءمة للتلخيص، في حين أن النماذج المعدلة على `plotqa` و`chartqa` تكون أكثر ملاءمة للإجابة على الأسئلة.

يمكنك استخدام هذه النماذج على النحو التالي (مثال على مجموعة بيانات ChatQA):

```python
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa").to(0)
processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt").to(0)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
```

## الضبط الدقيق

لضبط نموذج MatCha بدقة، راجع دفتر الملاحظات [fine-tuning notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb). بالنسبة لنماذج `Pix2Struct`، وجدنا أن ضبط النموذج بدقة باستخدام Adafactor وجدول التعلم بمعدل ثابت يؤدي إلى تقارب أسرع:

```python
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=0.01, weight_decay=1e-05)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=40000)
```

<Tip>

MatCha هو نموذج تم تدريبه باستخدام بنية `Pix2Struct`. يمكنك العثور على مزيد من المعلومات حول `Pix2Struct` في [وثائق Pix2Struct](https://huggingface.co/docs/transformers/main/en/model_doc/pix2struct).

</Tip>