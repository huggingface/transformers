# DINOv2

## نظرة عامة
اقترح نموذج DINOv2 في [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) من قبل Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, وPiotr Bojanowski.

DINOv2 هو ترقية لـ [DINO](https://arxiv.org/abs/2104.14294)، وهي طريقة ذاتية الإشراف تطبق على [محولات الرؤية](vit). تمكن هذه الطريقة من ميزات بصرية متعددة الأغراض، أي ميزات تعمل عبر توزيعات الصور والمهام دون الحاجة إلى الضبط الدقيق.

المقتطف من الورقة هو ما يلي:

*أدت الاختراقات الأخيرة في معالجة اللغة الطبيعية لنماذج ما قبل التدريب على كميات كبيرة من البيانات إلى فتح الطريق أمام نماذج الأساس المماثلة في رؤية الكمبيوتر. يمكن أن تبسط هذه النماذج بشكل كبير استخدام الصور في أي نظام عن طريق إنتاج ميزات بصرية متعددة الأغراض، أي ميزات تعمل عبر توزيعات الصور والمهام دون الحاجة إلى الضبط الدقيق. يُظهر هذا العمل أن طرق ما قبل التدريب الحالية، خاصة الطرق الخاضعة للإشراف الذاتي، يمكن أن تنتج مثل هذه الميزات إذا تم تدريبها على بيانات كافية ومنتقاة من مصادر متنوعة. نعيد النظر في النهج الحالية ونجمع بين تقنيات مختلفة لزيادة حجم ما قبل التدريب من حيث البيانات وحجم النموذج. تهدف معظم المساهمات التقنية إلى تسريع التدريب وتثبيته على نطاق واسع. من حيث البيانات، نقترح خط أنابيب تلقائي لبناء مجموعة بيانات مخصصة ومتنوعة ومنتقاة بدلاً من البيانات غير المنتقاة، كما هو الحال عادة في الأدبيات الخاضعة للإشراف الذاتي. من حيث النماذج، نقوم بتدريب نموذج ViT (Dosovitskiy et al.، 2020) مع 1B معلمات وتقطيرها إلى سلسلة من النماذج الأصغر التي تتفوق على أفضل الميزات متعددة الأغراض المتاحة، OpenCLIP (Ilharco et al.، 2021) في معظم المعايير المرجعية على مستويي الصورة والبكسل.*

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr).

يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/dinov2).

## نصائح الاستخدام

يمكن تتبع النموذج باستخدام `torch.jit.trace` الذي يستفيد من تجميع JIT لتحسين النموذج وجعله أسرع في التشغيل. لاحظ أن هذا لا يزال ينتج بعض العناصر غير المطابقة وأن الفرق بين النموذج الأصلي والنموذج الذي تم تتبعه هو من رتبة 1e-4.

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs[0]

# We have to force return_dict=False for tracing
model.config.return_dict = False

with torch.no_grad():
    traced_model = torch.jit.trace(model, [inputs.pixel_values])
    traced_outputs = traced_model(inputs.pixel_values)

print((last_hidden_states - traced_outputs[0]).abs().max())
```

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام DPT.

- يمكن العثور على دفاتر الملاحظات التوضيحية لـ DINOv2 [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DINOv2). 🌎

<PipelineTag pipeline="image-classification"/>

- [`Dinov2ForImageClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) هذا.

- راجع أيضًا: [دليل مهام التصنيف الصوري](../tasks/image_classification)

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب وسنقوم بمراجعته! يجب أن يوضح المورد المثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

## Dinov2Config

[[autodoc]] Dinov2Config

## Dinov2Model

[[autodoc]] Dinov2Model

- forward

## Dinov2ForImageClassification

[[autodoc]] Dinov2ForImageClassification

- forward