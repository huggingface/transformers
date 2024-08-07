# Grounding DINO

## نظرة عامة

اقتُرح نموذج Grounding DINO في ورقة بحثية بعنوان "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" من قبل Shilong Liu وآخرون. يوسع Grounding DINO نموذج اكتشاف الكائنات المغلق باستخدام مشفر نصي، مما يمكّن اكتشاف الكائنات المفتوح. ويحقق النموذج نتائج ملحوظة، مثل 52.5 AP على COCO zero-shot.

ملخص الورقة البحثية هو كما يلي:

> "في هذه الورقة، نقدم مكتشف كائنات مفتوح يسمى Grounding DINO، من خلال دمج مكتشف الكائنات القائم على Transformer DINO مع التدريب المسبق المستند إلى النص، والذي يمكنه اكتشاف الكائنات التعسفية مع مدخلات بشرية مثل أسماء الفئات أو التعبيرات المرجعية. يتمثل الحل الرئيسي لاكتشاف الكائنات المفتوح في تقديم اللغة إلى مكتشف الكائنات المغلق من أجل تعميم مفهوم المجموعة المفتوحة. ولدمج أوضاع اللغة والرؤية بشكل فعال، نقسم مفهومًا مكتشفًا مغلقًا إلى ثلاث مراحل ونقترح حل دمج محكم، والذي يتضمن محسن ميزات، واختيار استعلام موجه باللغة، وفك تشفير متعدد الوسائط لفك تشفير متعدد الوسائط. في حين أن الأعمال السابقة تقيم بشكل أساسي اكتشاف الكائنات المفتوح على الفئات الجديدة، نقترح أيضًا إجراء تقييمات على فهم التعبيرات المرجعية للكائنات المحددة بالسمات. يعمل Grounding DINO بشكل جيد للغاية في جميع الإعدادات الثلاثة، بما في ذلك المعايير المرجعية على COCO وLVIS وODinW وRefCOCO/+/g. يحقق Grounding DINO 52.5 AP على معيار نقل اكتشاف COCO zero-shot، أي بدون أي بيانات تدريب من COCO. ويحقق رقماً قياسياً جديداً على معيار ODinW zero-shot بمتوسط 26.1 AP."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/grouding_dino_architecture.png"
alt="drawing" width="600"/>

<small> نظرة عامة على Grounding DINO. مأخوذة من <a href="https://arxiv.org/abs/2303.05499">الورقة البحثية الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل EduardoPacheco وnielsr. يمكن العثور على الكود الأصلي [هنا](https://github.com/IDEA-Research/GroundingDINO).

## نصائح الاستخدام

- يمكن استخدام [`GroundingDinoProcessor`] لإعداد أزواج الصور والنصوص للنموذج.
- لفصل الفئات في النص، استخدم فترة، على سبيل المثال "قطة. كلب."
- عند استخدام فئات متعددة (على سبيل المثال "قطة. كلب.")، استخدم `post_process_grounded_object_detection` من [`GroundingDinoProcessor`] لمعالجة الإخراج. نظرًا لأن التسميات التي تم إرجاعها من `post_process_object_detection` تمثل المؤشرات من بُعد النموذج حيث تكون الاحتمالية أكبر من العتبة.

فيما يلي كيفية استخدام النموذج للكشف عن الكائنات بدون الإشراف:

```python
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection,

model_id = "IDEA-Research/grounding-dino-tiny"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
text = "a cat. a remote control."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
```

## Grounded SAM

يمكنك دمج Grounding DINO مع نموذج [Segment Anything](sam) للتنقيب القائم على النص كما هو مقدم في ورقة "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks". يمكنك الرجوع إلى هذا [دفتر الملاحظات التوضيحي](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb) 🌍 للحصول على التفاصيل.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/grounded_sam.png"
alt="drawing" width="900"/>

<small> نظرة عامة على Grounded SAM. مأخوذة من <a href="https://github.com/IDEA-Research/Grounded-Segment-Anything">المستودع الأصلي</a>.</small>

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام Grounding DINO. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- يمكن العثور على دفاتر الملاحظات التوضيحية المتعلقة بالاستدلال باستخدام Grounding DINO، وكذلك دمجه مع [SAM](sam) [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Grounding%20DINO). 🌎

## GroundingDinoImageProcessor

[[autodoc]] GroundingDinoImageProcessor

- preprocess
- post_process_object_detection

## GroundingDinoProcessor

[[autodoc]] GroundingDinoProcessor

- post_process_grounded_object_detection

## GroundingDinoConfig

[[autodoc]] GroundingDinoConfig

## GroundingDinoModel

[[autodoc]] GroundingDinoModel

- forward

## GroundingDinoForObjectDetection

[[autodoc]] GroundingDinoForObjectDetection

- forward