# Pyramid Vision Transformer V2 (PVTv2)

## نظرة عامة

تم اقتراح نموذج PVTv2 في ورقة "PVT v2: Improved Baselines with Pyramid Vision Transformer" من قبل Wenhai Wang وآخرون. وهو نسخة محسنة من PVT، حيث يتخلى عن ترميز الموضع، ويعتمد بدلاً من ذلك على المعلومات الموضعية المشفرة من خلال الصفر-الترقيع والترقيع المتداخل. وتبسيط البنية، والسماح بتشغيل الاستدلال بأي دقة دون الحاجة إلى استيفائها.

وقد تم نشر بنية مشفر PVTv2 بنجاح لتحقيق نتائج متقدمة في Segformer للتجزئة الدلالية، و GLPN للعمق الأحادي، و Panoptic Segformer للتجزئة الشاملة.

تتبع PVTv2 عائلة من النماذج تسمى المحولات الهرمية، والتي تجري تعديلات على طبقات المحول لإنشاء خرائط سمات متعددة المقاييس. على عكس البنية العمودية لمحول الرؤية (ViT) التي تفقد التفاصيل الدقيقة، من المعروف أن خرائط الميزات متعددة المقاييس تحافظ على هذه التفاصيل وتساعد في الأداء في مهام التنبؤ الكثيفة. في حالة PVTv2، يتم تحقيق ذلك من خلال إنشاء رموز رقعة الصورة باستخدام التراكيب المتداخلة في كل طبقة من طبقات المشفر.

تسمح الميزات متعددة المقاييس للمحولات الهرمية بالاستبدال بسهولة في النماذج الأساسية لرؤية الكمبيوتر التقليدية مثل ResNet في البنى الأكبر. وقد أظهر كل من Segformer و Panoptic Segformer أن التكوينات التي تستخدم PVTv2 كعمود فقري تفوقت باستمرار على تلك التي تحتوي على عمود فقري ResNet بنفس الحجم تقريبًا.

وتتمثل إحدى الميزات القوية الأخرى لـ PVTv2 في تقليل التعقيد في طبقات الاهتمام الذاتي المسماة Spatial Reduction Attention (SRA)، والتي تستخدم طبقات التراكيب ثنائية الأبعاد لتصور الحالات المخفية إلى دقة أصغر قبل الاهتمام بها باستخدام الاستعلامات، مما يحسن تعقيد الاهتمام الذاتي O(n^2) إلى O(n^2/R)، حيث R هو نسبة التخفيض المكاني ("sr_ratio"، المعروف أيضًا باسم حجم النواة والخطوة في التراكيب ثنائية الأبعاد).

تم تقديم SRA في PVT، وهي طريقة تقليل التعقيد الافتراضية المستخدمة في PVTv2. ومع ذلك، قدم PVTv2 أيضًا خيار استخدام آلية اهتمام ذاتي ذات تعقيد خطي يتعلق بحجم الصورة، والتي أطلقوا عليها اسم "Linear SRA". تستخدم هذه الطريقة تجميع المتوسط لتقليل الحالات المخفية إلى حجم ثابت لا يتأثر بدقتها الأصلية (على الرغم من أنها أكثر فداحة من SRA العادية). يمكن تمكين هذا الخيار عن طريق تعيين "linear_attention" إلى "True" في PVTv2Config.

## ملخص الورقة:

*في الآونة الأخيرة، قدم المحول تقدمًا مشجعًا في رؤية الكمبيوتر. في هذا العمل، نقدم خطوط أساس جديدة من خلال تحسين محول الرؤية الهرمية الأصلي (PVT v1) من خلال إضافة ثلاثة تصاميم، بما في ذلك (1) طبقة اهتمام ذات تعقيد خطي، (2) ترميز رقعة متداخلة، و(3) شبكة تغذية أمامية متراكبة. مع هذه التعديلات، يقلل PVT v2 التعقيد الحسابي لـ PVT v1 إلى خطي ويحقق تحسينات كبيرة في مهام الرؤية الأساسية مثل التصنيف والكشف والتجزئة. ومن الجدير بالذكر أن PVT v2 المقترح يحقق أداءً مماثلًا أو أفضل من الأعمال الحديثة مثل محول Swin. نأمل أن يسهل هذا العمل أبحاث المحول المتقدمة في رؤية الكمبيوتر. الكود متاح في https://github.com/whai362/PVT.*

تمت المساهمة بهذا النموذج من قبل [FoamoftheSea](https://huggingface.co/FoamoftheSea). يمكن العثور على الكود الأصلي [هنا](https://github.com/whai362/PVT).

## نصائح الاستخدام:

- [PVTv2] هو نموذج محول هرمي أظهر أداءً قويًا في تصنيف الصور ومهام متعددة أخرى، ويستخدم كعمود فقري للتجزئة الدلالية في [Segformer]، وتقدير العمق الأحادي في [GLPN]، والتجزئة الشاملة في [Panoptic Segformer]، مما يظهر باستمرار أداءً أعلى من تكوينات ResNet المماثلة.

- تحقق المحولات الهرمية مثل PVTv2 كفاءة بيانات ومعلمات متفوقة في بيانات الصور مقارنة ببنيات المحول النقي من خلال دمج عناصر التصميم من الشبكات العصبية التلافيفية (CNNs) في مشفراتها. يخلق هذا بنية هجينة تجمع بين التحيزات الاستقرائية المفيدة لـ CNNs مثل التكافؤ في الترجمة والمحلية في الشبكة أثناء الاستمتاع بفوائد الاستجابة الديناميكية للبيانات والنمذجة العلائقية العالمية التي توفرها آلية الاهتمام الذاتي [المحولات].

- يستخدم PVTv2 ترميزات رقعة متداخلة لإنشاء خرائط ميزات متعددة المقاييس، والتي يتم دمجها بمعلومات الموقع باستخدام الصفر-الترقيع والتركيبات العميقة.

- لتقليل التعقيد في طبقات الاهتمام، يقوم PVTv2 بإجراء تقليل مكاني للحالات المخفية باستخدام إما تراكيب ثنائية الأبعاد متسارعة (SRA) أو تجميع متوسط ثابت الحجم (Linear SRA). على الرغم من أن Linear SRA أكثر فداحة بطبيعته، إلا أنه يوفر أداءً رائعًا بتعقيد خطي فيما يتعلق بحجم الصورة. لاستخدام Linear SRA في طبقات الاهتمام الذاتي، قم بتعيين "linear_attention=True" في "PvtV2Config".

- [`PvtV2Model`] هو مشفر المحول الهرمي (الذي يشار إليه أيضًا باسم Mix Transformer أو MiT في الأدبيات). يضيف [`PvtV2ForImageClassification`] رأس مصنف بسيط في الأعلى لأداء تصنيف الصور. يمكن استخدام [`PvtV2Backbone`] مع نظام [`AutoBackbone`] في البنى الأكبر مثل Deformable DETR.

- يمكن العثور على الأوزان التي تم تدريبها مسبقًا على ImageNet لجميع أحجام النماذج على [hub].

أفضل طريقة للبدء مع PVTv2 هي تحميل نقطة التحقق التي تم تدريبها مسبقًا بالحجم الذي تختاره باستخدام `AutoModelForImageClassification`:

```python
import requests
import torch

from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

model = AutoModelForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0")
image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
processed = image_processor(image)
outputs = model(torch.tensor(processed["pixel_values"]))
```

لاستخدام PVTv2 كعمود فقري لبنى أكثر تعقيدًا مثل DeformableDETR، يمكنك استخدام AutoBackbone (سيحتاج هذا النموذج إلى ضبط دقيق حيث أنك تستبدل العمود الفقري في النموذج الذي تم تدريبه مسبقًا):

```python
import requests
import torch

from transformers import AutoConfig, AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image

model = AutoModelForObjectDetection.from_config(
    config=AutoConfig.from_pretrained(
        "SenseTime/deformable-detr",
        backbone_config=AutoConfig.from_pretrained("OpenGVLab/pvt_v2_b5"),
        use_timm_backbone=False
    ),
)

image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
processed = image_processor(image)
outputs = model(torch.tensor(processed["pixel_values"]))
```

أداء [PVTv2] على ImageNet-1K حسب حجم النموذج (B0-B5):

| الطريقة           | الحجم | Acc@1 | #Params (M) |
|------------------|:----:|:-----:|:-----------:|
| PVT-V2-B0        |  224 |  70.5 |     3.7     |
| PVT-V2-B1        |  224 |  78.7 |     14.0    |
| PVT-V2-B2-Linear |  224 |  82.1 |     22.6    |
| PVT-V2-B2        |  224 |  82.0 |     25.4    |
| PVT-V2-B3        |  224 |  83.1 |     45.2    |
| PVT-V2-B4        |  224 |  83.6 |     62.6    |
| PVT-V2-B5        |  224 |  83.8 |     82.0    |

## PvtV2Config

[[autodoc]] PvtV2Config

## PvtForImageClassification

[[autodoc]] PvtV2ForImageClassification

- forward

## PvtModel

[[autodoc]] PvtV2Model

- forward