# SAM

## نظرة عامة

اقترح SAM (Segment Anything Model) في [Segment Anything](https://arxiv.org/pdf/2304.02643v1.pdf) بواسطة Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick.

يمكن استخدام النموذج للتنبؤ بقناع التجزئة لأي كائن يهمه الأمر مع إعطاء صورة إدخال.

![مثال الصورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)

الملخص من الورقة هو ما يلي:

*نقدم مشروع Segment Anything (SA): مهمة جديدة ونموذج ومجموعة بيانات لتجزئة الصور. باستخدام نموذجنا الفعال في حلقة جمع البيانات، قمنا ببناء أكبر مجموعة بيانات للتجزئة حتى الآن (بفارق كبير)، مع أكثر من 1 مليار قناع على 11 مليون صورة مرخصة وتحترم الخصوصية. تم تصميم النموذج وتدريبه ليكون قابلاً للتشغيل، بحيث يمكنه النقل بدون تدريب إلى توزيعات الصور الجديدة والمهام الجديدة. نقيم قدراته على العديد من المهام ونجد أن أداءه بدون تدريب مثير للإعجاب - غالبًا ما يكون تنافسيًا أو حتى متفوقًا على النتائج الخاضعة للإشراف الكامل السابقة. نحن نطلق Segment Anything Model (SAM) ومجموعة البيانات المقابلة (SA-1B) التي تحتوي على 1 مليار قناع و11 مليون صورة على [https://segment-anything.com](https://segment-anything.com) لتعزيز البحث في نماذج الأساس للرؤية الحاسوبية.*

نصائح:

- يتنبأ النموذج بقناع ثنائي يشير إلى وجود كائن الاهتمام أو عدمه في الصورة.
- يتنبأ النموذج بنتائج أفضل بكثير إذا تم توفير نقاط ثنائية الأبعاد و/أو صناديق حد لتحديد الإدخال
- يمكنك مطالبة نقاط متعددة لنفس الصورة، والتنبؤ بقناع واحد.
- لا يتم دعم ضبط دقة النموذج بعد
- وفقًا للورقة، يجب أيضًا دعم الإدخال النصي. ومع ذلك، في وقت كتابة هذا التقرير، يبدو أنه غير مدعوم وفقًا لـ [المستودع الرسمي](https://github.com/facebookresearch/segment-anything/issues/4#issuecomment-1497626844).

تمت المساهمة بهذا النموذج من قبل [ybelkada](https://huggingface.co/ybelkada) و [ArthurZ](https://huggingface.co/ArthurZ).

يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/segment-anything).

فيما يلي مثال على كيفية تشغيل توليد القناع مع إعطاء صورة ونقطة ثنائية الأبعاد:

```python
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # موقع ثنائي الأبعاد لنافذة في الصورة

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

يمكنك أيضًا معالجة أقنعتك الخاصة جنبًا إلى جنب مع صور الإدخال في المعالج لتمريرها إلى النموذج.

```python
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
input_points = [[[450, 600]]]  # موقع ثنائي الأبعاد لنافذة في الصورة

inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام SAM.

- [دفتر الملاحظات التوضيحي](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) لاستخدام النموذج.
- [دفتر الملاحظات التوضيحي](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb) لاستخدام خط أنابيب توليد القناع التلقائي.
- [دفتر الملاحظات التوضيحي](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb) للاستدلال مع MedSAM، وهو إصدار مضبوط الضبط لـ SAM في المجال الطبي. 🌎
- [دفتر الملاحظات التوضيحي](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) لضبط دقة النموذج على بيانات مخصصة. 🌎

## SlimSAM

اقترح SlimSAM، وهو إصدار مقصوص من SAM، في [0.1% Data Makes Segment Anything Slim](https://arxiv.org/abs/2312.05284) بواسطة Zigeng Chen et al. يقلل SlimSAM حجم نماذج SAM بشكل كبير مع الحفاظ على نفس الأداء.

يمكن العثور على نقاط التفتيش على [المركز](https://huggingface.co/models?other=slimsam)، ويمكن استخدامها كبديل مباشر لـ SAM.

## Grounded SAM

يمكنك دمج [Grounding DINO](grounding-dino) مع SAM لتوليد القناع القائم على النص كما تم تقديمه في [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://arxiv.org/abs/2401.14159). يمكنك الرجوع إلى هذا [دفتر الملاحظات التوضيحي](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb) 🌍 لمزيد من التفاصيل.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/grounded_sam.png"
alt="drawing" width="900"/>

<small> نظرة عامة على Grounded SAM. مأخوذة من <a href="https://github.com/IDEA-Research/Grounded-Segment-Anything">المستودع الأصلي</a>. </small>

## SamConfig

[[autodoc]] SamConfig

## SamVisionConfig

[[autodoc]] SamVisionConfig

## SamMaskDecoderConfig

[[autodoc]] SamMaskDecoderConfig

## SamPromptEncoderConfig

[[autodoc]] SamPromptEncoderConfig

## SamProcessor

[[autodoc]] SamProcessor

## SamImageProcessor

[[autodoc]] SamImageProcessor

## SamModel

[[autodoc]] SamModel

- forward

## TFSamModel

[[autodoc]] TFSamModel

- call