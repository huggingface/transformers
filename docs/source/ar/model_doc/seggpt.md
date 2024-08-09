# SegGPT

## نظرة عامة

اقترح نموذج SegGPT في [SegGPT: Segmenting Everything In Context](https://arxiv.org/abs/2304.03284) بواسطة Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, Tiejun Huang. يستخدم SegGPT محولًا قائمًا على فك الترميز فقط يمكنه إنشاء قناع تجزئة بناءً على صورة دخل وصورة موجهة وقناع موجهة مطابق. يحقق النموذج نتائج لقطات رائعة بمعدل 56.1 mIoU على COCO-20 و 85.6 mIoU على FSS-1000.

الملخص من الورقة هو ما يلي:

*نقدم SegGPT، وهو نموذج عام لتقسيم كل شيء في السياق. نوحد مهام التجزئة المختلفة في إطار تعلم متخصص في السياق يستوعب أنواعًا مختلفة من بيانات التجزئة عن طريق تحويلها إلى نفس تنسيق الصور. تتم صياغة تدريب SegGPT على أنه مشكلة تلوين في السياق مع تعيين لون عشوائي لكل عينة بيانات. الهدف هو إنجاز مهام متنوعة وفقًا للسياق، بدلاً من الاعتماد على ألوان محددة. بعد التدريب، يمكن لـ SegGPT تنفيذ مهام تجزئة تعسفية في الصور أو مقاطع الفيديو عبر الاستدلال في السياق، مثل كائن مثيل، وملء، وجزء، ومحيط، ونص. يتم تقييم SegGPT على مجموعة واسعة من المهام، بما في ذلك التجزئة الدلالية لعدد قليل من اللقطات، وتجزئة كائن الفيديو، والتجزئة الدلالية، والتجزئة الشاملة. تظهر نتائجنا قدرات قوية في تجزئة المجال داخل المجال وخارجه*

نصائح:

- يمكن للمرء استخدام [`SegGptImageProcessor`] لتحضير إدخال الصورة والملصق والقناع للنموذج.
- يمكن للمرء إما استخدام خرائط التجزئة أو صور RGB كأقنعة موجهة. إذا كنت تستخدم هذا الأخير، فتأكد من تعيين `do_convert_rgb=False` في طريقة `preprocess`.
- من المستحسن بشدة تمرير `num_labels` عند استخدام `segmetantion_maps` (بدون اعتبار الخلفية) أثناء المعالجة المسبقة والمعالجة اللاحقة مع [`SegGptImageProcessor`] لحالتك الاستخدام.
- عند إجراء الاستدلال مع [`SegGptForImageSegmentation`] إذا كان `batch_size` أكبر من 1، فيمكنك استخدام تجميع الميزات عبر صورك عن طريق تمرير `feature_ensemble=True` في طريقة forward.

فيما يلي كيفية استخدام النموذج لتجزئة دلالية لقطات واحدة:

```python
import torch
from datasets import load_dataset
from transformers import SegGptImageProcessor, SegGptForImageSegmentation

checkpoint = "BAAI/seggpt-vit-large"
image_processor = SegGptImageProcessor.from_pretrained(checkpoint)
model = SegGptForImageSegmentation.from_pretrained(checkpoint)

dataset_id = "EduardoPacheco/FoodSeg103"
ds = load_dataset(dataset_id, split="train")
# Number of labels in FoodSeg103 (not including background)
num_labels = 103

image_input = ds[4]["image"]
ground_truth = ds[4]["label"]
image_prompt = ds[29]["image"]
mask_prompt = ds[29]["label"]

inputs = image_processor(
    images=image_input,
    prompt_images=image_prompt,
    segmentation_maps=mask_prompt,
    num_labels=num_labels,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = [image_input.size[::-1]]
mask = image_processor.post_process_semantic_segmentation(outputs, target_sizes, num_labels=num_labels)[0]
```

تمت المساهمة بهذا النموذج بواسطة [EduardoPacheco](https://huggingface.co/EduardoPacheco).
يمكن العثور على الكود الأصلي [هنا](https://github.com/baaivision/Painter/tree/main)).

## SegGptConfig

[[autodoc]] SegGptConfig

## SegGptImageProcessor

[[autodoc]] SegGptImageProcessor

- preprocess
- post_process_semantic_segmentation

## SegGptModel

[[autodoc]] SegGptModel

- forward

## SegGptForImageSegmentation

[[autodoc]] SegGptForImageSegmentation

- forward