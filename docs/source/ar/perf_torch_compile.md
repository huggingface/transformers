# تسريع الاستنتاج باستخدام torch.compile()

يهدف هذا الدليل إلى تقديم معيار لقياس التحسينات في سرعة الاستنتاج التي تم تقديمها مع [torch.compile()](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) [لنماذج الرؤية الحاسوبية في 🤗 Transformers](https://huggingface.co/models؟pipeline_tag=image-classification&library=transformers&sort=trending).

## فوائد torch.compile

بناءً على النموذج وبطاقة GPU، يوفر torch.compile() تسريعًا يصل إلى 30% أثناء الاستنتاج. لاستخدام torch.compile()، ما عليك سوى تثبيت أي إصدار من torch أعلى من 2.0.

يستغرق تجميع نموذج وقتًا، لذا فهو مفيد إذا كنت تقوم بتجميع النموذج مرة واحدة فقط بدلاً من كل مرة تقوم فيها بالاستنتاج. لتجميع أي نموذج رؤية حاسوبية من اختيارك، قم بالاستدعاء `torch.compile()` على النموذج كما هو موضح أدناه:

```diff
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to("cuda")
+ model = torch.compile(model)
```

يأتي `compile()` مع أوضاع متعددة للتجميع، والتي تختلف بشكل أساسي في وقت التجميع وعبء الاستدلال. يستغرق الأمر `max-autotune` وقتًا أطول من `reduce-overhead` ولكنه يؤدي إلى استدلال أسرع. الوضع الافتراضي هو الأسرع للتجميع ولكنه ليس بكفاءة `reduce-overhead` لوقت الاستدلال. في هذا الدليل، استخدمنا الوضع الافتراضي. يمكنك معرفة المزيد عنه [هنا](https://pytorch.org/get-started/pytorch-2.0/#user-experience).

قمنا باختبار `torch.compile` مع نماذج رؤية حاسوبية مختلفة، ومهمات، وأنواع الأجهزة، وأحجام الدفعات على إصدار `torch` 2.0.1.

## كود المعيار المرجعي

فيما يلي يمكنك العثور على كود المعيار المرجعي لكل مهمة. نقوم بتسخين وحدة معالجة الرسومات (GPU) قبل الاستدلال ونأخذ متوسط وقت 300 استدلال، باستخدام نفس الصورة في كل مرة.

### تصنيف الصور مع ViT

```python
import torch
from PIL import Image
import requests
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").to("cuda")
model = torch.compile(model)

processed_input = processor(image, return_tensors='pt').to(device="cuda")

with torch.no_grad():
    _ = model(**processed_input)
```

#### اكتشاف الكائنات مع DETR

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").to("cuda")
model = torch.compile(model)

texts = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=texts, images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    _ = model(**inputs)
```

#### تجزئة الصورة مع Segformer

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
with torch.no_grad():
    _ = model(**inputs)
```

#### تجزئة الصورة مع Segformer

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to("cuda")
model = torch.compile(model)
seg_inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    _ = model(**seg_inputs)
```

فيما يلي قائمة بالنماذج التي أجرينا عليها المعيار المرجعي.

**تصنيف الصور**
- [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- [microsoft/beit-base-patch16-224-pt22k-ft22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)
- [facebook/convnext-large-224](https://huggingface.co/facebook/convnext-large-224)
- [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)

**تجزئة الصورة**
- [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [facebook/mask2former-swin-tiny-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-tiny-coco-panoptic)
- [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade)
- [google/deeplabv3_mobilenet_v2_1.0_513](https://huggingface.co/google/deeplabv3_mobilenet_v2_1.0_513)

**اكتشاف الكائنات**
- [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32)
- [facebook/detr-resnet-101](https://huggingface.co/facebook/detr-resnet-101)
- [microsoft/conditional-detr-resnet-50](https://huggingface.co/microsoft/conditional-detr-resnet-50)

فيما يلي يمكنك العثور على رسومات بيانية لمدد الاستدلال مع وبدون `torch.compile()` والنسب المئوية للتحسين لكل نموذج في أجهزة وأحجام دفعات مختلفة.

<div class="flex">
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/a100_batch_comp.png" />
  </div>
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/v100_batch_comp.png" />
  </div>
   <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/t4_batch_comp.png" />
  </div>
</div>

<div class="flex">
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/A100_1_duration.png" />
  </div>
  <div>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/A100_1_percentage.png" />
  </div>
</div>


![مدة المقارنة على V100 بحجم دفعة 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/v100_1_duration.png)

![النسبة المئوية للتحسين على T4 بحجم دفعة 4](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/torch_compile/T4_4_percentage.png)

فيما يلي يمكنك العثور على مدد الاستدلال بالمللي ثانية لكل نموذج مع وبدون `compile()`. لاحظ أن OwlViT يؤدي إلى OOM في أحجام الدفعات الأكبر.

### A100 (حجم الدفعة: 1)

| المهمة/النموذج | الإصدار 2.0 من torch - <br>بدون تجميع | الإصدار 2.0 من torch - <br>تجميع |
|:---:|:---:|:---:|
| تصنيف الصور/ViT | 9.325 | 7.584 | 
| تجزئة الصورة/Segformer | 11.759 | 10.500 |
| اكتشاف الكائنات/OwlViT | 24.978 | 18.420 |
| تصنيف الصور/BeiT | 11.282 | 8.448 | 
| اكتشاف الكائنات/DETR | 34.619 | 19.040 |
| تصنيف الصور/ConvNeXT | 10.410 | 10.208 | 
| تصنيف الصور/ResNet | 6.531 | 4.124 |
| تجزئة الصورة/Mask2former | 60.188 | 49.117 |
| تجزئة الصورة/Maskformer | 75.764 | 59.487 | 
| تجزئة الصورة/MobileNet | 8.583 | 3.974 |
| اكتشاف الكائنات/Resnet-101 | 36.276 | 18.197 |
| اكتشاف الكائنات/Conditional-DETR | 31.219 | 17.993 |
### A100 (حجم الدفعة: 4)

| المهمة/النموذج | الإصدار 2.0 من torch - <br>بدون تجميع | الإصدار 2.0 من torch - <br>تجميع |
|:---:|:---:|:---:|
| تصنيف الصور/ViT | 14.832 | 14.499 | 
| تجزئة الصورة/Segformer | 18.838 | 16.476 |
| تصنيف الصور/BeiT | 13.205 | 13.048 | 
| اكتشاف الكائنات/DETR | 48.657 | 32.418|
| تصنيف الصور/ConvNeXT | 22.940 | 21.631 | 
| تصنيف الصور/ResNet | 6.657 | 4.268 |
| تجزئة الصورة/Mask2former | 74.277 | 61.781 |
| تجزئة الصورة/Maskformer | 180.700 | 159.116 | 
| تجزئة الصورة/MobileNet | 14.174 | 8.515 |
| اكتشاف الكائنات/Resnet-101 | 68.101 | 44.998 |
| اكتشاف الكائنات/Conditional-DETR | 56.470 | 35.552 |

### A100 (حجم الدفعة: 16)

| المهمة/النموذج | الإصدار 2.0 من torch - <br>بدون تجميع | الإصدار 2.0 من torch - <br>تجميع |
|:---:|:---:|:---:|
| تصنيف الصور/ViT | 40.944 | 40.010 | 
| تجزئة الصورة/Segformer | 37.005 | 31.144 |
| تصنيف الصور/BeiT | 41.854 | 41.048 | 
| اكتشاف الكائنات/DETR | 164.382 | 161.902 |
| تصنيف الصور/ConvNeXT | 82.258 | 75.561 | 
| تصنيف الصور/ResNet | 7.018 | 5.024 |
| تجزئة الصورة/Mask2former | 178.945 | 154.814 |
| تجزئة الصورة/Maskformer | 638.570 | 579.826 | 
| تجزئة الصورة/MobileNet | 51.693 | 30.310 |
| اكتشاف الكائنات/Resnet-101 | 232.887 | 155.021 |
| اكتشاف الكائنات/Conditional-DETR | 180.491 | 124.032 |

### V100 (حجم الدفعة: 1)

| المهمة/النموذج | الإصدار 2.0 من torch - <br>بدون تجميع | الإصدار 2.0 من torch - <br>تجميع |
|:---:|:---:|:---:|
| تصنيف الصور/ViT | 10.495 | 6.00 | 
| تجزئة الصورة/Segformer | 13.321 | 5.862 | 
| اكتشاف الكائنات/OwlViT | 25.769 | 22.395 | 
| تصنيف الصور/BeiT | 11.347 | 7.234 | 
| اكتشاف الكائنات/DETR | 33.951 | 19.388 |
| تصنيف الصور/ConvNeXT | 11.623 | 10.412 | 
| تصنيف الصور/ResNet | 6.484 | 3.820 |
| تجزئة الصورة/Mask2former | 64.640 | 49.873 |
| تجزئة الصورة/Maskformer | 95.532 | 72.207 | 
| تجزئة الصورة/MobileNet | 9.217 | 4.753 |
| اكتشاف الكائنات/Resnet-101 | 52.818 | 28.367 |
| اكتشاف الكائنات/Conditional-DETR | 39.512 | 20.816 |

### V100 (حجم الدفعة: 4)

| المهمة/النموذج | الإصدار 2.0 من torch - <br>بدون تجميع | الإصدار 2.0 من torch - <br>تجميع |
|:---:|:---:|:---:|
| تصنيف الصور/ViT | 15.181 | 14.501 | 
| تجزئة الصورة/Segformer | 16.787 | 16.188 |
| تصنيف الصور/BeiT | 15.171 | 14.753 | 
| اكتشاف الكائنات/DETR | 88.529 | 64.195 |
| تصنيف الصور/ConvNeXT | 29.574 | 27.085 | 
| تصنيف الصور/ResNet | 6.109 | 4.731 |
| تجزئة الصورة/Mask2former | 90.402 | 76.926 |
| تجزئة الصورة/Maskformer | 234.261 | 205.456 | 
| تجزئة الصورة/MobileNet | 24.623 | 14.816 |
| اكتشاف الكائنات/Resnet-101 | 134.672 | 101.304 |
| اكتشاف الكائنات/Conditional-DETR | 97.464 | 69.739 |

### T4 (batch size: 16)

| **Task/Model** | **torch 2.0 - <br>no compile** | **torch 2.0 - <br>compile** |
|:---:|:---:|:---:|
| Image Classification/ViT | 163.914 | 160.907 | 
| Image Segmentation/Segformer | 192.412 | 163.620 |
| Image Classification/BeiT | 188.978 | 187.976 | 
| Object Detection/DETR | OOM | OOM |
| Image Classification/ConvNeXT | 422.886 | 388.078 | 
| Image Classification/ResNet | 44.114 | 37.604 |
| Image Segmentation/Mask2former | 756.337 | 695.291 |
| Image Segmentation/Maskformer | 2842.940 | 2656.88 | 
| Image Segmentation/MobileNet | 299.003 | 201.942 |
| Object Detection/Resnet-101 |  1619.505 | 1262.758 | 
| Object Detection/Conditional-DETR | 1137.513 | 897.390|

## PyTorch Nightly
We also benchmarked on PyTorch nightly (2.1.0dev, find the wheel [here](https://download.pytorch.org/whl/nightly/cu118)) and observed improvement in latency both for uncompiled and compiled models. 

### A100

| **Task/Model** | **Batch Size** | **torch 2.0 - no compile** | **torch 2.0 -<br> compile** |
|:---:|:---:|:---:|:---:|
| Image Classification/BeiT | Unbatched | 12.462 | 6.954 | 
| Image Classification/BeiT | 4 | 14.109 | 12.851 | 
| Image Classification/BeiT | 16 | 42.179 | 42.147 | 
| Object Detection/DETR | Unbatched | 30.484 | 15.221 |
| Object Detection/DETR | 4 | 46.816 | 30.942 |
| Object Detection/DETR | 16 | 163.749 | 163.706  |

### T4

| **Task/Model** | **Batch Size** | **torch 2.0 - <br>no compile** | **torch 2.0 - <br>compile** |
|:---:|:---:|:---:|:---:|
| Image Classification/BeiT | Unbatched | 14.408 | 14.052 | 
| Image Classification/BeiT | 4 | 47.381 | 46.604 | 
| Image Classification/BeiT | 16 | 42.179 | 42.147  | 
| Object Detection/DETR | Unbatched | 68.382 | 53.481 |
| Object Detection/DETR | 4 | 269.615 | 204.785 |
| Object Detection/DETR | 16 | OOM | OOM   |

### V100

| **Task/Model** | **Batch Size** | **torch 2.0 - <br>no compile** | **torch 2.0 - <br>compile** |
|:---:|:---:|:---:|:---:|
| Image Classification/BeiT | Unbatched | 13.477 | 7.926 | 
| Image Classification/BeiT | 4 | 15.103 | 14.378 | 
| Image Classification/BeiT | 16 | 52.517 | 51.691  | 
| Object Detection/DETR | Unbatched | 28.706 | 19.077 |
| Object Detection/DETR | 4 | 88.402 | 62.949|
| Object Detection/DETR | 16 | OOM | OOM  |


## Reduce Overhead
We benchmarked `reduce-overhead` compilation mode for A100 and T4 in Nightly.

### A100

| **Task/Model** | **Batch Size** | **torch 2.0 - <br>no compile** | **torch 2.0 - <br>compile** |
|:---:|:---:|:---:|:---:|
| Image Classification/ConvNeXT | Unbatched | 11.758 | 7.335 | 
| Image Classification/ConvNeXT | 4 | 23.171 | 21.490 | 
| Image Classification/ResNet | Unbatched | 7.435 | 3.801 | 
| Image Classification/ResNet | 4 | 7.261 | 2.187 | 
| Object Detection/Conditional-DETR | Unbatched | 32.823 | 11.627  | 
| Object Detection/Conditional-DETR | 4 | 50.622 | 33.831  | 
| Image Segmentation/MobileNet | Unbatched | 9.869 | 4.244 |
| Image Segmentation/MobileNet | 4 | 14.385 | 7.946 |


### T4

| **Task/Model** | **Batch Size** | **torch 2.0 - <br>no compile** | **torch 2.0 - <br>compile** | 
|:---:|:---:|:---:|:---:|
| Image Classification/ConvNeXT | Unbatched | 32.137 | 31.84 | 
| Image Classification/ConvNeXT | 4 | 120.944 | 110.209 | 
| Image Classification/ResNet | Unbatched | 9.761 | 7.698 | 
| Image Classification/ResNet | 4 | 15.215 | 13.871 | 
| Object Detection/Conditional-DETR | Unbatched | 72.150 | 57.660  | 
| Object Detection/Conditional-DETR | 4 | 301.494 | 247.543  | 
| Image Segmentation/MobileNet | Unbatched | 22.266 | 19.339  |
| Image Segmentation/MobileNet | 4 | 78.311 | 50.983 |