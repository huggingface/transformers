<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# دليل مهمة صورة إلى صورة (Image-to-Image)

[[open-in-colab]]

مهمة صورة إلى صورة هي مهمة يستقبل فيها التطبيق صورة ويُخرج صورة أخرى. تشمل هذه المهمة مجموعة من المهام الفرعية مثل: تحسين الصورة (تعزيز الدقة الفائقة Super Resolution، تحسين الإضاءة المنخفضة، إزالة المطر، إلخ)، وإكمال/ترميم الصورة (Inpainting)، وغيرها.

يوضّح هذا الدليل كيفية:
- استخدام بايبلاين صورة إلى صورة لمهمة تعزيز الدقة الفائقة.
- تشغيل نماذج صورة إلى صورة لنفس المهمة دون استخدام بايبلاين.

ملاحظة: عند وقت إصدار هذا الدليل، يدعم بايبلاين `image-to-image` مهمة تعزيز الدقة الفائقة فقط.

لنبدأ بتثبيت المكتبات اللازمة.

```bash
pip install transformers
```

يمكننا الآن تهيئة البايبلاين باستخدام [نموذج Swin2SR](https://huggingface.co/caidas/swin2SR-lightweight-x2-64). بعد ذلك، نستطيع إجراء الاستدلال على صورة عبر تمريرها إلى البايبلاين. حاليًا، تدعم هذه البايبلاين فقط [نماذج Swin2SR](https://huggingface.co/models?sort=trending&search=swin2sr).

```python
from transformers import pipeline
import torch
from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
device, _, _ = get_backend()
pipe = pipeline(task="image-to-image", model="caidas/swin2SR-lightweight-x2-64", device=device)
```

الآن، دعنا نحمّل صورة.

```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
image = Image.open(requests.get(url, stream=True).raw)

print(image.size)
```
```bash
# (532, 432)
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg" alt="Photo of a cat"/>
</div>

يمكننا الآن إجراء الاستدلال باستخدام البايبلاين. سنحصل على نسخة مُكبّرة الدقة من صورة القطة.

```python
upscaled = pipe(image)
print(upscaled.size)
```
```bash
# (1072, 880)
```

إذا رغبت في إجراء الاستدلال بنفسك دون بايبلاين، يمكنك استخدام الصنفين `Swin2SRForImageSuperResolution` و`Swin2SRImageProcessor` من مكتبة Transformers. سنستخدم نفس نقطة التحقق للنموذج هنا. لنقم بتهيئة النموذج والمُعالج (Processor).

```python
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor 

model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64").to(device)
processor = Swin2SRImageProcessor("caidas/swin2SR-lightweight-x2-64")
```

يُجرد `pipeline` خطوات المعالجة المسبقة واللاحقة التي علينا القيام بها يدويًا، لذا دعنا ننفّذ المعالجة المسبقة للصورة. سنمرر الصورة إلى المعالج ثم ننقل قيم البكسل إلى وحدة المعالجة المناسبة (مثل GPU).

```python
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

pixel_values = pixel_values.to(device)
```

يمكننا الآن إجراء الاستدلال بتمرير قيم البكسل إلى النموذج.

```python
import torch

with torch.no_grad():
  outputs = model(pixel_values)
```
المخرجات كائن من النوع `ImageSuperResolutionOutput` ويبدو كما يلي 👇

```
(loss=None, reconstruction=tensor([[[[0.8270, 0.8269, 0.8275,  ..., 0.7463, 0.7446, 0.7453],
          [0.8287, 0.8278, 0.8283,  ..., 0.7451, 0.7448, 0.7457],
          [0.8280, 0.8273, 0.8269,  ..., 0.7447, 0.7446, 0.7452],
          ...,
          [0.5923, 0.5933, 0.5924,  ..., 0.0697, 0.0695, 0.0706],
          [0.5926, 0.5932, 0.5926,  ..., 0.0673, 0.0687, 0.0705],
          [0.5927, 0.5914, 0.5922,  ..., 0.0664, 0.0694, 0.0718]]]],
       device='cuda:0'), hidden_states=None, attentions=None)
```
نحتاج إلى الحصول على الحقل `reconstruction` ومعالجته لاحقًا من أجل العرض. لنتحقق من شكله.

```python
outputs.reconstruction.data.shape
# torch.Size([1, 3, 880, 1072])
```

سنقوم بعصر (squeeze) المخرجات لإزالة المحور 0، ثم قص القيم ضمن النطاق المناسب، وبعدها تحويلها إلى مصفوفة NumPy من نوع float. ثم سنعيد ترتيب المحاور لتصبح بالشكل [1072, 880]، وأخيرًا نعيد القيم إلى النطاق [0, 255].

```python
import numpy as np

# squeeze, take to CPU and clip the values
output = outputs.reconstruction.data.squeeze().cpu().clamp_(0, 1).numpy()
# rearrange the axes
output = np.moveaxis(output, source=0, destination=-1)
# bring values back to pixel values range
output = (output * 255.0).round().astype(np.uint8)
Image.fromarray(output)
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat_upscaled.png" alt="Upscaled photo of a cat"/>
</div>
