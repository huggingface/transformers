هذا دليل لمهمة Image-to-Image، والتي يتم فيها إدخال صورة إلى تطبيق وإخراج صورة أخرى. لديها العديد من المهام الفرعية، بما في ذلك تحسين الصورة (القرار الفائق، وتحسين الإضاءة المنخفضة، وإزالة المطر، وغير ذلك)، وإكمال الصورة، والمزيد.

سيوضح هذا الدليل كيفية:

- استخدام خط أنابيب image-to-image لمهمة super resolution
- تشغيل نماذج image-to-image لنفس المهمة بدون خط أنابيب

ملاحظة: اعتبارًا من وقت إصدار هذا الدليل، يدعم خط أنابيب "image-to-image" مهمة super resolution فقط.

لنبدأ بتثبيت المكتبات اللازمة.

```bash
pip install transformers
```

يمكننا الآن تهيئة خط الأنابيب باستخدام نموذج [Swin2SR](https://huggingface.co/caidas/swin2SR-lightweight-x2-64). بعد ذلك، يمكننا إجراء الاستدلال باستخدام خط الأنابيب عن طريق استدعائه مع صورة. في الوقت الحالي، تدعم خطوط الأنابيب هذه فقط نماذج [Swin2SR](https://huggingface.co/models?sort=trending&search=swin2sr).

```python
from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipeline(task="image-to-image", model="caidas/swin2SR-lightweight-x2-64", device=device)
```

الآن، دعنا نقوم بتحميل صورة.

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
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg" alt="صورة لقطة"/>
</div>

يمكننا الآن إجراء الاستدلال باستخدام خط الأنابيب. سنحصل على نسخة موسعة من صورة القطة.

```python
upscaled = pipe(image)
print(upscaled.size)
```
```bash
# (1072, 880)
```

إذا كنت ترغب في إجراء الاستدلال بنفسك بدون خط أنابيب، فيمكنك استخدام الفئات `Swin2SRForImageSuperResolution` و`Swin2SRImageProcessor` من مكتبة Transformers. سنستخدم نفس نقطة تفتيش النموذج لهذا الغرض. دعونا نقوم بتهيئة النموذج والمعالج.

```python
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64").to(device)
processor = Swin2SRImageProcessor("caidas/swin2SR-lightweight-x2-64")
```

يقوم `pipeline` بتبسيط خطوات ما قبل المعالجة وما بعد المعالجة التي يتعين علينا القيام بها بأنفسنا، لذلك دعنا نقوم بمعالجة الصورة مسبقًا. سنقوم بتمرير الصورة إلى المعالج، ثم نقل قيم البكسل إلى وحدة معالجة الرسومات (GPU).

```python
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

pixel_values = pixel_values.to(device)
```

الآن يمكننا استنتاج الصورة عن طريق تمرير قيم البكسل إلى النموذج.

```python
import torch

with torch.no_grad():
  outputs = model(pixel_values)
```
الإخراج عبارة عن كائن من النوع `ImageSuperResolutionOutput` يبدو كما يلي 👇

```python
import torch

with torch.no_grad():
  outputs = model(pixel_values)
```
الإخراج عبارة عن كائن من النوع `ImageSuperResolutionOutput` يبدو كما يلي 👇

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
نحتاج إلى الحصول على `reconstruction` ومعالجتها بعد المعالجة من أجل العرض المرئي. دعونا نرى كيف تبدو.

```python
outputs.reconstruction.data.shape
# torch.Size([1, 3, 880, 1072])
```

نحتاج إلى ضغط الإخراج والتخلص من المحور 0، وقص القيم، ثم تحويلها إلى float نومبي. بعد ذلك، سنقوم بترتيب المحاور بحيث يكون الشكل [1072، 880]، وأخيراً، إعادة إخراج القيم إلى النطاق [0، 255].

```python
import numpy as np

# ضغط، ونقل إلى وحدة المعالجة المركزية، وقص القيم
output = outputs.reconstruction.data.squeeze().cpu().clamp_(0, 1).numpy()
# إعادة ترتيب المحاور
output = np.moveaxis(output, source=0, destination=-1)
# إعادة القيم إلى نطاق قيم البكسل
output = (output * 255.0).round().astype(np.uint8)
Image.fromarray(output)
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat_upscaled.png" alt="صورة مكبرة لقطة"/>
</div>