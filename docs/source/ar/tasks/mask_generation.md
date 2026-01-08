<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# توليد الأقنعة

توليد الأقنعة هو مهمة إنشاء أقنعة ذات دلالة معنوية للصورة. هذه المهمة قريبة من [تقسيم الصورة](semantic_segmentation)، لكن توجد فروقات مهمة. تُدرَّب نماذج التقسيم على مجموعات بيانات معنونة ومحدودة بالفئات التي شاهدتها أثناء التدريب؛ إذ تُرجِع مجموعة أقنعة مع الفئات المقابلة للصورة المدخلة.

تُدرَّب نماذج توليد الأقنعة على كميات كبيرة من البيانات وتعمل بنمطين:
- نمط التلقين (Prompting): في هذا النمط يستقبل النموذج صورة ومُلقِّنًا (prompt)، حيث يمكن أن يكون المُلقِّن نقطة ثنائية الأبعاد (إحداثيات XY) داخل جسم ما، أو صندوق إحاطة يحيط بجسم. في نمط التلقين، يعيد النموذج القناع الخاص بالجسم الذي يُشير إليه المُلقِّن فقط.
- نمط "قسّم كل شيء" (Segment Everything): في هذا النمط، وبالإعطاء صورة، يولّد النموذج جميع الأقنعة في الصورة. لتحقيق ذلك، تُولَّد شبكة من النقاط وتُسقَط على الصورة أثناء الاستدلال.

تدعم مهمة توليد الأقنعة [نموذج Segment Anything (SAM)](model_doc/sam). وهو نموذج قوي يتكوّن من مُرمِّز صورة قائم على Vision Transformer، ومُرمِّز للمُلقِّن، ومُفكِّك أقنعة (mask decoder) يعتمد مُحوِّلًا ذهابًا وإيابًا (two-way transformer). تُرمَّز الصور والمُلقِّنات، ثم يتلقى المُفكِّك هذه التمثيلات ويولّد أقنعة صالحة.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sam.png" alt="بنية SAM"/>
</div>

يخدم SAM كنموذج تأسيسي قوي للتقسيم نظرًا لتغطيته الكبيرة للبيانات. فقد تم تدريبه على 
[SA-1B](https://ai.meta.com/datasets/segment-anything/)، وهي مجموعة بيانات تضم مليون صورة و1.1 مليار قناع.

في هذا الدليل ستتعلم كيف:
- تُجري الاستدلال في نمط "قسّم كل شيء" مع التدفّعات (batching)،
- تُجري الاستدلال بنمط التلقين بالنقاط،
- تُجري الاستدلال بنمط التلقين بالصناديق.

أولًا لنثبّت `transformers`:

```bash
pip install -q transformers
```

## خط أنابيب توليد الأقنعة

أسهل طريقة للاستدلال بنماذج توليد الأقنعة هي استخدام خط الأنابيب `mask-generation`.

```python
>>> from transformers import pipeline

>>> checkpoint = "facebook/sam-vit-base"
>>> mask_generator = pipeline(model=checkpoint, task="mask-generation")
```

لنطّلع على الصورة:

```python
from PIL import Image
import requests

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="صورة مثال"/>
</div>

لنقسّم كل شيء. يتيح `points-per-batch` إجراء الاستدلال المتوازي للنقاط في نمط "قسّم كل شيء"، ما يسرّع الاستدلال لكنه يستهلك ذاكرة أكبر. علاوة على ذلك، لا يدعم SAM التدفّع إلا عبر النقاط وليس الصور. أمّا `pred_iou_thresh` فهو حدّ ثقة IoU حيث تُعاد الأقنعة التي تتجاوز هذا الحد فقط.

```python
masks = mask_generator(image, points_per_batch=128, pred_iou_thresh=0.88)
```

تشبه بنية `masks` ما يلي:

```bash
{'masks': [array([[False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True,  True],
         ...,
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False]]),
  array([[False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         ...,
 'scores': tensor([0.9972, 0.9917,
        ...,
}
```

يمكننا عرضها بصريًا كالتالي:

```python
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')

for i, mask in enumerate(masks["masks"]):
    plt.imshow(mask, cmap='viridis', alpha=0.1, vmin=0, vmax=1)

plt.axis('off')
plt.show()
```

فيما يلي الصورة الأصلية بتدرج رمادي مع خرائط لونية فوقها. مبهر بالفعل.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee_segmented.png" alt="تصوّر النتائج"/>
</div>


## الاستدلال باستخدام النموذج مباشرةً

### تلقين بالنقطة

يمكنك أيضًا استخدام النموذج دون خط الأنابيب. لفعل ذلك، هيّئ النموذج والمعالج.

```python
from transformers import SamModel, SamProcessor
import torch
from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
device, _, _ = get_backend()
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
```

لإجراء التلقين بالنقطة، مرّر نقطة الإدخال إلى المعالج، ثم خذ مخرجاته ومرّرها إلى النموذج للاستدلال. لمعالجة المخرجات لاحقًا، مرّر كلًا من المخرجات و`original_sizes` و`reshaped_input_sizes` التي نحصل عليها من مخرجات المعالج الأولية. نحتاج ذلك لأن المعالج يُعيد تحجيم الصورة، ويجب إسقاط المخرجات على حجمها الأصلي.

```python
input_points = [[[2592, 1728]]] # point location of the bee

inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
```

يمكننا عرض الأقنعة الثلاثة في مخرجات `masks`.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')
mask_list = [masks[0][0][0].numpy(), masks[0][0][1].numpy(), masks[0][0][2].numpy()]

for i, mask in enumerate(mask_list, start=1):
    overlayed_image = np.array(image).copy()

    overlayed_image[:,:,0] = np.where(mask == 1, 255, overlayed_image[:,:,0])
    overlayed_image[:,:,1] = np.where(mask == 1, 0, overlayed_image[:,:,1])
    overlayed_image[:,:,2] = np.where(mask == 1, 0, overlayed_image[:,:,2])
    
    axes[i].imshow(overlayed_image)
    axes[i].set_title(f'Mask {i}')
for ax in axes:
    ax.axis('off')

plt.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/masks.png" alt="تصوّر الأقنعة"/>
</div>

### تلقين بالصندوق

يمكنك أيضًا إجراء التلقين بالصندوق بطريقة مشابهة للتلقين بالنقطة. ما عليك سوى تمرير صندوق الإدخال بالصيغة `[x_min, y_min, x_max, y_max]` مع الصورة إلى `processor`. خذ مخرجات المعالج ومرّرها مباشرةً إلى النموذج، ثم عالِج المخرجات لاحقًا.

```python
# bounding box around the bee
box = [2350, 1600, 2850, 2100]

inputs = processor(
        image,
        input_boxes={[[[box]]]},
        return_tensors="pt"
    ).to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

mask = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)[0][0][0].numpy()
```

يمكنك إظهار صندوق الإحاطة حول النحلة كما يلي.

```python
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.imshow(image)

rectangle = patches.Rectangle((2350, 1600), 500, 500, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rectangle)
ax.axis("off")
plt.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/bbox.png" alt="صندوق الإحاطة"/>
</div>

يمكنك مشاهدة ناتج الاستدلال أدناه.

```python
fig, ax = plt.subplots()
ax.imshow(image)
ax.imshow(mask, cmap='viridis', alpha=0.4)

ax.axis("off")
plt.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/box_inference.png" alt="تصوّر الاستدلال"/>
</div>
