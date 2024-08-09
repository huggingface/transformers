# إنشاء الأقنعة

تتمثل مهمة إنشاء القناع في إنشاء أقنعة ذات معنى دلالي لصورة ما. هذه المهمة مشابهة جدًا لقطاع [الصور](semantic_segmentation)، ولكن هناك العديد من الاختلافات. يتم تدريب نماذج تجزئة الصور على مجموعات بيانات موسومة ومقيدة بالطبقات التي شاهدتها أثناء التدريب؛ فهي تعيد مجموعة من الأقنعة والطبقات المقابلة، نظرًا للصورة.

يتم تدريب نماذج إنشاء القناع على كميات كبيرة من البيانات وتعمل في وضعين.

- وضع الإشارة: في هذا الوضع، يأخذ النموذج الصورة وإشارة، حيث يمكن أن تكون الإشارة موقع نقطة ثنائية الأبعاد (الإحداثيات XY) داخل كائن في الصورة أو مربع حدود يحيط بكائن. في وضع الإشارة، يعيد النموذج القناع فقط على الكائن الذي تشير إليه الإشارة.

- وضع تجزئة كل شيء: في تجزئة كل شيء، نظرًا للصورة، يقوم النموذج بإنشاء كل قناع في الصورة. للقيام بذلك، يتم إنشاء شبكة من النقاط ووضعها فوق الصورة للاستدلال.

تتم دعم مهمة إنشاء القناع بواسطة [نموذج تجزئة أي شيء (SAM)](model_doc/sam). إنه نموذج قوي يتكون من محول رؤية قائم على محول، ومشفر إشارة، وفك تشفير قناع محول ثنائي الاتجاه. يتم تشفير الصور والإشارات، ويأخذ فك التشفير هذه التضمينات وينشئ أقنعة صالحة.

![هندسة SAM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sam.png)

يعمل SAM كنموذج أساسي قوي للتجزئة نظرًا لتغطيته الكبيرة للبيانات. يتم تدريبه على [SA-1B](https://ai.meta.com/datasets/segment-anything/)، مجموعة بيانات تحتوي على مليون صورة و1.1 مليار قناع.

في هذا الدليل، ستتعلم كيفية:

- الاستنتاج في وضع تجزئة كل شيء مع التجميع،
- الاستنتاج في وضع إشارة النقطة،
- الاستنتاج في وضع إشارة المربع.

أولاً، دعنا نقوم بتثبيت `المحولات`:

```bash
pip install -q transformers
```

## خط أنابيب إنشاء القناع

أسهل طريقة للاستدلال على نماذج إنشاء القناع هي استخدام خط أنابيب `إنشاء القناع`.

```python
>>> from transformers import pipeline

>>> checkpoint = "facebook/sam-vit-base"
>>> mask_generator = pipeline(model=checkpoint, task="mask-generation")
```

دعنا نرى الصورة.

```python
from PIL import Image
import requests

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
```

![صورة مثال](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg)

دعنا نقسم كل شيء. `النقاط لكل دفعة` تمكن الاستدلال الموازي للنقاط في وضع تجزئة كل شيء. يمكّن هذا الاستدلال الأسرع، ولكنه يستهلك ذاكرة أكبر. علاوة على ذلك، لا تمكّن SAM التجميع عبر الصور فقط ولكن أيضًا عبر النقاط. `pred_iou_thresh` هو عتبة ثقة IoU حيث يتم إرجاع الأقنعة الموجودة فوق عتبة معينة فقط.

```python
masks = mask_generator(image, points_per_batch=128, pred_iou_thresh=0.88)
```

تبدو `الأقنعة` على النحو التالي:

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

يمكننا تصورها على النحو التالي:

```python
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')

for i, mask in enumerate(masks["masks"]):
    plt.imshow(mask, cmap='viridis', alpha=0.1, vmin=0, vmax=1)

plt.axis('off')
plt.show()
```

فيما يلي الصورة الأصلية باللون الرمادي مع خرائط ملونة فوقها. رائع جدا.

![مرئي](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee_segmented.png)

## استنتاج النموذج

### إشارة نقطة

يمكنك أيضًا استخدام النموذج بدون خط الأنابيب. للقيام بذلك، قم بتهيئة النموذج والمعالج.

```python
from transformers import SamModel, SamProcessor
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
```

للقيام بالإشارة النقطية، قم بتمرير نقطة الإدخال إلى المعالج، ثم خذ إخراج المعالج
وقم بتمريره إلى النموذج للاستدلال. لمعالجة إخراج النموذج، قم بتمرير الإخراج و
`الأحجام الأصلية` و`reshaped_input_sizes` نأخذ من الإخراج الأولي للمعالج. نحن بحاجة إلى تمرير هذه
نظرًا لأن المعالج يقوم بإعادة تحجيم الصورة، ويجب استقراء الإخراج.

```python
input_points = [[[2592, 1728]]] # point location of the bee

inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
```

يمكننا تصور الأقنعة الثلاثة في إخراج `الأقنعة`.

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

![مرئي](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/masks.png)

### إشارة المربع

يمكنك أيضًا إجراء إشارة المربع بطريقة مماثلة للإشارة النقطية. يمكنك ببساطة تمرير مربع الإدخال بتنسيق قائمة
`[x_min، y_min، x_max، y_max]` تنسيق جنبًا إلى جنب مع الصورة إلى `المعالج`. خذ إخراج المعالج ومرره مباشرةً
إلى النموذج، ثم قم بمعالجة الإخراج مرة أخرى.

```python
# bounding box around the bee
box = [2350, 1600, 2850, 2100]

inputs = processor(
        image,
        input_boxes=[[[box]]],
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

يمكنك تصور مربع الحدود حول النحلة كما هو موضح أدناه.

```python
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.imshow(image)

rectangle = patches.Rectangle((2350, 1600), 500, 500, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rectangle)
ax.axis("off")
plt.show()
```

![مرئي Bbox](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/bbox.png)

يمكنك رؤية إخراج الاستدلال أدناه.

```python
fig, ax = plt.subplots()
ax.imshow(image)
ax.imshow(mask, cmap='viridis', alpha=0.4)

ax.axis("off")
plt.show()
```

![مرئي الاستدلال](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/box_inference.png)