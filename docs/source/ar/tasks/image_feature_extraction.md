<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# استخراج خصائص الصور

[[open-in-colab]]

استخراج خصائص الصور هي مهمة استخراج ميزات ذات معنى دلالي معين من صورة معينة. ولهذه المهمة العديد من التطبيقات، بما في ذلك تحديد التشابه بين الصور واسترجاع الصور. علاوة على ذلك، يمكن استخدام معظم نماذج الرؤية الحاسوبية لاستخراج خصائص الصور، حيث يمكن إزالة الرأس الخاص بالمهمة (تصنيف الصور، أو اكتشاف الأشياء، وما إلى ذلك) والحصول على الميزات. هذه الميزات مفيدة جدًا على مستوى أعلى: ككشف الحواف، أو زوايا الصور، وما إلى ذلك. وقد تحتوي أيضًا على معلومات حول العالم الحقيقي (على سبيل المثال، شكل القطط) اعتمادًا على عمق النموذج. وبالتالي، يمكن استخدام هذه المخرجات لتدريب مصنفات جديدة على مجموعة بيانات محددة.

في هذا الدليل، سوف:

- تعلم كيفية بناء نظام تشابه صور بسيط يعتمد على خط أنابيب `image-feature-extraction`.
- إنجاز المهمة نفسها باستخدام استدلال النموذج الخام.

## تشابه الصور باستخدام خط أنابيب `image-feature-extraction`

لدينا صورتان لقطط تجلس فوق شباك صيد، إحداهما مُولدة.

```python
from PIL import Image
import requests

img_urls = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"]
image_real = Image.open(requests.get(img_urls[0], stream=True).raw).convert("RGB")
image_gen = Image.open(requests.get(img_urls[1], stream=True).raw).convert("RGB")
```

دعونا نرى خط الأنابيب في العمل. أولاً، قم بتهيئة خط الأنابيب. إذا لم تقم بتمرير أي نموذج إليه، فسيتم تهيئة خط الأنابيب تلقائيًا باستخدام [google/vit-base-patch16-224](google/vit-base-patch16-224). إذا كنت ترغب في حساب التشابه، قم بتعيين `pool` على True.

```python
import torch
from transformers import pipeline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device=DEVICE, pool=True)
```

لاستنتاج باستخدام `pipe` قم بتمرير كلتا الصورتين إليه.

```python
outputs = pipe([image_real, image_gen])
```

يحتوي الإخراج على تضمين مجمع لهذه الصورتين.

```python
# الحصول على طول إخراج واحد
print(len(outputs[0][0]))
# عرض الإخراج
print(outputs)

# 768
# [[[-0.03909236937761307, 0.43381670117378235, -0.06913255900144577,
```

لحساب درجة التشابه، نحتاج إلى تمريرها إلى دالة تشابه.

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(torch.Tensor(outputs[0]),
                                     torch.Tensor(outputs[1]), dim=1)

print(similarity_score)

# tensor([0.6043])
```

إذا كنت ترغب في الحصول على الحالات المخفية الأخيرة قبل التجميع، تجنب تمرير أي قيمة لمعلمة `pool`، حيث يتم تعيينها افتراضيًا على `False`. هذه الحالات المخفية مفيدة لتدريب مصنفات أو نماذج جديدة بناءً على الميزات من النموذج.

```python
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", device=DEVICE)
output = pipe(image_real)
```

نظرًا لأن المخرجات غير مجمعة، نحصل على الحالات المخفية الأخيرة حيث البعد الأول هو حجم الدفعة والأبعاد الأخيرة هي شكل التضمين.

```python
import numpy as np
print(np.array(outputs).shape)
# (1, 197, 768)
```

## الحصول على الميزات والتشابهات باستخدام `AutoModel`

يمكننا أيضًا استخدام فئة `AutoModel` في مكتبة `transformers` للحصول على الميزات. تقوم فئة `AutoModel` بتحميل أي نموذج من نماذج `transformers` بدون رأس خاص بمهمة معينة، ويمكننا استخدام هذا للحصول على الميزات.

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
```

دعونا نكتب دالة استدلال بسيطة. سنمرر المدخلات أولاً إلى المعالج `processor` ثم نمرر مخرجاته إلى النموذج `model`.

```python
def infer(image):
  inputs = processor(image, return_tensors="pt").to(DEVICE)
  outputs = model(**inputs)
  return outputs.pooler_output
```

يمكننا تمرير الصور مباشرة إلى هذه الدالة والحصول على التضمينات.

```python
embed_real = infer(image_real)
embed_gen = infer(image_gen)
```

يمكننا الحصول على التشابه مرة أخرى عبر التضمينات.

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(embed_real, embed_gen, dim=1)
print(similarity_score)

# tensor([0.6061], device='cuda:0', grad_fn=<SumBackward1>)
```