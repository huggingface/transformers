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

# استخراج ميزات الصور (Image Feature Extraction)

[[open-in-colab]]

استخراج ميزات الصور هو مهمة استخراج ميزات ذات دلالة دلالية من صورة مُعطاة. لهذه المهمة استخدامات عديدة، مثل تشابه الصور واسترجاع الصور. علاوة على ذلك، يمكن استخدام معظم نماذج الرؤية الحاسوبية لاستخراج ميزات الصور، حيث يمكن إزالة الرأس الخاص بالمهمة (تصنيف الصور، اكتشاف الكائنات، إلخ) والحصول على الميزات. تكون هذه الميزات مفيدة على مستوى أعلى: مثل كشف الحواف والزوايا، وغيرها. وقد تحتوي أيضًا على معلومات عن العالم الحقيقي (مثل شكل القطة) اعتمادًا على عمق النموذج. لذلك يمكن استخدام هذه المخرجات لتدريب مصنِّفات جديدة على مجموعة بيانات محددة.

في هذا الدليل ستقوم بـ:

- تعلّم بناء نظام بسيط لتشابه الصور بالاعتماد على بايبلاين `image-feature-extraction`.
- إنجاز المهمة نفسها عبر الاستدلال المباشر بالنموذج دون بايبلاين.

## تشابه الصور باستخدام بايبلاين `image-feature-extraction`

لدينا صورتان لقطط جالسة فوق شباك صيد، إحداهما مُولّدة.

```python
from PIL import Image
import requests

img_urls = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"]
image_real = Image.open(requests.get(img_urls[0], stream=True).raw).convert("RGB")
image_gen = Image.open(requests.get(img_urls[1], stream=True).raw).convert("RGB")
```

لنرَ البايبلاين أثناء العمل. أولًا، هيّئ البايبلاين. إذا لم تمرر أي نموذج له، فسيُهيّأ تلقائيًا باستخدام [google/vit-base-patch16-224](google/vit-base-patch16-224). إذا رغبت في حساب التشابه، اضبط `pool` على True.

```python
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
DEVICE, _, _ = get_backend()
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device=DEVICE, pool=True)
```

للاستدلال باستخدام `pipe` مرّر الصورتين إليه مباشرة.

```python
outputs = pipe([image_real, image_gen])
```

تحتوي المخرجات على متجهات مضغوطة (pooled embeddings) لكلتا الصورتين.

```python
# get the length of a single output
print(len(outputs[0][0]))
# show outputs
print(outputs)

# 768
# [[[-0.03909236937761307, 0.43381670117378235, -0.06913255900144577,
```

لحساب درجة التشابه، نحتاج إلى تمرير المتجهات إلى دالة تشابه.

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(torch.Tensor(outputs[0]),
                                     torch.Tensor(outputs[1]), dim=1)

print(similarity_score)

# tensor([0.6043])
```

إذا رغبت في الحصول على آخر الحالات المخفية قبل الدمج (pooling)، تجنّب تمرير أي قيمة للمعامل `pool`، إذ إن قيمته الافتراضية `False`. تكون هذه الحالات المخفية مفيدة لتدريب مصنِّفات أو نماذج جديدة بالاعتماد على ميزات النموذج.

```python
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", device=DEVICE)
outputs = pipe(image_real)
```

نظرًا لكون المخرجات غير مدمجة، سنحصل على آخر الحالات المخفية حيث يُمثّل البُعد الأول حجم الدُفعة (batch size)، وآخر بُعدين يمثلان شكل المتجه.

```python
import numpy as np
print(np.array(outputs).shape)
# (1, 197, 768)
```

## الحصول على الميزات والتشابه باستخدام `AutoModel`

يمكننا استخدام الصنف `AutoModel` من Transformers للحصول على الميزات. يقوم `AutoModel` بتحميل أي نموذج Transformers دون رأس خاص بمهمة معينة، ويمكننا استغلال ذلك لاستخراج الميزات.

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
```

لنكتب دالة بسيطة للاستدلال. سنمرر المدخلات أولًا إلى `processor` ثم نمرر مخرجاته إلى `model`.

```python
def infer(image):
  inputs = processor(image, return_tensors="pt").to(DEVICE)
  outputs = model(**inputs)
  return outputs.pooler_output
```

يمكننا تمرير الصور مباشرةً إلى هذه الدالة والحصول على المتجهات.

```python
embed_real = infer(image_real)
embed_gen = infer(image_gen)
```

يمكننا حساب التشابه مجددًا بالاعتماد على هذه المتجهات.

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(embed_real, embed_gen, dim=1)
print(similarity_score)

# tensor([0.6061], device='cuda:0', grad_fn=<SumBackward1>)
```
