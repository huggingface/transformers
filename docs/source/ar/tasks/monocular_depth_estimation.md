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

# تقدير العمق أحادي العين (Monocular depth estimation)

تقدير العمق أحادي العين هو مهمة في الرؤية الحاسوبية تتضمن التنبؤ بمعلومات العمق لمشهد انطلاقًا من صورة واحدة. بمعنى آخر، تقدير مسافة الأجسام في المشهد من منظور كاميرا واحد.

لتقدير العمق أحادي العين تطبيقات عدّة، منها إعادة البناء ثلاثي الأبعاد والواقع المعزّز والقيادة الذاتية والروبوتات. وهي مهمة صعبة لأنها تتطلب من النموذج فهم العلاقات المعقدة بين كائنات المشهد ومعلومات العمق المقابلة لها، والتي قد تتأثر بعوامل مثل الإضاءة والانسدال (occlusion) والملمس.

هناك فئتان رئيسيتان لتقدير العمق:

- **تقدير العمق المطلق**: يهدف هذا المتغير من المهمة إلى تقديم قياسات عمق دقيقة بالنسبة للكاميرا. ويُستخدم المصطلح بالتبادل مع تقدير العمق المتري، حيث يُعبَّر عن العمق بقراءات حقيقية بالأمتار أو الأقدام. تعيد نماذج العمق المطلق خرائط عمق بقيم رقمية تمثل مسافات واقعية.

- **تقدير العمق النسبي**: يهدف إلى توقّع ترتيب العمق بين الأجسام أو النقاط دون تقديم القياسات الدقيقة. تعيد هذه النماذج خريطة عمق تُظهر أي أجزاء أقرب أو أبعد نسبيًا دون ذكر المسافة الفعلية.

في هذا الدليل سنرى كيفية الاستدلال باستخدام [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)، وهو نموذج حديث للحالة الصفرية (zero-shot) لتقدير العمق النسبي، وباستخدام [ZoeDepth](https://huggingface.co/docs/transformers/main/en/model_doc/zoedepth)، وهو نموذج لتقدير العمق المطلق.

<Tip>

اطّلع على صفحة مهمة [تقدير العمق](https://huggingface.co/tasks/depth-estimation) للاطلاع على كل البنى والمعايير المتوافقة.

</Tip>

قبل البدء، نحتاج لتثبيت أحدث إصدار من Transformers:

```bash
pip install -q -U transformers
```

## خط أنابيب تقدير العمق

أسهل طريقة لتجربة الاستدلال بنموذج يدعم تقدير العمق هي استخدام [`pipeline`].
هيّئ خط أنابيب من [معيار (checkpoint) على Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads):

```py
>>> from transformers import pipeline
>>> import torch
>>> from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
>>> device, _, _ = get_backend()
>>> checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
>>> pipe = pipeline("depth-estimation", model=checkpoint, device=device)
```

بعدها اختر صورة لتحليلها:

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="صورة نحلة"/>
</div>

مرّر الصورة إلى خط الأنابيب.

```py
>>> predictions = pipe(image)
```

يعيد خط الأنابيب قاموسًا من مُدخلين. الأول `predicted_depth` وهو Tensor بقيم تمثل العمق بالأمتار لكل بكسل. والثاني `depth` وهو صورة PIL تُظهر تصور نتيجة تقدير العمق.

لنتفحص النتيجة المصوّرة:

```py
>>> predictions["depth"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png" alt="تصوّر تقدير العمق"/>
</div>

## الاستدلال يدويًا

بعد رؤية استخدام خط الأنابيب، لنكرر النتيجة يدويًا.

ابدأ بتحميل النموذج والمعالج المرتبط من [معيار على Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads). سنستخدم نفس المعيار السابق:

```py
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation

>>> checkpoint = "Intel/zoedepth-nyu-kitti"

>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)
```

حضّر مُدخل الصورة للنموذج باستخدام `image_processor` الذي سيتولى التحجيم والتطبيع وغيرهما:

```py
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
```

مرّر المدخلات المحضّرة عبر النموذج:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(pixel_values)
```

فلنُعالج النتائج لإزالة أي حشو وإعادة تحجيم خريطة العمق لحجم الصورة الأصلية. تُنتج `post_process_depth_estimation` قائمة قواميس تحتوي على "predicted_depth":

```py
>>> # ZoeDepth يضيف حشوًا ديناميكيًا للصورة؛ لذا نمرّر الحجم الأصلي إلى الدالة لإزالة الحشو وإعادة التحجيم.
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
... )

>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
>>> depth = depth.detach().cpu().numpy() * 255
>>> depth = Image.fromarray(depth.astype("uint8"))
```

<Tip>
<p>في <a href="https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L131">التطبيق الأصلي</a> يجري نموذج ZoeDepth الاستدلال على الصورة الأصلية وصورتها المعكوسة ويأخذ المتوسط. تستطيع دالة <code>post_process_depth_estimation</code> التعامل مع ذلك عبر تمرير المخرجات المعكوسة إلى المعامل الاختياري <code>outputs_flipped</code>:</p>
<pre><code class="language-Python">&gt;&gt;&gt; with torch.no_grad():   
...     outputs = model(pixel_values)
...     outputs_flipped = model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))
&gt;&gt;&gt; post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
...     outputs_flipped=outputs_flipped,
... )
</code></pre>
</Tip>

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization-zoe.png" alt="تصوّر تقدير العمق"/>
</div>
