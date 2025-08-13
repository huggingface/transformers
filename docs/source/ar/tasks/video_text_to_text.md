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

# الفيديو-إلى-نص (Video-text-to-text)

[[open-in-colab]]

نماذج الفيديو-إلى-نص، والمعروفة أيضًا باسم نماذج لغة الفيديو أو نماذج الرؤية-اللغة ذات مُدخلات الفيديو، هي نماذج لغوية تأخذ فيديو كمدخل. يمكن لهذه النماذج التعامل مع مهام متنوعة، من الإجابة على أسئلة الفيديو إلى توليد الأوصاف النصية للفيديو.

تشترك هذه النماذج في بنية قريبة جدًا من نماذج [الصورة-إلى-نص](../image_text_to_text) مع بعض التعديلات لقبول بيانات الفيديو، إذ إن بيانات الفيديو هي في جوهرها إطارات صور مع اعتماد زمني (temporal dependencies). بعض نماذج الصورة-إلى-نص تقبل صورًا متعددة، لكن ذلك وحده غير كافٍ لقبول الفيديوهات. علاوة على ذلك، غالبًا ما تُدرَّب نماذج الفيديو-إلى-نص على جميع أنماط الرؤية. قد يحتوي كل مثال على فيديوهات (متعددة)، وصور (متعددة). بعض هذه النماذج تدعم مدخلات متداخلة (interleaved). على سبيل المثال، يمكنك الإشارة إلى فيديو معين داخل سلسلة نصية بإضافة رمز فيديو في النص مثل: "What is happening in this video? `<video>`".

في هذا الدليل، نقدّم لمحة موجزة عن نماذج لغة الفيديو ونوضّح كيفية استخدامها مع Transformers للاستدلال.

للبدء، هناك عدة أنواع من نماذج لغة الفيديو:
- نماذج أساسية (base models) تُستخدم للضبط الموجّه (fine-tuning)
- نماذج مضبوطة للمحادثة (chat fine-tuned) للمحادثة
- نماذج مضبوطة بالتعليمات (instruction fine-tuned)

يركّز هذا الدليل على الاستدلال باستخدام نموذج مضبوط بالتعليمات، وهو [llava-hf/llava-interleave-qwen-7b-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf) القادر على قبول بيانات متداخلة. بديلًا من ذلك، يمكنك تجربة [llava-interleave-qwen-0.5b-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf) إذا كانت عتادك لا يسمح بتشغيل نموذج 7B.

لنبدأ بتثبيت التبعيات.

```bash
pip install -q transformers accelerate flash_attn 
```

لنُهيّئ النموذج والمعالِج (processor).

```python
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

processor = LlavaProcessor.from_pretrained(model_id)

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to("cuda") # يمكن أن تكون xpu أو mps أو npu... حسب المسرّع المتوفر لديك
```

بعض النماذج تستهلك رمز `<video>` مباشرةً، وأخرى تقبل عددًا من رموز `<image>` يساوي عدد الإطارات المُختارة (sampled frames). يتعامل هذا النموذج مع الفيديوهات بالطريقة الثانية. سنكتب أداة صغيرة للتعامل مع رموز الصور، وأخرى لجلب فيديو من عنوان URL وأخذ عينات من إطاراته.

```python
import uuid
import requests
import cv2
from PIL import Image

def replace_video_with_images(text, frames):
  return text.replace("<video>", "<image>" * frames)

def sample_frames(url, num_frames):

    response = requests.get(url)
    path_id = str(uuid.uuid4())

    path = f"./{path_id}.mp4" 

    with open(path, "wb") as f:
      f.write(response.content)

    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames[:num_frames]
```

لنحضّر المُدخلات. سنأخذ عينات من الإطارات ونقوم بربطها معًا.

```python
video_1 = "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4"
video_2 = "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_2.mp4"

video_1 = sample_frames(video_1, 6)
video_2 = sample_frames(video_2, 6)

videos = video_1 + video_2

videos

# [<PIL.Image.Image image mode=RGB size=1920x1080>,
# <PIL.Image.Image image mode=RGB size=1920x1080>,
# <PIL.Image.Image image mode=RGB size=1920x1080>, ...]
```

كلا الفيديوهين يعرض قططًا.

<div class="container">
  <div class="video-container">
    <video width="400" controls>
      <source src="https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4" type="video/mp4">
    </video>
  </div>

  <div class="video-container">
    <video width="400" controls>
      <source src="https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_2.mp4" type="video/mp4">
    </video>
  </div>
</div>

يمكننا الآن إجراء المعالجة المسبقة للمُدخلات.

يحتوي هذا النموذج على قالب محادثة (prompt template) كما يلي. أولًا، سنضع جميع الإطارات المُختارة في قائمة واحدة. بما أن لدينا ثمانية إطارات في كل فيديو، سنُدرج 12 رمز `<image>` في الموجه. أضف `assistant` في نهاية الموجه لتحفيز النموذج على إعطاء الإجابات. بعدها يمكننا إجراء المعالجة المسبقة.

```python
user_prompt = "Are these two cats in these two videos doing the same thing?"
toks = "<image>" * 12
prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
inputs = processor(text=prompt, images=videos, return_tensors="pt").to(model.device, model.dtype)
```

يمكننا الآن استدعاء [`~GenerationMixin.generate`] للاستدلال. يُنتج النموذج السؤال في مُدخلاتنا والإجابة، لذا سنأخذ فقط النص بعد جزء الموجه و`assistant` من مخرجات النموذج.

```python
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:])

# The first cat is shown in a relaxed state, with its eyes closed and a content expression, while the second cat is shown in a more active state, with its mouth open wide, possibly in a yawn or a vocalization.


```

وهكذا ببساطة!

لتعلّم المزيد حول قوالب المحادثة (chat templates) وبثّ الرموز (token streaming) لنماذج الفيديو-إلى-نص، ارجع إلى دليل مهمة [الصورة-إلى-نص](../tasks/image_text_to_text) لأن هذه النماذج تعمل بطريقة متشابهة.
