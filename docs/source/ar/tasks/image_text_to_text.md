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

# صورة-نص-إلى-نص (Image-text-to-text)

[[open-in-colab]]

نماذج صورة-نص-إلى-نص، والمعروفة أيضًا باسم نماذج اللغة البصرية (Vision Language Models - VLMs)، هي نماذج لغوية تستقبل صورة كمدخل. تستطيع هذه النماذج معالجة مهام متعددة، بدءًا من الإجابة عن الأسئلة البصرية وصولًا إلى تجزئة الصور. تشترك هذه المهمة في الكثير مع مهمة صورة-إلى-نص، مع تداخل في بعض حالات الاستخدام مثل توليد توصيف الصورة. تختلف نماذج صورة-إلى-نص بأنها لا تستقبل سوى الصور وعادةً ما تنفذ مهمة محددة، بينما تستقبل نماذج VLMs مدخلات مفتوحة الطرف من نصوص وصور وتكون أكثر عمومية.

في هذا الدليل، نقدّم لمحة موجزة عن نماذج VLMs ونوضح كيفية استخدامها مع Transformers من أجل الاستدلال (Inference).

لبداية الأمر، هناك عدة أنواع من نماذج VLMs:
- نماذج أساسية تُستخدَم من أجل الضبط الدقيق (fine-tuning)
- نماذج مضبوطة للمحادثة (chat fine-tuned) للمحادثة التفاعلية
- نماذج مضبوطة بالتعليمات (instruction fine-tuned)

يركّز هذا الدليل على الاستدلال باستخدام نموذج مضبوط بالتعليمات.

فلنبدأ بتثبيت الاعتمادات.

```bash
pip install -q transformers accelerate flash_attn
```

لنقم بتهيئة النموذج والمُعالِج (processor).

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

device = torch.device("cuda")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
```

يحتوي هذا النموذج على [قالب محادثة](./chat_templating) يساعد في تحليل مخرجات الدردشة. إضافةً إلى ذلك، يستطيع النموذج أيضًا استقبال عدة صور كمدخل ضمن محادثة واحدة أو رسالة واحدة. سنحضّر المدخلات الآن.

تبدو مدخلات الصور كما يلي.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png" alt="قطّتان تجلسان على شبكة"/>
</div>

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="نحلة على زهرة وردية"/>
</div>


```python
from PIL import Image
import requests

img_urls =["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
           "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"]
images = [Image.open(requests.get(img_urls[0], stream=True).raw),
          Image.open(requests.get(img_urls[1], stream=True).raw)]
```

فيما يلي مثال على قالب المحادثة. يمكننا تمرير أدوار المحادثة والرسالة الأخيرة بإلحاقها في نهاية القالب.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In this image we can see two cats on the nets."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ]
    },
]
```

سنستدعي الآن دالة [`~ProcessorMixin.apply_chat_template`] الخاصة بالمُعالِج لمعالجة مخرجاته مع مدخلات الصور.

```python
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[images[0], images[1]], return_tensors="pt").to(device)
```

يمكننا الآن تمرير المدخلات المعالجة مسبقًا إلى النموذج.

```python
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
## ['User: What do we see in this image? \nAssistant: In this image we can see two cats on the nets. \nUser: And how about this image? \nAssistant: In this image we can see flowers, plants and insect.']
```

## خط الأنابيب (Pipeline)

أسرع طريقة للبدء هي استخدام واجهة [`Pipeline`]. حدّد مهمة "image-text-to-text" والنموذج الذي تريد استخدامه.

```python
from transformers import pipeline
pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
```

يستخدم المثال أدناه قوالب المحادثة لتنسيق مدخلات النص.

```python
messages = [
     {
         "role": "user",
         "content": [
             {
                 "type": "image",
                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
             },
             {"type": "text", "text": "Describe this image."},
         ],
     },
     {
         "role": "assistant",
         "content": [
             {"type": "text", "text": "There's a pink flower"},
         ],
     },
 ]
```

مرّر نص القالب التنسيقي والصورة إلى [`Pipeline`] واضبط `return_full_text=False` لإزالة الإدخال من الإخراج المُولَّد.

```python
outputs = pipe(text=messages, max_new_tokens=20, return_full_text=False)
outputs[0]["generated_text"]
#  with a yellow center in the foreground. The flower is surrounded by red and white flowers with green stems
```

إذا رغبت، يمكنك أيضًا تحميل الصور بشكل منفصل وتمريرها إلى الـ pipeline كما يلي:

```python
pipe = pipeline("image-text-to-text", model="HuggingFaceTB/SmolVLM-256M-Instruct")

img_urls = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
]
images = [
    Image.open(requests.get(img_urls[0], stream=True).raw),
    Image.open(requests.get(img_urls[1], stream=True).raw),
]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": "What do you see in these images?"},
        ],
    }
]
outputs = pipe(text=messages, images=images, max_new_tokens=50, return_full_text=False)
outputs[0]["generated_text"]
" In the first image, there are two cats sitting on a plant. In the second image, there are flowers with a pinkish hue."
```

ستظل الصور مُتضمَّنة في الحقل "input_text" ضمن المخرجات:

```python
outputs[0]['input_text']
"""
[{'role': 'user',
  'content': [{'type': 'image',
    'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=622x412>},
   {'type': 'image',
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=5184x3456>},
   {'type': 'text', 'text': 'What do you see in these images?'}]}]## Streaming
"""
```

يمكننا استخدام [بثّ النص](./generation_strategies#streaming) لتحسين تجربة التوليد. يدعم Transformers البث عبر الصنفين [`TextStreamer`] أو [`TextIteratorStreamer`]. سنستخدم هنا [`TextIteratorStreamer`] مع IDEFICS-8B.

لنفترض أنّ لدينا تطبيقًا يحتفظ بسجلّ المحادثة ويتلقى مُدخل المستخدم الجديد. سنقوم بمعالجة المدخلات كالمعتاد ونهيّئ [`TextIteratorStreamer`] للتعامل مع التوليد في خيط (thread) منفصل. يتيح لك ذلك بثّ رموز النص المُولَّد في الزمن الحقيقي. يمكن تمرير أي وسيطات توليد إلى [`TextIteratorStreamer`].

```python
import time
from transformers import TextIteratorStreamer
from threading import Thread

def model_inference(
    user_prompt,
    chat_history,
    max_new_tokens,
    images
):
    user_prompt = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]
    }
    chat_history.append(user_prompt)
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        timeout=5.0,
    )

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "streamer": streamer,
        "do_sample": False
    }

    # add_generation_prompt=True makes model generate bot response
    prompt = processor.apply_chat_template(chat_history, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    ).to(device)
    generation_args.update(inputs)

    thread = Thread(
        target=model.generate,
        kwargs=generation_args,
    )
    thread.start()

    acc_text = ""
    for text_token in streamer:
        time.sleep(0.04)
        acc_text += text_token
        if acc_text.endswith("<end_of_utterance>"):
            acc_text = acc_text[:-18]
        yield acc_text

    thread.join()
```

فلنستدعِ الآن الدالة `model_inference` التي أنشأناها ونبثّ القيم.

```python
generator = model_inference(
    user_prompt="And what is in this image?",
    chat_history=messages[:2],
    max_new_tokens=100,
    images=images
)

for value in generator:
  print(value)

# In
# In this
# In this image ...
```

## ملاءمة النماذج للأجهزة الأصغر

تكون نماذج VLMs كبيرة غالبًا وتحتاج إلى تحسين لتناسب الأجهزة الأصغر. يدعم Transformers العديد من مكتبات تقليل الدقّة (quantization)، وهنا سنعرض فقط تقليل الدقّة إلى int8 باستخدام [Quanto](./quantization/quanto#quanto). يوفّر int8 تحسينات في الذاكرة تصل إلى 75 بالمئة (إذا تم تقليل دقّة جميع الأوزان). لكن لا وجبة مجانية هنا؛ إذ إن 8-بت ليست دقّة أصلية في CUDA، لذلك تُخفَّض الأوزان وتُعاد كمّها أثناء التشغيل، ما يضيف زمن تأخير (latency).

أولًا، ثبّت الاعتمادات.

```bash
pip install -U quanto bitsandbytes
```

لكي نُقلّل دقّة نموذج أثناء التحميل، نحتاج أولًا إلى إنشاء [`QuantoConfig`]. بعد ذلك نحمّل النموذج كالمعتاد، لكن نمرّر `quantization_config` أثناء تهيئة النموذج.

```python
from transformers import AutoModelForImageTextToText, QuantoConfig

model_id = "HuggingFaceM4/idefics2-8b"
quantization_config = QuantoConfig(weights="int8")
quantized_model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="cuda", quantization_config=quantization_config
)
```

وهذا كل شيء؛ يمكننا استخدام النموذج بالطريقة نفسها دون تغييرات.

## قراءة إضافية

إليك المزيد من الموارد حول مهمة صورة-نص-إلى-نص.

- [صفحة مهمة صورة-نص-إلى-نص](https://huggingface.co/tasks/image-text-to-text) تغطي أنواع النماذج وحالات الاستخدام ومجموعات البيانات والمزيد.
- [شرح نماذج اللغة البصرية](https://huggingface.co/blog/vlms) تدوينة تغطي كل ما يتعلق بنماذج اللغة البصرية والضبط الدقيق الخاضع للإشراف باستخدام [TRL](https://huggingface.co/docs/trl/en/index).
