# نماذج الصور والنص والنص

[[open-in-colab]]

نماذج الصور والنص والنص، المعروفة أيضًا باسم نماذج اللغة المرئية (VLMs)، هي نماذج لغة تأخذ إدخال صورة. يمكن لهذه النماذج معالجة مهام مختلفة، بدءًا من الإجابة على الأسئلة المرئية إلى تجزئة الصور. تشترك هذه المهمة في العديد من أوجه التشابه مع الصورة إلى النص، ولكن مع بعض حالات الاستخدام المتداخلة مثل تعليق الصور. تقبل نماذج الصورة إلى النص إدخالات الصور فقط وغالبًا ما تنجز مهمة محددة، في حين أن نماذج اللغة المرئية تقبل إدخالات نصية وصورية مفتوحة وتكون نماذج أكثر عمومية.

في هذا الدليل، نقدم نظرة عامة موجزة على نماذج اللغة المرئية ونظهر كيفية استخدامها مع Transformers للاستدلال.

بدايةً، هناك عدة أنواع من نماذج اللغة المرئية:

- النماذج الأساسية المستخدمة للضبط الدقيق
- النماذج المحسنة للمحادثة للمحادثة
- النماذج المحسنة للتعليمات

يركز هذا الدليل على الاستدلال باستخدام نموذج مضبوط للتعليمات.

دعونا نبدأ بتثبيت التبعيات.

```bash
pip install -q transformers accelerate flash_attn
```

دعونا نقوم بتهيئة النموذج والمعالج.

```python
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
import torch

device = torch.device("cuda")
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
```

يحتوي هذا النموذج على [قالب دردشة](./chat_templating) يساعد المستخدم على تحليل إخراج الدردشة. علاوة على ذلك، يمكن للنموذج أيضًا قبول صور متعددة كإدخال في محادثة واحدة أو رسالة واحدة. سنقوم الآن بإعداد الإدخالات.

تبدو إدخالات الصور كما يلي.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png" alt="قطتان تجلسان على شبكة"/>
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

فيما يلي مثال على قالب الدردشة. يمكننا إدخال دورات المحادثة والرسالة الأخيرة كإدخال عن طريق إضافتها في نهاية القالب.

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

سنقوم الآن باستدعاء طريقة [`~ProcessorMixin.apply_chat_template`] للمعالجات لمعالجة إخراجها جنبًا إلى جنب مع إدخالات الصور.

```python
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[images[0], images[1]], return_tensors="pt").to(device)
```

الآن يمكننا تمرير الإدخالات المعالجة مسبقًا إلى النموذج.

```python
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
## ['User: What do we see in this image? \nAssistant: In this image we can see two cats on the nets. \nUser: And how about this image? \nAssistant: In this image we can see flowers, plants and insect.']
```

## البث

يمكننا استخدام [بث النص](./generation_strategies#streaming) للحصول على تجربة إنشاء أفضل. تدعم Transformers البث باستخدام فئات [`TextStreamer`] أو [`TextIteratorStreamer`]. سنستخدم [`TextIteratorStreamer`] مع IDEFICS-8B.

لنفترض أن لدينا تطبيقًا يحتفظ بتاريخ الدردشة ويأخذ إدخال المستخدم الجديد. سنقوم بمعالجة الإدخالات كالمعتاد ونقوم بتهيئة [`TextIteratorStreamer`] للتعامل مع الإنشاء في خيط منفصل. يتيح لك ذلك بث رموز النص المولدة في الوقت الفعلي. يمكن تمرير أي حجج إنشاء إلى [`TextIteratorStreamer`].

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

الآن دعونا نستدعي وظيفة `model_inference` التي أنشأناها ونبث القيم.

```python
generator = model_inference(
    user_prompt="And what is in this image?",
    chat_history=messages,
    max_new_tokens=100,
    images=images
)

for value in generator:
  print(value)

# In
# In this
# In this image ...
```

## ملاءمة النماذج في الأجهزة الصغيرة

غالبًا ما تكون نماذج اللغة المرئية كبيرة وتحتاج إلى تحسين لتناسب الأجهزة الصغيرة. تدعم Transformers العديد من مكتبات كمية النموذج، وهنا سنعرض فقط التكميم int8 باستخدام [Quanto](./quantization/quanto#quanto). توفر الكمية int8 تحسينات في الذاكرة تصل إلى 75 بالمائة (إذا كانت جميع الأوزان مكدسة). ومع ذلك، فهو ليس مجانيًا، نظرًا لأن 8 بتات ليست دقة أصلية لـ CUDA، يتم إعادة تحجيم الأوزان ذهابًا وإيابًا أثناء التنقل، مما يؤدي إلى زيادة الكمون.

أولاً، قم بتثبيت التبعيات.

```bash
pip install -U quanto bitsandbytes
```

لكميات نموذج أثناء التحميل، نحتاج أولاً إلى إنشاء [`QuantoConfig`]. ثم قم بتحميل النموذج كالمعتاد، ولكن قم بتمرير `quantization_config` أثناء تهيئة النموذج.

```python
from transformers import Idefics2ForConditionalGeneration, AutoTokenizer, QuantoConfig

model_id = "HuggingFaceM4/idefics2-8b"
quantization_config = QuantoConfig(weights="int8")
quantized_model = Idefics2ForConditionalGeneration.from_pretrained(model_id, device_map="cuda", quantization_config=quantization_config)
```

وهذا كل شيء، يمكننا استخدام النموذج بنفس الطريقة دون إجراء أي تغييرات.

## قراءة إضافية

فيما يلي بعض الموارد الإضافية لمهمة الصورة والنص والنص.

- تغطي [صفحة مهمة الصورة والنص والنص](https://huggingface.co/tasks/image-text-to-text) أنواع النماذج وحالات الاستخدام ومجموعات البيانات والمزيد.
- [نماذج اللغة المرئية موضحة](https://huggingface.co/blog/vlms) هي مشاركة مدونة تغطي كل شيء عن نماذج اللغة المرئية والضبط الدقيق الخاضع للإشراف باستخدام [TRL](https://huggingface.co/docs/trl/en/index).