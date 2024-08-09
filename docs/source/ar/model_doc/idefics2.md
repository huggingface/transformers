# Idefics2

## نظرة عامة
اقترح نموذج Idefics2 في [ما الذي يهم عند بناء نماذج الرؤية واللغة؟](https://arxiv.org/abs/2405.02246) بواسطة ليو ترونشون، وهوجو لورينكون، وفيكتور سانه. ويمكن العثور على المنشور المرافق [هنا](https://huggingface.co/blog/idefics2).

Idefics2 هو نموذج مفتوح متعدد الوسائط يقبل تسلسلات تعسفية من إدخالات الصور والنصوص وينتج مخرجات نصية. يمكن للنموذج الإجابة على الأسئلة حول الصور، ووصف المحتوى المرئي، ورواية القصص المستندة إلى صور متعددة، أو التصرف ببساطة كنموذج لغة نقي بدون إدخالات بصرية. إنه يحسن IDEFICS-1، خاصة في فهم المستندات، أو التعرف الضوئي على الحروف، أو الاستدلال البصري. Idefics2 خفيف الوزن (8 مليار معلمة) ويعامل الصور بنسبة عرض إلى ارتفاع أصلية ودقة، مما يسمح بكفاءة استدلال متغيرة.

الملخص من الورقة هو ما يلي:

*لقد أدى الاهتمام المتزايد بنماذج اللغة المرئية (VLMs) إلى تحسينات في نماذج اللغة الكبيرة ومحولات الرؤية. وعلى الرغم من وفرة الأدبيات حول هذا الموضوع، فإننا نلاحظ أن القرارات الحاسمة المتعلقة بتصميم VLMs لا تكون مبررة في كثير من الأحيان. ونحن نجادل بأن هذه القرارات غير المدعومة تعرقل التقدم في هذا المجال من خلال جعل من الصعب تحديد الخيارات التي تحسن أداء النموذج. ولمعالجة هذه المشكلة، نجري تجارب واسعة النطاق حول النماذج المسبقة التدريب، واختيار الهندسة المعمارية، والبيانات، وأساليب التدريب. ويشمل توحيد نتائجنا تطوير Idefics2، وهو VLM أساسي فعال يبلغ 8 مليارات معلمة. ويحقق Idefics2 أداءً متميزًا ضمن فئة حجمه عبر العديد من المعايير متعددة الوسائط، ويكون غالبًا على قدم المساواة مع النماذج التي يبلغ حجمها أربعة أضعاف. ونقوم بإطلاق النموذج (الأساسي والموجه والدردشة) إلى جانب مجموعات البيانات التي تم إنشاؤها لتدريبه.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/idefics2_architecture.png" alt="drawing" width="600"/>

<small>هندسة Idefics2. مأخوذة من <a href="https://arxiv.org/abs/2405.02246">الورقة الأصلية.</a></small>

تمت المساهمة بهذا النموذج من قبل [amyeroberts](https://huggingface.co/amyeroberts).
يمكن العثور على الكود الأصلي [هنا](https://huggingface.co/HuggingFaceM4/idefics2).

## نصائح الاستخدام

- يمكن أن يحتوي كل عينة على صور متعددة، ويمكن أن يختلف عدد الصور بين العينات. وسوف يقوم المعالج بملء إدخالات إلى العدد الأقصى من الصور في دفعة للإدخال إلى النموذج.
- يوجد لدى المعالج خيار `do_image_splitting`. إذا كان `True`، فسيتم تقسيم كل صورة إدخال إلى 4 صور فرعية، وسيتم دمجها مع الأصل لتشكيل 5 صور. هذا مفيد لزيادة أداء النموذج. تأكد من تعيين `processor.image_processor.do_image_splitting` على `False` إذا لم يتم تدريب النموذج باستخدام هذا الخيار.
- يجب أن يكون `text` الذي تم تمريره إلى المعالج رموز `<image>` حيث يجب إدراج الصور. و `<end_of_utterance>` في نهاية كل عبارة إذا كان النص رسالة دردشة.
- يوجد لدى المعالج طريقة `apply_chat_template` الخاصة به لتحويل رسائل الدردشة إلى نص يمكن تمريره بعد ذلك كـ `text` إلى المعالج.

مثال على كيفية استخدام المعالج على رسائل الدردشة:

```python
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
        "role": "user",
        "content": [
                {"type": "text", "text": "What’s the difference between these two images?"},
                {"type": "image"},
                {"type": "image"},
        ],
}]

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

# at inference time, one needs to pass `add_generation_prompt=True` in order to make sure the model completes the prompt
text = processor.apply_chat_template(messages, add_generation_prompt=True)
print(text)
# 'User: What’s the difference between these two images?<image><image><end_of_utterance>\nAssistant:'

inputs = processor(images=images, text=text, return_tensors="pt").to(device)

generated_text = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
print("Generated text:", generated_text)
```

- أثناء التدريب، من المهم تحديد الرموز التي لا يجب على النموذج تعلمها. بالنسبة لـ Idefics2، ينخفض هذا عادةً إلى رموز الصور والتعبئة. وهذا يعني أنه يمكن إنشاء التسميات كما يلي:

```python
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s the difference between these two images?"},
            {"type": "image"},
            {"type": "image"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The difference is that one image is about dogs and the other one about cats."},
        ],
}]

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

text = processor.apply_chat_template(messages, add_generation_prompt=False)
inputs = processor(images=images, text=text, return_tensors="pt").to(device)

labels = inputs.input_ids.clone()
labels[labels == processor.tokenizer.pad_token_id] = -100
labels[labels == model.config.image_token_id] = -100

inputs["labels"] = labels

outputs = model(**inputs)
loss = outputs.loss
loss.backward()
```

يرجى ملاحظة أنه عند تدريب Idefics2 على المحادثات متعددة الأدوار بين المستخدم والمساعد، يتم أيضًا تعيين جميع الرموز المقابلة لرسائل المستخدم إلى -100.

## تحسين النموذج: Flash Attention

توضح مقتطفات الكود أعلاه الاستدلال بدون أي حيل تحسين. ومع ذلك، يمكن للمرء أن يسرع بشكل كبير من النموذج من خلال الاستفادة من [Flash Attention](../perf_train_gpu_one.md#flash-attention-2)، وهو تنفيذ أسرع لآلية الانتباه المستخدمة داخل النموذج.

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة نافذة الانزلاق.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن لديك أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لـ [مستودع الانتباه السريع](https://github.com/Dao-AILab/flash-attention). تأكد أيضًا من تحميل نموذجك في نصف الدقة (على سبيل المثال `torch.float16`).

لتحميل وتشغيل نموذج باستخدام Flash Attention-2، قم ببساطة بتغيير مقتطف الكود أعلاه بالتغيير التالي:

```diff
model = Idefics2ForConditionalGeneration.from_pretrained(
"HuggingFaceM4/idefics2-8b",
+    torch_dtype=torch.float16,
+    attn_implementation="flash_attention_2",
).to(device)
```

## تصغير حجم Idefics2 باستخدام التكميم

نظرًا لأن نموذج Idefics2 يحتوي على 8 مليارات معلمة، فهذا يتطلب حوالي 16 جيجابايت من ذاكرة GPU RAM في نصف الدقة (float16)، حيث يتم تخزين كل معلمة في بايتين. ومع ذلك، يمكن تقليل حجم النموذج باستخدام [التكميم](../quantization.md). إذا تم تكميم النموذج إلى 4 بتات (أو نصف بايت لكل معلمة)، فهذا يتطلب فقط حوالي 3.5 جيجابايت من ذاكرة الوصول العشوائي.

إن تكميم نموذج بسيط مثل تمرير `quantization_config` إلى النموذج. يمكنك تغيير مقتطف الكود أعلاه بالتغييرات أدناه. سنستفيد من تكوين BitsAndyBytes (ولكن راجع [هذه الصفحة](../quantization.md) لأساليب التكميم الأخرى):

```diff
+ from transformers import BitsAndBytesConfig

+ quantization_config = BitsAndBytesConfig(
+    load_in_4bit=True,
+    bnb_4bit_quant_type="nf4",
+    bnb_4bit_use_double_quant=True,
+    bnb_4bit_compute_dtype=torch.float16
+ )
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
+    torch_dtype=torch.float16,
+    quantization_config=quantization_config,
).to(device)
```

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام Idefics2. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يوضح المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- يمكن العثور على دفتر ملاحظات حول كيفية ضبط نموذج Idefics2 على مجموعة بيانات مخصصة باستخدام [المدرب](../main_classes/trainer.md) [هنا](https://colab.research.google.com/drive/1NtcTgRbSBKN7pYD3Vdx1j9m8pt3fhFDB?usp=sharing). وهو يدعم كل من الضبط الدقيق الكامل وكذلك (المكمم) LoRa.
- يمكن العثور على نص برمجي حول كيفية ضبط نموذج Idefics2 باستخدام مكتبة TRL [هنا](https://gist.github.com/edbeeching/228652fc6c2b29a1641be5a5778223cb).
- يمكن العثور على دفتر ملاحظات توضيحي حول ضبط نموذج Idefics2 لاستخدامات استخراج JSON [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2). 🌎

## Idefics2Config

[[autodoc]] Idefics2Config

## Idefics2Model

[[autodoc]] Idefics2Model

- forward

## Idefics2ForConditionalGeneration

[[autodoc]] Idefics2ForConditionalGeneration

- forward

## Idefics2ImageProcessor

[[autodoc]] Idefics2ImageProcessor

- preprocess

## Idefics2Processor

[[autodoc]] Idefics2Processor

- __call__