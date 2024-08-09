# LLaVA-NeXT

## نظرة عامة
اقترح نموذج LLaVA-NeXT في [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/) بواسطة Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, Yong Jae Lee. ويحسن LLaVa-NeXT (المعروف أيضًا باسم LLaVa-1.6) من [LLaVa](llava) عن طريق زيادة دقة صورة الإدخال والتدريب على مجموعة بيانات محسنة لضبط التعليمات المرئية لتحسين التعرف الضوئي على الحروف (OCR) والاستدلال بالمعرفة.

مقدمة المدونة هي كما يلي:

> "في أكتوبر 2023، أصدرنا LLaVA-1.5 بتصميم بسيط وفعال إلى جانب أداء رائع في مجموعة من 12 مجموعة بيانات. وقد شكل منذ ذلك الحين الأساس للعديد من الدراسات الشاملة للبيانات والنماذج وقدرات النماذج متعددة الوسائط الكبيرة (LMM)، وقد مكنت العديد من التطبيقات الجديدة.
> واليوم، يسرنا أن نقدم LLaVA-NeXT، مع تحسين الاستدلال والتعرف الضوئي على الحروف ومعرفة العالم. حتى أن LLaVA-NeXT يتفوق على Gemini Pro في العديد من المعايير المرجعية.
> مقارنة بـ LLaVA-1.5، يحتوي LLaVA-NeXT على العديد من التحسينات:
> - زيادة دقة صورة الإدخال إلى 4x بكسل أكثر. يسمح لها ذلك بالتقاط المزيد من التفاصيل المرئية. يدعم ثلاث نسب عرض إلى ارتفاع، تصل إلى 672x672، 336x1344، 1344x336 دقة.
> - قدرة استدلال بصرية أفضل وقدرة OCR مع مزيج محسن من بيانات ضبط التعليمات المرئية.
> - محادثة بصرية أفضل لمزيد من السيناريوهات، وتغطية تطبيقات مختلفة. معرفة عالمية واستدلال منطقي أفضل.
> - نشر فعال واستنتاج باستخدام SGLang.
> إلى جانب التحسينات في الأداء، يحافظ LLaVA-NeXT على التصميم الأنيق وكفاءة البيانات لـ LLaVA-1.5. فهو يعيد استخدام موصل pretrained من LLaVA-1.5، ولا يزال يستخدم أقل من 1 مليون عينة ضبط تعليمات بصرية. وينتهي أكبر متغير 34B من التدريب في ~1 يوم مع 32 A100s. "

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_overview.png"
alt="drawing" width="600"/>

<small> يدمج LLaVa-NeXT دقة إدخال أعلى عن طريق ترميز رقع مختلفة من صورة الإدخال. مأخوذة من <a href="https://arxiv.org/abs/2310.03744">الورقة الأصلية.</a> </small>

ساهم بهذا النموذج [nielsr](https://huggingface.co/nielsr).

يمكن العثور على الكود الأصلي [هنا](https://github.com/haotian-liu/LLaVA/tree/main).

## نصائح الاستخدام

- ننصح المستخدمين باستخدام `padding_side="left"` عند حساب التوليد الدفعي لأنه يؤدي إلى نتائج أكثر دقة. تأكد ببساطة من استدعاء `processor.tokenizer.padding_side = "left"` قبل التوليد.

- لاحظ أن كل نقطة تفتيش تم تدريبها بتنسيق موجه محدد، اعتمادًا على نموذج اللغة الكبير (LLM) المستخدم. أدناه، نقوم بإدراج تنسيقات الموجه الصحيحة لاستخدامها في موجه النص "What is shown in this image؟":

[llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) يتطلب التنسيق التالي:

```bash
"[INST] <image>\nWhat is shown in this image? [/INST]"
```

[llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) و [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) يتطلب التنسيق التالي:

```bash
"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
```

[llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) يتطلب التنسيق التالي:

```bash
"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
```

## مثال الاستخدام

### استنتاج صورة واحدة

فيما يلي كيفية تحميل النموذج وإجراء الاستدلال بنصف الدقة (`torch.float16`):

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
```

### استنتاج الصور المتعددة

يمكن لـ LLaVa-Next إجراء الاستدلال باستخدام صور متعددة كإدخال، حيث تنتمي الصور إما إلى نفس الموجه أو إلى موجهات مختلفة (في الاستدلال الدفعي). إليك كيفية القيام بذلك:

```python
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# Load the model in half-precision
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Get three different images
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image. Soehbat(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# Prepare a batched prompt, where the first one is a multi-turn conversation and the second is not
prompt = [
    "[INST] <image>\nWhat is shown in this image? [/INST] There is a red stop sign in the image. [INST] <image>\nWhat about this image? How many cats do you see [/INST]",
    "[INST] <image>\nWhat is shown in this image? [/INST]"
]

# We can simply feed images in the order they have to be used in the text prompt
# Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
inputs = processor(text=prompt, images=[image_stop, image_cats, image_snowman], padding=True, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

## تحسين النموذج

### التكميم باستخدام Bitsandbytes

يمكن تحميل النموذج في 8 أو 4 بتات، مما يقلل بشكل كبير من متطلبات الذاكرة مع الحفاظ على أداء النموذج الأصلي. تأكد أولاً من تثبيت bitsandbytes، `pip install bitsandbytes` وتأكد من إمكانية الوصول إلى جهاز GPU متوافق مع CUDA. ببساطة قم بتغيير المقتطف أعلاه إلى:

```python
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",  quantization_config=quantization_config, device_map="auto")
```

### استخدم Flash-Attention 2 لتسريع التوليد

تأكد أولاً من تثبيت flash-attn. راجع [المستودع الأصلي لـ Flash Attention](https://github.com/Dao-AILab/flash-attention) فيما يتعلق بتثبيت الحزمة. ببساطة قم بتغيير المقتطف أعلاه إلى:

```python
from transformers import LlavaNextForConditionalGeneration

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_flash_attention_2=True
).to(0)
```

## LlavaNextConfig

[[autodoc]] LlavaNextConfig

## LlavaNextImageProcessor

[[autodoc]] LlavaNextImageProcessor

- معالجة مسبقة

## LlavaNextProcessor

[[autodoc]] LlavaNextProcessor

## LlavaNextForConditionalGeneration

[[autodoc]] LlavaNextForConditionalGeneration

- forword