# Mistral

## نظرة عامة
تم تقديم ميسترال في [هذه التدوينة](https://mistral.ai/news/announcing-mistral-7b/) بواسطة ألبرت جيانج، وألكسندر سابليرولز، وأرثر مينش، وكريس بامفورد، وديفندرا سينغ تشابلوت، ودييجو دي لاس كاساس، وفلوريان بريساند، وجيانا لينجيل، وجيوم لابل، وليليو رينارد لافو، ولوسيل سولنييه، وماري-آن لاشو، وبيير ستوك، وتيفين لو سكاو، وتيبو لافري، وتوماس وانج، وتيموتي لاكروا، وويليام إل سيد.

تقول مقدمة التدوينة:

> *يفتخر فريق Mistral AI بتقديم Mistral 7B، أقوى نموذج لغوي حتى الآن بحجمه.*

ميسترال-7B هو أول نموذج لغوي كبير (LLM) أصدره [mistral.ai](https://mistral.ai/).

### التفاصيل المعمارية

ميسترال-7B هو محول فك تشفير فقط مع خيارات التصميم المعماري التالية:

- Sliding Window Attention - تم تدريبه باستخدام طول سياق 8k وحجم ذاكرة التخزين المؤقت الثابت، مع نطاق اهتمام نظري يبلغ 128K رمزًا

- GQA (Grouped Query Attention) - يسمح بإجراء استدلال أسرع وحجم ذاكرة تخزين مؤقت أقل.

- Byte-fallback BPE tokenizer - يضمن عدم تعيين الأحرف مطلقًا إلى رموز خارج المفردات.

للحصول على مزيد من التفاصيل، يرجى الرجوع إلى [تدوينة الإصدار](https://mistral.ai/news/announcing-mistral-7b/).

### الترخيص

تم إصدار `Mistral-7B` بموجب ترخيص Apache 2.0.

## نصائح الاستخدام

أصدر فريق ميسترال 3 نقاط تفتيش:

- نموذج أساسي، [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)، تم تدريبه مسبقًا للتنبؤ بالرمز التالي على بيانات بحجم الإنترنت.

- نموذج ضبط التعليمات، [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)، وهو النموذج الأساسي الذي تم تحسينه لأغراض الدردشة باستخدام الضبط الدقيق المُشرف (SFT) والتحسين المباشر للأفضليات (DPO).

- نموذج ضبط تعليمات محسن، [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)، والذي يحسن الإصدار 1.

يمكن استخدام النموذج الأساسي على النحو التالي:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"My favourite condiment is to ..."
```

يمكن استخدام نموذج ضبط التعليمات على النحو التالي:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"Mayonnaise can be made as follows: (...)"
```

كما هو موضح، يتطلب نموذج ضبط التعليمات [قالب دردشة](../chat_templating) للتأكد من إعداد المدخلات بتنسيق صحيح.

## تسريع ميسترال باستخدام Flash Attention

توضح مقتطفات الشفرة أعلاه الاستدلال بدون أي حيل للتحسين. ومع ذلك، يمكن للمرء تسريع النموذج بشكل كبير من خلال الاستفادة من [Flash Attention](../perf_train_gpu_one.md#flash-attention-2)، وهو تنفيذ أسرع لآلية الاهتمام المستخدمة داخل النموذج.

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة نافذة الانزلاق.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن لديك أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لـ [مستودع الاهتمام بالوميض](https://github.com/Dao-AILab/flash-attention). تأكد أيضًا من تحميل نموذجك في نصف الدقة (على سبيل المثال `torch.float16`)

للتحميل وتشغيل نموذج باستخدام Flash Attention-2، راجع المقتطف أدناه:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"My favourite condiment is to (...)"
```

### تسريع متوقع

فيما يلي رسم بياني للتسريع المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `mistralai/Mistral-7B-v0.1` وإصدار Flash Attention 2 من النموذج.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/mistral-7b-inference-large-seqlen.png">
</div>

### نافذة انزلاق الاهتمام

يدعم التنفيذ الحالي آلية اهتمام نافذة الانزلاق وإدارة ذاكرة التخزين المؤقت الفعالة من حيث الذاكرة.

لتمكين اهتمام نافذة الانزلاق، تأكد فقط من وجود إصدار `flash-attn` متوافق مع اهتمام نافذة الانزلاق (`>=2.3.0`).

يستخدم نموذج Flash Attention-2 أيضًا آلية تقطيع ذاكرة التخزين المؤقت الأكثر كفاءة من حيث الذاكرة - وفقًا للتنفيذ الرسمي لنموذج ميسترال الذي يستخدم آلية ذاكرة التخزين المؤقت المتداولة، نحتفظ بحجم ذاكرة التخزين المؤقت ثابتًا (`self.config.sliding_window`)، وندعم التوليد المجمع فقط لـ `padding_side="left"` ونستخدم الموضع المطلق للرمز الحالي لحساب التضمين الموضعي.

## تقليل حجم ميسترال باستخدام التكميم

نظرًا لأن نموذج ميسترال يحتوي على 7 مليارات معلمة، فسيحتاج ذلك إلى حوالي 14 جيجابايت من ذاكرة GPU RAM في الدقة النصفية (float16)، حيث يتم تخزين كل معلمة في 2 بايت. ومع ذلك، يمكن تقليل حجم النموذج باستخدام [التكميم](../quantization.md). إذا تم تكميم النموذج إلى 4 بتات (أو نصف بايت لكل معلمة)، فهذا يتطلب فقط حوالي 3.5 جيجابايت من ذاكرة الوصول العشوائي.

إن تكميم نموذج بسيط مثل تمرير `quantization_config` إلى النموذج. أدناه، سنستفيد من تكميم BitsAndyBytes (ولكن راجع [هذه الصفحة](../quantization.md) لأساليب التكميم الأخرى):

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

>>> # specify how to quantize the model
>>> quantization_config = BitsAndBytesConfig(
...         load_in_4bit=True,
...         bnb_4bit_quant_type="nf4",
...         bnb_4bit_compute_dtype="torch.float16",
... )

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

>>> prompt = "My favourite condiment is"

>>> messages = [
...     {"role": "user", "content": "What is your favourite condiment?"},
...     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
...     {"role": "user", "content": "Do you have mayonnaise recipes?"}
... ]

>>> model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected output"
```

تمت المساهمة بهذا النموذج من قبل [يونس بلقادة](https://huggingface.co/ybelkada) و[آرثر زوكر](https://huggingface.co/ArthurZ).

يمكن العثور على الكود الأصلي [هنا](https://github.com/mistralai/mistral-src).

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام ميسترال. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يوضح المورد بشكل مثالي شيء جديد بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="text-generation"/>

- يمكن العثور على دفتر ملاحظات توضيحي لأداء الضبط الدقيق المُشرف (SFT) لـ Mistral-7B [هنا](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb). 🌎

- [تدوينة](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) حول كيفية ضبط دقة LLMs في عام 2024 باستخدام أدوات Hugging Face. 🌎

- يتضمن [دليل المحاذاة](https://github.com/huggingface/alignment-handbook) من Hugging Face نصوصًا ووصفات لأداء الضبط الدقيق المُشرف (SFT) والتحسين المباشر للأفضليات باستخدام Mistral-7B. يتضمن هذا النصوص للضبط الدقيق الكامل، وQLoRa على GPU واحد بالإضافة إلى الضبط الدقيق متعدد GPU.

- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

## MistralConfig

[[autodoc]] MistralConfig

## MistralModel

[[autodoc]] MistralModel

- forward

## MistralForCausalLM

[[autodoc]] MistralForCausalLM

- forward

## MistralForSequenceClassification

[[autodoc]] MistralForSequenceClassification

- forward

## MistralForTokenClassification

[[autodoc]] MistralForTokenClassification

- forward

## FlaxMistralModel

[[autodoc]] FlaxMistralModel

- __call__

## FlaxMistralForCausalLM

[[autodoc]] FlaxMistralForCausalLM

- __call__

## TFMistralModel

[[autodoc]] TFMistralModel

- call

## TFMistralForCausalLM

[[autodoc]] TFMistralForCausalLM

- call

## TFMistralForSequenceClassification

[[autodoc]] TFMistralForSequenceClassification

- call