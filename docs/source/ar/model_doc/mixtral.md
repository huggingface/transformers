# Mixtral

## نظرة عامة
تم تقديم Mixtral-8x7B في [منشور مدونة Mixtral of Experts](https://mistral.ai/news/mixtral-of-experts/) بواسطة Albert Jiang، وAlexandre Sablayrolles، وArthur Mensch، وChris Bamford، وDevendra Singh Chaplot، وDiego de las Casas، وFlorian Bressand، وGianna Lengyel، وGuillaume Lample، وLélio Renard Lavaud، وLucile Saulnier، وMarie-Anne Lachaux، وPierre Stock، وTeven Le Scao، وThibaut Lavril، وThomas Wang، وTimothée Lacroix، وWilliam El Sayed.

تقول مقدمة منشور المدونة:

> "اليوم، يفخر الفريق بإطلاق Mixtral 8x7B، وهو نموذج عالي الجودة من نماذج المزج النادر للخبراء (SMoE) ذو أوزان مفتوحة. مرخص بموجب Apache 2.0. يتفوق Mixtral على Llama 2 70B في معظم المعايير المرجعية بسرعة استدلال أسرع 6 مرات. إنه أقوى نموذج ذو وزن مفتوح برخصة مسموحة، وأفضل نموذج بشكل عام فيما يتعلق بالمقايضات بين التكلفة والأداء. وعلى وجه الخصوص، فإنه يطابق أداء GPT3.5 أو يتفوق عليه في معظم المعايير المرجعية القياسية."

Mixtral-8x7B هو ثاني نموذج لغة كبير (LLM) أصدرته [mistral.ai](https://mistral.ai/)، بعد [Mistral-7B](mistral).

### التفاصيل المعمارية
Mixtral-8x7B هو محول فك تشفير فقط مع خيارات التصميم المعماري التالية:

- Mixtral هو نموذج مزيج من الخبراء (MoE) مع 8 خبراء لكل شبكة عصبية متعددة الطبقات (MLP)، بإجمالي 45 مليار معلمة. لمزيد من المعلومات حول المزج من الخبراء، يرجى الرجوع إلى [منشور المدونة](https://huggingface.co/blog/moe).

- على الرغم من أن النموذج يحتوي على 45 مليار معلمة، إلا أن الكمبيوتر المطلوب لإجراء تمرير أمامي واحد هو نفسه المطلوب لنموذج مع 14 مليار معلمة. ويرجع ذلك إلى أنه على الرغم من ضرورة تحميل كل خبير في ذاكرة الوصول العشوائي (متطلبات ذاكرة الوصول العشوائي مثل 70B)، يتم إرسال كل رمز من الرموز المخفية مرتين (التصنيف الأعلى 2) وبالتالي فإن الكمبيوتر (العملية المطلوبة في كل عملية حسابية أمامية) هو مجرد 2 X sequence_length.

تفاصيل التنفيذ التالية مشتركة مع النموذج الأول لـ Mistral AI [Mistral-7B](mistral):

- نافذة انزلاق الاهتمام - مدربة بطول سياق 8 كيلو بايت وحجم ذاكرة التخزين المؤقت ثابت، مع نطاق اهتمام نظري يبلغ 128 ألف رمز.

- GQA (مجموعة Query Attention) - تسمح باستدلال أسرع وحجم ذاكرة تخزين مؤقت أقل.

- محرف Byte-fallback BPE - يضمن عدم تعيين الأحرف مطلقًا إلى رموز خارج المفردات.

للحصول على مزيد من التفاصيل، يرجى الرجوع إلى [منشور المدونة](https://mistral.ai/news/mixtral-of-experts/).

### الترخيص
تم إصدار `Mixtral-8x7B` بموجب ترخيص Apache 2.0.

## نصائح الاستخدام
أصدر فريق Mistral نقطتي تفتيش:

- نموذج أساسي، [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)، تم تدريبه مسبقًا للتنبؤ بالرمز التالي على بيانات على نطاق الإنترنت.

- نموذج ضبط التعليمات، [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)، وهو النموذج الأساسي الذي تم تحسينه لأغراض الدردشة باستخدام الضبط الدقيق الخاضع للإشراف (SFT) والتحسين المباشر للأفضليات (DPO).

يمكن استخدام النموذج الأساسي على النحو التالي:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

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

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

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

كما هو موضح، يتطلب النموذج المضبوط للتعليمات [قالب دردشة](../chat_templating) للتأكد من إعداد الإدخالات بتنسيق صحيح.

## تسريع Mixtral باستخدام Flash Attention
توضح مقتطفات التعليمات البرمجية أعلاه الاستدلال بدون أي حيل للتحسين. ومع ذلك، يمكن للمرء أن يسرع بشكل كبير من النموذج من خلال الاستفادة من [Flash Attention](../perf_train_gpu_one.md#flash-attention-2)، وهو تنفيذ أسرع لآلية الاهتمام المستخدمة داخل النموذج.

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة نافذة الانزلاق.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن لديك أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لـ [مستودع الاهتمام بالوميض](https://github.com/Dao-AILab/flash-attention). تأكد أيضًا من تحميل نموذجك في نصف الدقة (على سبيل المثال `torch.float16`).

للتحميل وتشغيل نموذج باستخدام Flash Attention-2، راجع المقتطف أدناه:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected output"
```

### تسريع متوقع
فيما يلي رسم بياني للتسريع المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `mistralai/Mixtral-8x7B-v0.1` وإصدار Flash Attention 2 من النموذج.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/mixtral-7b-inference-large-seqlen.png">
</div>

### نافذة انزلاق الاهتمام
تدعم التنفيذ الحالي آلية اهتمام نافذة الانزلاق وإدارة ذاكرة التخزين المؤقت الفعالة من حيث الذاكرة.

لتمكين اهتمام نافذة الانزلاق، تأكد فقط من وجود إصدار `flash-attn` متوافق مع اهتمام نافذة الانزلاق (`>=2.3.0`).

يستخدم نموذج Flash Attention-2 أيضًا آلية تقطيع ذاكرة التخزين المؤقت الأكثر كفاءة من حيث الذاكرة - كما هو موصى به وفقًا للتنفيذ الرسمي لنموذج Mistral الذي يستخدم آلية التخزين المؤقت المتداول، نحافظ على حجم ذاكرة التخزين المؤقت ثابتًا (`self.config.sliding_window`)، وندعم التوليد المجمع فقط لـ `padding_side="left"` ونستخدم الموضع المطلق للرمز الحالي لحساب التضمين الموضعي.

## تقليل حجم Mixtral باستخدام التكميم
نظرًا لأن نموذج Mixtral يحتوي على 45 مليار معلمة، فسيحتاج ذلك إلى حوالي 90 جيجابايت من ذاكرة الوصول العشوائي للرسوميات في نصف الدقة (float16)، حيث يتم تخزين كل معلمة في بايتين. ومع ذلك، يمكنك تقليل حجم النموذج باستخدام [التكميم](../quantization.md). إذا تم تكميم النموذج إلى 4 بتات (أو نصف بايت لكل معلمة)، فإن A100 واحد بذاكرة وصول عشوائي سعة 40 جيجابايت يكفي لتناسب النموذج بالكامل، وفي هذه الحالة، تكون هناك حاجة إلى حوالي 27 جيجابايت فقط من ذاكرة الوصول العشوائي.

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

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", quantization_config=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

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

تمت المساهمة بهذا النموذج من قبل [Younes Belkada](https://huggingface.co/ybelkada) و[Arthur Zucker](https://huggingface.co/ArthurZ).

يمكن العثور على الكود الأصلي [هنا](https://github.com/mistralai/mistral-src).

## الموارد
قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام Mixtral. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يوضح المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="text-generation"/>

- يمكن العثور على دفتر ملاحظات توضيحي لإجراء الضبط الدقيق الخاضع للإشراف (SFT) لـ Mixtral-8x7B [هنا](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb). 🌎

- منشور [مدونة](https://medium.com/@prakharsaxena11111/finetuning-mixtral-7bx8-6071b0ebf114) حول الضبط الدقيق لـ Mixtral-8x7B باستخدام PEFT. 🌎

- يتضمن [دليل المحاذاة](https://github.com/huggingface/alignment-handbook) من Hugging Face نصوصًا ووصفات لأداء الضبط الدقيق الخاضع للإشراف (SFT) والتحسين المباشر للأفضليات باستخدام Mistral-7B. ويشمل ذلك نصوص الضبط الدقيق الكامل، وQLoRa على وحدة معالجة الرسومات (GPU) واحدة بالإضافة إلى الضبط الدقيق متعدد وحدات معالجة الرسومات (GPU).

- [مهمة نمذجة اللغة السببية](../tasks/language_modeling)

## MixtralConfig
[[autodoc]] MixtralConfig

## MixtralModel
[[autodoc]] MixtralModel

- forward

## MixtralForCausalLM
[[autodoc]] MixtralForCausalLM

- forward

## MixtralForSequenceClassification
[[autodoc]] MixtralForSequenceClassification

- forward

## MixtralForTokenClassification
[[autodoc]] MixtralForTokenClassification

- forward