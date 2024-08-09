# DBRX

## نظرة عامة

DBRX هو نموذج لغة كبير (LLM) قائم على محول فك التشفير فقط تم تدريبه باستخدام التنبؤ بالعلامة التالية. يستخدم النموذج بنية mixture-of-experts (MoE) ذات الحبيبات الدقيقة مع ما مجموعه 132 بليون معلمة، يتم تنشيط 36 بليون معلمة منها لأي إدخال. تم التدريب المسبق لـ DBRX على 12 طن من رموز نص وبيانات التعليمات البرمجية.

بالمقارنة مع النماذج المفتوحة الأخرى القائمة على MoE مثل Mixtral-8x7B و Grok-1، فإن DBRX دقيق، مما يعني أنه يستخدم عددًا أكبر من الخبراء الأصغر. يحتوي DBRX على 16 خبيرًا ويختار 4، بينما يحتوي Mixtral-8x7B و Grok-1 على 8 خبراء ويختاران 2. يوفر هذا 65 ضعف المزيد من مجموعات الخبراء الممكنة، وقد وجدنا أن هذا يحسن جودة النموذج.

يستخدم DBRX ترميزات الموضع الدوارة (RoPE) ووحدات الخطية المحكومة (GLU) واهتمام الاستعلام المجمع (GQA). إنه نموذج قائم على BPE ويستخدم محلل رموز GPT-4 كما هو موضح في مستودع [tiktoken](https://github.com/openai/tiktoken).

تم إجراء هذه الخيارات بناءً على تقييم شامل وتجارب التوسع.

تم التدريب المسبق لـ DBRX على 12 طن من الرموز من البيانات المختارة بعناية وطول سياق أقصى يبلغ 32000 رمز. نقدر أن هذه البيانات أفضل مرتين على الأقل من البيانات التي استخدمناها لتدريب عائلة نماذج MPT.

تم تطوير مجموعة البيانات الجديدة هذه باستخدام مجموعة أدوات Databricks الكاملة، بما في ذلك Apache Spark™ ودفاتر Databricks لمعالجة البيانات، وUnity Catalog لإدارة البيانات والحوكمة.

لقد استخدمنا التعلم المناهج لتدريب سابق، وتغيير مزيج البيانات أثناء التدريب بطرق وجدنا أنها تحسن بشكل كبير من جودة النموذج.

يمكن العثور على معلومات أكثر تفصيلاً حول DBRX Instruct و DBRX Base في منشور المدونة الفنية الخاص بنا [هنا](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).

تمت المساهمة بهذا النموذج من قبل [eitan-turok](https://huggingface.co/eitanturok) و [abhi-db](https://huggingface.co/abhi-db). يمكن العثور على الكود الأصلي [هنا](https://github.com/databricks/dbrx-instruct)، على الرغم من أنه قد لا يكون محدثًا.

## أمثلة الاستخدام

يمكن استخدام طريقة `generate()` لتوليد نص باستخدام DBRX. يمكنك إنشاء باستخدام تنفيذ الاهتمام القياسي، واهتمام الفلاش، واهتمام المنتج النقطي المميز PyTorch. يوفر الاهتمامان الأخيران تسريعًا.

```python
from transformers import DbrxForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token="YOUR_HF_TOKEN")
model = DbrxForCausalLM.from_pretrained(
    "databricks/dbrx-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token="YOUR_HF_TOKEN",
)

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

إذا كان لديك تثبيت اهتمام الفلاش (`pip install flash-attn`)، فيمكنك إنشاء أسرع. (يمكن العثور على وثائق HuggingFace لاهتمام الفلاش [هنا](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2).)

```python
from transformers import DbrxForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token="YOUR_HF_TOKEN")
model = DbrxForCausalLM.from_pretrained(
    "databricks/dbrx-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token="YOUR_HF_TOKEN",
    attn_implementation="flash_attention_2",
)

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

يمكنك أيضًا إنشاء أسرع باستخدام اهتمام المنتج النقطي لـ PyTorch. (يمكن العثور على وثائق HuggingFace لاهتمام المنتج النقطي المقيّم [هنا](https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention).)

```python
from transformers import DbrxForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token="YOUR_HF_TOKEN")
model = DbrxForCausalLM.from_pretrained(
    "databricks/dbrx-instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token="YOUR_HF_TOKEN",
    attn_implementation="sdpa",
)

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## DbrxConfig

[[autodoc]] DbrxConfig

## DbrxModel

[[autodoc]] DbrxModel

- forword

## DbrxForCausalLM

[[autodoc]] DbrxForCausalLM

- forword