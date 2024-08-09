# GPTBigCode

## نظرة عامة

اقتُرح نموذج GPTBigCode في ورقة [SantaCoder: don't reach for the stars!](https://arxiv.org/abs/2301.03988) بواسطة BigCode. المؤلفون هم: Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane Anderson, Yangtian Zi, Joel Lamy Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky, Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes, Daniel Fried, Arjun Guha, Harm de Vries, Leandro von Werra.

ملخص الورقة هو كما يلي:

*مشروع BigCode هو تعاون علمي مفتوح يعمل على التطوير المسؤول لنماذج اللغة الكبيرة للكود. ويصف هذا التقرير التقني التقدم الذي أحرزه التعاون حتى ديسمبر 2022، مُبرزًا الحالة الراهنة لأنبوب حجب المعلومات الشخصية (PII)، والتجارب التي أُجريت لتخفيف المخاطر عن بنية النموذج، والتجارب التي تبحث في طرق ما قبل المعالجة الأفضل لبيانات التدريب. نقوم بتدريب نماذج ذات 1.1 مليار معامل على مجموعات فرعية Java وJavaScript وPython من Stack وتقييمها على معيار MultiPL-E text-to-code. ونجد أن التصفية الأكثر عدوانية للنسخ المتماثلة القريبة يمكن أن تعزز الأداء بشكل أكبر، ومفاجئ، أن اختيار الملفات من المستودعات التي تحتوي على 5 نجوم GitHub أو أكثر يضر بالأداء بشكل كبير. يفوق نموذجنا الأفضل أداء نماذج إنشاء التعليمات البرمجية متعددة اللغات مفتوحة المصدر (InCoder-6.7B وCodeGen-Multi-2.7B) في كل من إنشاء من اليسار إلى اليمين والإكمال في الأجزاء Java وJavaScript وPython من MultiPL-E، على الرغم من كونه نموذجًا أصغر بكثير. يتم إصدار جميع النماذج بموجب ترخيص OpenRAIL في [هذا عنوان URL https.](https://huggingface.co/bigcode)*

النموذج هو [نموذج GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) مُحسّن مع دعم Multi-Query Attention.

## تفاصيل التنفيذ

الفروق الرئيسية مقارنة بـ GPT2:

- أضيف دعم Multi-Query Attention.
- استخدم `gelu_pytorch_tanh` بدلاً من `gelu` الكلاسيكي.
- تجنب المزامنات غير الضرورية (تمت إضافته منذ ذلك الحين إلى GPT2 في #20061، ولكنه لم يكن في كود المرجع).
- استخدم الطبقات الخطية بدلاً من Conv1D (تحسين جيد ولكنه يجعل نقاط التحقق غير متوافقة).
- دمج `_attn` و`_upcast_and_reordered_attn`. دائمًا قم بدمج الضرب النقطي مع التقييم. إعادة تسمية `reorder_and_upcast_attn`->`attention_softmax_in_fp32`
- قم بتخزين قناع الانتباه في الذاكرة المؤقتة لتجنب إعادة إنشائه في كل مرة.
- استخدم jit لدمج الانتباه fp32 casting وmasking وsoftmax وscaling.
- قم بدمج قناع الانتباه والقناع السببي في قناع واحد، محسوبة مسبقًا للنموذج بأكمله بدلاً من كل طبقة.
- دمج ذاكرة التخزين المؤقت الرئيسية وذاكرة التخزين المؤقت للقيمة في ذاكرة واحدة (يغير هذا تنسيق layer_past/present، فهل يخاطر ذلك بخلق مشاكل؟)
- استخدم تخطيط الذاكرة (self.num_heads، 3، self.head_dim) بدلاً من `(3، self.num_heads، self.head_dim)` لمصفوفة QKV مع MHA. (يمنع حدوث علو مع القيم الرئيسية والفرعية المدمجة، ولكنه يجعل نقاط التحقق غير متوافقة مع نموذج openai-community/gpt2 الأصلي).

يمكنك قراءة المزيد حول التحسينات في [طلب السحب الأصلي](https://github.com/huggingface/transformers/pull/22575)

## الجمع بين Starcoder وFlash Attention 2

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة نافذة الانزلاق.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن لديك أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لمستودع flash-attn. تأكد أيضًا من تحميل نموذجك في نصف الدقة (على سبيل المثال `torch.float16`)

لتحميل وتشغيل نموذج باستخدام Flash Attention 2، راجع المقتطف أدناه:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # الجهاز الذي سيتم تحميل النموذج عليه

>>> model = AutoModelForCausalLM.from_pretrained("bigcode/gpt_bigcode-santacoder", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
>>> tokenizer = AutoTokenizer.from_pretrained("bigcode/gpt_bigcode-santacoder")

>>> prompt = "def hello_world():"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
>>> tokenizer.batch_decode(generated_ids)[0]
'def hello_world():\n    print("hello world")\n\nif __name__ == "__main__":\n    print("hello world")\n<|endoftext|>'
```

### تسريع الأداء المتوقع

فيما يلي رسم بياني لتسريع الأداء المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في transformers باستخدام نقطة تفتيش `bigcode/starcoder` وإصدار Flash Attention 2 من النموذج باستخدام طولين تسلسليين مختلفين.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/starcoder-speedup.png">
</div>

## GPTBigCodeConfig

[[autodoc]] GPTBigCodeConfig

## GPTBigCodeModel

[[autodoc]] GPTBigCodeModel

- forward

## GPTBigCodeForCausalLM

[[autodoc]] GPTBigCodeForCausا

- forward

## GPTBigCodeForSequenceClassification

[[autodoc]] GPTBigCodeForSequenceClassification

- forward

## GPTBigCodeForTokenClassification

[[autodoc]] GPTBigCodeForTokenClassification

- forward