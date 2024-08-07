# Phi

## نظرة عامة
اقترح نموذج Phi-1 في ورقة "الكتب المدرسية هي كل ما تحتاجه" بواسطة سوريا جوناسيكر، وآخرون. اقترح نموذج Phi-1.5 في ورقة "الكتب المدرسية هي كل ما تحتاجه الجزء الثاني: التقرير الفني لـ phi-1.5" بواسطة يوانزهي لي، وآخرون.

### ملخص
في ورقتي Phi-1 وPhi-1.5، أظهر المؤلفون مدى أهمية جودة البيانات في التدريب بالنسبة لحجم النموذج. لقد اختاروا بيانات "كتب مدرسية" عالية الجودة إلى جانب بيانات مولدة بشكل صناعي لتدريب نموذجهم الصغير المستند إلى محول مع 1.3 مليار معلمة. وعلى الرغم من هذا الحجم الصغير، حقق النموذج Phi-1 دقة pass@1 بنسبة 50.6% على HumanEval و55.5% على MBPP.

اتبع المؤلفون نفس الاستراتيجية لـ Phi-1.5 وأنشأوا نموذجًا آخر بحجم 1.3 مليار معلمة بأداء في مهام اللغة الطبيعية قابل للمقارنة مع النماذج الأكبر حجمًا بمقدار 5 مرات، ويتفوق على معظم نماذج اللغة الكبيرة غير الرائدة. ويظهر Phi-1.5 العديد من سمات نماذج اللغة الكبيرة جدًا، مثل القدرة على "التفكير خطوة بخطوة" أو أداء بعض التعلم الأساسي في السياق.

من خلال هاتين التجربتين، أوضح المؤلفون بنجاح التأثير الهائل لجودة بيانات التدريب عند تدريب نماذج التعلم الآلي.

هذا هو الملخص من ورقة Phi-1:

*نقدم phi-1، وهو نموذج لغة جديد للترميز، بحجم أصغر بكثير من النماذج المنافسة: phi-1 هو نموذج قائم على محول مع 1.3 مليار معلمة، تم تدريبه لمدة 4 أيام على 8 A100s، باستخدام مجموعة من بيانات "جودة الكتب المدرسية" من الويب (6 مليار رمز) وبيانات تم إنشاؤها بشكل صناعي الكتب المدرسية والتدريبات باستخدام GPT-3.5 (1 مليار رمز). وعلى الرغم من هذا الحجم الصغير، يحقق phi-1 دقة pass@1 بنسبة 50.6% على HumanEval و55.5% على MBPP. كما يعرض خصائص ناشئة مفاجئة مقارنة بـ phi-1-base، وهو نموذجنا قبل مرحلة الضبط الدقيق على مجموعة بيانات من تمارين الترميز، وphi-1-small، وهو نموذج أصغر بحجم 350 مليون معلمة تم تدريبه باستخدام نفس خط الأنابيب مثل phi-1 والذي لا يزال يحقق 45% على HumanEval.*

هذا هو الملخص من ورقة Phi-1.5:

*نواصل التحقيق في قوة نماذج اللغة الأصغر المستندة إلى محول كما بدأتها TinyStories - وهو نموذج بحجم 10 ملايين معلمة يمكنه إنتاج الإنجليزية المتماسكة - والعمل اللاحق على phi-1، وهو نموذج بحجم 1.3 مليار معلمة بأداء الترميز Python قريب من أحدث التقنيات. اقترحت الورقة الأخيرة استخدام نماذج اللغة الكبيرة الحالية (LLMs) لتوليد بيانات "جودة الكتب المدرسية" كطريقة لتعزيز عملية التعلم مقارنة ببيانات الويب التقليدية. نتبع نهج "الكتب المدرسية هي كل ما تحتاجه"، مع التركيز هذه المرة على التفكير المنطقي الشائع في اللغة الطبيعية، وننشئ نموذجًا جديدًا بحجم 1.3 مليار معلمة يسمى phi-1.5، بأداء في مهام اللغة الطبيعية قابل للمقارنة مع النماذج الأكبر حجمًا بمقدار 5 مرات، ويتفوق على معظم نماذج اللغة الكبيرة غير الرائدة في مهام الاستدلال الأكثر تعقيدًا مثل الرياضيات في المرحلة الابتدائية والترميز الأساسي. وبشكل أكثر عمومية، يعرض phi-1.5 العديد من سمات نماذج اللغة الكبيرة جدًا، الجيدة منها -مثل القدرة على "التفكير خطوة بخطوة" أو أداء بعض التعلم الأساسي في السياق- والسيئة، بما في ذلك الهلوسة واحتمال التوليد السام والمتعصب -ومع ذلك، فإننا نشهد تحسنًا على تلك الجبهة بفضل غياب بيانات الويب. نقوم بإتاحة المصدر المفتوح لـ phi-1.5 لتعزيز مزيد من البحث في هذه الموضوعات الملحة.*

تمت المساهمة بهذا النموذج من قبل [Susnato Dhar](https://huggingface.co/susnato).

يمكن العثور على الكود الأصلي لـ Phi-1 وPhi-1.5 وPhi-2 [هنا](https://huggingface.co/microsoft/phi-1)، [هنا](https://huggingface.co/microsoft/phi-1_5) و [هنا](https://huggingface.co/microsoft/phi-2)، على التوالي.

## نصائح الاستخدام

- يشبه هذا النموذج إلى حد كبير نموذج "Llama" مع الاختلاف الرئيسي في [`PhiDecoderLayer`]، حيث تم استخدام طبقات [`PhiAttention`] و [`PhiMLP`] في تكوين متوازي.
- محول الرموز المستخدم لهذا النموذج مطابق لمحول رموز [`CodeGen`].

## كيفية استخدام Phi-2

<Tip warning={true}>

تم دمج Phi-2 في إصدار التطوير (4.37.0.dev) من `المحولات`. حتى يتم إصدار الإصدار الرسمي عبر `pip`، تأكد من قيامك بواحد مما يلي:

* عند تحميل النموذج، تأكد من تمرير `trust_remote_code=True` كحجة لدالة `from_pretrained()`.
* قم بتحديث إصدار "المحولات" المحلي إلى إصدار التطوير: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`. الأمر السابق هو بديل لإلغاء التثبيت والتثبيت من المصدر.

</Tip>

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

>>> inputs = tokenizer('Can you help me write a formal email to a potential business partner proposing a joint venture?', return_tensors="pt", return_attention_mask=False)

>>> outputs = model.generate(**inputs, max_length=30)
>>> text = tokenizer.batch_decode(outputs)[0]
>>> print(text)
Can you help me write a formal email to a potential business partner proposing a joint venture?
Input: Company A: ABC Inc.
Company B
```

### مثال:

```python
>>> from transformers import PhiForCausalLM, AutoTokenizer

>>> # define the model and tokenizer.
>>> model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5")
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

>>> # feel free to change the prompt to your liking.
>>> prompt = "If I were an AI that had just achieved"

>>> # apply the tokenizer.
>>> tokens = tokenizer(prompt, return_tensors="pt")

>>> # use the model to generate new tokens.
>>> generated_output = model.generate(**tokens, use_cache=True, max_new_tokens=10)

>>> tokenizer.batch_decode(generated_output)[0]
'If I were an AI that had just achieved a breakthrough in machine learning, I would be thrilled'
```

## الجمع بين Phi و Flash Attention 2

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة نافذة الانزلاق.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن لديك أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لمستودع flash-attn. تأكد أيضًا من تحميل نموذجك في نصف الدقة (على سبيل المثال `torch.float16`)

لتحميل وتشغيل نموذج باستخدام Flash Attention 2، راجع المقتطف أدناه:

```python
>>> import torch
>>> from transformers import PhiForCausalLM, AutoTokenizer

>>> # define the model and tokenizer and push the model and tokens to the GPU.
>>> model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to("cuda")  # doctest: +SKIP
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

>>> # feel free to change the prompt to your liking.
>>> prompt = "If I were an AI that had just achieved"

>>> # apply the tokenizer.
>>> tokens = tokenizer(prompt, return_tensors="pt").to("cuda")

>>> # use the model to generate new tokens.
>>> generated_output = model.generate(**tokens, use_cache=True, max_new_tokens=10)  # doctest: +SKIP

>>> tokenizer.batch_decode(generated_output)[0]  # doctest: +SKIP
'If I were an AI that had just achieved a breakthrough in machine learning, I would be thrilled'
```

### تسريع السرعة المتوقع

فيما يلي رسم بياني لتسريع السرعة المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `microsoft/phi-1` وإصدار Flash Attention 2 من النموذج باستخدام طول تسلسل يبلغ 2048.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/phi_1_speedup_plot.jpg">
</div>

## PhiConfig

[[autodoc]] PhiConfig

<frameworkcontent>
<pt>

## PhiModel

[[autodoc]] PhiModel

- forward

## PhiForCausalLM

[[autodoc]] PhiForCausalLM

- forward

- generate

## PhiForSequenceClassification

[[autodoc]] PhiForSequenceClassification

- forward

## PhiForTokenClassification

[[autodoc]] PhiForTokenClassification

- forward

</pt>
</frameworkcontent>