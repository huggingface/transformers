# قوالب نماذج الدردشة

## مقدمة

تعد **الدردشة** أحد استخدامات نماذج اللغات الكبيرة (LLMs) شائعة الاستخدام بشكل متزايد. ففي سياق الدردشة، وبدلاً من متابعة سلسلة نصية واحدة (كما هو الحال مع نماذج اللغات القياسية)، يواصل النموذج بدلاً من ذلك محادثة تتكون من رسالة واحدة أو أكثر، تتضمن كل منها دورًا، مثل "المستخدم" أو "المساعد"، بالإضافة إلى نص الرسالة.

وكما هو الحال مع تقسيم النص إلى رموز (tokenization)، تتوقع النماذج المختلفة تنسيقات إدخال مختلفة تمامًا للمحادثة. لهذا السبب أضفنا **قوالب الدردشة** كميزة جديدة. تُعد قوالب المحادثة جزءًا من tokenizer. تحدد هذه القوالب كيفية تحويل المحادثات، والتي يتم تمثيلها كقوائم من الرسائل، إلى سلسلة نصية واحدة قابلة للتقسيم إلى رموز بالتنسيق الذي يتوقعه النموذج.

دعونا نجعل هذا ملموسًا بمثال سريع باستخدام نموذج `BlenderBot`. لدى BlenderBot قالب افتراضي بسيط للغاية، والذي يضيف في الغالب مسافات بيضاء بين جولات الحوار:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...    {"role": "user", "content": "Hello, how are you?"},
...    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...    {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
```

لاحظ كيف تم ضغط الدردشة بأكملها في سلسلة واحدة. إذا استخدمنا `tokenize=True`، وهو الإعداد الافتراضي، فسيتم أيضًا تحليل السلسلة نحويًا نيابة عنا. ولكن، لنشاهد قالبًا أكثر تعقيدًا في العمل، دعونا نستخدم نموذج `mistralai/Mistral-7B-Instruct-v0.1`.

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]</s>"
```

لاحظ كيف أضاف المجزىء اللغوى tokenizer رموز التحكم [INST] و [/INST] للإشارة إلى بداية ونهاية رسائل المستخدم (ولكن ليس رسائل المساعد!) ، وتم تكثيف المحادثة بأكملها في سلسلة نصية واحدة. إذا استخدمنا tokenize=True ، وهو الإعداد الافتراضي ، فسيتم أيضًا تقسيم تلك السلسلة إلى رموز.

حاول الآن استخدام نفس الشفرة، لكن مع استبدال النموذج بـ HuggingFaceH4/zephyr-7b-beta ، وستحصل على:
```text
<|user|>
Hello, how are you?</s>
<|assistant|>
I'm doing great. How can I help you today?</s>
<|user|>
I'd like to show off how chat templating works!</s>

## كيف أستخدم قوالب الدردشة؟

كما رأيت في المثال السابق،  من السهل استخدام قوالب الدردشة. قم ببساطة بإنشاء قائمة من الرسائل، مع مفتاحي `role` و`content`، ثم قم بتمريرها إلى [`~PreTrainedTokenizer.apply_chat_template`] . بمجرد قيامك بذلك، ستحصل على مخرجات جاهزة للاستخدام! عند استخدام قوالب الدردشة كإدخال لتوليد نصوص بواسطة النموذج، فمن الجيد أيضًا استخدام `add_generation_prompt=True` لإضافة [مطالبات توليد النصوص](#what-are-generation-prompts).

فيما يلي مثال على إعداد الإدخال لـ `model.generate()`، باستخدام Zephyr مرة أخرى:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint) # قد ترغب في استخدام bfloat16 و/أو الانتقال إلى GPU هنا

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```
سيؤدي هذا إلى إنتاج سلسلة نصية بتنسيق الإدخال الذي يتوقعه Zephyr.

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

الآن بعد أن تم تنسيق الإدخال بشكل صحيح لـ Zephyr، يمكننا استخدام النموذج لإنشاء رد على سؤال المستخدم:

```python
outputs = model.generate(tokenized_chat, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

سيؤدي هذا إلى ما يلي:

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

كان ذلك سهلاً بعد كل شيء !



## هل هناك قنوات معالجة أوتوماتيكية للدردشة؟

نعم يوجد ! تدعم قنوات المعالجة توليد النصوص مدخلات الدردشة ، مما يُسهّل استخدام نماذج الدردشة . في الماضي ، كنا نستخدم فئة "ConversationalPipeline" المُخصّصة ، ولكن تم الآن إيقافها وتم دمج وظائفها في [TextGenerationPipeline]. دعونا نجرّب مثال Zephyr مرة أخرى ، ولكن هذه المرة باستخدام قناة معالجة:

```python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1]) # طباعة استجابة المساعد
```

```النص
{'role': 'assistant', 'content': "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."}
```

سيُراعي قناة المعالجة جميع تفاصيل تقسيم النص إلى رموز واستدعاء apply_chat_template نيابةً عنك - بمجرد أن يصبح لِدى النموذج قالب دردشة ، فكل ما تحتاج إلى القيام به هو تهيئة قناة معالجة وتمرير قائمة الرسائل إليها!

## ما هي "مطالبات التوليد"؟

قد تلاحظ أن طريقة `apply_chat_template` لها معامل `add_generation_prompt`. تخبر هذه المعامل القالب بإضافة رموز تشير إلى بداية رد البوت. على سبيل المثال، ضع في اعتبارك الدردشة التالية:

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

إليك كيف سيبدو ذلك بدون موجه توليد نصوص ، بالنسبة لنموذج يستخدم تنسيق "ChatML" القياسي :

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

وهكذا يبدو الأمر **مع** مطالبة التوليد:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
"""
```

لاحظ أننا أضفنا هذه المرة الرموز التي تشير إلى بداية رد البوت. يضمن هذا أنه عندما يُولّد النموذج نصًا فسيكتب رد البوت بدلاً من القيام بشيء غير متوقع، مثل الاستمرار في رسالة المستخدم. تذكر، أن نماذج الدردشة لا تزال مجرد نماذج للغة - فهي مدربة على متابعة النصوص، والدردشة هي مجرد نوع خاص من النصوص بالنسبة لها! يجب توجيهها برموز تحكم مناسبة، حتى تعرف ما الذي يجب عليها فعله.

لا تتطلب جميع النماذج الرموز التحكمية لتوليد نصوص . بعض النماذج ، مثل LLaMA ، ليس لديها أي رموز خاصة قبل ردود البوت . في هذه الحالات ، لن يكون لمعامل add_generation_prompt أي تأثير. يعتمد التأثير الدقيق الذي تُحدثه add_generation_prompt على القالب المستخدم .

## هل يمكنني استخدام قوالب الدردشة في التدريب؟

نعم ! تُعد هذه طريقة جيدة للتأكد من أن قالب الدردشة يتطابق مع الرموز التي يراها النموذج أثناء التدريب . نوصي بتطبيق قالب الدردشة كخطوة معالجة أولية لمجموعة بياناتك . بعد ذلك ، يمكنك ببساطة متابعة عملية التدريب كما هو الحال مع أي مهمة تدريب نماذج لغات أخرى . عند التدريب ، يجب أن تُعيّن عادةً  `add_generation_prompt=False` ، لأنه لن تكون الرموز المُضافة لتحفيز رد المساعد مفيدة أثناء التدريب . دعونا نرى مثالاً :

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```
ونحصل على:

```text
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

من هنا، استمر في التدريب كما تفعل مع مهمة نمذجة اللغة القياسية، باستخدام عمود `formatted_chat`.

<Tip>
بشكل افتراضي ، تضيف بعض *tokenizers* رموزًا خاصة مثل `<bos>` و `<eos>` إلى النص الذي تقوم بتقسيمه إلى رموز. يجب أن تتضمن قوالب المحادثة بالفعل جميع الرموز الخاصة التي تحتاجها ، وبالتالي فإن الرموز الخاصة الإضافية ستكون غالبًا غير صحيحة أو مُكررة ، مما سيؤثر سلبًا على أداء النموذج .

لذلك ، إذا قمت بتنسيق النص باستخدام  `apply_chat_template(tokenize=False)` ، فيجب تعيين المعامل add_special_tokens=False عندما تقوم بتقسيم ذلك النص إلى رموز لاحقًا . إذا كنت تستخدم apply_chat_template(tokenize=True) ، فلن تحتاج إلى القلق بشأن ذلك !
</Tip>

## متقدّم: مدخلات إضافية لِقوالب الدردشة


المعامل الوحيدة التي تتطلبها طريقة `apply_chat_template` هي `messages`. ومع ذلك، يمكنك تمرير أي معامل ككلمة مفتاحية إلى `apply_chat_template` وستكون متاحة داخل القالب. يمنحك هذا الكثير من المرونة لاستخدام قوالب الدردشة للعديد من الأشياء. لا توجد قيود على أسماء هذه المعامﻻت أو تنسيقاتها - يمكنك تمرير سلاسل نصية أو قوائم أو قواميس أو أي شيء آخر تريده.

ومع ذلك، هناك بعض الحالات الشائعة لاستخدام هذه المعامﻻت الإضافية، مثل تمرير أدوات لاستدعاء الوظائف، أو المستندات  لإنشاء النصوص المُعزّزة بالاسترجاع. في هذه الحالات الشائعة، لدينا بعض التوصيات المُحدّدة حول أسماء هذه المعامﻻت وتنسيقاتها، والتي يتم وصفها في الأقسام التالية. نشجع مطوّري النماذج على جعل قوالب الدردشة الخاصة بهم متوافقة مع هذا التنسيق، لتسهيل نقل التعليمات البرمجية لاستدعاء الأدوات بين النماذج.

## متقدم: استخدام الأداة / استدعاء الدالة

يمكن لنماذج "استخدام الأداة" اختيار استدعاء الدوال كأدوات خارجية قبل توليد الإجابة. عند تمرير الأدوات إلى نموذج استخدام الأدوات، يمكنك ببساطة تمرير قائمة من الوظائف إلى معامل `tools`:

```python
import datetime

def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

tools = [current_time, multiply]

model_input = tokenizer.apply_chat_template(
    messages,
    tools=tools
)
```

لكي يعمل هذا بشكل صحيح، يجب عليك كتابة وظائفك بالتنسيق السابق، حتى يمكن تحليلها بشكل صحيح كأدوات. على وجه التحديد، يجب عليك اتباع هذه القواعد:

- يجب أن يكون للدالة اسم وصفي.
- يجب أن يكون لكل معامل نوع للتلميح.
- يجب أن تحتوي الدالة على سلسلة مستندية بتنسيق Google القياسي (بمعنى وصف الدالة الأولي متبوعًا بكتلة `Args:` التي تصف المعاﻻت، ما لم تكن الدالة لا تحتوي على أي معامﻻت.
- لا تقم بتضمين الأنواع في كتلة `Args:` . بعبارة أخرى، اكتب `a: The first number to multiply`، وليس `a (int): The first number to multiply`. يجب أن تذهب تلميحات الأنواع في رأس الدالة بدلاً من ذلك.
- يمكن أن يكون للدالة نوع للإرجاع ومربع `Returns:` في السلسلة. ومع ذلك، فهذه اختيارية لأن معظم نماذج استخدام الأدوات تتجاهلها.

### تمرير نتائج الأداة إلى النموذج

يكفي الكود السابقة لسرد الأدوات المتاحة لنموذجك، ولكن ماذا يحدث إذا أراد النموذج استخدام واحدة منها؟ إذا حدث ذلك، فيجب عليك:

1. تحليل مخرجات النموذج للحصول على اسم (أسماء) الأدوات ومعامﻻتها.
2. أضف استدعاء (استدعاءات) النموذج لِلأدوات إلى المحادثة.
3. استدعاء الدالة (الدالات) المقابلة بتلك المعامﻻت.
4. أضف النتيجة (النتائج) إلى المحادثة

### مثال كامل على استخدام الأداة


سنستعرض مثالاً على استخدام الأدوات خطوة بخطوة . في هذا المثال ، سنستخدم نموذج Hermes-2-Pro بحجم 8 مليارات معامل ، نظرًا لأنه أحد أعلى نماذج استخدام الأدوات أداءً في فئة حجمه وقت كتابة هذا النص . إذا كان لديك الذاكرة الكافية ، فيمكنك النظر في استخدام نموذج أكبر بدلاً من ذلك مثل Command-R أو Mixtral-8x22B ، وكلاهما يدعم استخدام الأدوات ويوفر أداءً أقوى .


أولاً ، لنقم بتحميل نموذجنا و tokenizer الخاص بنا:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")

```python
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

الآن، لنقم نطبق قالب الدردشة ونولد رد:

```python
inputs = tokenizer.apply_chat_template(messages, chat_template="tool_use", tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

ونحصل على:

```text
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

لقد قام النموذج باستدعاء الدالة مع معامﻻت صحيحة، بالصيغة التي طلبتها توثيق الدالة. لقد استنتج أننا نشير على الأرجح إلى باريس في فرنسا، وتذكر أنه بكونها موطن وحدات القياس الدولية، يجب عرض درجة الحرارة في فرنسا بالدرجة المئوية.

دعنا نضيف استدعاء الأداة الخاص بالنموذج إلى المحادثة. لاحظ أننا نولد معرف استدعاء أداة عشوائيًا هنا. لا تستخدم جميع النماذج هذه المعرفات، ولكنها تسمح للنماذج بإصدار عدة استدعاءات للأدوات في نفس الوقت وتتبع الاستجابة المقابلة لكل استدعاء. يمكنك توليد هذه المعرفات بأي طريقة تريدها، ولكن يجب أن تكون فريدة داخل كل محادثة.

```python
tool_call_id = "vAHdf3"  # Random ID, should be unique for each tool call
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"id": tool_call_id, "type": "function", "function": tool_call}]})
```

الآن بعد أن أضفنا استدعاء الأداة إلى المحادثة، يمكننا استدعاء الدالة وإضافة النتيجة إلى المحادثة. نظرًا لأننا نستخدم دالة وهمية لهذا المثال والتي تعيد دائمًا 22.0، فيمكننا ببساطة إضافة تلك النتيجة مباشرةً. لاحظ معرف استدعاء الأداة - يجب أن يتطابق مع المعرف المستخدم في استدعاء الأداة أعلاه.

```python
messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": "get_current_temperature", "content": "22.0"})
```

أخيرًا، دعنا نجعل المساعد يقرأ مخرجات الدالة ويكمل الدردشة مع المستخدم:

```python
inputs = tokenizer.apply_chat_template(messages, chat_template="tool_use", tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

ونحصل على:

```text
The current temperature in Paris, France is 22.0 ° Celsius.<|im_end|>
```

على الرغم من أن هذا كان مجرد عرض توضيحيًا بسيطًا باستخدام أدوات وهمية ومحادثة ذات استدعاء واحد، إلا أن نفس التقنية تعمل مع أدوات حقيقية ومحادثات أطول. يمكن أن تكون هذه طريقة فعالة لتوسيع قدرات الوكلاء المحادثة بمعلومات حقيقية وأدوات حسابية مثل الآلات الحاسبة، أو الوصول إلى قواعد بيانات كبيرة.

<Tip>
لا تستخدم جميع نماذج استخدام الأدوات جميع ميزات استدعاء الأدوات الموضحة أعلاه. يستخدم البعض معرفات استدعاء الأدوات، بينما يستخدم البعض الآخر ببساطة اسم الدالة ويقارن استدعاءات الأدوات بالنتائج باستخدام الترتيب، وهناك عدة نماذج لا تستخدم أيًا منهما ولا تصدر سوى استدعاء أداة واحد في كل مرة لتجنب الارتباك. إذا كنت تريد أن يكون رمزك متوافقًا مع أكبر عدد ممكن من النماذج، فإننا نوصي بهيكلة استدعاءات الأدوات الخاصة بك كما هو موضح هنا، وإعادة نتائج الأدوات بالترتيب الذي أصدرها النموذج. يجب أن تتعامل قوالب الدردشة على كل نموذج مع الباقي.
</Tip>

### فهم مخططات الأدوات

يتم تحويل كل دالة تقوم بتمريرها إلى معامل "tools" في دالة "apply_chat_template" إلى [مخطط JSON](https://json-schema.org/learn/getting-started-step-by-step). يتم بعد ذلك تمرير هذه المخططات إلى قالب الدردشة النموذج. وبعبارة أخرى، فإن نماذج استخدام الأدوات لا ترى دوالك مباشرة، ولا ترى مطلقًا الكود الموجود بداخلها. ما يهمها هو**تعريفات** الدوال و**المعامﻻت** التي تحتاج إلى تمريرها إليها - فهي تهتم بما تفعله الأدوات وكيفية استخدامها، وليس بكيفية عملها! يقع على عاتقك قراءة مخرجاتها، والكشف عما إذا كانت قد طلبت استخدام أداة، وتمرير المعامﻻت إلى دالة الأداة، وإرجاع الرد في الدردشة.

يجب أن يكون إنشاء مخططات JSON لتمريرها إلى القالب تلقائيًا وغير مرئي طالما أن دوالك تتبع المواصفات الموضحة أعلاه، ولكن إذا واجهت مشكلات، أو إذا كنت تريد ببساطة مزيدًا من التحكم في التحويل، فيمكنك التعامل مع التحويل يدويًا. فيما يلي مثال على تحويل مخطط يدوي:

```python
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

schema = get_json_schema(multiply)
print(schema)
```

سيؤدي هذا إلى ما يلي:

```json
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "A function that multiplies two numbers", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "The first number to multiply"
        }, 
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

إذا كنت ترغب في ذلك، يمكنك تحرير هذه المخططات، أو حتى كتابتها من البداية بنفسك دون استخدام `get_json_schema` على الإطلاق. يمكن تمرير مخططات JSON مباشرةً إلى معامل  `tools` في `apply_chat_template` - يمنحك هذا الكثير من القوة لتعريف مخططات دقيقة لوظائف أكثر تعقيدًا. ولكن كن حذرًا - كلما زاد تعقيد مخططاتك، زاد احتمال ارتباك النموذج عند التعامل معها! نوصي  بتوقيعات دوال بسيطة حيثما أمكن، مع تقليل المعامﻻت (وخاصة المعامﻻت المعقدة والمتداخلة) إلى الحد الأدنى.

فيما يلي مثال على تعريف المخططات يدويًا، وتمريرها مباشرةً إلى `apply_chat_template`:

```python
# A simple function that takes no arguments
current_time = {
  "type": "function", 
  "function": {
    "name": "current_time",
    "description": "Get the current local time as a string.",
    "parameters": {
      'type': 'object',
      'properties': {}
    }
  }
}

# A more complete function that takes two numerical arguments
multiply = {
  'type': 'function',
  'function': {
    'name': 'multiply',
    'description': 'A function that multiplies two numbers', 
    'parameters': {
      'type': 'object', 
      'properties': {
        'a': {
          'type': 'number',
          'description': 'The first number to multiply'
        }, 
        'b': {
          'type': 'number', 'description': 'The second number to multiply'
        }
      }, 
      'required': ['a', 'b']
    }
  }
}

model_input = tokenizer.apply_chat_template(
    messages,
    tools = [current_time, multiply]
)
```


## ما القالب الذي يجب أن أستخدمه؟

عند تعيين قالب لنموذج تم تدريبه بالفعل على الدردشة، يجب التأكد من أن القالب يتطابق تمامًا مع تنسيق الرسالة الذي شاهده النموذج أثناء التدريب، وإلا فمن المحتمل أن تواجه تدهورًا في الأداء. هذا صحيح حتى إذا كنت تدرب النموذج بشكل إضافي - فمن المحتمل أن تحصل على أفضل أداء إذا قمت بإبقاء رموز الدردشة ثابتة.  يُشبه هذا إلى حد كبير عملية التجزئة - فأنت تحصل بشكل عام على أفضل أداء للاستدلال أو الضبط الدقيق عندما تتطابق بدقة مع التجزئة المستخدمة أثناء التدريب.

من ناحية أخرى، إذا كنت تُدرّب نموذجًا من البداية، أو تقوم بضبط دقيق لنموذج لغة أساسي للدردشة، لديك حرية اختيار قالب مناسب! تتمتع LLMs بالذكاء الكافي للتعامل مع العديد من تنسيقات الإدخال المختلفة. أحد الخيارات الشائعة هو تنسيق "ChatML"، وهو خيار جيد ومرن للعديد من حالات الاستخدام. يبدو كالتالي:

```
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
```

إذا أعجبك هذا، فإليك نسخة جاهزة لوضعها في كودك. يتضمن الخط المفرد أيضًا دعمًا مفيدًا [لإرشادات التوليد](#what-are-generation-prompts)، ولكن لاحظ أنه لا يضيف رموز BOS أو EOS! إذا كان نموذجك يتوقع هذه الرموز، فلن يتم إضافتها تلقائيًا بواسطة "apply_chat_template" - بمعنى آخر، سيتم تجزئة النص باستخدام "add_special_tokens=False". هذا لتجنب التعارضات المحتملة بين القالب ومنطق "add_special_tokens". إذا كان نموذجك يتوقع رموزًا خاصة، فتأكد من إضافتها إلى القالب!

```python
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

يُحيط هذا القالب كل رسالة بين الرمزين "<|im_start|>" و "<|im_end|>"، ويكتب ببساطة الدور كسلسلة نصية، مما يسمح بالمرونة في الأدوار التي تتدرب عليها. يبدو الناتج كما يلي:

```text
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

تعد أدوار "user" و "system" و "assistant" هي المعيار للدردشة، ونوصي باستخدامها عندما يكون ذلك منطقيًا، خاصة إذا كنت تريد أن يعمل نموذجك بشكل جيد مع [`TextGenerationPipeline`]. ومع ذلك، فأنت لست مقيدًا بهذه الأدوار - القوالب مرنة للغاية، ويمكن أن تكون أي سلسلة دورًا.


## أريد إضافة بعض قوالب الدردشة! كيف أبدأ؟

إذا كان لديك أي نماذج دردشة، فيجب عليك تعيين سمة "tokenizer.chat_template" الخاصة بها واختبارها باستخدام [`~PreTrainedTokenizer.apply_chat_template`]، ثم دفع محدد التحليل اللغوي المحدث إلى Hub. ينطبق هذا حتى إذا لم تكن مالك النموذج - إذا كنت تستخدم نموذجًا بقالب دردشة فارغ، أو لا يزال يستخدم قالب الفئة الافتراضية، فيرجى فتح طلب سحب إلى مستودع النموذج حتى يمكن تعيين السمة بشكل صحيح!

بمجرد تعيين السمة، هذا كل شيء، لقد انتهيت! ستعمل "tokenizer.apply_chat_template" الآن بشكل صحيح لهذا النموذج، مما يعني أنها مدعومة أيضًا بشكل تلقائي في أماكن مثل "TextGenerationPipeline"!

من خلال التأكد من أن النماذج لديها هذه السمة، يمكننا التأكد من أن المجتمع بأكمله يستخدم القوة الكاملة للنماذج مفتوحة المصدر. لقد كانت عدم تطابق التنسيق تطارد المجال وتضر الأداء بهدوء لفترة طويلة جدًا - لقد حان الوقت لوضع حد لها!

## متقدم: نصائح لكتابة القوالب

إذا كنت غير معتاد على Jinja، فإننا نجد بشكل عام أن أسهل طريقة لكتابة قالب دردشة هي أولاً كتابة نص برمجي Python قصير يقوم بتنسيق الرسائل بالطريقة التي تريدها، ثم تحويل هذا النص إلى قالب.

تذكر أن برنامج التعامل مع القوالب سيستلم سجل المحادثة كمتغير يسمى "messages". ستتمكن من الوصول إلى "messages" في قالبك تمامًا كما هو الحال في Python، مما يعني أنه يمكنك التكرار عبره باستخدام "{% for message in messages %}" أو الوصول إلى الرسائل الفردية باستخدام "{{ messages[0] }}"، على سبيل المثال.

يمكنك أيضًا استخدام النصائح التالية لتحويل كودك إلى Jinja:

### تقليم المسافات البيضاء

بشكل افتراضي، ستطبع Jinja أي مسافات بيضاء تأتي قبل أو بعد كتلة. يمكن أن يكون هذا مشكلة لقوالب الدردشة، والتي تريد عادةً أن تكون دقيقة جدًا مع المسافات البيضاء! لتجنب ذلك، نوصي بشدة بكتابة قوالبك على النحو التالي:

```
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

بدلاً من ذلك:

```
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

سيؤدي إضافة "-" إلى إزالة أي مسافات بيضاء تأتي قبل الكتلة. يبدو المثال الثاني بريئًا، ولكن قد يتم تضمين السطر الجديد والمسافة البادئة في الإخراج، وهو على الأرجح ليس ما تريده!

### حلقات for

تبدو حلقات for في Jinja على النحو التالي:

```
{%- for message in messages %}
    {{- message['content'] }}
{%- endfor %}
```

لاحظ أن ما بداخل كتلة `{{ expression }}` سيتم طباعته في الإخراج. يمكنك استخدام عوامل التشغيل مثل `+` لدمج السلاسل داخل كتل التعبير.

### جمل الشرط

تبدو جمل الشرط في Jinja على النحو التالي:

```
{%- if message['role'] == 'user' %}
    {{- message['content'] }}
{%- endif %}
```

لاحظ كيف يستخدم Python المسافات البيضاء للإشارة إلى بدايات ونهايات كتل "for" و "if"، تتطلب Jinja منك إنهاءها بشكل صريح باستخدام "{% endfor %}" و "{% endif %}".

### المتغيرات الخاصة

داخل قالبك، ستتمكن من الوصول إلى قائمة "messages"، ولكن يمكنك أيضًا الوصول إلى العديد من المتغيرات الخاصة الأخرى. تشمل هذه الرموز الخاصة مثل "bos_token" و "eos_token"، بالإضافة إلى متغير "add_generation_prompt" الذي ناقشناه أعلاه. يمكنك أيضًا استخدام متغير "loop" للوصول إلى معلومات حول تكرار الحلقة الحالي، على سبيل المثال باستخدام "{% if loop.last %}" للتحقق مما إذا كانت الرسالة الحالية هي الرسالة الأخيرة في المحادثة. إليك مثال يجمع هذه الأفكار معًا لإضافة موجه إنشاء في نهاية المحادثة إذا كان "add_generation_prompt" هو "True":

```
{%- if loop.last and add_generation_prompt %}
    {{- bos_token + 'Assistant:\n' }}
{%- endif %}
```

### التوافق مع Jinja غير Python
```
{%- if loop.last and add_generation_prompt %}
    {{- bos_token + 'Assistant:\n' }}
{%- endif %}
```

### التوافق مع Jinja غير Python

هناك العديد من تطبيقات Jinja بلغات مختلفة. عادة ما يكون لها نفس بناء الجملة، ولكن الاختلاف الرئيسي هو أنه عند كتابة قالب في Python، يمكنك استخدام طرق Python، مثل ".lower()" على السلاسل أو ".items()" على القواميس. سيؤدي هذا إلى كسر إذا حاول شخص ما استخدام قالبك في تنفيذ غير Python لـ Jinja. تعد التطبيقات غير Python شائعة بشكل خاص في بيئات النشر، حيث تعد JS و Rust شائعة جدًا.

لا تقلق، على الرغم من ذلك! هناك بعض التغييرات البسيطة التي يمكنك إجراؤها على قوالبك لضمان توافقها عبر جميع تطبيقات Jinja:

- استبدل طرق Python بمرشحات Jinja. عادة ما يكون لها نفس الاسم، على سبيل المثال، يصبح "string.lower()" عبارة عن "string|lower"، ويصبح "dict.items()" عبارة عن "dict|items". أحد التغييرات الملحوظة هو أن "string.strip()" يصبح "string|trim". راجع [قائمة المرشحات المدمجة](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters) في وثائق Jinja لمزيد من المعلومات.
- استبدل "True" و "False" و "None"، وهي خاصة بـ Python، بـ "true" و "false" و "none".
- قد يؤدي عرض قاموس أو قائمة مباشرة إلى نتائج مختلفة في التطبيقات الأخرى (على سبيل المثال، قد تتغير الإدخالات ذات الاقتباس المفرد إلى إدخالات مزدوجة الاقتباس). يمكن أن يساعد إضافة مرشح "tojson" في ضمان الاتساق هنا.