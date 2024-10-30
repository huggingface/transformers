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

لاحظ كيف أضاف المجزىء اللغوى tokenizer رموز التحكم `[INST]` و `[/INST]` للإشارة إلى بداية ونهاية رسائل المستخدم (ولكن ليس رسائل المساعد!) ، وتم تكثيف المحادثة بأكملها في سلسلة نصية واحدة. إذا استخدمنا `tokenize=True` ، وهو الإعداد الافتراضي ، فسيتم أيضًا تقسيم تلك السلسلة إلى رموز.

حاول الآن استخدام نفس الشفرة، لكن مع استبدال النموذج بـ `HuggingFaceH4/zephyr-7b-beta` ، وستحصل على:
```text
<|user|>
Hello, how are you?</s>
<|assistant|>
I'm doing great. How can I help you today?</s>
<|user|>
I'd like to show off how chat templating works!</s>
```
تم ضبط كل من Zephyr و Mistral-Instruct من نفس النموذج الأصلي ، Mistral-7B-v0.1. ومع ذلك ، فقد تم تدريبهم بتنسيقات دردشة مختلفة تمامًا. بدون قوالب المحادثة، ستضطر إلى كتابة شفرة تنسيق يدويًا لكل نموذج ، ومن السهل جدًا ارتكاب أخطاء بسيطة تؤثر على الأداء! تُدير قوالب المحادثة تفاصيل التنسيق نيابةً عنك ، مما يُتيح لك كتابة شفرة عامة تعمل مع أي نموذج.

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

نعم يوجد ! تدعم قنوات المعالجة توليد النصوص مدخلات الدردشة ، مما يُسهّل استخدام نماذج الدردشة . في الماضي ، كنا نستخدم فئة "ConversationalPipeline" المُخصّصة ، ولكن تم الآن إيقافها وتم دمج وظائفها في [`TextGenerationPipeline`]. دعونا نجرّب مثال Zephyr مرة أخرى ، ولكن هذه المرة باستخدام قناة معالجة:

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

لا تتطلب جميع النماذج الرموز التحكمية لتوليد نصوص . بعض النماذج ، مثل LLaMA ، ليس لديها أي رموز خاصة قبل ردود البوت . في هذه الحالات ، لن يكون لمعامل `add_generation_prompt` أي تأثير. يعتمد التأثير الدقيق الذي تُحدثه `add_generation_prompt` على القالب المستخدم .

## ما وظيفة "continue_final_message"؟

عند تمرير قائمة من الرسائل إلى `apply_chat_template` أو `TextGenerationPipeline` ، يمكنك اختيار تنسيق المحادثة بحيث يواصل النموذج الرسالة الأخيرة في المحادثة بدلاً من بدء رسالة جديدة. يتم ذلك عن طريق إزالة أي رموز نهاية التسلسل التي تشير إلى نهاية الرسالة الأخيرة ، بحيث يقوم النموذج ببساطة بتمديد الرسالة الأخيرة عندما يبدأ في توليد النص . يُعد هذا أمرًا مفيدًا "لِمَلء بداية" رد النموذج مُسبقًا.

وهنا مثال:
```python
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
```
سيقوم النموذج بتوليد نص يكمل سلسلة JSON ، بدلاً من بدء رسالة جديدة . يمكن أن يكون هذا النهج مفيدًا جدًا لتحسين دقة اتباع النموذج للإرشادات عندما تعرف كيف تريد أن يبدأ ردوده .
.

نظرًا لأن `add_generation_prompt` تضيف الرموز التي تبدأ رسالة جديدة ، و `continue_final_message` تزيل أي رموز نهاية الرسالة من الرسالة الأخيرة ، فليس من المنطقي استخدامهما معًا . ونتيجة لذلك ، ستتلقّى خطأً إذا حاولت ذلك !

السلوك الافتراضي لِـ `TextGenerationPipeline` هو تعيين `add_generation_prompt=True` بحيث تبدأ رسالة جديدة . ومع ذلك ، إذا كانت الرسالة الأخيرة في المحادثة التي تم إدخالها لديها دور "assistant" ، فسوف تفترض أن هذه الرسالة هي "مَلء بداية" وتتحوّل إلى `continue_final_message=True` بدلاً من ذلك ، لأن مُعظم النماذج لا تدعم عدة رسائل متتالية للمساعد . يمكنك تجاوز هذا السلوك عن طريق تمرير معامل `continue_final_message` بشكل صريح عند استدعاء قناة المعالجة .



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

لذلك ، إذا قمت بتنسيق النص باستخدام  `apply_chat_template(tokenize=False)` ، فيجب تعيين المعامل `add_special_tokens=False` عندما تقوم بتقسيم ذلك النص إلى رموز لاحقًا . إذا كنت تستخدم `apply_chat_template(tokenize=True)` ، فلن تحتاج إلى القلق بشأن ذلك !
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


سنستعرض مثالاً على استخدام الأدوات خطوة بخطوة . في هذا المثال ، سنستخدم نموذج `Hermes-2-Pro` بحجم 8 مليارات معامل ، نظرًا لأنه أحد أعلى نماذج استخدام الأدوات أداءً في فئة حجمه وقت كتابة هذا النص . إذا كان لديك الذاكرة الكافية ، فيمكنك النظر في استخدام نموذج أكبر بدلاً من ذلك مثل `Command-R` أو `Mixtral-8x22B` ، وكلاهما يدعم استخدام الأدوات ويوفر أداءً أقوى .


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

<Tip>
لا تستخدم جميع نماذج استخدام الأدوات جميع ميزات استدعاء الأدوات الموضحة أعلاه. يستخدم البعض معرفات استدعاء الأدوات، بينما يستخدم البعض الآخر ببساطة اسم الدالة ويقارن استدعاءات الأدوات بالنتائج باستخدام الترتيب، وهناك عدة نماذج لا تستخدم أيًا منهما ولا تصدر سوى استدعاء أداة واحد في كل مرة لتجنب الارتباك. إذا كنت تريد أن يكون رمزك متوافقًا مع أكبر عدد ممكن من النماذج، فإننا نوصي بهيكلة استدعاءات الأدوات الخاصة بك كما هو موضح هنا، وإعادة نتائج الأدوات بالترتيب الذي أصدرها النموذج. يجب أن تتعامل قوالب الدردشة على كل نموذج مع الباقي.
</Tip>

### فهم مخططات الأدوات

يتم تحويل كل دالة تقوم بتمريرها إلى معامل `tools` في دالة `apply_chat_template` إلى [مخطط JSON](https://json-schema.org/learn/getting-started-step-by-step). يتم بعد ذلك تمرير هذه المخططات إلى قالب الدردشة النموذج. وبعبارة أخرى، فإن نماذج استخدام الأدوات لا ترى دوالك مباشرة، ولا ترى مطلقًا الكود الموجود بداخلها. ما يهمها هو**تعريفات** الدوال و**المعامﻻت** التي تحتاج إلى تمريرها إليها - فهي تهتم بما تفعله الأدوات وكيفية استخدامها، وليس بكيفية عملها! يقع على عاتقك قراءة مخرجاتها، والكشف عما إذا كانت قد طلبت استخدام أداة، وتمرير المعامﻻت إلى دالة الأداة، وإرجاع الرد في الدردشة.

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

## متقدم: توليد قائم على الاسترجاع
يمكن لنماذج اللغة الكبيرة من نوع "توليد قائم على الاسترجاع" أو "RAG" البحث في مجموعة نصوص عن معلومات قبل الرد على الاستعلام. يسمح هذا للنماذج بتوسيع قاعدة معارفها بشكل كبير إلى ما هو أبعد من حجم سياقها المحدود. توصيتنا لنماذج RAG هي أن يقبل قالبها وسيطة `documents`. يجب أن تكون هذه قائمة من المستندات، حيث يكون كل "مستند" عبارة عن قاموس واحد بمفاتيح `title` و `contents`، وكلاهما سلاسل نصية. نظرًا لأن هذا التنسيق أبسط بكثير من مخططات JSON المستخدمة للأدوات، فلا توجد حاجة إلى دوال مساعدة.

فيما يلي مثال على قالب RAG بالفعل:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# تحميل النموذج والمجزىء اللغوي
model_id = "CohereForAI/c4ai-command-r-v01-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
device = model.device # الحصول على الجهاز الذي تم تحميل النموذج عليه

# تعريف مُدخلات المحادثة
conversation = [
    {"role": "user", "content": "What has Man always dreamed of?"}
]

# تعريف المستندات لتوليد قائم على الاسترجاع
documents = [
    {
        "title": "The Moon: Our Age-Old Foe", 
        "text": "Man has always dreamed of destroying the moon. In this essay, I shall..."
    },
    {
        "title": "The Sun: Our Age-Old Friend",
        "text": "Although often underappreciated, the sun provides several notable benefits..."
    }
]
# معالجة المحادثة والمستندات باستخدام قالب RAG، وإرجاع موترات PyTorch.
input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    documents=documents,
    chat_template="rag",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt").to(device)

# توليد الرد
gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

# فك تشفير النص المُوَلّد وطباعته
gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```
إن مُدخل documents للتوليد القائم على الاسترجاع غير مدعوم على نطاق واسع، والعديد من النماذج لديها قوالب دردشة تتجاهل هذا المُدخل ببساطة.

للتحقق مما إذا كان النموذج يدعم مُدخل `documents`، يمكنك قراءة بطاقة النموذج الخاصة به، أو `print(tokenizer.chat_template)` لمعرفة ما إذا كان مفتاح `documents` مستخدمًا في أي مكان.
<Tip>
ومع ذلك، فإن أحد فئات النماذج التي تدعمه هي [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) و [Command-R+](https://huggingface.co/CohereForAI/c4ai-command-r-pluse-08-2024) من Cohere، من خلال قالب الدردشة rag الخاص بهم. يمكنك رؤية أمثلة إضافية على التوليد باستخدام هذه الميزة في بطاقات النموذج الخاصة بهم.
</Tip>

## متقدم: كيف تعمل قوالب الدردشة؟
يتم تخزين قالب الدردشة للنموذج في الخاصية `tokenizer.chat_template`. إذا لم يتم تعيين قالب دردشة، فسيتم استخدام القالب الافتراضي لفئة النموذج هذه بدلاً من ذلك. دعونا نلقي نظرة على قالب دردشة `Zephyr`، ولكن لاحظ أن هذا القالب مُبسّط قليلاً عن القالب الفعلي!

```
{%- for message in messages %}
    {{- '<|' + message['role'] + |>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```
إذا لم تكن قد رأيت أحد هذه القوالب من قبل، فهذا [قالب Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) .Jinja هي لغة قوالب تسمح لك بكتابة تعليمات برمجية بسيطة تُوَلّد نصًا. من نواحٍ عديدة، يُشبه الرمز والتركيب للغة Python. أما في لغة Python، سيبدو هذا القالب كما يلي:

```python
for message in messages:
    print(f'<|{message["role"]}|>')
    print(message['content'] + eos_token)
if add_generation_prompt:
    print('<|assistant|>')
```
يقوم القالب بثلاثة أشياء بشكل فعال:

- لكل رسالة، بطبع الدور مُحاطًا بـ `<|` و `|>`، مثل `<|user|>` أو `<|assistant|>`.
- بعد ذلك، يطبع محتوى الرسالة، متبوعًا برمز نهاية التسلسل `eos_token` .
- أخيرًا، إذا تم تعيين `add_generation_prompt` ، يطبع الرمز المساعد، حتى يعرف النموذج أنه يجب أن يبدأ في توليد استجابة المساعد.
  
هذا قالب بسيط جدًا، لكن Jinja تمنحك الكثير من المرونة للقيام بأشياء أكثر تعقيدًا! دعونا نرى قالب Jinja يُمكنه تنسيق المُدخلات بطريقة تُشبه الطريقة التي تُنسّق بها LLaMA مُدخلاتها (لاحظ أن قالب LLaMA الحقيقي يتضمن معالجة لرسائل النظام الافتراضية ومعالجة رسائل النظام بشكل مختلف قليلاً بشكل عام - لا تستخدم هذا القالب في التعليمات البرمجية الفعلية الخاصة بك!)
```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + message['content'] + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
```
نأمل أنه إذا حدقت في هذا لفترة قصيرة، يمكنك أن ترى ما يفعله هذا القالب - فهو يُضيف رموزًا مُحددة مثل `[INST]` و `[/INST]` بناءً على دور كل رسالة. يمكن تمييز رسائل المستخدم والمساعد والنظام بوضوح للنموذج بسبب الرموز التي تُحيط بها.

## متقدم: إضافة وتعديل قوالب الدردشة

### كيف أنشئ قالب دردشة؟
ببساطة، اكتب قالب Jinja واضبط `tokenizer.chat_template`. قد تجد أنه من الأسهل البدء بقالب موجود من نموذج آخر وتحريره ببساطة ليناسب احتياجاتك! على سبيل المثال، يمكننا أن نأخذ قالب LLaMA أعلاه ونضيف `[ASST]` و `[/ASST]` إلى رسائل المساعد:

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}
```

الآن، اضبط ببساطة الخاصية `tokenizer.chat_template`. في المرة القادمة التي تستخدم فيها [`~PreTrainedTokenizer.apply_chat_template`] ، سيستخدم القالب الجديد الخاص بك! سيتم حفظ هذه الخاصية في ملف `tokenizer_config.json`، حتى تتمكن من استخدام  [`~utils.PushToHubMixin.push_to_hub`] لتحميل قالبك الجديد إلى Hub والتأكد من أن الجميع يستخدم القالب الصحيح لنموذجك!

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # تغيير رمز النظام
tokenizer.chat_template = template  # تعيين القالب الجديد
tokenizer.push_to_hub("model_name")  # تحميل القالب الجديد إلى Hub!
```

يتم استدعاء الدالة [`~PreTrainedTokenizer.apply_chat_template`] الذي نستخدم قالب الدردشة الخاص بك بواسطة فئة [`TextGenerationPipeline`] لذلك بمجرد تعيين قالب الدردشة الصحيح، سيصبح نموذجك متوافقًا تلقائيًا مع [`TextGenerationPipeline`].

<Tip>
إذا كنت تُجري ضبطًا دقيقًا لنموذج للدردشة، بالإضافة إلى تعيين قالب دردشة، فربما يجب عليك إضافة أي رموز تحكم دردشة جديدة كرموز خاصة في المجزىء اللغوي. لا يتم تقسيم الرموز الخاصة أبدًا، مما يضمن معالجة رموز التحكم الخاصة بك دائمًا كرموز فردية بدلاً من تجزئتها إلى أجزاء. يجب عليك أيضًا تعيين خاصية `eos_token` للمجزىء اللغوي إلى الرمز الذي يُشير إلى نهاية توليدات المساعد في قالبك. سيضمن هذا أن أدوات توليد النصوص يمكنها تحديد وقت إيقاف توليد النص بشكل صحيح.
</Tip>

### لماذا تحتوي بعض النماذج على قوالب متعددة؟
تستخدم بعض النماذج قوالب مختلفة لحالات استخدام مختلفة. على سبيل المثال، قد تستخدم قالبًا واحدًا للدردشة العادية وآخر لاستخدام الأدوات، أو التوليد القائم على الاسترجاع. في هذه الحالات، تكون `tokenizer.chat_template` قاموسًا. يمكن أن يتسبب هذا في بعض الارتباك، وحيثما أمكن، نوصي باستخدام قالب واحد لجميع حالات الاستخدام. يمكنك استخدام عبارات Jinja مثل `if tools is defined` وتعريفات `{% macro %}` لتضمين مسارات تعليمات برمجية متعددة بسهولة في قالب واحد.

عندما يحتوي المعالج اللغوي على قوالب متعددة، ستكون `tokenizer.chat_template dict`، حيث يكون كل مفتاح هو اسم قالب. يحتوي أسلوب `apply_chat_template` على معالجة خاصة لأسماء قوالب مُعينة: على وجه التحديد، سيبحث عن قالب باسم `default` في معظم الحالات، وسيُثير خطأً إذا لم يتمكن من العثور على واحد. ومع ذلك، إذا كان هناك قالب باسم `tool_use` عندما قام المستخدم بتمرير وسيطة `tools`، فسيستخدم هذا القالب بدلاً من ذلك. للوصول إلى قوالب بأسماء أخرى، مرر اسم القالب الذي تُريده إلى وسيطة `chat_template` لـ `apply_chat_template()`.

نجد أن هذا قد يكون مُربكًا بعض الشيء للمستخدمين - لذلك إذا كنت تكتب قالبًا بنفسك، فننصحك بمحاولة وضعه كله في قالب واحد حيثما أمكن!

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

تعد أدوار "user" و "system" و "assistant" هي الأدوار القياسية للدردشة، ونوصي باستخدامها عندما يكون ذلك منطقيًا، خاصة إذا كنت تريد أن يعمل نموذجك بشكل جيد مع [`TextGenerationPipeline`]. ومع ذلك، فأنت لست مقيدًا بهذه الأدوار - فإن القوالب مرنة للغاية، ويمكن أن تكون أي سلسلة نصية دورًا.


## أريد إضافة بعض قوالب الدردشة! كيف أبدأ؟

إذا كان لديك أي نماذج دردشة، فيجب عليك تعيين الخاصية "tokenizer.chat_template" الخاصة بها واختبارها باستخدام [`~PreTrainedTokenizer.apply_chat_template`]، ثم رفع  المجزىء اللغوي المُحدّث إلى Hub. ينطبق هذا حتى إذا لم تكن مالك النموذج - إذا كنت تستخدم نموذجًا بقالب دردشة فارغ، أو لا يزال يستخدم قالب الفئة الافتراضية، فيرجى فتح [طلب سحب](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)  إلى مستودع النموذج حتى يمكن تعيين الخاصية بشكل صحيح!

بمجرد تعيين الخاصية، هذا كل شيء، لقد انتهيت! ستعمل "tokenizer.apply_chat_template" الآن بشكل صحيح لهذا النموذج، مما يعني أنها مدعومة أيضًا بشكل تلقائي في أماكن مثل "TextGenerationPipeline"!

من خلال ضمان امتلاك النماذج لهذه الخاصية، يُمكننا التأكد من أن المجتمع بأكمله يستخدم القوة الكاملة للنماذج مفتوحة المصدر. لقد كانت عدم تطابق التنسيق تطارد المجال وأضرت الأداء بصمت لفترة طويلة جدًا - لقد حان الوقت لوضع حد لها!

## متقدم: نصائح لكتابة القوالب

<Tip>
أسهل طريقة للبدء في كتابة قوالب Jinja هي إلقاء نظرة على بعض القوالب الموجودة. يمكنك استخدام `print(tokenizer.chat_template)` لأي نموذج دردشة لمعرفة القالب الذي يستخدمه. بشكل عام، تحتوي النماذج التي تدعم استخدام الأدوات على قوالب أكثر تعقيدًا بكثير من النماذج الأخرى - لذلك عندما تبدأ للتو، فمن المحتمل أنها مثال سيئ للتعلم منه! يمكنك أيضًا إلقاء نظرة على [وثائق Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis) للحصول على تفاصيل حول تنسيق Jinja العام وتركيبه.

</Tip>

تُطابق قوالب Jinja في `transformers` قوالب Jinja في أي مكان آخر. الشيء الرئيسي الذي يجب معرفته هو أن سجل الدردشة سيكون متاحًا داخل قالبك كمتغير يسمى `messages`. ستتمكن من الوصول إلى `messages` في قالبك تمامًا كما يمكنك في Python، مما يعني أنه يمكنك التكرار خلاله باستخدام `{% for message in messages %}` أو الوصول إلى رسائل فردية باستخدام `{{ messages[0] }}`، على سبيل المثال.

يمكنك أيضًا استخدام النصائح التالية لكتابة قوالب Jinja نظيفة وفعالة:

### إقتطاع المسافات الفارغة

بشكل افتراضي، ستطبع Jinja أي مسافات فارغة تأتي قبل أو بعد كتلة. يمكن أن يكون هذا مشكلة لقوالب الدردشة، والتي تريد عادةً أن تكون دقيقة جدًا مع المسافات! لتجنب ذلك، نوصي بشدة بكتابة قوالبك على النحو التالي:

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

سيؤدي إضافة "-" إلى إزالة أي مسافات تأتي قبل الكتلة. يبدو المثال الثاني عادية، ولكن قد يتم تضمين السطر الجديد والمسافة البادئة في المخرجات، وهو على الأرجح ليس ما تُريده!


### المتغيرات الخاصة

 داخل قالبك، سيكون لديك حق الوصول إلى العديد من المتغيرات الخاصة. أهمها هو `messages`، والذي يحتوي على سجل الدردشة كقائمة من قواميس الرسائل. ومع ذلك، هناك العديد من المتغيرات الأخرى. لن يتم استخدام كل متغير في كل قالب. المتغيرات الأكثر شيوعًا هي:

- `tools` تحتوي على قائمة بالأدوات بتنسيق مخطط JSON. ستكون `None` أو غير مُعرّفة إذا لم يتم تمرير أي أدوات.
- `documents` تحتوي على قائمة من المستندات بالتنسيق `{"title": "العنوان", "contents": "المحتويات"}`، تُستخدم للتوليد المُعزز بالاسترجاع. ستكون `None` أو غير مُعرّفة إذا لم يتم تمرير أي مستندات.
- `add_generation_prompt` هي قيمة منطقية تكون `True` إذا طلب المستخدم مُطالبة توليد، و `False` بخلاف ذلك. إذا تم تعيين هذا، فيجب أن يُضيف قالبك رأس رسالة مساعد إلى نهاية المحادثة. إذا لم يكن لدى نموذجك رأس مُحدد لرسائل المساعد، فيمكنك تجاهل هذا العلم.
- **الرموز الخاصة** مثل `bos_token` و `eos_token`. يتم استخراجها من `tokenizer.special_tokens_map`. ستختلف الرموز الدقيقة المتاحة داخل كل قالب اعتمادًا على المجزىء اللغوي الأصلي.


<Tip>

يمكنك في الواقع تمرير أي `kwarg` إلى `apply_chat_template`، وستكون متاحة داخل القالب كمتغير. بشكل عام، نوصي بمحاولة الالتزام بالمتغيرات الأساسية المذكورة أعلاه، لأن ذلك سيجعل نموذجك أكثر صعوبة في الاستخدام إذا كان على المستخدمين كتابة تعليمات برمجية مخصصة لتمرير `kwargs` خاصة بالنموذج. ومع ذلك، فنحن نُدرك أن هذا المجال يتحرك بسرعة، لذلك إذا كانت لديك حالة استخدام جديدة لا تتناسب مع واجهة برمجة التطبيقات الأساسية، فلا تتردد في استخدام `kwarg`  معامل جديد لها! إذا أصبح `kwarg` المعامل الجديد شائعًا، فقد نقوم بترقيته إلى واجهة برمجة التطبيقات الأساسية وإنشاء  وتوثيق الخاص به.

</Tip>

### دوال قابلة للاستدعاء

هناك أيضًا قائمة قصيرة من الدوال القابلة للاستدعاء المتاحة لك داخل قوالبك. هذه هي:

- `raise_exception(msg)`: تُثير `TemplateException`. هذا مفيد لتصحيح الأخطاء، ولإخبار المستخدمين عندما يفعلون شيئًا لا يدعمه قالبك.
- `strftime_now(format_str)`: تُكافئ `datetime.now().strftime(format_str)` في Python. يُستخدم هذا للحصول على التاريخ/الوقت الحالي بتنسيق مُحدد، والذي يتم تضمينه أحيانًا في رسائل النظام.

### التوافق مع Jinja غير Python

هناك تطبيقات متعددة لـ Jinja بلغات مختلفة. عادة ما يكون لها نفس التركيب، ولكن الاختلاف الرئيسي هو أنه عند كتابة قالبًا في Python، يمكنك استخدام أساليب Python، مثل ".lower()" على السلاسل أو ".items()" على القواميس. سيؤدي هذا إلى كسر إذا حاول شخص ما استخدام قالبك في تنفيذ غير Python لـ Jinja. تعد التطبيقات غير Python شائعة بشكل خاص في بيئات النشر، حيث تعد JS و Rust شائعة جدًا.

لا تقلق، على الرغم من ذلك! هناك بعض التغييرات البسيطة التي يمكنك إجراؤها على قوالبك لضمان توافقها عبر جميع تطبيقات Jinja:

- استبدل أساليب Python بمرشحات Jinja. عادة ما يكون لها نفس الاسم، على سبيل المثال، يصبح "string.lower()" عبارة عن "string|lower"، ويصبح "dict.items()" عبارة عن "dict|items". أحد التغييرات الملحوظة هو أن "string.strip()" يصبح "string|trim". راجع [قائمة المرشحات المدمجة](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters) في وثائق Jinja لمزيد من المعلومات.
- استبدل "True" و "False" و "None"، وهي خاصة بـ Python، بـ "true" و "false" و "none".
- قد يؤدي عرض قاموس أو قائمة مباشرة إلى نتائج مختلفة في التطبيقات الأخرى (على سبيل المثال، قد تتغير  مدخﻻت السلسلة النصية من علامات اقتباس مفردة ' إلى علامات اقتباس مزدوجة "). يمكن أن يساعد إضافة "tojson" في ضمان الاتساق هنا.

## كتابة مطالبات التوليد
لقد ذكرنا أعلاه أن add_generation_prompt هو متغير خاص يمكن الوصول إليه داخل قالبك، ويتحكم فيه المستخدم من خلال تعيين معامل add_generation_prompt. إذا كان نموذجك يتوقع عنوان لرسائل المساعد، فيجب أن يدعم قالبك إضافة العنوان عند تعيين add_generation_prompt.

فيما يلي مثال على قالب يُنسّق الرسائل بأسلوب ChatML، مع دعم مُطالبة التوليد:

```text
{{- bos_token }}
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```
سيعتمد المحتوى الدقيق لعنوان المساعد على نموذجك المُحدد، ولكن يجب أن يكون دائمًا السلسلة النصية التي تُمثل بداية رسالة المساعد، بحيث إذا قام المستخدم بتطبيق قالبك باستخدام add_generation_prompt=True ثم قام بتوليد نص، سيكتب النموذج استجابة المساعد. لاحظ أيضًا أن بعض النماذج لا تحتاج إلى مُطالبة توليد، لأن رسائل المساعد تبدأ دائمًا فورًا بعد رسائل المستخدم. هذا شائع بشكل خاص لنماذج LLaMA و Mistral، حيث تبدأ رسائل المساعد فورًا بعد رمز [/INST] الذي ينهي رسائل المستخدم. في هذه الحالات، يمكن للقالب تجاهل معامل add_generation_prompt.

مُطالبات التوليد مُهمة! إذا كان نموذجك يتطلب مُطالبة توليد ولكنها غير مُعيّنة في القالب، فمن المُحتمل أن تتدهور عمليات توليد النموذج بشدة، أو قد يُظهر النموذج سلوكًا غير عادي مثل متابعة رسالة المستخدم الأخيرة!

### كتابة قوالب أكبر وتصحيحها
عندما تم تقديم هذه الميزة، كانت معظم القوالب صغيرة جدًا، أي ما يُعادل نص برمجي "من سطر واحد" في Jinja. ومع ذلك، مع النماذج والميزات الجديدة مثل استخدام الأدوات و RAG، يمكن أن يصل طول بعض القوالب إلى 100 سطر أو أكثر. عند كتابة قوالب كهذه، من الجيد كتابتها في ملف مُنفصل، باستخدام مُحرر نصوص. يمكنك بسهولة استخراج قالب دردشة إلى ملف:

```python
open("template.jinja", "w").write(tokenizer.chat_template)
```
أو تحميل القالب المُحرر مرة أخرى إلى المعالج اللغوي:

```python
tokenizer.chat_template = open("template.jinja").read()
```
كميزة إضافية، عندما تكتب قالبًا طويلاً متعدد الأسطر في ملف مُنفصل، ستتوافق أرقام الأسطر في هذا الملف تمامًا مع أرقام الأسطر في أخطاء تحليل القالب أو تنفيذه. سيُسهّل هذا كثيرًا تحديد مكان المشكلات.

### كتابة قوالب للأدوات
على الرغم من أن قوالب الدردشة لا تفرض واجهة برمجة تطبيقات مُحددة للأدوات (أو لأي شيء حقًا)، فإننا نوصي مؤلفي القوالب بمحاولة الالتزام بواجهة برمجة تطبيقات قياسية حيثما أمكن. الهدف النهائي لقوالب الدردشة هو السماح بنقل التعليمات البرمجية عبر النماذج، لذا فإن الانحراف عن واجهة برمجة تطبيقات الأدوات القياسية يعني أن المستخدمين سيضطرون إلى كتابة تعليمات برمجية مخصصة لاستخدام الأدوات مع نموذجك. في بعض الأحيان يكون ذلك أمرًا لا مفر منه، ولكن غالبًا ما يكون من الممكن استخدام واجهة برمجة التطبيقات القياسية من خلال استخدام قوالب ذكية!

أدناه، سنُدرج عناصر واجهة برمجة التطبيقات القياسية، ونقدم نصائح حول كتابة قوالب ستعمل بشكل جيد معها.

#### تعريفات الأدوات
يجب أن يتوقع قالبك أن يكون المتغير tools إما فارغًا (إذا لم يتم تمرير أي أدوات)، أو قائمة من قواميس مخطط JSON. تسمح أساليب قالب الدردشة الخاصة بنا للمستخدمين بتمرير الأدوات إما كمخطط JSON أو كدوال Python، ولكن عندما يتم تمرير الدوال، فإننا نقوم تلقائيًا بإنشاء مخطط JSON وتمريره إلى قالبك. نتيجة لذلك، سيكون متغير tools الذي يستقبله قالبك دائمًا قائمة من مخططات JSON. هنا مخطط JSON أداة نموذجي:

```json
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "دالة تضرب عددين", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "الرقم الأول للضرب"
        }, 
        "b": {
          "type": "number", 
          "description": "الرقم الثاني للضرب"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

وهنا بعض الأمثلة البرمجية للتعامل مع الأدوات في قالب الدردشة الخاص بك. تذكر أن هذا مجرد مثال لتنسيق مُحدد - من المحتمل أن يحتاج نموذجك إلى تنسيق مختلف!
```text
{%- if tools %}
    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endif %}
{%- endif %}
```

يجب بالطبع اختيار الرموز المحددة ووصف الأدوات التي يُعرضها قالبك لتتناسب مع تلك التي تم تدريب نموذجك عليها. لا يوجد شرط أن يفهم نموذجك مُدخلات مخطط JSON، فقط أن يتمكن قالبك من ترجمة مخطط JSON إلى تنسيق نموذجك. على سبيل المثال، تم تدريب Command-R باستخدام أدوات مُعرّفة باستخدام رؤوس دوال Python، ولكن يقبل قالب أداة Command-R مخطط JSON، ويُحوّل الأنواع داخليًا ويُعرض أدوات الإدخال كعناوين Python. يمكنك فعل الكثير باستخدام القوالب!

#### استدعاءات الأدوات
استدعاءات الأدوات، إذا كانت موجودة، ستكون قائمة مُرفقة برسالة بدور "assistant". لاحظ أن tool_calls هي دائمًا قائمة، على الرغم من أن معظم نماذج استدعاء الأدوات تدعم فقط استدعاءات أدوات فردية في كل مرة، مما يعني أن القائمة ستحتوي عادةً على عنصر واحد فقط. هنا قاموس رسالة نموذجي يحتوي على استدعاء أداة:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "arguments": {
          "a": 5,
          "b": 6
        }
      }
    }
  ]
}
```
والنمط الشائع للتعامل معها سيكون كهذا:

```text
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
        {%- endif %}
    {%- endfor %}
{%- endif %}
```

مرة أخرى، يجب عليك عرض استدعاء الأداة بالتنسيق والرموز الخاصة التي يتوقعها نموذجك.

#### استجابات الأدوات
استجابات الأدوات لها تنسيق بسيط: إنها قاموس رسالة بدور "tool"، ومفتاح "name" يُعطي اسم الدالة المُستدعاة، ومفتاح "content" يحتوي على نتيجة استدعاء الأداة. هنا استجابة أداة نموذجية:

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```
لست بحاجة إلى استخدام جميع المفاتيح في استجابة الأداة. على سبيل المثال، إذا كان نموذجك لا يتوقع تضمين اسم الدالة في استجابة الأداة، فيمكن أن يكون عرضها بسيطًا مثل:

```text
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

مرة أخرى، تذكر أن التنسيق الفعلي والرموز الخاصة خاصة بالنموذج - يجب أن تُولي عناية كبيرة لضمان أن الرموز والمسافات الفارغة وكل شيء آخر يتطابق تمامًا مع التنسيق الذي تم تدريب نموذجك عليه!
