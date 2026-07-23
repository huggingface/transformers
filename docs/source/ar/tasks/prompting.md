<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# هندسة المحفّزات (Prompt engineering)

[[open-in-colab]]

تعني هندسة المحفّزات أو "التلقين" (prompting) استخدام اللغة الطبيعية لتحسين أداء نماذج اللغة الكبيرة (LLMs) في مجموعة واسعة من المهام. يمكن للمحفّز توجيه النموذج نحو توليد المخرجات المطلوبة. في كثير من الحالات، لا تحتاج حتى إلى نموذج [مُدرَّب خصيصًا](#finetuning) للمهمة؛ كل ما تحتاجه هو محفّز جيد.

جرِّب تلقين نموذج لغة كبير لتصنيف نص. عند إنشاء محفّز، من المهم تقديم تعليمات محددة جدًا بشأن المهمة وكيف يجب أن تبدو النتيجة.

```py
from transformers import pipeline
import torch

pipeline = pipeline(task="text-generation", model="mistralai/Mistal-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Classify the text into neutral, negative or positive.
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
"""

outputs = pipeline(prompt, max_new_tokens=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

يكمن التحدّي في تصميم محفّزات تنتج النتائج المتوقعة نظرًا للتعقيد الهائل والدقّة العالية للّغة الطبيعية.

يغطي هذا الدليل أفضل الممارسات والتقنيات والأمثلة حول كيفية حل مهام اللغة والاستدلال.

## أفضل الممارسات

1. حاول اختيار أحدث النماذج لأفضل أداء. ضع في اعتبارك أن نماذج LLM تأتي بنسختين: [أساسية](https://hf.co/mistralai/Mistral-7B-v0.1) و[مضبوطة وفق التعليمات](https://hf.co/mistralai/Mistral-7B-Instruct-v0.1) (أو "مخصّصة للمحادثة").

    النماذج الأساسية ممتازة في إتمام النص عند تزويدها بمحفّز أولي، لكنها ليست جيدة في اتباع التعليمات. أمّا النماذج المضبوطة وفق التعليمات فهي نسخ مُدرَّبة من النماذج الأساسية على بيانات تعليمية أو محادثية، ما يجعلها أنسب للتلقين.

    > [!WARNING]
    > عادةً ما تكون نماذج LLM الحديثة من نوع "مُفكِّك فقط" (decoder-only)، ولكن هناك بعض النماذج "مُرمِّز-مُفكِّك" (encoder-decoder) مثل [Flan-T5](../model_doc/flan-t5) أو [BART](../model_doc/bart) التي يمكن استخدامها للتلقين. بالنسبة لنماذج المرمّز-المفكّك، تأكد من ضبط معرّف مهمة البايبلاين إلى `text2text-generation` بدلًا من `text-generation`.

2. ابدأ بمحفّز قصير وبسيط، وكرّر تحسينه للحصول على نتائج أفضل.

3. ضع التعليمات في بداية المحفّز أو نهايته. بالنسبة للمحفّزات الطويلة، قد تطبّق النماذج تحسينات لمنع زيادة التعقيد التربيعي للاهتمام (attention)، ما يضع تركيزًا أكبر على البداية والنهاية.

4. افصل بوضوح بين التعليمات ونص الاهتمام (النص المراد معالجته).

5. كن محددًا ووصفيًا بشأن المهمة والمخرجات المطلوبة، بما في ذلك الشكل والطول والنمط واللغة. تجنّب التعليمات والعبارات الغامضة.

6. ركّز التعليمات على "ما يجب فعله" بدلًا من "ما لا يجب فعله".

7. وجّه النموذج نحو المخرجات الصحيحة بكتابة الكلمة الأولى أو حتى الجملة الأولى.

8. استخدم [التلقين بعدة أمثلة (few-shot)](#few-shot) عند الحاجة لمنح النموذج سياقًا حول كيفية أداء المهمة.

### مثال: استخراج تواريخ

```python
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon.
Date:"""

outputs = pipeline(prompt, max_new_tokens=12, do_sample=True, top_k=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
```

بنية المحادثة: نظّم محفّزك كمحادثة تبادلية واستخدم الطريقة [`apply_chat_template`] لتنسيقه وترميزه.

```python
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {"role": "user", "content": "Text: The first human went into space and orbited the Earth on April 12, 1961."},
    {"role": "assistant", "content": "Date: 04/12/1961"},
    {"role": "user", "content": "Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon."},
    {"role": "assistant", "content": "Date:"},
]

outputs = pipeline(messages, max_new_tokens=6, do_sample=True, top_k=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
```

استشر دائمًا توثيق النموذج المضبوط بالتعليمات لمعرفة المزيد عن تنسيق قالب المحادثة بما يسمح لك ببناء تلقين few-shot وفقًا لذلك.

### سلسلة التفكير (Chain-of-thought)

تُعد سلسلة التفكير (CoT) فعالة في توليد مخرجات أكثر تماسكًا وحسن استدلال عبر توفير سلسلة من المحفّزات التي تساعد النموذج على "التفكير" بعمق حول موضوع ما.

```text
You and your partner are making muffins.
1. You need 6 eggs and you already have 2. How many more eggs do you need?
2. You buy a carton of a dozen eggs. How many eggs do you have now?
3. You need 6 cups of flour and you have 2.5 cups. How many more cups do you need?
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
If you eat 6 muffins, how many are left?
Answer: 6
```

مثل [few-shot](#few-shot)، سلبيات CoT تتمثل في الحاجة لجهد أكبر لتصميم سلسلة من المحفّزات التي تساعد النموذج على الاستدلال في مهمة معقدة، كما أن طول المحفّز يزيد زمن الاستجابة.

## الضبط الخاص (Fine-tuning)

إذا لم تنجح هندسة المحفّزات في الحصول على النتائج المطلوبة، فكّر في الضبط الخاص للنموذج في الحالات التالية:

- إذا كانت لديك قيود مجال محدّد يتطلّب معرفة متخصّصة.
- ترغب في سلوك أكثر ثباتًا أو قابلية للتنبؤ.
- تحتاج مخرجات متسقة أو صيغة محددة للغاية.
- تستخدم نموذجًا صغيرًا بسبب التكلفة أو الخصوصية أو البنية التحتية أو غيرها.

في جميع هذه السيناريوهات، تأكد من امتلاك مجموعة بيانات كافية خاصة بالمجال لتدريب نموذجك، وتأكد من توفّر الوقت والموارد، وأن تكلفة الضبط الخاص مبرَّرة. وإلا فقد يكون من الأفضل تحسين المحفّز.

## أمثلة

تُظهر الأمثلة التالية كيفية تلقين نموذج لغة كبير لمهام مختلفة.

<hfoptions id="tasks">
<hfoption id="named entity recognition">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Return a list of named entities in the text.
Text: The company was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf in New York City, originally as a company that developed a chatbot app targeted at teenagers.
Named entities:
"""

outputs = pipeline(prompt, max_new_tokens=50, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result:  [Clément Delangue, Julien Chaumond, Thomas Wolf, company, New York City, chatbot app, teenagers]
```

</hfoption>
<hfoption id="translation">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Translate the English text to French.
Text: Sometimes, I've believed as many as six impossible things before breakfast.
Translation:
"""

outputs = pipeline(prompt, max_new_tokens=20, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: À l'occasion, j'ai croyu plus de six choses impossibles
```

</hfoption>
<hfoption id="summarization">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
Write a summary of the above text.
Summary:
"""

outputs = pipeline(prompt, max_new_tokens=30, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Permaculture is the design process that involves mimicking natural ecosystems to provide sustainable solutions to basic needs. It is a holistic approach that comb
```

</hfoption>
<hfoption id="question answering">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Answer the question using the context below.
Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
Question: What modern tool is used to make gazpacho?
Answer:
"""

outputs = pipeline(prompt, max_new_tokens=10, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: A blender or food processor is the modern tool
```

</hfoption>
</hfoptions>
