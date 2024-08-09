# الوكلاء والأدوات

[[open-in-colab]]

### ما هو الوكيل؟

يمكن للنظم اللغوية الكبيرة (LLMs) التي تم تدريبها على أداء [نمذجة اللغة السببية](./tasks/language_modeling.) التعامل مع مجموعة واسعة من المهام، ولكنها غالبًا ما تواجه صعوبات في المهام الأساسية مثل المنطق والحساب والبحث. وعندما يتم استدعاؤها في مجالات لا تؤدي فيها أداءً جيدًا، فإنها غالبًا ما تفشل في توليد الإجابة التي نتوقعها منها.

يتمثل أحد النهج للتغلب على هذا القصور في إنشاء "وكيل".

الوكيل هو نظام يستخدم LLM كمحرك له، ولديه حق الوصول إلى وظائف تسمى "أدوات".

هذه "الأدوات" هي وظائف لأداء مهمة، وتحتوي على جميع الأوصاف اللازمة للوكيل لاستخدامها بشكل صحيح.

يمكن برمجة الوكيل للقيام بما يلي:
- وضع سلسلة من الإجراءات/الأدوات وتشغيلها جميعًا في نفس الوقت مثل [`CodeAgent`] على سبيل المثال
- التخطيط للاجراءات/الأدوات وتنفيذها واحدة تلو الأخرى والانتظار حتى انتهاء كل إجراء قبل إطلاق التالي مثل [`ReactJsonAgent`] على سبيل المثال

### أنواع الوكلاء

#### وكيل الشفرة

يتمتع هذا الوكيل بخطوة تخطيط، ثم يقوم بتوليد شفرة Python لتنفيذ جميع إجراءاته في نفس الوقت. وهو يتعامل بشكل أصلي مع أنواع الإدخال والإخراج المختلفة لأدواته، وبالتالي فهو الخيار الموصى به للمهام متعددة الوسائط.

#### وكلاء التفاعل

هذا هو الوكيل الذي يتم اللجوء إليه لحل مهام الاستدلال، حيث يجعل إطار ReAct ([Yao et al.، 2022](https://huggingface.co/papers/2210.03629)) من الكفاءة حقًا التفكير على أساس ملاحظاته السابقة.

نقوم بتنفيذ إصدارين من ReactJsonAgent: 
- [`ReactJsonAgent`] يقوم بتوليد استدعاءات الأدوات كـ JSON في إخراجها.
- [`ReactCodeAgent`] هو نوع جديد من ReactJsonAgent يقوم بتوليد استدعاءات أدواته كمقاطع من التعليمات البرمجية، والتي تعمل بشكل جيد حقًا مع LLMs التي تتمتع بأداء ترميز قوي.

> [!TIP]
> اقرأ منشور المدونة [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) لمعرفة المزيد عن وكيل ReAct.

![إطار عمل وكيل ReAct](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)

على سبيل المثال، إليك كيف يعمل وكيل ReAct Code طريقه من خلال السؤال التالي.

```py3
>>> agent.run(
...     "How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?",
... )
=====New task=====
How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?
====Agent is executing the code below:
bert_blocks = search(query="number of blocks in BERT base encoder")
print("BERT blocks:", bert_blocks)
====
Print outputs:
BERT blocks: twelve encoder blocks

====Agent is executing the code below:
attention_layer = search(query="number of layers in Attention is All You Need")
print("Attention layers:", attention_layer)
====
Print outputs:
Attention layers: Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- 2 Page 3 Figure 1: The Transformer - model architecture.

====Agent is executing the code below:
bert_blocks = 12
attention_layers = 6
diff = bert_blocks - attention_layers
print("Difference in blocks:", diff)
final_answer(diff)
====

Print outputs:
Difference in blocks: 6

Final answer: 6
```

### كيف يمكنني بناء وكيل؟

لتهيئة وكيل، تحتاج إلى هذه الحجج:

- LLM لتشغيل وكيلك - الوكيل ليس هو LLM بالضبط، إنه أكثر مثل برنامج الوكيل الذي يستخدم LLM كمحرك له.
- موجه النظام: ما الذي سيتم استدعاء محرك LLM به لتوليد الإخراج الخاص به
- صندوق أدوات يختار الوكيل منه الأدوات لتنفيذها
- محلل لاستخراج الأدوات التي يجب استدعاؤها من إخراج LLM والحجج التي يجب استخدامها

عند تهيئة نظام الوكيل، يتم استخدام سمات الأداة لتوليد وصف للأداة، ثم يتم تضمينها في موجه `system_prompt` للوكيل لإعلامه بالأدوات التي يمكنه استخدامها ولماذا.

للبدء، يرجى تثبيت `agents` الإضافية لتثبيت جميع التبعيات الافتراضية.

```bash
pip install transformers[agents]
```

قم ببناء محرك LLM الخاص بك من خلال تعريف طريقة `llm_engine` التي تقبل قائمة من [الرسائل](./chat_templating.) وتعيد النص. يجب أن تقبل هذه الدالة القابلة للاستدعاء أيضًا وسيط `stop` يشير إلى متى يجب التوقف عن التوليد.

```python
from huggingface_hub import login, InferenceClient

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct")

def llm_engine(messages, stop_sequences=["Task"]) -> str:
    response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
    answer = response.choices[0].message.content
    return answer
```

يمكنك استخدام أي طريقة `llm_engine` طالما أنها:
1. يتبع تنسيق [رسائل](./chat_templating.md) لإدخاله (`List [Dict [str، str]]`) ويعيد `str`
2. يتوقف عن توليد الإخراج عند التسلسلات التي تم تمريرها في وسيط `stop`

أنت بحاجة أيضًا إلى وسيط "الأدوات" الذي يقبل قائمة من "الأدوات". يمكنك توفير قائمة فارغة لـ "الأدوات"، ولكن استخدم صندوق الأدوات الافتراضي مع وسيط اختياري `add_base_tools=True`.

الآن يمكنك إنشاء وكيل، مثل [`CodeAgent`], وتشغيله. وللراحة، نقدم أيضًا فئة [`HfEngine`] التي تستخدم `huggingface_hub.InferenceClient` تحت الغطاء.

```python
from transformers import CodeAgent, HfEngine

llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)

agent.run(
    "Could you translate this sentence from French, say it out loud and return the audio.",
    sentence="Où est la boulangerie la plus proche?",
)
```

سيكون هذا مفيدًا في حالة الطوارئ عند الحاجة إلى الخبز الفرنسي! يمكنك حتى ترك وسيط `llm_engine` غير محدد، وسيتم إنشاء [`HfEngine`] افتراضيًا.

```python
from transformers import CodeAgent

agent = CodeAgent(tools=[], add_base_tools=True)

agent.run(
    "Could you translate this sentence from French, say it out loud and give me the audio.",
    sentence="Où est la boulangerie la plus proche?",
)
```

لاحظ أننا استخدمنا وسيط "جملة" إضافي: يمكنك تمرير النص كوسيط إضافي إلى النموذج.

يمكنك أيضًا استخدام هذا للإشارة إلى مسار الملفات المحلية أو البعيدة للنموذج لاستخدامها:

```py
from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)

agent.run("Why does Mike not know many people in New York?", audio="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3")
```


تم تحديد موجه النظام ومحلل الإخراج تلقائيًا، ولكن يمكنك فحصهما بسهولة عن طريق استدعاء `system_prompt_template` على وكيلك.

```python
print(agent.system_prompt_template)
```

من المهم أن تشرح بأكبر قدر ممكن من الوضوح المهمة التي تريد تنفيذها.
كل عملية [`~Agent.run`] مستقلة، وبما أن الوكيل مدعوم من LLM، فقد تؤدي الاختلافات الطفيفة في موجهك إلى نتائج مختلفة تمامًا.
يمكنك أيضًا تشغيل وكيل بشكل متتالي لمهام مختلفة: في كل مرة يتم فيها إعادة تهيئة سمات `agent.task` و`agent.logs`.


#### تنفيذ التعليمات البرمجية

يقوم مفسر Python بتنفيذ التعليمات البرمجية على مجموعة من الإدخالات التي يتم تمريرها جنبًا إلى جنب مع أدواتك.
يجب أن يكون هذا آمنًا لأن الوظائف الوحيدة التي يمكن استدعاؤها هي الأدوات التي قدمتها (خاصة إذا كانت أدوات من Hugging Face فقط) ووظيفة الطباعة، لذا فأنت مقيد بالفعل بما يمكن تنفيذه.

مفسر Python لا يسمح أيضًا بالاستيراد بشكل افتراضي خارج قائمة آمنة، لذا فإن جميع الهجمات الأكثر وضوحًا لا ينبغي أن تكون مشكلة.
يمكنك أيضًا الإذن باستيرادات إضافية عن طريق تمرير الوحدات النمطية المصرح بها كقائمة من السلاسل في وسيط `additional_authorized_imports` عند تهيئة [`ReactCodeAgent`] أو [`CodeAgent`]:

```py
>>> from transformers import ReactCodeAgent

>>> agent = ReactCodeAgent(tools=[], additional_authorized_imports=['requests', 'bs4'])
>>> agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")

(...)
'Hugging Face – Blog'
```

سيتم إيقاف التنفيذ عند أي رمز يحاول تنفيذ عملية غير قانونية أو إذا كان هناك خطأ Python عادي في التعليمات البرمجية التي تم إنشاؤها بواسطة الوكيل.

> [!WARNING]
> يمكن لـ LLM توليد رمز تعسفي سيتم تنفيذه بعد ذلك: لا تقم بإضافة أي استيرادات غير آمنة!

### موجه النظام

ينشئ الوكيل، أو بالأحرى LLM الذي يقود الوكيل، إخراجًا بناءً على موجه النظام. يمكن تخصيص موجه النظام وتصميمه للمهام المقصودة. على سبيل المثال، تحقق من موجه النظام لـ [`ReactCodeAgent`] (الإصدار أدناه مبسط قليلاً).

```text
You will be given a task to solve as best you can.
You have access to the following tools:
<<tool_descriptions>>

To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task, then the tools that you want to use.
Then in the 'Code:' sequence, you shold write the code in simple Python. The code sequence must end with '/End code' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then be available in the 'Observation:' field, for using this information as input for the next step.

In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
{examples}

Above example were using notional tools that might not exist for you. You only have acces to those tools:
<<tool_names>>
You also can perform computations in the python code you generate.

Always provide a 'Thought:' and a 'Code:\n```py' sequence ending with '```<end_code>' sequence. You MUST provide at least the 'Code:' sequence to move forward.

Remember to not perform too many operations in a single code block! You should split the task into intermediate code blocks.
Print results at the end of each step to save the intermediate results. Then use final_answer() to return the final result.

Remember to make sure that variables you use are all defined.

Now Begin!
```

يتضمن موجه النظام:
- *مقدمة* تشرح كيف يجب أن يتصرف الوكيل والأدوات التي يجب عليه استخدامها.
- وصف لجميع الأدوات التي يتم تحديدها بواسطة رمز `<<tool_descriptions>>` الذي يتم استبداله ديناميكيًا في وقت التشغيل بالأدوات التي يحددها المستخدم أو يختارها.
    - يأتي وصف الأداة من سمات الأداة، `name`، و`description`، و`inputs` و`output_type`، وقالب `jinja2` بسيط يمكنك تحسينه.
- تنسيق الإخراج المتوقع.

يمكنك تحسين موجه النظام، على سبيل المثال، عن طريق إضافة شرح لتنسيق الإخراج.

للحصول على أقصى قدر من المرونة، يمكنك الكتابة فوق قالب موجه النظام بالكامل عن طريق تمرير موجه مخصص كوسيط إلى معلمة `system_prompt`.

```python
from transformers import ReactJsonAgent
from transformers.agents import PythonInterpreterTool

agent = ReactJsonAgent(tools=[PythonInterpreterTool()], system_prompt="{your_custom_prompt}")
```

> [!WARNING]
> يرجى التأكد من تحديد سلسلة `<<tool_descriptions>>` في مكان ما في `template` حتى يكون الوكيل على علم 
بالأدوات المتاحة.


### فحص تشغيل الوكيل

فيما يلي بعض السمات المفيدة لفحص ما حدث بعد التشغيل:
- `agent.logs` تخزين سجلات مفصلة للوكيل. في كل خطوة من خطوات تشغيل الوكيل، يتم تخزين كل شيء في قاموس يتم إلحاقه بعد ذلك بـ `agent.logs`.
- يقوم تشغيل `agent.write_inner_memory_from_logs()` بإنشاء ذاكرة داخلية لسجلات الوكيل للنظام LLM لعرضها، كقائمة من رسائل الدردشة. تنتقل هذه الطريقة عبر كل خطوة من سجل الوكيل ولا تخزن سوى ما يهمها كرسالة: على سبيل المثال، سيحفظ موجه النظام والمهمة في رسائل منفصلة، ثم لكل خطوة سيخزن إخراج LLM كرسالة، وإخراج استدعاء الأداة كرسالة أخرى. استخدم هذا إذا كنت تريد عرضًا عامًا لما حدث - ولكن لن يتم نسخ كل سجل بواسطة هذه الطريقة.

## الأدوات

الأداة هي وظيفة ذرية لاستخدامها بواسطة وكيل.

يمكنك على سبيل المثال التحقق من [`PythonInterpreterTool`]: لديه اسم ووصف ووصف الإدخال ونوع الإخراج، وطريقة `__call__` لأداء الإجراء.

عند تهيئة الوكيل، يتم استخدام سمات الأداة لتوليد وصف للأداة يتم تضمينه في موجه النظام الخاص بالوكيل. يتيح هذا للوكيل معرفة الأدوات التي يمكنه استخدامها ولماذا.

### صندوق الأدوات الافتراضي

يأتي Transformers مع صندوق أدوات افتراضي لتمكين الوكلاء، والذي يمكنك إضافته إلى وكيلك عند التهيئة باستخدام وسيط `add_base_tools = True`:

- **الإجابة على أسئلة المستند**: الإجابة على سؤال حول المستند (مثل ملف PDF) بتنسيق صورة ([Donut](./model_doc/donut))
- **الإجابة على أسئلة الصور**: الإجابة على سؤال حول صورة ([VILT](./model_doc/vilt))
- **التحدث إلى النص**: قم بتفريغ الكلام إلى نص ([Whisper](./model_doc/whisper))
- **النص إلى كلام**: تحويل النص إلى كلام ([SpeechT5](./model_doc/speecht5))
- **الترجمة**: ترجمة جملة معينة من لغة المصدر إلى لغة الهدف.
- **مفسر كود Python**: تشغيل كود Python الذي تم إنشاؤه بواسطة LLM في بيئة آمنة. لن يتم إضافة هذه الأداة إلى [`ReactJsonAgent`] إلا إذا استخدمت `add_base_tools=True`، نظرًا لأن الأدوات المستندة إلى التعليمات البرمجية يمكنها بالفعل تنفيذ كود Python
لا تترجم النصوص الخاصة ولا الأكواد البرمجية ولا الروابط ولا رموز HTML وCSS:

يمكنك استخدام أداة يدويًا عن طريق استدعاء دالة [`load_tool`] وتحديد مهمة لتنفيذها.

```python
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

### إنشاء أداة جديدة

يمكنك إنشاء أداتك الخاصة لحالات الاستخدام التي لا تغطيها الأدوات الافتراضية من Hugging Face.
على سبيل المثال، دعنا نقوم بإنشاء أداة تعيد النموذج الأكثر تنزيلًا لمهمة معينة من Hub.

ابدأ بالشيفرة أدناه.

```python
from huggingface_hub import list_models

task = "text-classification"

model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(model.id)
```

يمكن تحويل هذه الشيفرة إلى فئة ترث من الفئة العليا [`Tool`].

تحتاج الأداة المخصصة إلى:

- خاصية `name`، والتي تمثل اسم الأداة نفسها. عادةً ما يصف الاسم ما تقوم به الأداة. بما أن الشيفرة تعيد النموذج الأكثر تنزيلًا لمهمة ما، دعنا نطلق عليها اسم `model_download_counter`.
- تستخدم خاصية `description` لملء موجه نظام الوكيل.
- خاصية `inputs`، والتي هي عبارة عن قاموس بمفاتيح "type" و"description". يحتوي على معلومات تساعد المفسر Python على اتخاذ خيارات مستنيرة بشأن الإدخال.
- خاصية `output_type`، والتي تحدد نوع الإخراج.
- طريقة `forward` والتي تحتوي على شيفرة الاستدلال التي سيتم تنفيذها.

```python
from transformers import Tool
from huggingface_hub import list_models

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = (
        "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. "
        "It returns the name of the checkpoint."
    )

    inputs = {
        "task": {
            "type": "text",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "text"

    def forward(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

الآن بعد أن أصبحت فئة `HfModelDownloadsTool` المخصصة جاهزة، يمكنك حفظها في ملف باسم `model_downloads.py` واستيرادها للاستخدام.

```python
from model_downloads import HFModelDownloadsTool

tool = HFModelDownloadsTool()
```

يمكنك أيضًا مشاركة أداتك المخصصة في Hub عن طريق استدعاء [`~Tool.push_to_hub`] على الأداة. تأكد من أنك قمت بإنشاء مستودع لها على Hub وأنك تستخدم رمز وصول للقراءة.

```python
tool.push_to_hub("{your_username}/hf-model-downloads")
```

قم بتحميل الأداة باستخدام دالة [`~Tool.load_tool`] ومررها إلى معلمة `tools` في الوكيل الخاص بك.

```python
from transformers import load_tool, CodeAgent

model_download_tool = load_tool("m-ric/hf-model-downloads")
agent = CodeAgent(tools=[model_download_tool], llm_engine=llm_engine)
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```

ستحصل على ما يلي:

```text
======== New task ========
Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?
==== Agent is executing the code below:
most_downloaded_model = model_download_counter(task="text-to-video")
print(f"The most downloaded model for the 'text-to-video' task is {most_downloaded_model}.")
====
```

والناتج:

`"النموذج الأكثر تنزيلًا لمهمة `text-to-video` هو ByteDance/AnimateDiff-Lightning."`

### إدارة صندوق أدوات الوكيل الخاص بك

إذا كنت قد قمت بالفعل بتهيئة وكيل، فمن غير الملائم إعادة تهيئته من الصفر باستخدام أداة تريد استخدامها. باستخدام مكتبة Transformers، يمكنك إدارة صندوق أدوات الوكيل عن طريق إضافة أداة أو استبدالها.

دعنا نضيف `model_download_tool` إلى وكيل موجود تمت تهيئته بصندوق الأدوات الافتراضي فقط.

```python
from transformers import CodeAgent

agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)
agent.toolbox.add_tool(model_download_tool)
```

الآن يمكننا الاستفادة من الأداة الجديدة وأداة تحويل النص إلى كلام السابقة:

```python
    agent.run(
        "Can you read out loud the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub and return the audio?"
    )
```

| **Audio**                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
| <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/damo.wav" type="audio/wav"/> |

> [!WARNING]
> احترس عند إضافة أدوات إلى وكيل يعمل بالفعل لأنه يمكن أن يؤثر على اختيار الأداة لصالح أداتك أو اختيار أداة أخرى غير المحددة بالفعل.

استخدم طريقة `agent.toolbox.update_tool()` لاستبدال أداة موجودة في صندوق أدوات الوكيل.
هذا مفيد إذا كانت أداتك الجديدة بديلاً مباشرًا للأداة الموجودة لأن الوكيل يعرف بالفعل كيفية تنفيذ تلك المهمة المحددة.
تأكد فقط من اتباع الأداة الجديدة لنفس واجهة برمجة التطبيقات مثل الأداة المستبدلة أو قم بتكييف قالب موجه النظام لضمان تحديث جميع الأمثلة التي تستخدم الأداة المستبدلة.

### استخدام مجموعة من الأدوات

يمكنك الاستفادة من مجموعات الأدوات باستخدام كائن ToolCollection، مع علامة مجموعة الأدوات التي تريد استخدامها.
ثم قم بتمريرها كقائمة لتهيئة الوكيل الخاص بك، وبدء استخدامها!

```py
from transformers import ToolCollection, ReactCodeAgent

image_tool_collection = ToolCollection(collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
agent = ReactCodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

agent.run("Please draw me a picture of rivers and lakes.")
```

لتسريع البداية، يتم تحميل الأدوات فقط إذا استدعاها الوكيل.

ستحصل على هذه الصورة:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" />

### استخدام gradio-tools

[gradio-tools](https://github.com/freddyaboulton/gradio-tools) هي مكتبة قوية تتيح استخدام Hugging
Face Spaces كأدوات. تدعم العديد من المساحات الموجودة بالإضافة إلى مساحات مخصصة.

تدعم مكتبة Transformers `gradio_tools` باستخدام طريقة [`Tool.from_gradio`] في الفئة. على سبيل المثال، دعنا نستخدم [`StableDiffusionPromptGeneratorTool`](https://github.com/freddyaboulton/gradio-tools/blob/main/gradio_tools/tools/prompt_generator.py) من مجموعة أدوات `gradio-tools` لتحسين المطالبات لإنشاء صور أفضل.

استورد وقم بتهيئة الأداة، ثم مررها إلى طريقة `Tool.from_gradio`:

```python
from gradio_tools import StableDiffusionPromptGeneratorTool
from transformers import Tool, load_tool, CodeAgent

gradio_prompt_generator_tool = StableDiffusionPromptGeneratorTool()
prompt_generator_tool = Tool.from_gradio(gradio_prompt_generator_tool)
```

الآن يمكنك استخدامه مثل أي أداة أخرى. على سبيل المثال، دعنا نحسن المطالبة `a rabbit wearing a space suit`.

```python
image_generation_tool = load_tool('huggingface-tools/text-to-image')
agent = CodeAgent(tools=[prompt_generator_tool, image_generation_tool], llm_engine=llm_engine)

agent.run(
    "Improve this prompt, then generate an image of it.", prompt='A rabbit wearing a space suit'
)
```

يستفيد النموذج بشكل كافٍ من الأداة:

```text
======== New task ========
Improve this prompt, then generate an image of it.
You have been provided with these initial arguments: {'prompt': 'A rabbit wearing a space suit'}.
==== Agent is executing the code below:
improved_prompt = StableDiffusionPromptGenerator(query=prompt)
while improved_prompt == "QUEUE_FULL":
    improved_prompt = StableDiffusionPromptGenerator(query=prompt)
print(f"The improved prompt is {improved_prompt}.")
image = image_generator(prompt=improved_prompt)
====
```

قبل إنشاء الصورة أخيرًا:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png" />

> [!WARNING]
> تتطلب gradio-tools إدخالات وإخراجات *نصية* حتى عند العمل مع طرائق مختلفة مثل كائنات الصور والصوت. الإدخالات والإخراجات الصورية والصوتية غير متوافقة حاليًا.

### استخدام أدوات LangChain

نحن نحب Langchain ونعتقد أنها تحتوي على مجموعة أدوات قوية للغاية.
لاستيراد أداة من LangChain، استخدم الطريقة `from_langchain()`.

فيما يلي كيفية استخدامها لإعادة إنشاء نتيجة البحث في المقدمة باستخدام أداة بحث الويب LangChain.

```python
from langchain.agents import load_tools
from transformers import Tool, ReactCodeAgent

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = ReactCodeAgent(tools=[search_tool])

agent.run("How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?")
```

## واجهة Gradio

يمكنك الاستفادة من `gradio.Chatbot` لعرض أفكار الوكيل الخاص بك باستخدام `stream_to_gradio`، إليك مثال:

```py
import gradio as gr
from transformers import (
    load_tool,
    ReactCodeAgent,
    HfEngine,
    stream_to_gradio,
)

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image")

llm_engine = HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")

# Initialize the agent with the image generation tool
agent = ReactCodeAgent(tools=[image_generation_tool], llm_engine=llm_engine)


def interact_with_agent(task):
    messages = []
    messages.append(gr.ChatMessage(role="user", content=task))
    yield messages
    for msg in stream_to_gradio(agent, task):
        messages.append(msg)
        yield messages + [
            gr.ChatMessage(role="assistant", content="⏳ Task not finished yet!")
        ]
    yield messages


with gr.Blocks() as demo:
    text_input = gr.Textbox(lines=1, label="Chat Message", value="Make me a picture of the Statue of Liberty.")
    submit = gr.Button("Run illustrator agent!")
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
    )
    submit.click(interact_with_agent, [text_input], [chatbot])

if __name__ == "__main__":
    demo.launch()
```