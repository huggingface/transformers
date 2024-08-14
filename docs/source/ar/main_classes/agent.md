# الوكلاء والأدوات

لمعرفة المزيد عن الوكلاء والأدوات، تأكد من قراءة [الدليل التمهيدي](../transformers_agents). تحتوي هذه الصفحة على وثائق API للفئات الأساسية.

## الوكلاء

نقدم نوعين من الوكلاء، بناءً على الفئة الأساسية [`Agent`]:

- يعمل [`CodeAgent`] في خطوة واحدة، من خلال توليد الكود لحل المهمة، ثم تنفيذه على الفور.

- يعمل [`ReactAgent`] خطوة بخطوة، وتتكون كل خطوة من فكرة واحدة، ثم استدعاء أداة واحدة وتنفيذها. ولديه فئتان:

- [`ReactJsonAgent`] تكتب استدعاءات أدواتها بتنسيق JSON.

- [`ReactCodeAgent`] تكتب استدعاءات أدواتها بلغة Python البرمجية.

### الوكيل

[[autodoc]] Agent

### CodeAgent

[[autodoc]] CodeAgent

### وكلاء التفاعل

[[autodoc]] ReactAgent

[[autodoc]] ReactJsonAgent

[[autodoc]] ReactCodeAgent

## الأدوات

### load_tool

[[autodoc]] load_tool

### أداة

[[autodoc]] Tool

### صندوق الأدوات

[[autodoc]] Toolbox

### PipelineTool

[[autodoc]] PipelineTool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

### ToolCollection

[[autodoc]] ToolCollection

## المحركات

يمكنك إنشاء واستخدام محركاتك الخاصة لتكون قابلة للاستخدام بواسطة إطار عمل الوكلاء.

تتبع هذه المحركات المواصفات التالية:

1. اتبع [تنسيق الرسائل](../chat_templating.md) لإدخالها (`List [Dict [str، str]]`) وأعد سلسلة.

2. توقف عن إنشاء الإخراج *قبل* التسلسلات التي تم تمريرها في وسيط `stop_sequences`

### HfEngine

للراحة، أضفنا `HfEngine` الذي ينفذ النقاط المذكورة أعلاه ويستخدم نقطة نهاية الاستدلال لتنفيذ LLM.

```python
>>> from transformers import HfEngine

>>> messages = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "No need to help, take it easy."},
... ]

>>> HfEngine()(messages, stop_sequences=["conversation"])

"That's very kind of you to say! It's always nice to have a relaxed "
```

[[autodoc]] HfEngine

## أنواع الوكلاء

يمكن للوكلاء التعامل مع أي نوع من الكائنات بين الأدوات؛ يمكن للأدوات، التي تكون متعددة الوسائط تمامًا، قبول وإرجاع النص والصورة والصوت والفيديو، من بين أنواع أخرى. لزيادة التوافق بين الأدوات، وكذلك لعرض هذه الإرجاعات بشكل صحيح في ipython (jupyter، colab، دفاتر ipython، ...)، نقوم بتنفيذ فئات wrapper حول هذه الأنواع.

يجب أن تستمر الكائنات الملفوفة في التصرف كما كانت في البداية؛ يجب أن يظل كائن النص يتصرف كسلسلة، ويجب أن يظل كائن الصورة يتصرف كـ `PIL.Image`.

لهذه الأنواع ثلاثة أغراض محددة:

- استدعاء `to_raw` على النوع يجب أن يعيد الكائن الأساسي

- استدعاء `to_string` على النوع يجب أن يعيد الكائن كسلسلة: يمكن أن تكون السلسلة في حالة `AgentText` ولكن ستكون مسار الإصدار المسلسل للكائن في حالات أخرى

- عرضه في نواة ipython يجب أن يعرض الكائن بشكل صحيح

### AgentText

[[autodoc]] transformers.agents.agent_types.AgentText

### AgentImage

[[autodoc]] transformers.agents.agent_types.AgentImage

### AgentAudio

[[autodoc]] transformers.agents.agent_types.AgentAudio