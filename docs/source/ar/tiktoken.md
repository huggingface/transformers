# Tiktoken والتفاعل مع Transformers

يتم دمج دعم ملفات نموذج tiktoken بسلاسة في 🤗 transformers عند تحميل النماذج
`from_pretrained` مع ملف `tokenizer.model` tiktoken على Hub، والذي يتم تحويله تلقائيًا إلى [المحلل اللغوي السريع](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

### النماذج المعروفة التي تم إصدارها مع `tiktoken.model`:
	- gpt2
	- llama3

## مثال على الاستخدام

من أجل تحميل ملفات `tiktoken` في `transformers`، تأكد من أن ملف `tokenizer.model` هو ملف tiktoken وسيتم تحميله تلقائيًا عند التحميل `from_pretrained`. إليك كيفية تحميل مجزىء لغوي ونموذج، والذي
يمكن تحميله من نفس الملف بالضبط:

```py
from transformers import AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="original")
```
## إنشاء مجزىء لغوي tiktoken

لا يحتوي ملف `tokenizer.model` على أي معلومات حول الرموز أو الأنماط الإضافية. إذا كانت هذه الأمور مهمة، قم بتحويل المحلل اللغوي إلى `tokenizer.json`، وهو التنسيق المناسب لـ [`PythonBackend`].

قم بتوليد ملف `tokenizer.model` باستخدام [tiktoken.get_encoding](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/tiktoken/registry.py#L63) ثم قم بتحويله إلى `tokenizer.json` باستخدام [`convert_tiktoken_to_fast`].

```py

from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding

# يمكنك تحميل ترميزك المخصص أو الترميز الذي توفره OpenAI
encoding = get_encoding("gpt2")
convert_tiktoken_to_fast(encoding, "config/save/dir")
```

يتم حفظ ملف `tokenizer.json` الناتج في الدليل المحدد ويمكن تحميله باستخدام [`PythonBackend`].

```py
tokenizer = PythonBackend.from_pretrained("config/save/dir")
```
