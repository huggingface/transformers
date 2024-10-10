# GGUF وتفاعلها مع المحولات

تُستخدم صيغة ملف GGUF لتخزين النماذج للاستنتاج مع [GGML](https://github.com/ggerganov/ggml) والمكتبات الأخرى التي تعتمد عليها، مثل [llama.cpp](https://github.com/ggerganov/llama.cpp) أو [whisper.cpp](https://github.com/ggerganov/whisper.cpp) الشهيرة جدًا.

إنها صيغة ملف [مدعومة من قبل Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf) مع ميزات تسمح بالتفتيش السريع للتوابع والبيانات الوصفية داخل الملف.

تم تصميم تنسيق الملف هذا كـ "تنسيق ملف واحد" حيث يحتوي ملف واحد عادةً على كل من سمات التكوين ومفردات المحلل اللغوي والسمات الأخرى، بالإضافة إلى جميع التوابع التي سيتم تحميلها في النموذج. تأتي هذه الملفات بتنسيقات مختلفة وفقًا لنوع التكميم في الملف. نلقي نظرة موجزة على بعضها [هنا](https://huggingface.co/docs/hub/en/gguf#quantization-types).

## الدعم داخل المحولات

أضفنا القدرة على تحميل ملفات `gguf` داخل `المحولات` لتوفير قدرات تدريب/ضبط إضافية لنماذج gguf، قبل إعادة تحويل تلك النماذج إلى `gguf` لاستخدامها داخل نظام بيئي `ggml`. عند تحميل نموذج، نقوم أولاً بإلغاء تكميمه إلى fp32، قبل تحميل الأوزان لاستخدامها في PyTorch.

> [!ملاحظة]
> لا يزال الدعم استكشافيًا للغاية ونرحب بالمساهمات لتثبيته عبر أنواع التكميم وبنيات النماذج.

فيما يلي، بنيات النماذج وأنواع التكميم المدعومة:

### أنواع التكميم المدعومة

تُحدد أنواع التكميم المدعومة مبدئيًا وفقًا لملفات التكميم الشائعة التي تمت مشاركتها على Hub.

- F32
- Q2_K
- Q3_K
- Q4_0
- Q4_K
- Q5_K
- Q6_K
- Q8_0

نأخذ مثالاً من المحلل الممتاز [99991/pygguf](https://github.com/99991/pygguf) Python لإلغاء تكميم الأوزان.

### بنيات النماذج المدعومة

في الوقت الحالي، بنيات النماذج المدعومة هي البنيات التي كانت شائعة جدًا على Hub، وهي:

- LLaMa
- ميسترال
- Qwen2

## مثال الاستخدام

لتحميل ملفات `gguf` في `المحولات`، يجب تحديد وسيط `gguf_file` لأساليب `from_pretrained` لكل من المحللات اللغوية والنماذج. فيما يلي كيفية تحميل محلل لغوي ونموذج، يمكن تحميلهما من نفس الملف بالضبط:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
```

الآن لديك حق الوصول إلى الإصدار الكامل غير المكمم للنموذج في نظام PyTorch البيئي، حيث يمكنك دمجه مع مجموعة من الأدوات الأخرى.

لإعادة التحويل إلى ملف `gguf`، نوصي باستخدام ملف [`convert-hf-to-gguf.py`](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py) من llama.cpp.

فيما يلي كيفية إكمال البرنامج النصي أعلاه لحفظ النموذج وتصديره مرة أخرى إلى `gguf`:

```py
tokenizer.save_pretrained('directory')
model.save_pretrained('directory')

!python ${path_to_llama_cpp}/convert-hf-to-gguf.py ${directory}
```