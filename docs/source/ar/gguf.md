# GGUF وتفاعلها مع المحولات

تُستخدم صيغة ملف GGUF لتخزين النماذج للاستدلال باستخدام [GGML](https://github.com/ggerganov/ggml) والمكتبات الأخرى التي تعتمد عليه، مثل [llama.cpp](https://github.com/ggerganov/llama.cpp) أو [whisper.cpp](https://github.com/ggerganov/whisper.cpp) الشهيرة جدًا.

إنها صيغة ملف [مدعومة من قبل Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf) مع ميزات تسمح بالفحص السريع للموترات والبيانات الوصفية داخل الملف.

تم تصميم تنسيق الملف هذا كـ "تنسيق ملف واحد" حيث يحتوي ملف واحد عادةً على كل من سمات التكوين ومفردات المجزىء اللغوي والخصائص الأخرى، بالإضافة إلى جميع الموترات التي سيتم تحميلها في النموذج. تأتي هذه الملفات بتنسيقات مختلفة وفقًا لنوع التكميم في الملف. نلقي نظرة موجزة على بعضها [هنا](https://huggingface.co/docs/hub/en/gguf#quantization-types).

## الدعم داخل المحولات

أضفنا القدرة على تحميل ملفات `gguf` داخل `المحولات` لتوفير قدرات تدريب/ضبط إضافية لنماذج gguf، قبل إعادة تحويل تلك النماذج إلى `gguf` لاستخدامها داخل نظام `ggml`. عند تحميل نموذج، نقوم أولاً بإلغاء تكميمه إلى fp32، قبل تحميل الأوزان لاستخدامها في PyTorch.

> [!NOTE]
> لا يزال الدعم تجريبيًا للغاية ونرحب بالمساهمات من أجل ترسيخه عبر أنواع التكميم وبنى النماذج.

فيما يلي، بنيات النماذج وأنواع التكميم المدعومة:

### أنواع التكميم المدعومة

تُحدد أنواع التكميم المدعومة مبدئيًا وفقًا لملفات التكميم الشائعة التي تمت مشاركتها على Hub.

- F32
- F16
- BF16
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q2_K
- Q3_K
- Q4_K
- Q5_K
- Q6_K
- IQ1_S
- IQ1_M
- IQ2_XXS
- IQ2_XS
- IQ2_S
- IQ3_XXS
- IQ3_S
- IQ4_XS
- IQ4_NL

> [!NOTE]
> لدعم إلغاء تكميم gguf، يلزم تثبيت `gguf>=0.10.0`.

### بنيات النماذج المدعومة

في الوقت الحالي، بنيات النماذج المدعومة هي البنيات التي كانت شائعة جدًا على Hub، وهي:

- LLaMa
- Mistral
- Qwen2
- Qwen2Moe
- Phi3
- Bloom
- Falcon
- StableLM
- GPT2
- Starcoder2
- T5

## مثال الاستخدام

لتحميل ملفات `gguf` في `transformers`، يجب تحديد معامل `gguf_file` فى دالة `from_pretrained` لكل من المُجزّئ اللغوية والنموذج. فيما يلي كيفية تحميل المُجزّئ اللغوي ونموذج، يمكن تحميلهما من نفس الملف:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
```

الآن لديك إمكانية الوصول إلى النسخة الكامل غير المكممة للنموذج في بيئة PyTorch، حيث يمكنك دمجه مع مجموعة كبيرة من الأدوات الأخرى.

لإعادة التحويل إلى ملف `gguf`، نوصي باستخدام ملف [`convert-hf-to-gguf.py`](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py) من llama.cpp.

فيما يلي كيفية إكمال البرنامج النصي أعلاه لحفظ النموذج وإعادة تصديره مرة أخرى إلى `gguf`:

```py
tokenizer.save_pretrained('directory')
model.save_pretrained('directory')

!python ${path_to_llama_cpp}/convert-hf-to-gguf.py ${directory}
```
