# GPTQ

<Tip>

جرب تكميم GPTQ مع PEFT في هذا [notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) وتعرف على المزيد حول تفاصيله في هذه [التدوينة](https://huggingface.co/blog/gptq-integration)!

</Tip>

تنفذ مكتبة [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) خوارزمية GPTQ، وهي تقنية تكميم بعد التدريب حيث يتم تكميم كل صف في مصفوفة الأوزان بشكل مستقل لإيجاد نسخة من الأوزان التي تقلل من الخطأ. يتم تكميم هذه الأوزان إلى int4، ولكن يتم استعادتها إلى fp16 أثناء الاستدلال. يمكن أن يوفر هذا استخدام الذاكرة الخاصة بك بمقدار 4x لأن أوزان int4 يتم إلغاء تكميمها في نواة مدمجة بدلاً من الذاكرة العالمية لوحدة معالجة الرسومات (GPU)، ويمكنك أيضًا توقع تسريع في الاستدلال لأن استخدام عرض نطاق ترددي أقل يستغرق وقتًا أقل في التواصل.

قبل البدء، تأكد من تثبيت المكتبات التالية:

```bash
pip install auto-gptq
pip install --upgrade accelerate optimum transformers
```

ولتكميم نموذج (مدعوم حاليًا للنصوص فقط)، يلزمك إنشاء فئة [`GPTQConfig`] وتعيين عدد البتات التي سيتم تكميمها، ومجموعة بيانات لمعايرة الأوزان من أجل التكميم، ومحلل رموز لإعداد مجموعة البيانات.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```

يمكنك أيضًا تمرير مجموعة البيانات الخاصة بك كقائمة من السلاسل النصية، ولكن يوصى بشدة باستخدام نفس مجموعة البيانات من ورقة GPTQ.

```py
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
```

قم بتحميل نموذج لتكميمه ومرر `gptq_config` إلى طريقة [`~AutoModelForCausalLM.from_pretrained`]. قم بتعيين `device_map="auto"` لنقل النموذج تلقائيًا إلى وحدة المعالجة المركزية (CPU) للمساعدة في تثبيت النموذج في الذاكرة، والسماح بنقل وحدات النموذج بين وحدة المعالجة المركزية ووحدة معالجة الرسومات (GPU) للتكميم.

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
```

إذا كنت تعاني من نفاد الذاكرة لأن مجموعة البيانات كبيرة جدًا، فإن النقل إلى القرص غير مدعوم. إذا كان الأمر كذلك، فحاول تمرير معلمة `max_memory` لتحديد مقدار الذاكرة التي سيتم استخدامها على جهازك (GPU وCPU):

```py
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"}, quantization_config=gptq_config)
```

<Tip warning={true}>

اعتمادًا على عتادك، قد يستغرق تكميم نموذج من الصفر بعض الوقت. قد يستغرق تكميم نموذج [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) حوالي 5 دقائق على وحدة معالجة الرسومات (GPU) من Google Colab من الفئة المجانية، ولكنه سيستغرق حوالي 4 ساعات لتكميم نموذج بمعلمات 175B على NVIDIA A100. قبل تكميم نموذج، من الجيد التحقق من Hub لمعرفة ما إذا كان هناك بالفعل إصدار مُكمم من GPTQ للنموذج.

</Tip>
</Tip>

بمجرد تكميم نموذجك، يمكنك دفعه إلى Hub مع محلل الرموز حيث يمكن مشاركته والوصول إليه بسهولة. استخدم طريقة [`~PreTrainedModel.push_to_hub`] لحفظ [`GPTQConfig`]:

```py
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

يمكنك أيضًا حفظ نموذجك المُكمم محليًا باستخدام طريقة [`~PreTrainedModel.save_pretrained`]. إذا تم تكميم النموذج باستخدام معلمة `device_map`، فتأكد من نقل النموذج بالكامل إلى وحدة معالجة الرسومات (GPU) أو وحدة المعالجة المركزية (CPU) قبل حفظه. على سبيل المثال، لحفظ النموذج على وحدة المعالجة المركزية (CPU):

```py
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")

# إذا تم التكميم باستخدام device_map
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

قم بتحميل نموذج مُكمم باستخدام طريقة [`~PreTrainedModel.from_pretrained`]. قم بتعيين `device_map="auto"` لتوزيع النموذج تلقائيًا على جميع وحدات معالجة الرسومات (GPU) المتوفرة لتحميل النموذج بشكل أسرع دون استخدام ذاكرة أكثر من اللازم.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

## ExLlama

[ExLlama](https://github.com/turboderp/exllama) هو تنفيذ Python/C++/CUDA لنموذج [Llama](model_doc/llama) مصمم للاستدلال بشكل أسرع مع أوزان GPTQ ذات 4 بتات (تحقق من هذه [الاختبارات المعيارية](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark)). يتم تنشيط نواة ExLlama بشكل افتراضي عند إنشاء كائن [`GPTQConfig`]. لزيادة تسريع الاستدلال، استخدم نواة [ExLlamaV2](https://github.com/turboderp/exllamav2) عن طريق تكوين معلمة `exllama_config`:

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config=gptq_config)
```

<Tip warning={true}>

يتم دعم النماذج ذات 4 بتات فقط، ونوصي بتعطيل نواة ExLlama إذا كنت تقوم بتعديل نموذج مُكمم باستخدام PEFT.

</Tip>

تدعم نواة ExLlama فقط عندما يكون النموذج بالكامل على وحدة معالجة الرسومات (GPU). إذا كنت تقوم بالاستدلال على وحدة المعالجة المركزية (CPU) باستخدام AutoGPTQ (الإصدار > 0.4.2)، فستحتاج إلى تعطيل نواة ExLlama. وهذا يكتب فوق السمات المتعلقة بنواة ExLlama في تكوين التكميم لملف config.json.

```py
import torch
from transformers import AutoModelForCausalLM, GPTQConfig
gptq_config = GPTQConfig(bits=4, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="cpu", quantization_config=gptq_config)
```