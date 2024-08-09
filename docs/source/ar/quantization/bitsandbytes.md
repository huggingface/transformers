# bitsandbytes

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) هي أسهل طريقة لضغط نموذج إلى 8 بت و4 بت. تضاعف الضغط إلى 8 بت من تأثير القيم الشاذة على أداء النموذج من خلال ضرب القيم الشاذة في fp16 مع غير الشاذة في int8، وتحويل قيم غير الشاذة مرة أخرى إلى fp16، ثم جمعها معًا لإرجاع الأوزان في fp16. يضغط الضغط إلى 4 بت النموذج بشكل أكبر، ويستخدم عادةً مع [QLoRA](https://hf.co/papers/2305.14314) لضبط دقة النماذج الضخمة للغة (LLMs).

لاستخدام bitsandbytes، تأكد من تثبيت المكتبات التالية:

<hfoptions id="bnb">
<hfoption id="8-bit">

```bash
pip install transformers accelerate bitsandbytes>0.37.0
```

</hfoption>
<hfoption id="4-bit">

```bash
pip install bitsandbytes>=0.39.0
pip install --upgrade accelerate transformers
```

</hfoption>
</hfoptions>

الآن يمكنك ضغط نموذج عن طريق تمرير `BitsAndBytesConfig` إلى طريقة [`~PreTrainedModel.from_pretrained`] . يعمل هذا مع أي نموذج في أي طريقة، طالما أنه يدعم التحميل باستخدام Accelerate ويحتوي على طبقات `torch.nn.Linear` .

<hfoptions id="bnb">
<hfoption id="8-bit">

يضاعف الضغط إلى 8 بت من استخدام الذاكرة، وبالنسبة للنماذج الكبيرة، قم بتعيين `device_map="auto"` لاستخدام وحدات معالجة الرسومات (GPUs) المتوفرة بكفاءة:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7", 
    quantization_config=quantization_config
)
```

بشكل افتراضي، يتم تحويل جميع الوحدات النمطية الأخرى مثل `torch.nn.LayerNorm` إلى `torch.float16`. يمكنك تغيير نوع بيانات هذه الوحدات النمطية باستخدام معلمة `torch_dtype` إذا أردت:

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", 
    quantization_config=quantization_config, 
    torch_dtype=torch.float32
)
model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

بمجرد ضغط نموذج إلى 8 بت، لا يمكنك دفع الأوزان المضغوطة إلى Hub إلا إذا كنت تستخدم أحدث إصدار من Transformers وbitsandbytes. إذا كان لديك أحدث الإصدارات، فيمكنك دفع النموذج 8 بت إلى Hub باستخدام طريقة [`~PreTrainedModel.push_to_hub`] . يتم أولاً دفع ملف تكوين الضغط، يليه أوزان النموذج المضغوط.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", 
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

</hfoption>
<hfoption id="4-bit">

يقلل الضغط إلى 4 بت من استخدام الذاكرة بمقدار 4 مرات، وبالنسبة للنماذج الكبيرة، قم بتعيين `device_map="auto"` لاستخدام وحدات معالجة الرسومات (GPUs) المتوفرة بكفاءة:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_Multiplier = BitsAndBytesConfig(load_in_4bit=True)
```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    quantization_config=quantization_config
)
```

بشكل افتراضي، يتم تحويل جميع الوحدات النمطية الأخرى مثل `torch.nn.LayerNorm` إلى `torch.float16`. يمكنك تغيير نوع بيانات هذه الوحدات النمطية باستخدام معلمة `torch_dtype` إذا أردت:

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    quantization_config=quantization_config, 
    torch_dtype=torch.float32
)
model_4bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

إذا كان لديك `bitsandbytes>=0.41.3`، فيمكنك تسلسل نماذج 4 بت ودفعها على Hugging Face Hub. ما عليك سوى استدعاء `model.push_to_hub()` بعد تحميله بدقة 4 بت. يمكنك أيضًا حفظ نماذج 4 بت المتسلسلة محليًا باستخدام أمر `model.save_pretrained()` .

</hfoption>
</hfoptions>

<Tip warning={true}>

يتم دعم التدريب باستخدام أوزان 8 بت و4 بت فقط لتدريب المعلمات *الإضافية* .

</Tip>

يمكنك التحقق من بصمة الذاكرة الخاصة بك باستخدام طريقة `get_memory_footprint` :

```py
print(model.get_memory_footprint())
```

يمكن تحميل النماذج المضغوطة من طريقة [`~PreTrainedModel.from_pretrained`] دون الحاجة إلى تحديد معلمات `load_in_8bit` أو `load_in_4bit` :

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```

## 8-بت (LLM.int8() خوارزمية)

<Tip>

تعرف على المزيد حول تفاصيل الضغط إلى 8 بت في [منشور المدونة](https://huggingface.co/blog/hf-bitsandbytes-integration) هذا!

</Tip>

يستكشف هذا القسم بعض الميزات المحددة لنماذج 8 بت، مثل التفريغ وعتبات القيم الشاذة ومتجاوزة تحويل الوحدة النمطية والضبط الدقيق.

### التفريغ

8 بت يمكن أن تقوم النماذج بتفريغ الأوزان بين وحدة المعالجة المركزية (CPU) ووحدات معالجة الرسومات (GPU) لدعم تناسب النماذج الكبيرة جدًا في الذاكرة. يتم تخزين الأوزان المرسلة إلى وحدة المعالجة المركزية (CPU) فعليًا في **float32**، ولا يتم تحويلها إلى 8 بت. على سبيل المثال، لتمكين التفريغ لنموذج [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7) ، ابدأ بإنشاء [`BitsAndBytesConfig`]:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

قم بتصميم خريطة جهاز مخصصة لتناسب كل شيء على وحدة معالجة الرسومات (GPU) الخاصة بك باستثناء `lm_head` ، والتي ستقوم بتفريغها إلى وحدة المعالجة المركزية (CPU):

```py
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

الآن قم بتحميل نموذجك باستخدام `device_map` مخصص و `quantization_config` :

```py
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

### عتبة القيم الشاذة

"القيمة الشاذة" هي قيمة حالة مخفية أكبر من عتبة معينة، ويتم حساب هذه القيم في fp16. في حين أن القيم موزعة عادة ([-3.5، 3.5])، يمكن أن يكون هذا التوزيع مختلفًا جدًا للنماذج الكبيرة ([-60، 6] أو [6، 60]). يعمل الضغط إلى 8 بت بشكل جيد للقيم ~5، ولكن بعد ذلك، هناك عقوبة أداء كبيرة. العتبة الافتراضية جيدة هي 6، ولكن قد تكون هناك حاجة إلى عتبة أقل للنماذج الأقل استقرارًا (نماذج صغيرة أو ضبط دقيق).

لإيجاد أفضل عتبة لنموذجك، نوصي بالتجربة باستخدام معلمة `llm_int8_threshold` في [`BitsAndBytesConfig`]:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)
```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
```

### تخطي تحويل الوحدة النمطية

بالنسبة لبعض النماذج، مثل [Jukebox](model_doc/jukebox)، لا تحتاج إلى ضغط كل وحدة نمطية إلى 8 بت والتي يمكن أن تسبب عدم استقرار في الواقع. مع Jukebox، هناك عدة وحدات `lm_head` يجب تخطيها باستخدام معلمة `llm_int8_skip_modules` في [`BitsAndBytesConfig`]:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)
```

### الضبط الدقيق

مع مكتبة [PEFT](https://github.com/huggingface/peft) ، يمكنك ضبط دقة النماذج الكبيرة مثل [flan-t5-large](https://huggingface.co/google/flan-t5-large) و [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) باستخدام الضغط إلى 8 بت. لا تحتاج إلى تمرير معلمة `device_map` للتدريب لأنه سيقوم تلقائيًا بتحميل نموذجك على وحدة معالجة الرسومات (GPU). ومع ذلك، يمكنك لا تزال تخصيص خريطة الجهاز مع معلمة `device_map` إذا كنت ترغب في ذلك (`device_map="auto"` يجب أن تستخدم فقط للاستدلال).

## 4-بت (خوارزمية QLoRA)

<Tip>

جرب الضغط إلى 4 بت في هذا [دفتر الملاحظات](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf) وتعرف على المزيد حول تفاصيله في [منشور المدونة](https://huggingface.co/blog/4bit-transformers-bitsandbytes) هذا.

</Tip>

يستكشف هذا القسم بعض الميزات المحددة لنماذج 4 بت، مثل تغيير نوع بيانات الحساب، واستخدام نوع بيانات Normal Float 4 (NF4)، واستخدام الضغط المتداخل.

### نوع بيانات الحساب

لزيادة سرعة الحساب، يمكنك تغيير نوع البيانات من float32 (القيمة الافتراضية) إلى bf16 باستخدام معلمة `bnb_4bit_compute_dtype` في [`BitsAndBytesConfig`]:

```py
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

### Normal Float 4 (NF4)

NF4 هو نوع بيانات 4 بت من ورقة [QLoRA](https://hf.co/papers/2305.14314) ، تم تكييفه مع الأوزان المبدئية من توزيع عادي. يجب عليك استخدام NF4 للتدريب على نماذج الأساس 4 بت. يمكن تكوين هذا باستخدام معلمة `bnb_4bit_quant_type` في [`BitsAndBytesConfig`]:

```py
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

بالنسبة للاستدلال، لا يكون لنوع `bnb_4bit_quant_type` تأثير كبير على الأداء. ومع ذلك، للبقاء متسقًا مع أوزان النموذج، يجب عليك استخدام قيم `bnb_4bit_compute_dtype` و `torch_dtype` .

### الضغط المتداخل

الضغط المتداخل هو تقنية يمكن أن توفر ذاكرة إضافية دون تكلفة أداء إضافية. تقوم هذه الميزة بضغط الأوزان المضغوطة بالفعل لتوفر 0.4 بت/معلمة إضافية. على سبيل المثال، مع الضغط المتداخل، يمكنك ضبط نموذج [Llama-13b](https://huggingface.co/meta-llama/Llama-2-13b) بدقة على وحدة معالجة الرسومات (GPU) NVIDIA T4 بسعة 16 جيجابايت مع طول تسلسل يبلغ 1024، وحجم دفعة يبلغ 1، وتمكين تراكم التدرجات مع 4 خطوات.

```py
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b", quantization_config=double_quant_config)
```

## فك ضغط نماذج `bitsandbytes`

بمجرد ضغط نموذج، يمكنك فك ضغطه إلى الدقة الأصلية ولكن قد يؤدي ذلك إلى فقدان جودة النموذج قليلاً. تأكد من أن لديك ذاكرة وصول عشوائي (RAM) لوحدة معالجة الرسومات (GPU) كافية لتناسب النموذج الذي تم فك ضغطه.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

model_id = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id, BitsAndBytesConfig(load_in_4bit=True))
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.dequantize()

text = tokenizer("Hello my name is", return_tensors="pt").to(0)

out = model.generate(**text)
print(tokenizer.decode(out[0]))
```