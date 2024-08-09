# تحميل المحولات باستخدام 🤗 PEFT

[[open-in-colab]]

تقوم طرق [التدريب الدقيق الفعال للبارامترات (PEFT)](https://huggingface.co/blog/peft) بتجميد معلمات النموذج المُدرب مسبقًا أثناء الضبط الدقيق وإضافة عدد صغير من المعلمات القابلة للتدريب (المحولات) فوقه. يتم تدريب المحولات لتعلم معلومات خاصة بالمهام. وقد ثبت أن هذا النهج فعال للغاية من حيث الذاكرة مع انخفاض استخدام الكمبيوتر أثناء إنتاج نتائج قابلة للمقارنة مع نموذج مضبوط دقيقًا تمامًا.

عادة ما تكون المحولات المدربة باستخدام PEFT أصغر أيضًا بمقدار درجة من حيث الحجم من النموذج الكامل، مما يجعل من السهل مشاركتها وتخزينها وتحميلها.

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">تبلغ أوزان المحول لطراز OPTForCausalLM المخزن على Hub حوالي 6 ميجابايت مقارنة بالحجم الكامل لأوزان النموذج، والتي يمكن أن تكون حوالي 700 ميجابايت.</figcaption>
</div>

إذا كنت مهتمًا بمعرفة المزيد عن مكتبة 🤗 PEFT، فراجع [الوثائق](https://huggingface.co/docs/peft/index).

## الإعداد

ابدأ بتثبيت 🤗 PEFT:

```bash
pip install peft
```

إذا كنت تريد تجربة الميزات الجديدة تمامًا، فقد تكون مهتمًا بتثبيت المكتبة من المصدر:

```bash
pip install git+https://github.com/huggingface/peft.git
```

## نماذج PEFT المدعومة

يدعم 🤗 Transformers بشكلٍ أصلي بعض طرق PEFT، مما يعني أنه يمكنك تحميل أوزان المحول المخزنة محليًا أو على Hub وتشغيلها أو تدريبها ببضع سطور من التعليمات البرمجية. الطرق المدعومة هي:

- [محولات الرتبة المنخفضة](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [AdaLoRA](https://arxiv.org/abs/2303.10512)

إذا كنت تريد استخدام طرق PEFT الأخرى، مثل تعلم المحث أو ضبط المحث، أو حول مكتبة 🤗 PEFT بشكل عام، يرجى الرجوع إلى [الوثائق](https://huggingface.co/docs/peft/index).

## تحميل محول PEFT

لتحميل نموذج محول PEFT واستخدامه من 🤗 Transformers، تأكد من أن مستودع Hub أو الدليل المحلي يحتوي على ملف `adapter_config.json` وأوزان المحول، كما هو موضح في صورة المثال أعلاه. بعد ذلك، يمكنك تحميل نموذج محول PEFT باستخدام فئة `AutoModelFor`. على سبيل المثال، لتحميل نموذج محول PEFT للنمذجة اللغوية السببية:

1. حدد معرف نموذج PEFT
2. مرره إلى فئة [`AutoModelForCausalLM`]

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```

<Tip>

يمكنك تحميل محول PEFT باستخدام فئة `AutoModelFor` أو فئة النموذج الأساسي مثل `OPTForCausalLM` أو `LlamaForCausalLM`.

</Tip>

يمكنك أيضًا تحميل محول PEFT عن طريق استدعاء طريقة `load_adapter`:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

راجع قسم [وثائق API](#transformers.integrations.PeftAdapterMixin) أدناه لمزيد من التفاصيل.

## التحميل في 8 بت أو 4 بت

راجع قسم [وثائق API](#transformers.integrations.PeftAdapterMixin) أدناه لمزيد من التفاصيل.

## التحميل في 8 بت أو 4 بت

تدعم تكامل `bitsandbytes` أنواع بيانات الدقة 8 بت و4 بت، والتي تكون مفيدة لتحميل النماذج الكبيرة لأنها توفر الذاكرة (راجع دليل تكامل `bitsandbytes` [guide](./quantization#bitsandbytes-integration) لمعرفة المزيد). أضف `load_in_8bit` أو `load_in_4bit` المعلمات إلى [`~PreTrainedModel.from_pretrained`] وقم بتعيين `device_map="auto"` لتوزيع النموذج بشكل فعال على الأجهزة لديك:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

## إضافة محول جديد

يمكنك استخدام [`~peft.PeftModel.add_adapter`] لإضافة محول جديد إلى نموذج بمحول موجود طالما أن المحول الجديد هو من نفس نوع المحول الحالي. على سبيل المثال، إذا كان لديك محول LoRA موجود ملحق بنموذج:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import LoraConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```

لإضافة محول جديد:

```py
# قم بتعليق محول جديد بنفس التكوين
model.add_adapter(lora_config, adapter_name="adapter_2")
```

الآن يمكنك استخدام [`~peft.PeftModel.set_adapter`] لتعيين المحول الذي سيتم استخدامه:

```py
# استخدم adapter_1
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))

# استخدم adapter_2
model.set_adapter("adapter_2")
output_enabled = model.generate(**inputs)
print(tokenizer.decode(output_enabled[0], skip_special_tokens=True))
```

## تمكين وتعطيل المحولات

بمجرد إضافة محول إلى نموذج، يمكنك تمكين أو تعطيل وحدة المحول. لتمكين وحدة المحول:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)

# لبدء تشغيله بأوزان عشوائية
peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_adapters()
output = model.generate(**inputs)
```

لإيقاف تشغيل وحدة المحول:

```py
model.disable_adapters()
output = model.generate(**inputs)
```

## تدريب محول PEFT

يدعم محول PEFT فئة [`Trainer`] بحيث يمكنك تدريب محول لحالتك الاستخدام المحددة. فهو يتطلب فقط إضافة بضع سطور أخرى من التعليمات البرمجية. على سبيل المثال، لتدريب محول LoRA:

<Tip>

إذا لم تكن معتادًا على ضبط نموذج دقيق باستخدام [`Trainer`، فراجع البرنامج التعليمي](training) لضبط نموذج مُدرب مسبقًا.

</Tip>

1. حدد تكوين المحول باستخدام نوع المهمة وفرط المعلمات (راجع [`~peft.LoraConfig`] لمزيد من التفاصيل حول ما تفعله فرط المعلمات).

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"،
)
```

2. أضف المحول إلى النموذج.

```py
model.add_adapter(peft_config)
```

3. الآن يمكنك تمرير النموذج إلى [`Trainer`]!

```py
trainer = Trainer(model=model, ...)
trainer.train()
```

لحفظ محول المدرب وتحميله مرة أخرى:

```py
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```

## إضافة طبقات قابلة للتدريب إضافية إلى محول PEFT

```py
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```

## إضافة طبقات قابلة للتدريب إضافية إلى محول PEFT

يمكنك أيضًا ضبط طبقات قابلة للتدريب إضافية أعلى نموذج به محولات مرفقة عن طريق تمرير `modules_to_save` في تكوين PEFT الخاص بك. على سبيل المثال، إذا كنت تريد أيضًا ضبط `lm_head` أعلى نموذج بمحول LoRA:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import LoraConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    modules_to_save=["lm_head"]،
)

model.add_adapter(lora_config)
```

## وثائق API

[[autodoc]] integrations.PeftAdapterMixin
    - load_adapter
    - add_adapter
    - set_adapter
    - disable_adapters
    - enable_adapters
    - active_adapters
    - get_adapter_state_dict




<!--
TODO: (@younesbelkada @stevhliu)
-   Link to PEFT docs for further details
-   Trainer
-   8-bit / 4-bit examples ?
-->