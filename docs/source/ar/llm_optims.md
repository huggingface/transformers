# تحسين الاستنتاج لنماذج اللغة الضخمة

دفعت نماذج اللغة الضخمة (LLMs) تطبيقات توليد النصوص، مثل نماذج الدردشة واستكمال الأكواد، إلى المستوى التالي من خلال إنتاج نصوص تظهر مستوى عاليًا من الفهم والطلاقة. ولكن ما يجعل نماذج اللغة الضخمة قوية - أي حجمها - يطرح أيضًا تحديات للاستنتاج.

الاستنتاج الأساسي بطيء لأن نماذج اللغة الضخمة يجب أن تُستدعى بشكل متكرر لتوليد الرمز التالي. تتزايد تسلسل المدخلات مع تقدم التوليد، الأمر الذي يستغرق وقتًا أطول وأطول لنماذج اللغة الضخمة لمعالجتها. تمتلك نماذج اللغة الضخمة أيضًا مليارات من المعلمات، مما يجعل من الصعب تخزين ومعالجة جميع هذه الأوزان في الذاكرة.

سيوضح هذا الدليل كيفية استخدام تقنيات التحسين المتاحة في مكتبة Transformers لتسريع الاستنتاج لنماذج اللغة الضخمة.

> [!TIP]
> توفر Hugging Face أيضًا [Text Generation Inference (TGI)](https://hf.co/docs/text-generation-inference)، وهي مكتبة مخصصة لنشر وخدمة نماذج اللغة الضخمة المحسنة للغاية للاستنتاج. تتضمن ميزات التحسين الموجهة للنشر غير المدرجة في مكتبة Transformers، مثل التجميع المستمر لزيادة الإنتاجية ومتوازية التنسور لاستنتاج متعدد وحدات معالجة الرسومات (GPU).

## ذاكرة التخزين المؤقت الثابتة لـ key-value و `torch.compile`

أثناء فك التشفير، يحسب نموذج اللغة الضخمة قيم key-value (kv) لكل رمز من رموز المدخلات، وبما أنه تنبؤي ذاتيًا، فإنه يحسب نفس قيم kv في كل مرة لأن الإخراج المولد يصبح الآن جزءًا من المدخلات. هذا غير فعال لأنك تقوم بإعادة حساب نفس قيم kv في كل مرة.

لتحسين ذلك، يمكنك استخدام ذاكرة التخزين المؤقت لـ kv لتخزين المفاتيح والقيم السابقة بدلاً من إعادة حسابها في كل مرة. ومع ذلك، نظرًا لأن ذاكرة التخزين المؤقت لـ kv تنمو مع كل خطوة من خطوات التوليد وهي ديناميكية، فإنها تمنعك من الاستفادة من [`torch.compile`](./perf_torch_compile)، وهي أداة تحسين قوية تقوم بدمج كود PyTorch في نواة سريعة ومحسنة.

تعالج ذاكرة التخزين المؤقت الثابتة لـ kv هذه المشكلة من خلال تخصيص حجم ذاكرة التخزين المؤقت لـ kv مسبقًا إلى قيمة قصوى، مما يتيح لك دمجها مع `torch.compile` للتسريع بمقدار 4 مرات. قد يختلف تسريعك اعتمادًا على حجم النموذج (تمتلك النماذج الأكبر تسريعًا أصغر) والأجهزة.

> [!WARNING]
> حاليًا، تدعم نماذج [Llama](./model_doc/llama2] وبعض النماذج الأخرى فقط ذاكرة التخزين المؤقت الثابتة لـ kv و`torch.compile`. تحقق من [هذه المشكلة](https://github.com/huggingface/transformers/issues/28981) للحصول على قائمة توافق النماذج المباشرة.

هناك ثلاثة نكهات من استخدام ذاكرة التخزين المؤقت الثابتة لـ kv، اعتمادًا على مدى تعقيد مهمتك:
1. الاستخدام الأساسي: قم ببساطة بتعيين علامة في `generation_config` (يوصى بها)؛
2. الاستخدام المتقدم: التعامل مع كائن ذاكرة التخزين المؤقت للتوليد متعدد الأدوار أو حلقة التوليد المخصصة؛
3. الاستخدام المتقدم: قم بتجميع دالة `generate` بأكملها في رسم بياني واحد، إذا كان وجود رسم بياني واحد ذي صلة بالنسبة لك.

حدد علامة التبويب الصحيحة أدناه للحصول على مزيد من التعليمات حول كل من هذه النكهات.

> [!TIP]
> بغض النظر عن الاستراتيجية المستخدمة مع `torch.compile`، يمكنك تجنب إعادة التجميع المتعلقة بالشكل إذا قمت بمحاذاة إدخالات نموذج اللغة الضخمة إلى مجموعة محدودة من القيم. علم التوكيد [`pad_to_multiple_of`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.pad_to_multiple_of) هو صديقك!
<hfoptions id="static-kv">
<hfoption id="basic usage: generation_config">

في هذا المثال، دعنا نستخدم نموذج [Gemma](https://hf.co/google/gemma-2b). كل ما نحتاج إلى فعله هو:
1. الوصول إلى سمة `generation_config` للنموذج وتعيين `cache_implementation` إلى "static"؛
2. استدعاء `torch.compile` على النموذج لتجميع عملية التمرير للأمام مع ذاكرة التخزين المؤقت الثابتة لـ kv.

وهذا كل شيء!

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # لمنع التحذيرات الطويلة :)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

model.generation_config.cache_implementation = "static"

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference']
```

تحت الغطاء، ستحاول دالة `generate` إعادة استخدام كائن ذاكرة التخزين المؤقت نفسه، مما يزيل الحاجة إلى إعادة التجميع في كل استدعاء. تجنب إعادة التجميع أمر بالغ الأهمية للحصول على أقصى استفادة من `torch.compile`، ويجب أن تكون على دراية بما يلي:
1. إذا تغير حجم الدفعة أو زاد طول الإخراج الأقصى بين الاستدعاءات، فسيتعين إعادة تهيئة ذاكرة التخزين المؤقت، مما يؤدي إلى تشغيل تجميع جديد؛
2. تكون أولى الاستدعاءات القليلة للدالة المجمعة أبطأ، حيث يجري تجميع الدالة.

> [!WARNING]
> للاستخدام الأكثر تقدمًا لذاكرة التخزين المؤقت الثابتة، مثل المحادثات متعددة الأدوار، نوصي بإنشاء كائن ذاكرة التخزين المؤقت والتعامل معه خارج [`~GenerationMixin.generate`]. راجع علامة التبويب "الاستخدام المتقدم".

</hfoption>
<hfoption id="advanced usage: control Static Cache">

يمكن تمرير كائن [`StaticCache`] إلى دالة [`~GenerationMixin.generate`] الخاصة بالنموذج في إطار وسيط `past_key_values`. سيحتفظ كائن ذاكرة التخزين المؤقت بمحتويات ذاكرة التخزين المؤقت، لذا يمكنك تمريره إلى استدعاء جديد لـ [`~GenerationMixin.generate`] لمواصلة التوليد، كما تفعل مع ذاكرة التخزين المؤقت الديناميكية.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # لمنع التحذيرات الطويلة :)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
prompt_length = input_ids.input_ids.shape[1]
model.generation_config.max_new_tokens = 16

past_key_values = StaticCache(
    config=model.config,
    max_batch_size=1,
    # إذا كنت تخطط لإعادة استخدام ذاكرة التخزين المؤقت، فتأكد من أن طول ذاكرة التخزين المؤقت كبير بما يكفي لجميع الحالات
max_cache_len=prompt_length+(model.generation_config.max_new_tokens*2),
    device=model.device,
    dtype=model.dtype
)
outputs = model.generate(**input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference frames. 2']

# قم بتمرير النص المولد ونفس كائن ذاكرة التخزين المؤقت للاستمرار في التوليد من حيث توقف. اختياريًا، في
# محادثة متعددة الأدوار، أضف إدخال المستخدم الجديد إلى النص المولد.
new_input_ids = outputs
outputs = model.generate(new_input_ids, past_key_values=past_key_values)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference frames. 2. The speed of light is constant in all inertial reference frames. 3.']
```
> [!TIP]
> إذا كنت تريد إعادة استخدام نفس كائن [`StaticCache`] على موجه جديد، فتأكد من إعادة تعيين محتوياته باستخدام طريقة `.reset()` بين الاستدعاءات

إذا كنت تريد الذهاب إلى مستوى أعمق، فيمكن أيضًا تمرير كائن [`StaticCache`] إلى تمرير النموذج للأمام في إطار وسيط `past_key_values` نفسه. باستخدام هذه الاستراتيجية، يمكنك كتابة دالتك الخاصة لفك تشفير الرمز التالي نظرًا للرمز الحالي والموضع وموضع ذاكرة التخزين المؤقت للرموز المولدة سابقًا.

```py
from transformers import LlamaTokenizer, LlamaForCausalLM, StaticCache, logging
from transformers.testing_utils import CaptureLogger
import torch

prompts = [
    "Simply put, the theory of relativity states that ",
    "My favorite all time favorite condiment is ketchup.",
]

NUM_TOKENS_TO_GENERATE = 40
torch_device = "cuda"

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="sequential")
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token
```

هناك بعض الأشياء المهمة التي يجب عليك القيام بها لتمكين ذاكرة التخزين المؤقت الثابتة لـ kv و`torch.compile` مع طريقة `StaticCache`:
1. قم بإنشاء مثيل [`StaticCache`] قبل استخدام النموذج للاستنتاج. هناك يمكنك تكوين معلمات مثل حجم الدفعة القصوى وطول التسلسل.
2. استدعاء `torch.compile` على النموذج لتجميع عملية التمرير للأمام مع ذاكرة التخزين المؤقت الثابتة لـ kv.
3. قم بتعيين `enable_math=True` في سياق [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) لتمكين تنفيذ C++ الأصلي لـ PyTorch للاهتمام بمنتج النقاط المحدد لحجمها لتسريع الاستنتاج أكثر.

```py
batch_size, seq_length = inputs["input_ids"].shape
with torch.no_grad():
    past_key_values = StaticCache(
        config=model.config, max_batch_size=2, max_cache_len=4096, device=torch_device, dtype=model.dtype
    )
    cache_position = torch.arange(seq_length, device=torch_device)
    generated_ids = torch.zeros(
        batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=torch.int, device=torch_device
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(torch_device).to(torch.int)

    logits = model(
        **inputs, cache_position=cache_position, past_key_values=past_key_values,return_dict=False, use_cache=True
    )[0]
    next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    generated_ids[:, seq_length] = next_token[:, 0]

    decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead", fullgraph=True)
    cache_position = torch.tensor([seq_length + 1], device=torch_device)
    for _ in range(1, NUM_TOKENS_TO_GENERATE):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one_tokens(model, next_token.clone(), None, cache_position, past_key_values)
            generated_ids[:, cache_position] = next_token.int()
        cache_position += 1

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
text
['Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light is the same for all observers, and 3) the laws of physics are the same for all observers.',
 'My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p']
```

</hfoption>
<hfoption id="advanced usage: end-to-end generate compilation">
</hfoption>
<hfoption id="advanced usage: end-to-end generate compilation">

من حيث الكود، فإن تجميع دالة `generate` بأكملها أبسط حتى من الاستخدام الأساسي: قم باستدعاء `torch.compile` على `generate` لتجميع الدالة بأكملها. لا يلزم تحديد استخدام ذاكرة التخزين المؤقت الثابتة: على الرغم من أنها متوافقة، إلا أن ذاكرة التخزين المؤقت الديناميكية (الافتراضية) كانت أسرع في مقاييسنا.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # لمنع التحذيرات الطويلة :)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

model.generate = torch.compile(model.generate, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference']
```

ونتيجة لذلك، فإننا لا نقوم بتجميع عملية التمرير للأمام للنموذج فحسب، بل نقوم أيضًا بتجميع جميع عمليات إعداد الإدخال وعمليات معالج الرموز وما إلى ذلك. يجب أن يكون الاستدعاء الناتج لـ `generate` أبطأ قليلاً مقارنة بمثال الاستخدام الأساسي، وقد يكون الرسم البياني المجمع أكثر ملاءمة لأجهزة الأجهزة أو حالات الاستخدام الأكثر غرابة. ومع ذلك، هناك عيوب شديدة في استخدام هذا النهج:
1. التجميع أبطأ بكثير؛
2. يجب إجراء جميع معلمات `generate` من خلال `generation_config`؛
3. يتم قمع العديد من التحذيرات والاستثناءات - نقترح عليك اختباره أولاً بشكل غير مجمع؛
4. على الرغم من أننا نعمل عليه، إلا
## تحسين الانتباه

هناك مشكلة معروفة في نماذج المحولات وهي أن آلية الانتباه الذاتي تنمو بشكل تربيعي في الحساب والذاكرة مع عدد الرموز المميزة للإدخال. يتم تضخيم هذا القيد فقط في نماذج اللغة الكبيرة التي تتعامل مع تسلسلات أطول. لمعالجة هذا الأمر، جرب FlashAttention2 أو التنفيذ المُحسَّن للانتباه المُوزَّع المُقَيَّد المُقَدَّم في PyTorch، واللذان يُعدَّان أكثر كفاءة في الذاكرة ويمكن أن يُسرِّعا الاستنتاج.

### FlashAttention-2

يقسم FlashAttention و [FlashAttention-2](./perf_infer_gpu_one#flashattention-2) حساب الانتباه إلى أجزاء أصغر ويقلل عدد عمليات القراءة/الكتابة الوسيطة إلى ذاكرة GPU لتسريع الاستنتاج. ويحسن FlashAttention-2 من خوارزمية FlashAttention الأصلية من خلال الموازاة أيضًا على بعد طول التسلسل وتقسيم العمل بشكل أفضل على الأجهزة لتقليل النفقات العامة للاتصال والتزامن.

لاستخدام FlashAttention-2، قم بتعيين `attn_implementation="flash_attention_2"` في طريقة [`~PreTrainedModel.from_pretrained`] .

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### الانتباه الموزع المُقَيَّد المُقَدَّم في PyTorch

يتم تمكين الانتباه الموزع المُقَيَّد المُقَدَّم في PyTorch 2.0 افتراضيًا وهو يدعم FlashAttention و xFormers وتنفيذ PyTorch في C++. يختار الانتباه الموزع المُقَيَّد المُقَدَّم في PyTorch أكثر خوارزميات الانتباه أداءً إذا كنت تستخدم backend CUDA. وبالنسبة إلى backends الأخرى، فإن الانتباه الموزع المُقَيَّد المُقَدَّم في PyTorch يستخدم التنفيذ الافتراضي لـ C++.

> [!TIP]
> يدعم الانتباه الموزع المُقَيَّد المُقَدَّم في PyTorch FlashAttention-2 طالما أن لديك أحدث إصدار من PyTorch مثبتًا.

استخدم مدير سياق [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) لتمكين أو تعطيل أي من خوارزميات الانتباه الثلاث بشكل صريح. على سبيل المثال، قم بتعيين `enable_flash=True` لتمكين FlashAttention.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    torch_dtype=torch.bfloat16,
)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)
```

## التكميم

يقلل التكميم من حجم أوزان نماذج اللغة الكبيرة من خلال تخزينها في دقة أقل. وهذا يترجم إلى استخدام ذاكرة أقل ويجعل تحميل نماذج اللغة الكبيرة للاستنتاج أكثر سهولة إذا كنت مقيدًا بذاكرة GPU الخاصة بك. إذا لم تكن محدودًا بـ GPU الخاص بك، فلا يلزم بالضرورة تكميم نموذجك لأنه قد يتكبد تكلفة صغيرة في الكمون (باستثناء وحدات AWQ و AWQ المدمجة) بسبب الخطوة الإضافية المطلوبة لكممة وإلغاء كممة الأوزان.

> [!TIP]
> هناك العديد من مكتبات التكميم (راجع دليل [Quantization](./quantization) للحصول على مزيد من التفاصيل) المتاحة، مثل Quanto و AQLM و AWQ و AutoGPTQ. لا تتردد في تجربتها وشاهد أيها يعمل بشكل أفضل لحالتك الاستخدامية. نوصي أيضًا بقراءة منشور المدونة [نظرة عامة على مخططات التكميم المدعومة أصلاً في 🤗 Transformers](https://hf.co/blog/overview-quantization-transformers) الذي يقارن AutoGPTQ و bitsandbytes.

استخدم آلة حاسبة ذاكرة النموذج أدناه لتقدير ومقارنة مقدار الذاكرة المطلوبة لتحميل نموذج. على سبيل المثال، جرب تقدير مقدار الذاكرة التي يتكلفها تحميل [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).

<iframe
	src="https://hf-accelerate-model-memory-usage.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

لتحميل Mistral-7B-v0.1 بنصف الدقة، قم بتعيين معلمة `torch_dtype` في طريقة [`~transformers.AutoModelForCausalLM.from_pretrained`] إلى `torch.bfloat16`. يتطلب هذا 13.74 جيجابايت من الذاكرة.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", torch_dtype=torch.bfloat16, device_map="auto",
)
```


لتحميل نموذج كمي (8 بت أو 4 بت) للاستدلال، جرب [bitsandbytes](https://hf.co/docs/bitsandbytes) وقم بتعيين معلمات "load_in_4bit" أو "load_in_8bit" إلى "True". يتطلب تحميل النموذج في 8 بتات فقط 6.87 جيجابايت من الذاكرة.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=quant_config, device_map="auto"
)
```