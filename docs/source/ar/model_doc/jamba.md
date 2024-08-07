# Jamba

## نظرة عامة

Jamba عبارة عن شبكة عصبية هجينة SSM-Transformer LLM رائدة في مجالها. إنه أول تطبيق لمقياس Mamba، والذي يفتح فرصًا مثيرة للبحث والتطبيق. في حين أن هذه التجربة الأولية تظهر مكاسب مشجعة، فإننا نتوقع أن يتم تعزيزها بشكل أكبر مع الاستكشافات والتحسينات المستقبلية.

للحصول على التفاصيل الكاملة حول هذا النموذج، يرجى قراءة [منشور المدونة](https://www.ai21.com/blog/announcing-jamba) الخاص بالإصدار.

### تفاصيل النموذج

Jamba هو نموذج نصي تنبئي مسبق التدريب، يعتمد على مزيج من الخبراء (MoE)، ويحتوي على 12 بليون معامل نشط و52 بليون معامل إجمالي عبر جميع الخبراء. يدعم النموذج طول سياق يصل إلى 256 ألفًا، ويمكنه استيعاب ما يصل إلى 140 ألف رمز على وحدة معالجة رسومية (GPU) واحدة بسعة 80 جيجابايت.

كما هو موضح في الرسم البياني أدناه، تتميز بنية Jamba بنهج الكتل والطبقات الذي يسمح لـ Jamba بدمج بنيات Transformer وMamba بنجاح. تحتوي كل كتلة من Jamba إما على طبقة اهتمام أو طبقة Mamba، تليها شبكة عصبية متعددة الطبقات (MLP)، مما ينتج عنه نسبة إجمالية تبلغ طبقة Transformer واحدة من كل ثماني طبقات إجمالية.

![رسم توضيحي لبنية Jamba](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/jamba_architecture.png)

## الاستخدام

### المتطلبات الأساسية

يتطلب Jamba استخدام إصدار `transformers` 4.39.0 أو أعلى:

```bash
pip install transformers>=4.39.0
```

لتشغيل تطبيقات Mamba المحسّنة، يجب أولاً تثبيت `mamba-ssm` و`causal-conv1d`:

```bash
pip install mamba-ssm causal-conv1d>=1.2.0
```

يجب أيضًا أن يكون لديك النموذج على جهاز CUDA.

يمكنك تشغيل النموذج دون استخدام نوى Mamba المحسنة، ولكن لا يوصى بذلك لأنه سيؤدي إلى انخفاض كبير في الكمون. للقيام بذلك، ستحتاج إلى تحديد `use_mamba_kernels=False` عند تحميل النموذج.

### تشغيل النموذج

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1")
tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

input_ids = tokenizer("In the recent Super Bowl LVIII,", return_tensors='pt').to(model.device)["input_ids"]

outputs = model.generate(input_ids, max_new_tokens=216)

print(tokenizer.batch_decode(outputs))
# ["<|startoftext|>In the recent Super Bowl LVIII, the Kansas City Chiefs emerged victorious, defeating the San Francisco 49ers in a thrilling overtime showdown. The game was a nail-biter, with both teams showcasing their skills and determination.\n\nThe Chiefs, led by their star quarterback Patrick Mahomes, displayed their offensive prowess, while the 49ers, led by their strong defense, put up a tough fight. The game went into overtime, with the Chiefs ultimately securing the win with a touchdown.\n\nThe victory marked the Chiefs' second Super Bowl win in four years, solidifying their status as one of the top teams in the NFL. The game was a testament to the skill and talent of both teams, and a thrilling end to the NFL season.\n\nThe Super Bowl is not just about the game itself, but also about the halftime show and the commercials. This year's halftime show featured a star-studded lineup, including Usher, Alicia Keys, and Lil Jon. The show was a spectacle of music and dance, with the performers delivering an energetic and entertaining performance.\n"]
```

<details>

<summary><strong>تحميل النموذج بنصف الدقة</strong></summary>

تم حفظ نقطة التفتيش المنشورة في BF16. لتحميله في ذاكرة الوصول العشوائي (RAM) في BF16/FP16، تحتاج إلى تحديد `torch_dtype`:

```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1", torch_dtype=torch.bfloat16)
# يمكنك أيضًا استخدام torch_dtype=torch.float16
```

عند استخدام نصف الدقة، يمكنك تمكين تنفيذ [FlashAttention2](https://github.com/Dao-AILab/flash-attention) لكتل الاهتمام. لاستخدامه، تحتاج أيضًا إلى وجود النموذج على جهاز CUDA. نظرًا لأن النموذج في هذه الدقة كبير جدًا بحيث لا يمكنه الاندماج في وحدة معالجة رسومية (GPU) واحدة بسعة 80 جيجابايت، فستحتاج أيضًا إلى موازاته باستخدام [accelerate](https://huggingface.co/docs/accelerate/index):

```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto")
```

</details>

<details>

<summary><strong>تحميل النموذج بدقة 8 بت</strong></summary>

**باستخدام دقة 8 بت، من الممكن استيعاب أطوال تسلسل تصل إلى 140 ألفًا على وحدة معالجة رسومية (GPU) واحدة بسعة 80 جيجابايت.** يمكنك بسهولة تحويل النموذج إلى 8 بت باستخدام [bitsandbytes](https://huggingface.co/docs/bitsandbytes/index). للحفاظ على جودة النموذج، نوصي باستبعاد كتل Mamba من التقريب:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["mamba"])
model = AutoModelForCausalLM.from_pretrained(
"ai21labs/Jamba-v0.1", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", quantization_config=quantization_config
)
```

</details>

## JambaConfig

[[autodoc]] JambaConfig

## JambaModel

[[autodoc]] JambaModel

- forward

## JambaForCausalLM

[[autodoc]] JambaForCausalLM

- forward

## JambaForSequenceClassification

[[autodoc]] transformers.JambaForSequenceClassification

- forward