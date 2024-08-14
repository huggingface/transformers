# AWQ

[Activation-aware Weight Quantization (AWQ)](https://hf.co/papers/2306.00978) لا تقوم بضغط جميع الأوزان في النموذج، وبدلاً من ذلك، تحافظ على نسبة صغيرة من الأوزان المهمة لأداء LLM. وهذا يقلل بشكل كبير من فقدان الضغط بحيث يمكنك تشغيل النماذج بدقة 4 بت دون أي تدهور في الأداء.

هناك العديد من المكتبات لضغط النماذج باستخدام خوارزمية AWQ، مثل [llm-awq](https://github.com/mit-han-lab/llm-awq) أو [autoawq](https://github.com/casper-hansen/AutoAWQ) أو [optimum-intel](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc). تدعم مكتبة Transformers تحميل النماذج المضغوطة باستخدام مكتبتي llm-awq وautoawq. سيُظهر لك هذا الدليل كيفية تحميل النماذج المضغوطة باستخدام مكتبة autoawq، ولكن العملية مماثلة للنماذج المضغوطة باستخدام llm-awq.

تأكد من تثبيت autoawq:

```bash
pip install autoawq
```

يمكن تحديد النماذج المضغوطة باستخدام AWQ عن طريق التحقق من سمة `quantization_config` في ملف [config.json](https://huggingface.co/TheBloke/zephyr-7B-alpha-AWQ/blob/main/config.json) للنموذج:

```json
{
  "_name_or_path": "/workspace/process/huggingfaceh4_zephyr-7b-alpha/source",
  "architectures": [
    "MistralForCausalLM"
  ],
  ...
  ...
  ...
  "quantization_config": {
    "quant_method": "awq",
    "zero_point": true,
    "group_size": 128,
    "bits": 4,
    "version": "gemm"
  }
}
```

يتم تحميل النموذج المضغوط باستخدام طريقة [`~PreTrainedModel.from_pretrained`]. إذا قمت بتحميل نموذجك على وحدة المعالجة المركزية (CPU)، فتأكد من نقله إلى جهاز GPU أولاً. استخدم معلمة `device_map` لتحديد مكان وضع النموذج:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TheBloke/zephyr-7B-alpha-AWQ"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")
```

يتم تلقائيًا تعيين الأوزان الأخرى إلى fp16 بشكل افتراضي عند تحميل نموذج مضغوط باستخدام AWQ لأسباب تتعلق بالأداء. إذا كنت تريد تحميل هذه الأوزان الأخرى بتنسيق مختلف، فاستخدم معلمة `torch_dtype`:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TheBloke/zephyr-7B-alpha-AWQ"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
```

يمكن أيضًا دمج ضغط AWQ مع [FlashAttention-2](../perf_infer_gpu_one#flashattention-2) لتسريع الاستنتاج بشكل أكبر:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-alpha-AWQ", attn_implementation="flash_attention_2", device_map="cuda:0")
```

## الوحدات المندمجة

تقدم الوحدات المندمجة (Fused modules) دقة وأداء محسنين، وهي مدعومة بشكل افتراضي لوحدات AWQ لمعماريتي [Llama](https://huggingface.co/meta-llama) و[Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)، ولكن يمكنك أيضًا دمج وحدات AWQ للمعماريات غير المدعومة.

<Tip warning={true}>

لا يمكن دمج الوحدات المندمجة مع تقنيات التحسين الأخرى مثل FlashAttention-2.

</Tip>

<hfoptions id="fuse">
<hfoption id="supported architectures">
<Tip warning={true}>

لا يمكن دمج الوحدات المندمجة مع تقنيات التحسين الأخرى مثل FlashAttention-2.

</Tip>

لتمكين الوحدات المندمجة للمعماريات المدعومة، قم بإنشاء تكوين [`AwqConfig`] وتعيين معلمتي `fuse_max_seq_len` و`do_fuse=True`. معلمة `fuse_max_seq_len` هي الطول الإجمالي للتسلسل، ويجب أن تشمل طول السياق وطول التوليد المتوقع. يمكنك تعيينه إلى قيمة أكبر ليكون آمنًا.

على سبيل المثال، لدمج وحدات AWQ لنموذج [TheBloke/Mistral-7B-OpenOrca-AWQ](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-AWQ).

```python
import torch
from transformers import AwqConfig, AutoModelForCausalLM

model_id = "TheBloke/Mistral-7B-OpenOrca-AWQ"

quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    do_fuse=True,
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config).to(0)
```

تم اختبار نموذج [TheBloke/Mistral-7B-OpenOrca-AWQ](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-AWQ) باستخدام `batch_size=1` مع الوحدات المندمجة وغير المندمجة.

<figcaption class="text-center text-gray-500 text-lg">وحدة غير مندمجة</figcaption>

|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)   |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:----------------|
|            1 |               32 |              32 |            60.0984 |           38.4537 | 4.50 GB (5.68%) |
|            1 |               64 |              64 |          1333.67   |           31.6604 | 4.50 GB (5.68%) |
|            1 |              128 |             128 |          2434.06   |           31.6272 | 4.50 GB (5.68%) |
|            1 |              256 |             256 |          3072.26   |           38.1731 | 4.50 GB (5.68%) |
|            1 |              512 |             512 |          3184.74   |           31.6819 | 4.59 GB (5.80%) |
|            1 |             1024 |            1024 |          3148.18   |           36.8031 | 4.81 GB (6.07%) |
|            1 |             2048 |            2048 |          2927.33   |           35.2676 | 5.73 GB (7.23%) |

<figcaption class="text-center text-gray-500 text-lg">وحدة مندمجة</figcaption>

|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)   |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:----------------|
|            1 |               32 |              32 |            81.4899 |           80.2569 | 4.00 GB (5.05%) |
|            1 |               64 |              64 |          1756.1    |          106.26   | 4.00 GB (5.05%) |
|            1 |              128 |             128 |          2479.32   |          105.631  | 4.00 GB (5.06%) |
|            1 |              256 |             256 |          1813.6    |           85.7485 | 4.01 GB (5.06%) |
|            1 |              512 |             512 |          2848.9    |           97.701  | 4.11 GB (5.19%) |
|            1 |             1024 |            1024 |          3044.35   |           87.7323 | 4.41 GB (5.57%) |
|            1 |             2048 |            2048 |          2715.11   |           89.4709 | 5.57 GB (7.04%) |

تم أيضًا اختبار سرعة وسرعة نقل الوحدات المندمجة وغير المندمجة باستخدام مكتبة [optimum-benchmark](https://github.com/huggingface/optimum-benchmark).

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_forward_memory_plot.png" alt="generate throughput per batch size" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">forward peak memory/batch size</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_generate_throughput_plot.png" alt="forward latency per batch size" />
    <figcaption class="mt-ladbach-text-sm text-gray-500">generate throughput/batch size</figcaption>
  </div>
</div>

</hfoption>
<hfoption id="unsupported architectures">

بالنسبة للمعماريات التي لا تدعم الوحدات المندمجة بعد، يجب عليك إنشاء خريطة دمج مخصصة لتحديد الوحدات التي تحتاج إلى الدمج باستخدام معلمة `modules_to_fuse`. على سبيل المثال، لدمج وحدات AWQ لنموذج [TheBloke/Yi-34B-AWQ](https://huggingface.co/TheBloke/Yi-34B-AWQ).

```python
import torch
from transformers import AwqConfig, AutoModelForCausalLM

model_id = "TheBloke/Yi-34B-AWQ"

quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    modules_to_fuse={
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "layernorm": ["ln1", "ln2", "norm"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "use_alibi": False,
        "num_attention_heads": 56,
        "num_key_value_heads": 8,
        "hidden_size": 7168
    }
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config).to(0)
```

يجب أن تشمل معلمة `modules_to_fuse` ما يلي:

- `"attention"`: أسماء طبقات الاهتمام التي سيتم دمجها بالترتيب التالي: query، key، value، وطبقة الإسقاط الإخراجي. إذا كنت لا تريد دمج هذه الطبقات، قم بتمرير قائمة فارغة.
- `"layernorm"`: أسماء جميع طبقات LayerNorm التي تريد استبدالها بطبقة LayerNorm مدمجة مخصصة. إذا كنت لا تريد دمج هذه الطبقات، قم بتمرير قائمة فارغة.
- `"mlp"`: أسماء طبقات MLP التي تريد دمجها في طبقة MLP واحدة بالترتيب التالي: (gate (dense، layer، post-attention) / up / down layers).
- `"use_alibi"`: إذا كان نموذجك يستخدم ALiBi positional embedding.
- `"num_attention_heads"`: عدد رؤوس الاهتمام.
- `"num_key_value_heads"`: عدد رؤوس القيمة الرئيسية التي يجب استخدامها لتنفيذ Query Attention مجمعة (GQA). إذا كان `num_key_value_heads=num_attention_heads`، فسيستخدم النموذج Multi Head Attention (MHA)، إذا كان `num_key_value_heads=1`، فسيستخدم النموذج Multi Query Attention (MQA)، وإلا سيتم استخدام GQA.
- `"hidden_size"`: بُعد التمثيلات المخفية.

</hfoption>
</hfoptions>



## دعم ExLlama-v2

تدعم الإصدارات الحديثة من `autoawq` نواة ExLlama-v2 لعمليات التعبئة فك التشفير بشكل أسرع. للبدء، قم أولاً بتثبيت أحدث إصدار من `autoawq` عن طريق تشغيل:

```bash
pip install git+https://github.com/casper-hansen/AutoAWQ.git
```

ابدأ بالمرور عبر تكوين `AwqConfig()` مع `version="exllama"`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

quantization_config = AwqConfig(version="exllama")

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    quantization_config=quantization_config,
    device_map="auto",
)

input_ids = torch.randint(0, 100, (1, 128), dtype=torch.long, device="cuda")
output = model(input_ids)
print(output.logits)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-AWQ")
input_ids = tokenizer.encode("How to make a cake", return_tensors="pt").to(model.device)
output = model.generate(input_ids, do_sample=True, max_length=50, pad_token_id=50256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

<Tip warning={true}>

ملاحظة: هذه الميزة مدعومة على معالجات الرسومات AMD.

</Tip>