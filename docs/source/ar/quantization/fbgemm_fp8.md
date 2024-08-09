<!--Copyright 2024 فريق HuggingFace. جميع الحقوق محفوظة.

مرخص بموجب رخصة أباتشي، الإصدار 2.0 ("الرخصة")؛ لا يجوز لك استخدام هذا الملف إلا وفقًا
لرخصة. يمكنك الحصول على نسخة من الترخيص في

http://www.apache.org/licenses/LICENSE-2.0

ما لم يتطلب القانون المعمول به أو يتفق عليه كتابةً، يتم توزيع البرنامج الموزع بموجب الرخصة "كما هو"
أساس، بدون ضمانات أو شروط من أي نوع، سواء كانت صريحة أو ضمنية. راجع الترخيص للحصول على
لغة محددة تحكم الأذونات والقيود بموجب الترخيص.

⚠️ لاحظ أن هذا الملف مكتوب بتنسيق Markdown ولكنه يحتوي على بناء جملة محدد لمولد المستندات (مشابه لـ MDX)
قد لا يتم عرضها بشكل صحيح في عارض Markdown الخاص بك.

-->

# FBGEMM FP8

مع طريقة FBGEMM FP8 quantization، يمكنك تحويل نموذجك إلى صيغة FP8 (W8A8):

- سيتم تحويل أوزان النموذج إلى 8 بت (FP8) لكل قناة.
- سيتم تحويل تنشيط النموذج إلى 8 بت (FP8) لكل رمز.

تعتمد هذه الطريقة على مكتبة FBGEMM [https://github.com/pytorch/FBGEMM] التي توفر عمليات فعالة لضرب المصفوفات ذات الدقة المنخفضة للدفعات الصغيرة، كما تدعم تقنيات تقليل فقدان الدقة مثل التحويل الكمي row-wise والتحويل الكمي outlier-aware.

> [!TIP]
> أنت بحاجة إلى وحدة معالجة رسومات GPU ذات قدرة حاسوبية >=9 (مثل H100)

قبل البدء، تأكد من تثبيت المكتبات التالية بأحدث إصداراتها:

```bash
pip install --upgrade accelerate fbgemm-gpu torch
```

إذا واجهتك مشكلات مع مكتبة fbgemm-gpu وtorch، فقد تحتاج إلى تثبيت الإصدار الليلي. يمكنك اتباع التعليمات [هنا](https://pytorch.org/FBGEMM/fbgemm_gpu-development/InstallationInstructions.html#fbgemm-gpu-install-libraries:~:text=found%20here.-,Install%20the%20FBGEMM_GPU%20Package,-Install%20through%20PyTorch).


```py
from transformers import FbgemmFp8Config, AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = FbgemmFp8Config()
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "ما هو عشاءنا الليلة؟"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

يمكن حفظ النموذج المحول باستخدام `saved_pretrained` وإعادة استخدامه مرة أخرى عبر `from_pretrained`.

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```