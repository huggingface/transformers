<!--Copyright 2024 فريق HuggingFace. جميع الحقوق محفوظة.

مرخص بموجب رخصة أباتشي، الإصدار 2.0 ("الرخصة")؛ لا يجوز لك استخدام هذا الملف إلا وفقًا
لرخصة. يمكنك الحصول على نسخة من الترخيص في

http://www.apache.org/licenses/LICENSE-2.0

ما لم يتطلب القانون المعمول به أو يتم الاتفاق عليه كتابيًا، يتم توزيع البرنامج تحت
الرخصة على أساس "كما هي" بدون ضمانات أو شروط من أي نوع، سواء كانت صريحة أو ضمنية. راجع الترخيص للحصول على
اللغة المحددة التي تحكم الأذونات والقيود بموجب الترخيص.

⚠️ لاحظ أن هذا الملف مكتوب بتنسيق Markdown ولكنه يحتوي على بناء جملة محدد لمولد المستندات (مشابه لـ MDX) والذي قد لا يكون
يتم عرضها بشكل صحيح في عارض Markdown الخاص بك.

-->

# HQQ 

ينفذ Half-Quadratic Quantization (HQQ) التكميم أثناء التنقل من خلال التحسين السريع والمتين. لا يتطلب بيانات المعايرة ويمكن استخدامه لتكميم أي نموذج.
يرجى الرجوع إلى <a href="https://github.com/mobiusml/hqq/">الحزمة الرسمية</a> لمزيد من التفاصيل.

بالنسبة للتثبيت، نوصي باستخدام النهج التالي للحصول على أحدث إصدار وبناء نواة CUDA المقابلة:
```
pip install hqq
```

لتكميم نموذج، تحتاج إلى إنشاء [`HqqConfig`]. هناك طريقتان للقيام بذلك:
``` Python
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

# الطريقة 1: ستستخدم جميع الطبقات الخطية نفس تكوين التكميم
quant_config = HqqConfig(nbits=8, group_size=64, quant_zero=False, quant_scale=False, axis=0) #يتم استخدام axis=0 بشكل افتراضي
```

``` Python
# الطريقة 2: ستستخدم كل طبقة خطية بنفس العلامة تكوين تكميم مخصص
q4_config = {'nbits':4, 'group_size':64, 'quant_zero':False, 'quant_scale':False}
q3_config = {'nbits':3, 'group_size':32, 'quant_zero':False, 'quant_scale':False}
quant_config  = HqqConfig(dynamic_config={
  'self_attn.q_proj':q4_config,
  'self_attn.k_proj':q4_config,
  'self_attn.v_proj':q4_config,
  'self_attn.o_proj':q4_config,

  'mlp.gate_proj':q3_config,
  'mlp.up_proj'  :q3_config,
  'mlp.down_proj':q3_config,
})
```

النهج الثاني مثير للاهتمام بشكل خاص لتكميم Mixture-of-Experts (MoEs) لأن الخبراء أقل تأثرًا بإعدادات التكميم المنخفضة.

بعد ذلك، قم ببساطة بتكميم النموذج كما يلي
``` Python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)
```

## وقت التشغيل الأمثل

يدعم HQQ واجهات برمجة تطبيقات خلفية مختلفة، بما في ذلك PyTorch النقي ونواة CUDA المخصصة لإلغاء التكميم. هذه الواجهات الخلفية مناسبة للجيل القديم من وحدات معالجة الرسومات (GPUs) وتدريب peft/QLoRA.
للحصول على استدلال أسرع، يدعم HQQ نواة 4-bit المدمجة (TorchAO وMarlin)، والتي تصل إلى 200 رمز/ثانية على 4090 واحد.
للحصول على مزيد من التفاصيل حول كيفية استخدام واجهات برمجة التطبيقات الخلفية، يرجى الرجوع إلى https://github.com/mobiusml/hqq/?tab=readme-ov-file#backend