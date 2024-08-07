# EETQ

تدعم مكتبة [EETQ](https://github.com/NetEase-FuXi/EETQ) التجزئة ذات الوزن فقط لكل قناة int8 لمعالجات NVIDIA GPU. وتُستمد أنوية GEMM و GEMV عالية الأداء من FasterTransformer و TensorRT-LLM. ولا يتطلب الأمر وجود مجموعة بيانات معايرة، كما لا يلزم إجراء التجزئة المسبقة لنموذجك. وعلاوة على ذلك، يكون تدهور الدقة طفيفًا بفضل التجزئة لكل قناة.

تأكد من تثبيت eetq من [صفحة الإصدار](https://github.com/NetEase-FuXi/EETQ/releases)

```
pip install --no-cache-dir https://github.com/NetEase-FuXi/EETQ/releases/download/v1.0.0/EETQ-1.0.0+cu121+torch2.1.2-cp310-cp310-linux_x86_64.whl
```

أو عبر كود المصدر https://github.com/NetEase-FuXi/EETQ. يتطلب EETQ قدرة CUDA <= 8.9 و >= 7.0

```
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
```

يمكن تجزئة نموذج غير مجزأ عبر "from_pretrained".

```py
from transformers import AutoModelForCausalLM, EetqConfig
path = "/path/to/model"
quantization_config = EetqConfig("int8")
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", quantization_config=quantization_config)
```

يمكن حفظ النموذج المجزأ عبر "save_pretrained" وإعادة استخدامه مرة أخرى عبر "from_pretrained".

```py
quant_path = "/path/to/save/quantized/model"
model.save_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
```