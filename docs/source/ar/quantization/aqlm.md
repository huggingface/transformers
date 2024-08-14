# AQLM

> [!TIP]
> جرب AQLM على [Google Colab](https://colab.research.google.com/drive/1-xZmBRXT5Fm3Ghn4Mwa2KRypORXb855X?usp=sharing)!

Additive Quantization of Language Models ([AQLM](https://arxiv.org/abs/2401.06118)) هي طريقة لضغط نماذج اللغة الكبيرة. فهو يقوم بضغط العديد من الأوزان معًا ويستفيد من أوجه الترابط بينها. يمثل AQLM مجموعات من 8-16 وزنًا على أنها مجموع العديد من الرموز الشفرية.

يتم تحقيق دعم الاستدلال لـ AQLM في مكتبة `aqlm`. تأكد من تثبيته لتشغيل النماذج (ملاحظة: تعمل aqlm فقط مع python>=3.10):

```bash
pip install aqlm[gpu,cpu]
```

توفر المكتبة نوى فعالة لكل من GPU وCPU الاستدلال والتدريب.

يمكن العثور على التعليمات حول كيفية ضغط النماذج بنفسك، بالإضافة إلى جميع التعليمات البرمجية ذات الصلة، في مستودع GitHub [المناسب](https://github.com/Vahe1994/AQLM). لتشغيل نماذج AQLM، قم ببساطة بتحميل نموذج تمت ضغطه باستخدام AQLM:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf",
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf")
```

## PEFT

بدءًا من الإصدار `aqlm 1.0.2`، يدعم AQLM الضبط الدقيق الفعال للبارامترات في شكل [LoRA](https://huggingface.co/docs/peft/package_reference/lora) المدمج في مكتبة [PEFT](https://huggingface.co/blog/peft).

## تكوينات AQLM

تختلف إعدادات ضغط AQLM بشكل أساسي في عدد كتب الرموز المستخدمة بالإضافة إلى أحجام كتب الرموز بالبتات. فيما يلي أكثر الإعدادات شيوعًا، بالإضافة إلى نوى الاستدلال التي تدعمها:

| Kernel | عدد كتب الرموز | حجم كتاب الرموز، بت | التدوين | الدقة | تسريع | استدلال GPU سريع | استدلال CPU سريع |
|---|---------------------|---------------------|----------|-------------|-------------|--------------------|--------------------|
| Triton | K                   | N                  | KxN     | -        | حتى ~0.7x | ✅                  | ❌                  |
| CUDA | 1                   | 16                  | 1x16     | الأفضل        | حتى ~1.3x | ✅                  | ❌                  |
| CUDA | 2                   | 8                   | 2x8      | جيد          | حتى ~3.0x | ✅                  | ❌                  |
| Numba | K                   | 8                   | Kx8      | جيد        | حتى ~4.0x | ❌                  | ✅                  |