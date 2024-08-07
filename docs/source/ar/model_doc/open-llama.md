# Open-Llama

**ملاحظة هامة:**
هذا النموذج في وضع الصيانة فقط، ولن نقبل أي طلبات سحب (PRs) جديدة لتعديل شفرته.
إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعمه: v4.31.0. يمكنك القيام بذلك عن طريق تشغيل الأمر التالي:

```
pip install -U transformers==4.31.0
```

**ملاحظة:** يختلف هذا النموذج عن [نماذج OpenLLaMA](https://huggingface.co/models?search=openllama) على Hugging Face Hub، والتي تستخدم بشكل أساسي [بنية LLaMA](llama).

## نظرة عامة
اقترح نموذج Open-Llama في مشروع Open-Llama مفتوح المصدر من قبل مطور المجتمع s-JoL.
يستند النموذج بشكل أساسي إلى LLaMA مع بعض التعديلات، حيث يدمج الانتباه الكفء من حيث الذاكرة من Xformers، والتضمين المستقر من Bloom، والتضمين المشترك للإدخال والإخراج من PaLM.
كما أنه مدرب مسبقًا على كل من اللغتين الصينية والإنجليزية، مما يمنحه أداءً أفضل في مهام اللغة الصينية.

ساهم بهذا النموذج [s-JoL](https://huggingface.co/s-JoL).
تم إصدار الشفرة الأصلية على GitHub بواسطة [s-JoL](https://github.com/s-JoL)، ولكن تمت إزالتها الآن.

## OpenLlamaConfig
[[autodoc]] OpenLlamaConfig

## OpenLlamaModel
[[autodoc]] OpenLlamaModel

- forward

## OpenLlamaForCausalLM
[[autodoc]] OpenLlamaForCausalLM

- forward

## OpenLlamaForSequenceClassification
[[autodoc]] OpenLlamaForSequenceClassification

- forward