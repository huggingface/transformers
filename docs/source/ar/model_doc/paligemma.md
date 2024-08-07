# PaliGemma

## نظرة عامة

اقترح نموذج PaliGemma في [PaliGemma – Google's Cutting-Edge Open Vision Language Model](https://huggingface.co/blog/paligemma) بواسطة Google. إنه نموذج للرؤية اللغوية يبلغ حجمه 3B، ويتكون من [SigLIP](siglip) encoder للرؤية وفك تشفير اللغة [Gemma](gemma) متصلين بواسطة إسقاط خطي متعدد الوسائط. ويقطع الصورة إلى عدد ثابت من رموز VIT ويضيفها إلى موجه اختياري. ومن خصوصياته أن النموذج يستخدم انتباه الكتلة الكاملة على جميع رموز الصورة بالإضافة إلى رموز النص المدخلة. وهو متوفر بثلاثة دقات، 224x224 و448x448 و896x896 مع 3 نماذج أساسية، و55 نسخة مُعدّلة مسبقًا لمهمام مختلفة، ونموذجين مختلطين.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/paligemma_arch.png"
alt="drawing" width="600"/>

<small> هندسة PaliGemma. مأخوذة من <a href="https://huggingface.co/blog/paligemma">منشور المدونة.</a> </small>

تمت المساهمة بهذا النموذج بواسطة [Molbap](https://huggingface.co/Molbap).

## نصائح الاستخدام

يمكن إجراء الاستنتاج باستخدام PaliGemma على النحو التالي:

```python
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "What is on the flower?"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(prompt, raw_image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
```

- لا يقصد بـ PaliGemma الاستخدام المحادثي، وهو يعمل بشكل أفضل عند الضبط الدقيق لحالة استخدام محددة. بعض المهام النهائية التي يمكن ضبط PaliGemma الدقيق لها تشمل إنشاء تعليقات توضيحية للصور، والإجابة على الأسئلة المرئية (VQA)، وكشف الأجسام، وتحديد أجزاء الكلام، وفهم المستندات.

- يمكن استخدام `PaliGemmaProcessor` لإعداد الصور والنص والعلامات الاختيارية للنموذج. عند الضبط الدقيق لنموذج PaliGemma، يمكن تمرير وسيط `suffix` إلى المعالج الذي يقوم بإنشاء `labels` للنموذج:

```python
prompt = "What is on the flower?"
answer = "a bee"
inputs = processor(text=prompt, images=raw_image, suffix=answer, return_tensors="pt")
```

## الموارد

قائمة بموارد Hugging Face الرسمية والمجتمعية (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام PaliGemma. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب Pull Request وسنراجعه! ويفضل أن يُظهر المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

- يمكن العثور على منشور مدونة يقدم جميع ميزات PaliGemma [هنا](https://huggingface.co/blog/paligemma).

- يمكن العثور على دفاتر الملاحظات التوضيحية حول كيفية الضبط الدقيق لـ PaliGemma لـ VQA باستخدام واجهة برمجة تطبيقات Trainer إلى جانب الاستنتاج [هنا](https://github.com/huggingface/notebooks/tree/main/examples/paligemma).

- يمكن العثور على دفاتر الملاحظات التوضيحية حول كيفية الضبط الدقيق لـ PaliGemma على مجموعة بيانات مخصصة (صورة الإيصال -> JSON) إلى جانب الاستنتاج [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma). 🌎

## PaliGemmaConfig

[[autodoc]] PaliGemmaConfig

## PaliGemmaProcessor

[[autodoc]] PaliGemmaProcessor

## PaliGemmaForConditionalGeneration

[[autodoc]] PaliGemmaForConditionalGeneration

- forward