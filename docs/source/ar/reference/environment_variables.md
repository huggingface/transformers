# متغيرات البيئة

## HF_ENABLE_PARALLEL_LOADING

يكون هذا الخيار معطّلًا افتراضيًا. يتيح تحميل أوزان النماذج المبنية على torch وsafetensor بشكل متوازٍ، ما قد يُقلّل وقت تحميل النماذج الكبيرة بشكل ملحوظ، وغالبًا ما يحقق تسارعًا يقارب ~50%.

يمكن ضبطه كسلسلة تساوي "false" أو "true". مثلًا: os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true".

على سبيل المثال، يمكن جعل facebook/opt-30b على خادوم AWS EC2 من نوع g4dn.metal يُحمَّل في ~30s عند تفعيل هذا الخيار مقارنةً بـ ~55s عند تعطيله.

يُنصح بالقيام بعملية قياس أداء (Profiling) قبل اعتماد هذا المتغير؛ إذ قد لا يحقق تسارعًا للنماذج الأصغر.

```py
import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"

from transformers import pipeline

model = pipeline(task="text-generation", model="facebook/opt-30b", device_map="auto")
```

## HF_PARALLEL_LOADING_WORKERS

يُحدّد عدد الخيوط (threads) التي ينبغي استخدامها عند تمكين التحميل المتوازي. القيمة الافتراضية هي 8.

إذا كان عدد الملفات الجاري تحميلها أقل من عدد الخيوط المحدّد، فسيتم إطلاق عدد من الخيوط يساوي عدد الملفات فعليًا.

على سبيل المثال: إذا حدّدت 8 عمّال (workers)، وكان هناك ملفّان فقط، فسيتم إطلاق عاملين اثنين فقط.

قم بالضبط والمواءمة بحسب حاجتك.

```py
import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "4"

from transformers import pipeline

model = pipeline(task="text-generation", model="facebook/opt-30b", device_map="auto")
```
