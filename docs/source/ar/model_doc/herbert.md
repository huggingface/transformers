# HerBERT

## نظرة عامة
تم اقتراح نموذج HerBERT في الورقة البحثية [KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://www.aclweb.org/anthology/2020.acl-main.111.pdf) بواسطة Piotr Rybak, Robert Mroczkowski, Janusz Tracz, و Ireneusz Gawlik. وهو نموذج لغة يعتمد على BERT تم تدريبه على نصوص اللغة البولندية باستخدام هدف MLM مع التمويه الديناميكي للكلمات الكاملة.

الملخص المستخرج من الورقة البحثية هو كما يلي:

*في السنوات الأخيرة، حققت سلسلة من النماذج القائمة على محول تحسينات كبيرة في مهام فهم اللغة الطبيعية العامة (NLU). إن وتيرة البحث السريعة هذه ما كانت لتكون ممكنة بدون معايير NLU العامة، والتي تسمح بمقارنة عادلة للطرق المقترحة. ومع ذلك، فإن هذه المعايير متاحة لعدد قليل من اللغات فقط. وللتخفيف من هذه المشكلة، نقدم معيارًا متعدد المهام شاملًا لفهم اللغة البولندية، مصحوبًا بلوحة قيادة عبر الإنترنت. ويتكون من مجموعة متنوعة من المهام، تم تبنيها من مجموعات البيانات الحالية للتعرف على الكيانات المسماة، والأسئلة والأجوبة، والاستدلال النصي، وغيرها. كما نقدم مهمة جديدة لتحليل المشاعر لمجال التجارة الإلكترونية، تسمى Allegro Reviews (AR). ولضمان وجود مخطط تقييم مشترك وتشجيع النماذج التي تعمم على مهام NLU المختلفة، يتضمن المعيار مجموعات بيانات من مجالات وتطبيقات مختلفة. بالإضافة إلى ذلك، نقدم HerBERT، وهو نموذج محول تم تدريبه خصيصًا للغة البولندية، والذي يحقق أفضل أداء متوسط ويحصل على أفضل النتائج في ثلاث مهام من أصل تسع مهام. وأخيرًا، نقدم تقييمًا شاملاً، بما في ذلك العديد من خطوط الأساس القياسية والنماذج متعددة اللغات المقترحة مؤخرًا والقائمة على المحول.*

تمت المساهمة بهذا النموذج من قبل [rmroczkowski](https://huggingface.co/rmroczkowski). يمكن العثور على الكود الأصلي [هنا](https://github.com/allegro/HerBERT).

## مثال على الاستخدام

```python
>>> from transformers import HerbertTokenizer, RobertaModel

>>> tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

>>> encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors="pt")
>>> outputs = model(encoded_input)

>>> # يمكن أيضًا تحميل HerBERT باستخدام AutoTokenizer وAutoModel:
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
```

<Tip>

يتماثل تنفيذ Herbert مع تنفيذ `BERT` باستثناء طريقة التمييز. راجع [توثيق BERT](bert) للحصول على مرجع API وأمثلة.

</Tip>

## HerbertTokenizer

[[autodoc]] HerbertTokenizer

## HerbertTokenizerFast

[[autodoc]] HerbertTokenizerFast