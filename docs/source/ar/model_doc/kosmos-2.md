# KOSMOS-2

## نظرة عامة
اقترح نموذج KOSMOS-2 في ورقة بحثية بعنوان "Kosmos-2: Grounding Multimodal Large Language Models to the World" من قبل Zhiliang Peng وآخرون.

KOSMOS-2 هو نموذج لغوي سببي قائم على محول، تم تدريبه باستخدام مهمة التنبؤ بالكلمة التالية على مجموعة بيانات واسعة النطاق من أزواج الصور والنصوص المترابطة. يتم تحويل الإحداثيات المكانية لصناديق الحدود في مجموعة البيانات إلى تسلسل من رموز المواقع، والتي يتم إلحاقها بنص الكيان المقابل (على سبيل المثال، "a snowman" متبوعًا بـ "<patch_index_0044><patch_index_0863>"). يشبه تنسيق البيانات "الروابط التشعبية" التي تربط مناطق الكائنات في صورة بنصها في التعليق التوضيحي المقابل.

ملخص الورقة البحثية هو كما يلي:

> "نقدم Kosmos-2، وهو نموذج لغة متعدد الوسائط (MLLM)، يمكّن قدرات جديدة لفهم أوصاف الكائنات (مثل صناديق الحدود) وربط النص بالعالم المرئي. وعلى وجه التحديد، فإننا نمثل تعبيرات الإشارة على أنها روابط في Markdown، أي "[نص التعليمة البرمجية](صناديق الحدود)"، حيث تكون أوصاف الكائنات عبارة عن تسلسلات من رموز المواقع. إلى جانب مجموعات البيانات متعددة الوسائط، نقوم ببناء بيانات كبيرة الحجم من أزواج الصور والنصوص المترابطة (تسمى GrIT) لتدريب النموذج. بالإضافة إلى القدرات الحالية لـ MLLMs (مثل إدراك الوسائط العامة، واتباع التعليمات، والتعلم في السياق)، يدمج Kosmos-2 قدرات الربط في التطبيقات النهائية. نقيّم Kosmos-2 على مجموعة واسعة من المهام، بما في ذلك (1) الربط متعدد الوسائط، مثل فهم تعبيرات الإشارة، وربط العبارات، (2) الإشارة متعددة الوسائط، مثل توليد تعبيرات الإشارة، (3) مهام اللغة الإدراكية، و (4) فهم اللغة وتوليد النصوص. يضع هذا العمل الأساس لتطوير الذكاء الاصطناعي التجسيدي ويلقي الضوء على التقارب الكبير بين اللغة، والإدراك متعدد الوسائط، والعمل، ونمذجة العالم، والتي تعد خطوة رئيسية نحو الذكاء الاصطناعي العام. الكود والنماذج المسبقة التدريب متوفرة في https://aka.ms/kosmos-2."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos_2_overview.jpg" alt="drawing" width="600"/>

<small>نظرة عامة على المهام التي يمكن أن يتعامل معها KOSMOS-2. مأخوذة من <a href="https://arxiv.org/abs/2306.14824">الورقة البحثية الأصلية</a>.</small>

## مثال

```python
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

>>> model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
>>> processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

>>> url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> prompt = "<grounding> An image of"

>>> inputs = processor(text=prompt, images=image, return_tensors="pt")

>>> generated_ids = model.generate(
...     pixel_values=inputs["pixel_values"],
...     input_ids=inputs["input_ids"],
...     attention_mask=inputs["attention_mask"],
...     image_embeds=None,
...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
...     use_cache=True,
...     max_new_tokens=64,
... )
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
>>> processed_Multiplier
'<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.'

>>> caption, entities = processor.post_process_generation(generated_text)
>>> caption
'An image of a snowman warming himself by a fire.'

>>> entities
[('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]
```

تمت المساهمة بهذا النموذج من قبل [Yih-Dar SHIEH](https://huggingface.co/ydshieh). يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/unilm/tree/master/kosmos-2).

## Kosmos2Config

[[autodoc]] Kosmos2Config

## Kosmos2ImageProcessor

## Kosmos2Processor

[[autodoc]] Kosmos2Processor

- __call__

## Kosmos2Model

[[autodoc]] Kosmos2Model

- forward

## Kosmos2ForConditionalGeneration

[[autodoc]] Kosmos2ForConditionalGeneration

- forward