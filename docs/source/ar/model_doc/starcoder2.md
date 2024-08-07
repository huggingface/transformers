# StarCoder2

## نظرة عامة

StarCoder2 هي عائلة من شهادات LLM المفتوحة للتعليمات البرمجية وتأتي بثلاثة أحجام مختلفة مع معلمات 3B و7B و15B. تم تدريب نموذج StarCoder2-15B الرائد على أكثر من 4 تريليون رمز وأكثر من 600 لغة برمجة من The Stack v2. تستخدم جميع النماذج انتباه الاستعلام المجمع، وهي نافذة سياق مكونة من 16,384 رمزًا مميزًا مع نافذة انزلاقية تبلغ 4,096 رمزًا مميزًا، وتم تدريبها باستخدام هدف ملء الوسط. تم إصدار النماذج مع الورقة البحثية [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/abs/2402.19173) بقلم أنطون لوزكوف، وريموند لي، ولبنى بن علال، وفيديريكو كاسانو، وجويل لامي- بوارييه، نعمان تازي، آو تانغ، دميترو بيكتار، جياوي ليو، يوشيانغ وي، تيانيانغ ليو، ماكس تيان، دينيس كوسيتكوف، آرثر زوكر، يونس بلكادا، زيجيان وانغ، تشيان ليو، ديمتري أبوخانوف، إندرانيل بول، تشوانغ لي، وين دينغ لي، ميجان ريسدال، جيا لي، جيان تشو، تيري يو تشو، إيفجيني زيلتونوجسكي، ني أوساي أوساي ديد، وينهاو يو، لوكاس كراوس، نامان جاين، ييشوان سو، شوانلي هي، مانان داي، إدواردو أباتي، ييكون تشاي، نيكلاس مونيجوف، شيانغرو تانغ، مهتشام أوبلوكولوف، كريستوفر أكيكي، مارك مارون، تشينغهاو مو، مايانك ميشرا، أليكس جو، بينيوان هوي، تري داو، أرميل زيباز، أوليفييه ديهيني، نيكولاس باتري، كانوين شو، جوليان ماكولي، هان هو، تورستن شولاك، سيباستيان باكيه ، جينيفر روبنسون، كارولين جين أندرسون، نيكولا تشابادوس، مصطفى باتواري، نيما تاجبخش، ياسين جيرنيت، كارلوس مونيوز فيرانديس، لينغمينغ تشانغ، شون هيوز، توماس وولف، أرجون جوها، لياندرو فون ويرا، وهارم دي فريس.

ملخص الورقة هو ما يلي:

> يقدم مشروع BigCode، وهو عبارة عن تعاون علمي مفتوح يركز على التطوير المسؤول لنماذج اللغات الكبيرة للبرمجة (Code LLMs)، StarCoder2. بالشراكة مع Software Heritage (SWH)، قمنا ببناء The Stack v2 على رأس المشاعات الرقمية لأرشيف التعليمات البرمجية المصدر الخاص بهم. إلى جانب مستودعات SWH التي تغطي 619 لغة برمجة، نختار بعناية مصادر بيانات أخرى عالية الجودة، مثل طلبات سحب GitHub، ودفاتر ملاحظات Kaggle، ووثائق التعليمات البرمجية. وينتج عن ذلك مجموعة تدريب أكبر بأربع مرات من مجموعة بيانات StarCoder الأولى. نقوم بتدريب نماذج StarCoder2 بمعلمات 3B و7B و15B على 3.3 إلى 4.3 تريليون رمز ونقوم بتقييمها بدقة بناءً على مجموعة شاملة من معايير Code LLM. لقد وجدنا أن نموذجنا الصغير، StarCoder2-3B، يتفوق في الأداء على Code LLMs الأخرى ذات الحجم المماثل في معظم المعايير، ويتفوق أيضًا على StarCoderBase-15B. يتفوق طرازنا الكبير، StarCoder2- 15B، بشكل كبير على النماذج الأخرى ذات الحجم المماثل. بالإضافة إلى ذلك، فهو يطابق أو يتفوق على CodeLlama-34B، وهو نموذج يزيد حجمه عن ضعف حجمه. على الرغم من أن DeepSeekCoder- 33B هو النموذج الأفضل أداءً في إكمال التعليمات البرمجية للغات عالية الموارد، إلا أننا نجد أن StarCoder2-15B يتفوق عليه في معايير الرياضيات واستدلال التعليمات البرمجية، بالإضافة إلى العديد من اللغات منخفضة الموارد. نحن نجعل أوزان النماذج متاحة بموجب ترخيص OpenRAIL ونضمن الشفافية الكاملة فيما يتعلق ببيانات التدريب من خلال إصدار معرفات SoftWare Heritage المستمرة (SWHIDs) لبيانات كود المصدر.
## رخصة

تم ترخيص النماذج بموجب [اتفاقية ترخيص BigCode OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement).

## نصائح الاستخدام

يمكن العثور على نماذج StarCoder2 في [HuggingFace hub](https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a). يمكنك العثور على بعض الأمثلة للاستدلال والضبط الدقيق في [GitHub repo] الخاص بـ [StarCoder2](https://github.com/bigcode-project/starcoder2).

يمكن تنزيل نقاط التفتيش الجاهزة للاستخدام هذه واستخدامها عبر HuggingFace Hub:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-7b", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")

>>> prompt = "def print_hello_world():"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
>>> tokenizer.batch_decode(generated_ids)[0]
'def print_hello_world():\n    print("Hello World!")\n\ndef print'
```

## Starcoder2Config

[[autodoc]] Starcoder2Config

## Starcoder2Model

[[autodoc]] Starcoder2Model
    - forward

## Starcoder2ForCausalLM

[[autodoc]] Starcoder2ForCausalLM
    - forward

## Starcoder2ForSequenceClassification

[[autodoc]] Starcoder2ForSequenceClassification
    - forward

## Starcoder2ForTokenClassification

[[autodoc]] Starcoder2ForTokenClassification
    - forward