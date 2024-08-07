# ByT5

## نظرة عامة

تم تقديم نموذج ByT5 في ورقة بحثية بعنوان: [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) بواسطة Linting Xue و Aditya Barua و Noah Constant و Rami Al-Rfou و Sharan Narang و Mihir Kale و Adam Roberts و Colin Raffel.

وفيما يلي الملخص المستخرج من الورقة البحثية:

*تعمل معظم نماذج اللغة المُدربة مسبقًا على نطاق واسع على تسلسلات من الرموز التي تتوافق مع وحدات الكلمات أو الوحدات الفرعية للكلمات. يتطلب ترميز النص كتسلسل من الرموز محللًا نحويًا، والذي يتم إنشاؤه عادةً كأثر مستقل عن النموذج. وللنّماذج الخالية من الرموز والتي تعمل مباشرةً على النص الخام (البايتات أو الأحرف) العديد من المزايا: فيمكنها معالجة النص بأي لغة بشكل افتراضي، وهي أكثر مقاومة للضوضاء، وتقلل من الديون التقنية عن طريق إزالة أنابيب المعالجة المسبقة للنص المعقدة والمعرضة للأخطاء. وبما أن تسلسلات البايتات أو الأحرف أطول من تسلسلات الرموز، فقد قدم العمل السابق حول النماذج الخالية من الرموز في كثير من الأحيان تصميمات معمارية جديدة للنماذج تهدف إلى استهلاك تكلفة العمل مباشرةً على النص الخام. وفي هذه الورقة، نُظهر أنه يمكن استخدام بنية Transformer القياسية مع تعديلات طفيفة لمعالجة تسلسلات البايتات. ونقوم بتحديد المقايضات بدقة من حيث عدد المعلمات، وFLOPs التدريب، وسرعة الاستدلال، ونُظهر أن النماذج على مستوى البايت تنافسية مع نظيراتها على مستوى الرموز. كما نُظهر أن النماذج على مستوى البايت أكثر مقاومة للضوضاء بشكل كبير، وتؤدي أداءً أفضل في المهام التي تتأثر بالإملاء والنطق. وكجزء من مساهمتنا، نقوم بإطلاق مجموعة جديدة من النماذج المُدربة مسبقًا على مستوى البايت والقائمة على بنية Transformer، بالإضافة إلى جميع الرموز والبيانات المستخدمة في تجاربنا.*

تمت المساهمة بهذا النموذج من قبل [patrickvonplaten](https://huggingface.co/patrickvonplaten). ويمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/byt5).

<Tip>

تستند بنية ByT5 إلى نموذج T5v1.1، راجع [صفحة وثائق T5v1.1](t5v1.1) للمرجع الخاص بواجهة برمجة التطبيقات (API). ويختلفان فقط في كيفية إعداد المدخلات للنموذج، راجع أمثلة الكود أدناه.

</Tip>

نظرًا لأن ByT5 تم تدريبه بشكل غير خاضع للإشراف، فلا توجد ميزة حقيقية لاستخدام بادئة المهمة أثناء الضبط الدقيق أحادي المهمة. إذا كنت تقوم بالضبط الدقيق متعدد المهام، فيجب عليك استخدام بادئة.

## مثال على الاستخدام

يعمل ByT5 على بايتات UTF-8 الخام، لذلك يمكن استخدامه بدون محلل نحوي:

```python
>>> from transformers import T5ForConditionalGeneration
>>> import torch

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

>>> num_special_tokens = 3
>>> # Model has 3 special tokens which take up the input ids 0,1,2 of ByT5.
>>> # => Need to shift utf-8 character encodings by 3 before passing ids to model.

>>> input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens

>>> labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

>>> loss = model(input_ids, labels=labels).loss
>>> loss.item()
2.66
```

ومع ذلك، يُنصح باستخدام المحلل النحوي للضبط الدقيق والتنبؤ بالدفعات:

```python
>>> from transformers import T5ForConditionalGeneration, AutoTokenizer

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

>>> model_inputs = tokenizer(
...     ["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt"
... )
>>> labels_dict = tokenizer(
...     ["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt"
... )
>>> labels = labels_dict.input_ids

>>> loss = model(**model_inputs, labels=labels).loss
>>> loss.item()
17.9
```

على غرار [T5](t5)، تم تدريب ByT5 على مهمة إخفاء القناع. ومع ذلك، نظرًا لأن النموذج يعمل مباشرةً على الأحرف، فإن مهمة التدريب المسبق مختلفة قليلًا. دعونا نقوم بإفساد بعض أحرف الجملة التالية: `"The dog chases a ball in the park."` ونسأل ByT5 أن يتنبأ بها من أجلنا.

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")

>>> input_ids_prompt = "The dog chases a ball in the park."
>>> input_ids = tokenizer(input_ids_prompt).input_ids

>>> # Note that we cannot add "{extra_id_...}" to the string directly
>>> # as the Byte tokenizer would incorrectly merge the tokens
>>> # For ByT5, we need to work directly on the character level
>>> # Contrary to T5, ByT5 does not use sentinel tokens for masking, but instead
>>> # uses final utf character ids.
>>> # UTF-8 is represented by 8 bits and ByT5 has 3 special tokens.
>>> # => There are 2**8+2 = 259 input ids and mask tokens count down from index 258.
>>> # => mask to "The dog [258]a ball [257]park."

>>> input_ids = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
>>> input_ids
tensor([[ 87, 107, 104,  35, 103, 114, 106,  35, 258,  35, 100,  35, 101, 100, 111, 111, 257,  35, 115, 100, 111, 110,  49,   1]])

>>> # ByT5 produces only one char at a time so we need to produce many more output characters here -> set `max_length=100`.
>>> output_ids = model.generate(input_ids, max_length=100)[0].tolist()
>>> output_ids
[0, 258, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118, 257,  35, 108, 113,  35, 119, 107, 104,  35, 103, 108, 118, 102, 114, 256, 108, 113,  35, 119, 107, 104, 35, 115, 100, 117, 110,  49,  35,  87, 107, 104,  35, 103, 114, 106, 35, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118,  35, 100,  35, 101, 100, 111, 111,  35, 108, 113, 255,  35, 108, 113,  35, 119, 107, 104,  35, 115, 100, 117, 110,  49]

>>> # ^- Note how 258 descends to 257, 256, 255

>>> # Now we need to split on the sentinel tokens, let's write a short loop for this
>>> output_ids_list = []
>>> start_token = 0
>>> sentinel_token = 258
>>> while sentinel_token in output_ids:
...     split_idx = output_ids.index(sentinel_token)
...     output_ids_list.append(output_ids[start_token:split_idx])
...     start_token = split_idx
...     sentinel_token -= 1

>>> output_ids_list.append(output_ids[start_token:])
>>> output_string = tokenizer.batch_decode(output_ids_list)
>>> output_string
['<pad>', 'is the one who does', ' in the disco', 'in the park. The dog is the one who does a ball in', ' in the park.']
```

## ByT5Tokenizer

[[autodoc]] ByT5Tokenizer

راجع [`ByT5Tokenizer`] للحصول على جميع التفاصيل.