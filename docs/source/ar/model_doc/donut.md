# Donut

## نظرة عامة

اقتُرح نموذج Donut في ورقة بحثية بعنوان "محول فهم المستندات دون استخدام تقنية التعرف البصري على الحروف" من قبل جيووك كيم، وتيكغيو هونغ، وموونبين ييم، وجيونجيون نام، وجينييونغ بارك، وجينيونج ييم، وونسويك هوانج، وسانغدو يون، ودونغيون هان، وسيونغهيون بارك.

يتكون نموذج Donut من محول ترميز الصور ومحول ترميز النصوص التلقائي لإنجاز مهام فهم المستندات مثل تصنيف صور المستندات، وفهم النماذج، والإجابة على الأسئلة البصرية.

وفيما يلي الملخص المستخرج من الورقة البحثية:

*يعد فهم صور المستندات (مثل الفواتير) مهمة أساسية ولكنها صعبة لأنها تتطلب وظائف معقدة مثل قراءة النص وفهم المستند ككل. وتستعين الطرق الحالية لفهم المستندات المرئية بتقنية التعرف البصري على الحروف المتوفرة تجاريًا للقيام بمهمة قراءة النص، وتركز على مهمة الفهم باستخدام مخرجات التعرف البصري على الحروف. وعلى الرغم من أن هذه الطرق التي تعتمد على تقنية التعرف البصري على الحروف أظهرت أداءً واعدًا، إلا أنها تعاني من 1) ارتفاع التكاليف الحسابية لاستخدام تقنية التعرف البصري على الحروف؛ 2) عدم مرونة نماذج التعرف البصري على الحروف في اللغات أو أنواع المستندات؛ 3) انتشار أخطاء التعرف البصري على الحروف في العملية اللاحقة. ولمعالجة هذه القضايا، نقدم في هذه الورقة نموذجًا جديدًا لفهم المستندات المرئية دون استخدام تقنية التعرف البصري على الحروف يسمى Donut، وهو اختصار لـ "محول فهم المستندات". وكخطوة أولى في أبحاث فهم المستندات المرئية دون استخدام تقنية التعرف البصري على الحروف، نقترح بنية بسيطة (أي محول) مع هدف تدريب مسبق (أي دالة الخسارة المتقاطعة). ويتميز نموذج Donut ببساطته وفعاليته في نفس الوقت. ومن خلال التجارب والتحليلات المستفيضة، نثبت أن نموذج Donut البسيط لفهم المستندات المرئية دون استخدام تقنية التعرف البصري على الحروف يحقق أداءً متميزًا من حيث السرعة والدقة في العديد من مهام فهم المستندات المرئية. بالإضافة إلى ذلك، نقدم مولد بيانات تركيبية يساعد في مرونة التدريب المسبق للنموذج في مختلف اللغات والمجالات.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg" alt="drawing" width="600"/>

<small>نظرة عامة عالية المستوى لنموذج Donut. مأخوذة من <a href="https://arxiv.org/abs/2111.15664">الورقة البحثية الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). ويمكن العثور على الكود الأصلي [هنا](https://github.com/clovaai/donut).

## نصائح الاستخدام

- أسرع طريقة للبدء في استخدام نموذج Donut هي من خلال الاطلاع على [دفاتر الملاحظات التعليمية](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut)، والتي توضح كيفية استخدام النموذج في مرحلة الاستدلال وكذلك الضبط الدقيق على بيانات مخصصة.

- يتم استخدام نموذج Donut دائمًا ضمن إطار عمل [VisionEncoderDecoder](vision-encoder-decoder).

## أمثلة الاستدلال

يقبل نموذج [`VisionEncoderDecoder`] في نموذج Donut الصور كمدخلات ويستخدم [`~generation.GenerationMixin.generate`] لتوليد النص تلقائيًا بناءً على صورة المدخلات.

تتولى فئة [`DonutImageProcessor`] مسؤولية معالجة صورة المدخلات، ويقوم [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] بفك ترميز رموز الهدف المتوقعة إلى سلسلة الهدف. وتجمع فئة [`DonutProcessor`] بين [`DonutImageProcessor`] و [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`] في مثيل واحد لاستخراج ميزات المدخلات وفك ترميز رموز الهدف المتوقعة.

- خطوة بخطوة لتصنيف صورة المستند:

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # تحميل صورة المستند
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[1]["image"]

>>> # إعداد مدخلات فك الترميز
>>> task_prompt = "<s_rvlcdip>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # إزالة رمز بدء المهمة الأول
>>> print(processor.token2json(sequence))
{'class': 'advertisement'}
```

- خطوة بخطوة لفهم المستند:

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # تحميل صورة المستند
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[2]["image"]

>>> # إعداد مدخلات فك الترميز
>>> task_prompt = "<s_cord-v2>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # إزالة رمز بدء المهمة الأول
>>> print(processor.token2json(sequence))
{'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
```

- خطوة بخطوة للإجابة على الأسئلة البصرية في المستند (DocVQA):

```py
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # تحميل صورة المستند من مجموعة بيانات DocVQA
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[0]["image"]

>>> # إعداد مدخلات فك الترميز
>>> task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
>>> question = "When is the coffee break?"
>>> prompt = task_prompt.replace("{user_input}", question)
>>> decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # إزالة رمز بدء المهمة الأول
>>> print(processor.token2json(sequence))
{'question': 'When is the coffee break?', 'answer': '11-14 to 11:39 a.m.'}
```

يمكنك الاطلاع على [مركز النماذج](https://huggingface.co/models?filter=donut) للبحث عن نقاط تفتيش لنموذج Donut.

## التدريب

يمكن الرجوع إلى [دفاتر الملاحظات التعليمية](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut).

## DonutSwinConfig

[[autodoc]] DonutSwinConfig

## DonutImageProcessor

[[autodoc]] DonutImageProcessor

- preprocess

## DonutFeatureExtractor

[[autodoc]] DonutFeatureExtractor

- __call__

## DonutProcessor

[[autodoc]] DonutProcessor

- __call__

- from_pretrained

- save_pretrained

- batch_decode

- decode

## DonutSwinModel

[[autodoc]] DonutSwinModel

- forward