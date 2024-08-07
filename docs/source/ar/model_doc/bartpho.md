# BARTpho

## نظرة عامة

اقترح نجوين لونغ تران، ودوونغ مينه لي، ودات كوك نجوين نموذج BARTpho في ورقة بحثية بعنوان: "BARTpho: نماذج تسلسل إلى تسلسل مسبقة التدريب للغة الفيتنامية".

ويوضح الملخص المستخرج من الورقة ما يلي:

"نقدم BARTpho بإصدارين - BARTpho_word وBARTpho_syllable - أول نموذج تسلسل إلى تسلسل أحادي اللغة واسع النطاق مُدرب مسبقًا للغة الفيتنامية. يستخدم نموذجنا BARTpho البنية والطريقة المُدربة مسبقًا "large" لنموذج BART للتسلسل إلى تسلسل لإزالة التشويش، مما يجعله مناسبًا بشكل خاص لمهام NLP التوليدية. وتظهر التجارب على مهمة فرعية لتلخيص النص الفيتنامي أن BARTpho الخاص بنا يتفوق على خط الأساس القوي mBART ويحسن الحالة الراهنة في كل من التقييمات الآلية والبشرية. ونقدم BARTpho لتسهيل الأبحاث والتطبيقات المستقبلية لمهام NLP التوليدية باللغة الفيتنامية."

ساهم في هذا النموذج [dqnguyen](https://huggingface.co/dqnguyen). ويمكن العثور على الكود الأصلي [هنا](https://github.com/VinAIResearch/BARTpho).

## مثال على الاستخدام

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")

>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

>>> line = "Chúng tôi là những nghiên cứu viên."

>>> input_ids = tokenizer(line, return_tensors="pt")

>>> with torch.no_grad():
...     features = bartpho(**input_ids) # مخرجات النماذج هي الآن عبارة عن مجموعات

>>> # مع TensorFlow 2.0+:
>>> from transformers import TFAutoModel

>>> bartpho = TFAutoModel.from_pretrained("vinai/bartpho-syllable")
>>> input_ids = tokenizer(line, return_tensors="tf")
>>> features = bartpho(**input_ids)
```

## نصائح الاستخدام

- على غرار mBART، يستخدم BARTpho البنية "large" من BART مع طبقة إضافية للتحويل الطبيعي على رأس كل من المشفر وفك المشفر. لذلك، يجب ضبط أمثلة الاستخدام في [توثيق BART](bart)، عند التكيف مع الاستخدام مع BARTpho، عن طريق استبدال الفئات المتخصصة في BART بالفئات المناظرة المتخصصة في mBART. على سبيل المثال:

```python
>>> from transformers import MBartForConditionalGeneration

>>> bartpho = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
>>> TXT = "Chúng tôi là <mask> nghiên cứu viên."
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = bartpho(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)
>>> print(tokenizer.decode(predictions).split())
```

- هذا التنفيذ مخصص للتشفير فقط: "monolingual_vocab_file" يتكون من الأنواع المتخصصة باللغة الفيتنامية المستخرجة من نموذج SentencePiece المُدرب مسبقًا "vocab_file" والمتاح من XLM-RoBERTa متعدد اللغات. ويمكن للغات الأخرى، إذا استخدمت ملف "vocab_file" هذا لتقسيم الكلمات إلى أجزاء أصغر متعددة اللغات، أن تعيد استخدام BartphoTokenizer مع ملف "monolingual_vocab_file" الخاص بها والمتخصص بلغتها.

## BartphoTokenizer

[[autodoc]] BartphoTokenizer