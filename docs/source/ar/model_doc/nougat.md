# Nougat

## نظرة عامة
اقتُرح نموذج Nougat في ورقة "Nougat: Neural Optical Understanding for Academic Documents" من قبل Lukas Blecher و Guillem Cucurull و Thomas Scialom و Robert Stojnic. يستخدم Nougat نفس البنية المعمارية لـ Donut، مما يعني أنه يستخدم مشفر محول الصور وفك تشفير محول النص التلقائي لترجمة ملفات PDF العلمية إلى Markdown، مما يسهل الوصول إليها.

وفيما يلي الملخص المستخلص من الورقة:

*تُخزن المعرفة العلمية في المقام الأول في الكتب والمجلات العلمية، وغالطًا ما تكون على شكل ملفات PDF. ومع ذلك، يؤدي تنسيق PDF إلى فقدان المعلومات الدلالية، خاصة بالنسبة للتعبيرات الرياضية. نقترح نموذج Nougat (Neural Optical Understanding for Academic Documents)، وهو نموذج محول بصري يقوم بمهمة التعرف البصري على الأحرف (OCR) لمعالجة المستندات العلمية إلى لغة ترميز، ونثبت فعالية نموذجنا على مجموعة بيانات جديدة من المستندات العلمية. ويقدم النهج المقترح حلاً واعدًا لتعزيز إمكانية الوصول إلى المعرفة العلمية في العصر الرقمي، من خلال سد الفجوة بين المستندات التي يمكن للإنسان قراءتها والنص الذي يمكن للآلة قراءته.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/nougat_architecture.jpg"
alt="drawing" width="600"/>

<small>نظرة عامة عالية المستوى على Nougat. مأخوذة من <a href="https://arxiv.org/abs/2308.13418">الورقة الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). ويمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/nougat).

## نصائح الاستخدام

- أسرع طريقة للبدء مع Nougat هي التحقق من [دفاتر الملاحظات التعليمية](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Nougat)، والتي توضح كيفية استخدام النموذج في وقت الاستدلال وكذلك الضبط الدقيق على البيانات المخصصة.

- يتم استخدام Nougat دائمًا ضمن إطار عمل [VisionEncoderDecoder](vision-encoder-decoder). النموذج مطابق لـ [Donut](donut) من حيث البنية المعمارية.

## الاستنتاج

يقبل نموذج [`VisionEncoderDecoder`] في Nougat الصور كمدخلات ويستخدم [`~generation.GenerationMixin.generate`] لتوليد النص تلقائيًا بناءً على صورة المدخلات.

تتولى فئة [`NougatImageProcessor`] معالجة الصورة المدخلة، بينما تقوم فئة [`NougatTokenizerFast`] بفك ترميز رموز الهدف المولدة إلى سلسلة الهدف. وتجمع فئة [`NougatProcessor`] بين فئتي [`NougatImageProcessor`] و [`NougatTokenizerFast`] في مثيل واحد لاستخراج ميزات المدخلات وفك ترميز رموز الهدف المتوقعة.

- نسخ المستندات خطوة بخطوة من ملفات PDF

```py
>>> from huggingface_hub import hf_hub_download
>>> import re
>>> from PIL import Image

>>> from transformers import NougatProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = NougatProcessor.from_pretrained("facebook/nougat-base")
>>> model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> # prepare PDF image for the model
>>> filepath = hf_hub_download(repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_paper.png", repo_type="dataset")
>>> image = Image.open(filepath)
>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> # generate transcription (here we only generate 30 tokens)
>>> outputs = model.generate(
...     pixel_values.to(device),
...     min_length=1,
...     max_new_tokens=30,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
... )

>>> sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
>>> sequence = processor.post_process_generation(sequence, fix_markdown=False)
>>> # note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
>>> print(repr(sequence))
'\n\n# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blecher\n\nCorrespondence to: lblecher@'
```

راجع [مركز النماذج](https://huggingface.co/models?filter=nougat) للبحث عن نقاط تفتيش Nougat.

<Tip>

النموذج مطابق لـ [Donut](donut) من حيث البنية المعمارية.

</Tip>

## NougatImageProcessor

[[autodoc]] NougatImageProcessor

- preprocess

## NougatTokenizerFast

[[autodoc]] NougatTokenizerFast

## NougatProcessor

[[autodoc]] NougatProcessor

- __call__

- from_pretrained

- save_pretrained

- batch_decode

- decode

- post_process_generation