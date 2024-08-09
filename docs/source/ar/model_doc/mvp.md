# MVP

## نظرة عامة
تم اقتراح نموذج MVP في [MVP: Multi-task Supervised Pre-training for Natural Language Generation](https://arxiv.org/abs/2206.12131) بواسطة Tianyi Tang و Junyi Li و Wayne Xin Zhao و Ji-Rong Wen.

وفقًا للملخص،

- يتبع MVP بنية Transformer encoder-decoder القياسية.
- يتم التدريب المسبق لـ MVP باستخدام مجموعات بيانات موسومة.
- يحتوي MVP أيضًا على موجهات ناعمة خاصة بمهام معينة لتحفيز قدرة النموذج على أداء مهمة معينة.
- تم تصميم MVP خصيصًا لتوليد اللغة الطبيعية ويمكن تكييفه مع مجموعة واسعة من مهام التوليد، بما في ذلك على سبيل المثال لا الحصر، الملخص، وتوليد النص من البيانات، ونظام الحوار المفتوح، وتوليد القصص، والأسئلة والأجوبة، وتوليد الأسئلة، ونظام الحوار الموجه للمهمة، وتوليد الحس السليم، وتوليد إعادة الصياغة، ونقل أسلوب النص، وتبسيط النص. يمكن أيضًا تكييف نموذجنا مع مهام فهم اللغة الطبيعية مثل تصنيف التسلسلات والإجابة عن الأسئلة (الاستخراجية).

تمت المساهمة بهذا النموذج بواسطة [Tianyi Tang](https://huggingface.co/StevenTang). يمكن العثور على المعلومات والتعليمات التفصيلية [هنا](https://github.com/RUCAIBox/MVP).

## نصائح الاستخدام

- لقد أصدرنا سلسلة من النماذج [هنا](https://huggingface.co/models?filter=mvp)، بما في ذلك MVP، و MVP مع موجهات خاصة بمهام معينة، والمتغيرات متعددة المهام مسبقة التدريب.
- إذا كنت تريد استخدام نموذج بدون موجهات (محول قياسي)، فيمكنك تحميله عبر `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp')`.
- إذا كنت تريد استخدام نموذج مع موجهات خاصة بمهام معينة، مثل الملخص، فيمكنك تحميله عبر `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp-summarization')`.
- يدعم نموذجنا الضبط الخفيف للموجهات باتباع [Prefix-tuning](https://arxiv.org/abs/2101.00190) مع طريقة `set_lightweight_tuning()`.

## أمثلة الاستخدام

بالنسبة للملخص، يعد هذا مثالًا على استخدام MVP و MVP مع موجهات خاصة بالملخص.

```python
>>> from transformers import MvpTokenizer, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_prompt = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp-summarization")

>>> inputs = tokenizer(
...     "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["Why You Shouldn't Quit Your Job"]

>>> generated_ids = model_with_prompt.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["Don't do it if these are your reasons"]
```

بالنسبة لتوليد النص من البيانات، يعد هذا مثالًا على استخدام MVP والمتغيرات متعددة المهام مسبقة التدريب.

```python
>>> from transformers import MvpTokenizerFast, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizerFast.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_mtl = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")

>>> inputs = tokenizer(
...     "Describe the following data: Iron Man | instance of | Superhero [SEP] Stan Lee | creator | Iron Man",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['Stan Lee created the character of Iron Man, a fictional superhero appearing in American comic']

>>> generated_ids = model_with_mtl.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['Iron Man is a fictional superhero appearing in American comic books published by Marvel Comics.']
```

بالنسبة للضبط الخفيف، أي تثبيت النموذج وضبط الموجهات فقط، يمكنك تحميل MVP بموجهات مبدئية بشكل عشوائي أو بموجهات خاصة بمهام معينة. تدعم الشفرة الخاصة بنا أيضًا Prefix-tuning مع BART باتباع [الورقة الأصلية](https://arxiv.org/abs/2101.00190).

```python
>>> from transformers import MvpForConditionalGeneration

>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp", use_prompt=True)
>>> # عدد المعلمات القابلة للتدريب (الضبط الكامل)
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
468116832

>>> # الضبط الخفيف مع موجهات مبدئية بشكل عشوائي
>>> model.set_lightweight_tuning()
>>> # عدد المعلمات القابلة للتدريب (الضبط الخفيف)
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
61823328

>>> # الضبط الخفيف مع موجهات خاصة بمهام معينة
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")
>>> model.set_lightweight_tuning()
>>> # Prefix-tuning الخفيف الأصلي
>>> model = MvpForConditionalGeneration.from_pretrained("facebook/bart-large", use_prompt=True)
>>> model.set_lightweight_tuning()
```

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة الأسئلة والأجوبة](../tasks/question_answering)
- [دليل مهمة النمذجة اللغوية السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المُقنعة](../tasks/masked_language_modeling)
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)

## MvpConfig

[[autodoc]] MvpConfig

## MvpTokenizer

[[autodoc]] MvpTokenizer

## MvpTokenizerFast

[[autodoc]] MvpTokenizerFast

## MvpModel

[[autodoc]] MvpModel

- forward

## MvpForConditionalGeneration

[[autodoc]] MvpForConditionalGeneration

- forward

## MvpForSequenceClassification

[[autodoc]] MvpForSequenceClassification

- forward

## MvpForQuestionAnswering

[[autodoc]] MvpForQuestionAnswering

- forward

## MvpForCausalLM

[[autodoc]] MvpForCausalLM

- forward