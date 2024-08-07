# MarkupLM

## نظرة عامة

اقتُرح نموذج MarkupLM في ورقة بحثية بعنوان "MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document Understanding" من قبل Junlong Li وYiheng Xu وLei Cui وFuru Wei. ويعد نموذج MarkupLM نسخة مطورة من نموذج BERT، ولكنه يطبق على صفحات HTML بدلاً من وثائق النص الخام. ويضيف النموذج طبقات تضمين إضافية لتحسين الأداء، على غرار نموذج LayoutLM.

يمكن استخدام النموذج في مهام مثل الإجابة على الأسئلة على صفحات الويب أو استخراج المعلومات من صفحات الويب. ويحقق نتائج متقدمة في معيارين مهمين:

- WebSRC، وهو مجموعة بيانات للقراءة التركيبية القائمة على الويب (مشابهة لأسئلة SQuAD ولكن لصفحات الويب)
- SWDE، وهي مجموعة بيانات لاستخراج المعلومات من صفحات الويب (تعرّف الكيانات المسماة على صفحات الويب بشكل أساسي)

وفيما يلي ملخص الورقة البحثية:

*أحرزت الطرق متعددة الوسائط في مرحلة ما قبل التدريب باستخدام النص والتخطيط والصورة تقدماً ملحوظاً في فهم المستندات الغنية بالمعلومات المرئية (VrDU)، خاصة المستندات ذات التخطيط الثابت مثل صور المستندات الممسوحة ضوئياً. ومع ذلك، لا يزال هناك عدد كبير من المستندات الرقمية التي لا يكون فيها معلومات التخطيط ثابتة ويجب عرضها بشكل تفاعلي وديناميكي، مما يجعل من الصعب تطبيق طرق ما قبل التدريب القائمة على التخطيط. وفي هذه الورقة، نقترح نموذج MarkupLM لمهام فهم المستندات التي تستخدم لغات الترميز كعمود فقري، مثل المستندات القائمة على HTML/XML، حيث يتم التدريب المشترك للمعلومات النصية ومعلومات الترميز. وتظهر نتائج التجارب أن نموذج MarkupLM المُدرب مسبقًا يتفوق بشكل كبير على نماذج الخط الأساسي القوية الحالية في عدة مهام لفهم المستندات. وسيتم إتاحة النموذج المُدرب مسبقًا والشفرة البرمجية للجمهور.*

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). ويمكن العثور على الشفرة البرمجية الأصلية [هنا](https://github.com/microsoft/unilm/tree/master/markuplm).

## نصائح الاستخدام

- بالإضافة إلى `input_ids`، يتوقع [`~MarkupLMModel.forward`] إدخالين إضافيين، وهما `xpath_tags_seq` و`xpath_subs_seq`. وهما علامات XPATH والمخطوطات الفرعية لكل رمز في تسلسل الإدخال على التوالي.
- يمكن استخدام [`MarkupLMProcessor`] لإعداد جميع البيانات للنموذج. راجع [دليل الاستخدام](#usage-markuplmprocessor) للحصول على مزيد من المعلومات.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/markuplm_architecture.jpg" alt="drawing" width="600"/>

<small> بنية نموذج MarkupLM. مأخوذة من <a href="https://arxiv.org/abs/2110.08518">الورقة البحثية الأصلية.</a> </small>

## الاستخدام: MarkupLMProcessor

أسهل طريقة لإعداد البيانات للنموذج هي استخدام [`MarkupLMProcessor`]، والذي يجمع بين مستخرج الميزات ([`MarkupLMFeatureExtractor`]) ومُرمز ([`MarkupLMTokenizer`] أو [`MarkupLMTokenizerFast`]) داخليًا. ويستخدم مستخرج الميزات لاستخراج جميع العقد ومسارات لغة الترميز الموحدة (XPath) من سلاسل HTML، والتي يتم توفيرها بعد ذلك إلى المرمز، والذي يحولها إلى إدخالات على مستوى الرموز للنموذج (`input_ids`، إلخ). لاحظ أنه يمكنك استخدام مستخرج الميزات والمرمز بشكل منفصل إذا كنت تريد التعامل مع إحدى المهمتين فقط.

```python
from transformers import MarkupLMFeatureExtractor, MarkupLMTokenizerFast, MarkupLMProcessor

feature_extractor = MarkupLMFeatureExtractor()
tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base")
processor = MarkupLMProcessor(feature_extractor, tokenizer)
```

وباختصار، يمكنك توفير سلاسل HTML (وربما بيانات إضافية) إلى [`MarkupLMProcessor`]، وسينشئ الإدخالات التي يتوقعها النموذج. داخليًا، يستخدم المعالج أولاً [`MarkupLMFeatureExtractor`] للحصول على قائمة العقد ومسارات لغة الترميز الموحدة (XPath) المقابلة. ثم يتم توفير العقد ومسارات لغة الترميز الموحدة (XPath) إلى [`MarkupLMTokenizer`] أو [`MarkupLMTokenizerFast`]، والذي يحولها إلى `input_ids` على مستوى الرموز و`attention_mask` و`token_type_ids` و`xpath_subs_seq` و`xpath_tags_seq`.

اختياريًا، يمكنك توفير تسميات العقد للمعالج، والتي يتم تحويلها إلى `labels` على مستوى الرموز.

يستخدم [`MarkupLMFeatureExtractor`] مكتبة [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)، وهي مكتبة بايثون لاستخراج البيانات من ملفات HTML وXML، في الخلفية. لاحظ أنه يمكنك استخدام حل التوصيل الخاص بك وتوفير العقد ومسارات لغة الترميز الموحدة (XPath) بنفسك إلى [`MarkupLMTokenizer`] أو [`MarkupLMTokenizerFast`].

هناك خمس حالات استخدام مدعومة من قبل المعالج. فيما يلي قائمة بها جميعًا. لاحظ أن كل حالة من هذه الحالات الاستخدامية تعمل مع الإدخالات المجمعة وغير المجمعة (نوضحها لإدخالات غير مجمعة).

**حالة الاستخدام 1: تصنيف صفحات الويب (التدريب والاستنتاج) + تصنيف الرموز (الاستنتاج)، parse_html = True**

هذه هي الحالة الأكثر بساطة، والتي سيستخدم فيها المعالج مستخرج الميزات للحصول على جميع العقد ومسارات لغة الترميز الموحدة (XPath) من HTML.

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")

>>> html_string = """
...  <!DOCTYPE html>
...  <html>
...  <head>
...  <title>Hello world</title>
...  </head>
...  <body>
...  <h1>Welcome</hihmc
...  <p>Here is my website.</p>
...  </body>
...  </html>"""

>>> # لاحظ أنه يمكنك أيضًا توفير جميع معلمات المرمز هنا مثل الحشو أو الاقتصاص
>>> encoding = processor(html_string, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

**حالة الاستخدام 2: تصنيف صفحات الويب (التدريب والاستنتاج) + تصنيف الرموز (الاستنتاج)، parse_html=False**

في حالة الحصول مسبقًا على جميع العقد ومسارات لغة الترميز الموحدة (XPath)، لا تحتاج إلى مستخرج الميزات. في هذه الحالة، يجب توفير العقد ومسارات لغة الترميز الموحدة (XPath) المقابلة بنفسك للمعالج، والتأكد من تعيين `parse_html` إلى `False`.

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
>>> processor.parse_html = False

>>> nodes = ["hello", "world", "how", "are"]
>>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
>>> encoding = processor(nodes=nodes, xpaths=xpaths, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

**حالة الاستخدام 3: تصنيف الرموز (التدريب)، parse_html=False**

لمهام تصنيف الرموز (مثل [SWDE](https://paperswithcode.com/dataset/swde))، يمكنك أيضًا توفير تسميات العقد المقابلة لتدريب النموذج. وسيحولها المعالج بعد ذلك إلى `labels` على مستوى الرموز.

وبشكل افتراضي، سيقوم بتسمية أول رمز من رموز الكلمة فقط، وتسمية رموز الكلمات المتبقية بـ -100، وهو `ignore_index` في وظيفة PyTorch's CrossEntropyLoss. إذا كنت تريد تسمية جميع رموز الكلمات، فيمكنك تهيئة المرمز مع تعيين `only_label_first_subword` إلى `False`.

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
>>> processor.parse_html = False

>>> nodes = ["hello", "world", "how", "are"]
>>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
>>> node_labels = [1, 2, 2, 1]
>>> encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq', 'labels'])
```

**حالة الاستخدام 4: الإجابة على أسئلة صفحات الويب (الاستنتاج)، parse_html=True**

لمهام الإجابة على الأسئلة على صفحات الويب، يمكنك توفير سؤال للمعالج. وبشكل افتراضي، سيستخدم المعالج مستخرج الميزات للحصول على جميع العقد ومسارات لغة الترميز الموحدة (XPath)، وإنشاء رموز [CLS] ورموز الأسئلة [SEP] ورموز الكلمات [SEP].

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")

>>> html_string = """
...  <!DOCTYPE html>
...  <html>
...  <head>
...  <title>Hello world</title>
...  </head>
...  <body>
...  <h1>Welcome</h1>
...  <p>My name is Niels.</p>
...  </body>
...  </html>"""

>>> question = "What's his name?"
>>> encoding = processor(html_string, questions=question, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

**حالة الاستخدام 5: الإجابة على أسئلة صفحات الويب (الاستنتاج)، parse_html=False**

لمهام الإجابة على الأسئلة (مثل WebSRC)، يمكنك توفير سؤال للمعالج. إذا قمت باستخراج جميع العقد ومسارات لغة الترميز الموحدة (XPath) بنفسك، فيمكنك توفيرها مباشرة إلى المعالج. تأكد من تعيين `parse_html` إلى `False`.

```python
>>> from transformers import MarkupLMProcessor

>>> processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
>>> processor.parse_html = False

>>> nodes = ["hello", "world", "how", "are"]
>>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
>>> question = "What's his name?"
>>> encoding = processor(nodes=nodes, xpaths=xpaths, questions=question, return_tensors="pt")
>>> print(encoding.keys())
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])
```

## الموارد

- [دفاتر الملاحظات التوضيحية](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MarkupLM)
- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)

## MarkupLMConfig

[[autodoc]] MarkupLMConfig
- all

## MarkupLMFeatureExtractor

[[autodoc]] MarkupLMFeatureExtractor
- __call__

## MarkupLMTokenizer

[[autodoc]] MarkupLMTokenizer
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## MarkupLMTokenizerFast

[[autodoc]] MarkupLMTokenizerFast
- all

## MarkupLMProcessor

[[autodoc]] MarkupLMProcessor
- __call__

## MarkupLMModel

[[autodoc]] MarkupLMModel
- forward

## MarkupLMForSequenceClassification

[[autodoc]] MarkupLMForSequenceClassification
- forward

## MarkupLMForTokenClassification

[[autodoc]] MarkupLMForTokenClassification
- forward

## MarkupLMForQuestionAnswering

[[autodoc]] MarkupLMForQuestionAnswering
- forward