# OpenAI GPT

## نظرة عامة
اقترح نموذج OpenAI GPT في ورقة "تحسين فهم اللغة عن طريق التدريب المسبق التوليدي" من قبل Alec Radford و Karthik Narasimhan و Tim Salimans و Ilya Sutskever. وهو محول أحادي الاتجاه تم تدريبه مسبقًا باستخدام نمذجة اللغة على مجموعة بيانات كبيرة بها تبعيات طويلة المدى، وهي Toronto Book Corpus.

ملخص الورقة البحثية هو كما يلي:

> "يشتمل فهم اللغة الطبيعية على مجموعة واسعة من المهام المتنوعة مثل الاستنتاج النصي، والأسئلة والأجوبة، وتقييم التشابه الدلالي، وتصنيف المستندات. على الرغم من وفرة مجموعات البيانات النصية الكبيرة غير الموسومة، إلا أن البيانات الموسومة لتعلم هذه المهام المحددة نادرة، مما يجعل من الصعب على النماذج التي تم تدريبها بشكل تمييزي أن تؤدي بشكل كاف. نحن نثبت أنه يمكن تحقيق مكاسب كبيرة في هذه المهام من خلال التدريب المسبق التوليدي لنموذج اللغة على مجموعة بيانات متنوعة من النصوص غير الموسومة، يليه الضبط الدقيق التمييزي لكل مهمة محددة. وعلى عكس الأساليب السابقة، نستخدم تحويلات المدخلات الواعية بالمهام أثناء الضبط الدقيق لتحقيق نقل فعال مع إجراء تغييرات طفيفة على بنية النموذج. نثبت فعالية نهجنا على مجموعة واسعة من المعايير المرجعية لفهم اللغة الطبيعية. يفوق نموذجنا العام غير المخصص للمهمة النماذج التي تم تدريبها بشكل تمييزي والتي تستخدم البنى المصممة خصيصًا لكل مهمة، مما يحسن بشكل كبير حالة الفن في 9 من أصل 12 مهمة تمت دراستها."

"Write With Transformer" هو تطبيق ويب تم إنشاؤه واستضافته بواسطة Hugging Face لعرض القدرات التوليدية لعدة نماذج. يعد GPT أحد هذه النماذج.

تمت المساهمة بهذا النموذج من قبل [thomwolf](https://huggingface.co/thomwolf). يمكن العثور على الكود الأصلي [هنا](https://github.com/openai/finetune-transformer-lm).

## نصائح الاستخدام

- GPT هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بإضافة حشو إلى المدخلات من اليمين بدلاً من اليسار.

- تم تدريب GPT باستخدام هدف نمذجة اللغة السببية (CLM) وبالتالي فهو قوي في التنبؤ بالرمز التالي في تسلسل. الاستفادة من هذه الميزة تسمح لـ GPT-2 بتوليد نص متماسك من الناحية التركيبية كما يمكن ملاحظته في مثال نص البرنامج النصي *run_generation.py*.

ملاحظة:

إذا كنت تريد إعادة إنتاج عملية الترميز الأصلية لورقة *OpenAI GPT*، فستحتاج إلى تثبيت `ftfy` و`SpaCy`:

```bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

إذا لم تقم بتثبيت `ftfy` و`SpaCy`، فسيتم تعيين [`OpenAIGPTTokenizer`] بشكل افتراضي للترميز باستخدام `BasicTokenizer` الخاص بـ BERT متبوعًا بالترميز Byte-Pair (الذي يجب أن يكون مناسبًا لمعظم الاستخدامات، لا داعي للقلق).

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام OpenAI GPT. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب وسنراجعه! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- منشور مدونة حول [تفوق أداء OpenAI GPT-3 باستخدام SetFit لتصنيف النصوص](https://www.philschmid.de/getting-started-setfit).

- راجع أيضًا: [دليل مهام تصنيف النصوص](../tasks/sequence_classification)

- مدونة حول كيفية [ضبط نموذج GPT-2 غير الإنجليزي باستخدام Hugging Face](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface).

- مدونة حول [كيفية توليد النص: استخدام طرق فك مختلفة لتوليد اللغة مع المحولات](https://huggingface.co/blog/how-to-generate) مع GPT-2.

- مدونة حول [تدريب CodeParrot 🦜 من الصفر](https://huggingface.co/blog/codeparrot)، وهو نموذج GPT-2 كبير.

- مدونة حول [توليد النص بشكل أسرع باستخدام TensorFlow و XLA](https://huggingface.co/blog/tf-xla-generate) مع GPT-2.

- مدونة حول [كيفية تدريب نموذج اللغة باستخدام Megatron-LM](https://huggingface.co/blog/megatron-training) مع نموذج GPT-2.

- دفتر ملاحظات حول كيفية [ضبط دقيق GPT2 لتوليد كلمات الأغاني على غرار فنانك المفضل](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb). 🌎

- دفتر ملاحظات حول كيفية [ضبط دقيق GPT2 لتوليد التغريدات على غرار مستخدم Twitter المفضل لديك](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb). 🌎

- فصل [نمذجة اللغة السببية](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) من دورة 🤗 Hugging Face Course.

- [`OpenAIGPTLMHeadModel`] مدعوم بواسطة مثال نص البرنامج النصي [نمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)، ونص البرنامج النصي [توليد النص](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

- [`TFOpenAIGPTLMHeadModel`] مدعوم بواسطة مثال نص البرنامج النصي [نمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

- راجع أيضًا: [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

- مواد الدورة التدريبية حول ترميز [ترميز زوج البايت](https://huggingface.co/course/en/chapter6/5).

## OpenAIGPTConfig

[[autodoc]] OpenAIGPTConfig

## OpenAIGPTTokenizer

[[autodoc]] OpenAIGPTTokenizer

- save_vocabulary

## OpenAIGPTTokenizerFast

[[autodoc]] OpenAIGPTTokenizerFast

## المخرجات الخاصة بـ OpenAI

[[autodoc]] models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput

[[autodoc]] models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput

<frameworkcontent>
<pt>

## OpenAIGPTModel

[[autodoc]] OpenAIGPTModel

- forward

## OpenAIGPTLMHeadModel

[[autodoc]] OpenAIGPTLMHeadModel

- forward

## OpenAIGPTDoubleHeadsModel

[[autodoc]] OpenAIGPTDoubleHeadsModel

- forward

## OpenAIGPTForSequenceClassification

[[autodoc]] OpenAIGPTForSequenceClassification

- forward

</pt>
<tf>

## TFOpenAIGPTModel

[[autodoc]] TFOpenAIGPTModel

- call

## TFOpenAIGPTLMHeadModel

[[autodoc]] TFOpenAIGPTLMHeadModel

- call

## TFOpenAIGPTDoubleHeadsModel

[[autodoc]] TFOpenAIGPTDoubleHeadsModel

- call

## TFOpenAIGPTForSequenceClassification

[[autodoc]] TFOpenAIGPTForSequenceClassification

- call

</tf>
</frameworkcontent>