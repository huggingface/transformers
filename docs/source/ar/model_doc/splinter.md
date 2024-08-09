# Splinter

## نظرة عامة
اقترح نموذج Splinter في [Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/abs/2101.00438) بواسطة Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy. Splinter هو محول للتشفير فقط (مشابه لـ BERT) تم تدريبه مسبقًا باستخدام مهمة اختيار النطاق المتكرر على مجموعة كبيرة من النصوص تتكون من Wikipedia وToronto Book Corpus.

ملخص الورقة البحثية هو كما يلي:

في العديد من معايير تقييم الإجابة على الأسئلة، وصلت النماذج التي تم تدريبها مسبقًا إلى مستوى أداء البشر من خلال الضبط الدقيق على ما يقرب من 100000 سؤال وجواب معنون. نستكشف إعدادًا أكثر واقعية، وهو الإعداد القائم على القليل من الأمثلة، حيث لا تتوفر سوى بضع مئات من الأمثلة التدريبية، ونلاحظ أن النماذج القياسية تؤدي أداءً ضعيفًا، مما يسلط الضوء على عدم التوافق بين أهداف التدريب الحالية والإجابة على الأسئلة. نقترح نظامًا جديدًا للتدريب المسبق مصممًا خصيصًا للإجابة على الأسئلة: اختيار النطاق المتكرر. بالنظر إلى فقرة تحتوي على مجموعات متعددة من النطاقات المتكررة، نقوم في كل مجموعة بقناع جميع النطاقات المتكررة باستثناء واحد، ونسأل النموذج لتحديد النطاق الصحيح في الفقرة لكل نطاق مقنع. يتم استبدال النطاقات المقنعة برمز خاص، يتم اعتباره كتمثيل للسؤال، والذي يتم استخدامه لاحقًا أثناء الضبط الدقيق لاختيار نطاق الإجابة. يحقق النموذج الناتج نتائج جيدة جدًا في العديد من معايير التقييم (على سبيل المثال، 72.7 F1 على SQuAD باستخدام 128 مثال تدريبي فقط)، مع الحفاظ على أداء تنافسي في إعداد الموارد العالية.

تمت المساهمة بهذا النموذج من قبل [yuvalkirstain](https://huggingface.co/yuvalkirstain) و [oriram](https://huggingface.co/oriram). يمكن العثور على الكود الأصلي [هنا](https://github.com/oriram/splinter).

## نصائح الاستخدام

- تم تدريب Splinter للتنبؤ بنطاقات الإجابات المشروطة برمز [QUESTION] خاص. تقوم هذه الرموز بسياق التمثيلات الخاصة بالأسئلة المستخدمة للتنبؤ بالإجابات. تسمى هذه الطبقة QASS، وهي السلوك الافتراضي في فئة [`SplinterForQuestionAnswering`]. لذلك:

- استخدم [`SplinterTokenizer`] (بدلاً من [`BertTokenizer`])، حيث يحتوي بالفعل على هذا الرمز الخاص. أيضًا، السلوك الافتراضي هو استخدام هذا الرمز عند إعطاء تسلسلين (على سبيل المثال، في نص *run_qa.py*).

- إذا كنت تخطط لاستخدام Splinter خارج نص *run_qa.py*، يرجى مراعاة رمز السؤال - فقد يكون مهمًا لنجاح نموذجك، خاصة في إعداد القليل من الأمثلة.

- يرجى ملاحظة وجود نقطتي تفتيش مختلفتين لكل حجم من Splinter. كلاهما متشابهان بشكل أساسي، باستثناء أن أحدهما يحتوي أيضًا على أوزان مدربة مسبقًا لطبقة QASS (*tau/splinter-base-qass* و *tau/splinter-large-qass*) ولا يحتوي الآخر (*tau/splinter-base* و *tau/splinter-large*). يتم ذلك لدعم تهيئة هذه الطبقة بشكل عشوائي عند الضبط الدقيق، حيث ثبت أنها تحقق نتائج أفضل لبعض الحالات في الورقة البحثية.

## الموارد

- [دليل مهام الإجابة على الأسئلة](../tasks/question-answering)

## SplinterConfig

[[autodoc]] SplinterConfig

## SplinterTokenizer

[[autodoc]] SplinterTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## SplinterTokenizerFast

[[autodoc]] SplinterTokenizerFast

## SplinterModel

[[autodoc]] SplinterModel

- forward

## SplinterForQuestionAnswering

[[autodoc]] SplinterForQuestionAnswering

- forward

## SplinterForPreTraining

[[autodoc]] SplinterForPreTraining

- forward