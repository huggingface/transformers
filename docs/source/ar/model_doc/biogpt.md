# BioGPT

## نظرة عامة

اقترح نموذج BioGPT في [BioGPT: generative pre-trained transformer for biomedical text generation and mining](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9) بواسطة Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu. BioGPT هو نموذج لغة محول مُدرب مسبقًا مخصص لمجال معين لتوليد النصوص واستخراجها في المجال الطبي الحيوي. يتبع BioGPT العمود الفقري لنموذج لغة المحول، وهو مُدرب مسبقًا من البداية على 15 مليون ملخص PubMed.

الملخص من الورقة هو ما يلي:

*لقد جذبت نماذج اللغة المدربة مسبقًا اهتمامًا متزايدًا في المجال الطبي الحيوي، مستوحاة من نجاحها الكبير في مجال اللغة الطبيعية العامة. من بين الفرعين الرئيسيين لنماذج اللغة المُدربة مسبقًا في مجال اللغة العامة، أي BERT (وأشكاله المختلفة) و GPT (وأشكاله المختلفة)، تمت دراسة الأول على نطاق واسع في المجال الطبي الحيوي، مثل BioBERT و PubMedBERT. على الرغم من أنهم حققوا نجاحًا كبيرًا في مجموعة متنوعة من المهام الطبية الحيوية التمييزية، فإن الافتقار إلى القدرة على التوليد يقيد نطاق تطبيقها. في هذه الورقة، نقترح BioGPT، وهو نموذج لغة محول مخصص لمجال معين ومدرب مسبقًا على الأدب الطبي الحيوي واسع النطاق. نقوم بتقييم BioGPT على ست مهام معالجة اللغة الطبيعية الطبية الحيوية ونثبت أن نموذجنا يتفوق على النماذج السابقة في معظم المهام. خاصة، نحصل على 44.98٪، 38.42٪ و 40.76٪ F1 score على BC5CDR، KD-DTI و DDI end-to-end relation extraction tasks، على التوالي، و 78.2٪ دقة على PubMedQA، مما يخلق رقما قياسيا جديدا. توضح دراستنا الإضافية حول توليد النصوص ميزة BioGPT على الأدب الطبي الحيوي لتوليد أوصاف سلسة للمصطلحات الطبية الحيوية.*

تمت المساهمة بهذا النموذج من قبل [kamalkraj](https://huggingface.co/kamalkraj). يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/BioGPT).

## نصائح الاستخدام

- BioGPT هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بتعبئة المدخلات من اليمين بدلاً من اليسار.

- تم تدريب BioGPT بهدف نمذجة اللغة السببية (CLM) وبالتالي فهو قوي في التنبؤ بالرمز التالي في تسلسل. الاستفادة من هذه الميزة تسمح لـ BioGPT بتوليد نص متماسك من الناحية التركيبية كما يمكن ملاحظته في مثال run_generation.py script.

- يمكن للنموذج أن يأخذ `past_key_values` (للبيثون) كإدخال، وهو أزواج الاهتمام الرئيسية / القيم المحسوبة مسبقًا. باستخدام هذه القيمة (past_key_values أو past) يمنع النموذج من إعادة حساب القيم المحسوبة مسبقًا في سياق توليد النص. بالنسبة إلى PyTorch، راجع حجة past_key_values لطريقة BioGptForCausalLM.forward() للحصول على مزيد من المعلومات حول استخدامها.

## الموارد

- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

## BioGptConfig

[[autodoc]] BioGptConfig

## BioGptTokenizer

[[autodoc]] BioGptTokenizer

- save_vocabulary

## BioGptModel

[[autodoc]] BioGptModel

- forward

## BioGptForCausalLM

[[autodoc]] BioGptForCausalLM

- forward

## BioGptForTokenClassification

[[autodoc]] BioGptForTokenClassification

- forward

## BioGptForSequenceClassification

[[autodoc]] BioGptForSequenceClassification

- forward