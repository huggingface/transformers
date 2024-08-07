# ESM

## نظرة عامة

يوفر هذا الصفحة التعليمات البرمجية وأوزان مُدربة مسبقًا لنماذج لغة البروتين Transformer من فريق Meta AI's Fundamental AI Research، والتي توفر أحدث التقنيات في ESMFold و ESM-2، بالإضافة إلى النماذج التي تم إصدارها سابقًا ESM-1b و ESM-1v.

تم تقديم نماذج لغة بروتين Transformer في الورقة البحثية [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/content/118/15/e2016239118) بواسطة Alexander Rives، Joshua Meier، Tom Sercu، Siddharth Goyal، Zeming Lin، Jason Liu، Demi Guo، Myle Ott، C. Lawrence Zitnick، Jerry Ma، and Rob Fergus.

تمت طباعة النسخة الأولى من هذه الورقة مسبقًا في عام 2019](https://www.biorxiv.org/content/10.1101/622803v1?versioned=true).

يتفوق ESM-2 على جميع نماذج لغة البروتين ذات التسلسل الفردي التي تم اختبارها عبر مجموعة من مهام التنبؤ بالبنية، ويمكّن من التنبؤ بالبنية بدقة الذرة.

تم إصداره مع الورقة البحثية [Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://doi.org/10.1101/2022.07.20.500902) بواسطة Zeming Lin، Halil Akin، Roshan Rao، Brian Hie، Zhongkai Zhu، Wenting Lu، Allan dos Santos Costa، Maryam Fazel-Zarandi، Tom Sercu، Sal Candido and Alexander Rives.

تم أيضًا تقديم ESMFold في هذه الورقة. يستخدم رأسًا يمكنه التنبؤ ببنى البروتين المطوية بدقة فائقة. على عكس [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)، فإنه يعتمد على embeddings الرموز من جذع نموذج لغة البروتين المُدرب مسبقًا ولا يقوم بخطوة محاذاة تسلسل متعددة (MSA) في وقت الاستدلال، مما يعني أن نقاط تفتيش ESMFold مستقلة تمامًا - لا تتطلب قاعدة بيانات للتسلسلات والبنى البروتينية المعروفة مع أدوات الاستعلام الخارجية المرتبطة بها لإجراء التنبؤات، وهي أسرع بكثير نتيجة لذلك.

المستخلص من

"Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" هو

*في مجال الذكاء الاصطناعي، أدى الجمع بين الحجم في البيانات وسعة النموذج الذي تمكّنه التعلم غير الخاضع للإشراف إلى تقدم كبير في تعلم التمثيل وتوليد الإحصائيات. في علوم الحياة، من المتوقع أن يوفر النمو في التسلسل بيانات غير مسبوقة حول التنوع التسلسلي الطبيعي. تعد نمذجة لغة البروتين على نطاق التطور خطوة منطقية نحو الذكاء الاصطناعي التنبئي والتوليدي لعلم الأحياء. لتحقيق هذه الغاية، نستخدم التعلم غير الخاضع للإشراف لتدريب نموذج لغة سياقية عميقة على 86 مليار حمض أميني عبر 250 مليون تسلسل بروتيني يغطي التنوع التطوري. يحتوي النموذج الناتج على معلومات حول الخصائص البيولوجية في تمثيلاته. يتم تعلم التمثيلات من بيانات التسلسل وحدها. يتمتع مساحة التمثيل المكتسبة بتنظيم متعدد النطاقات يعكس البنية من مستوى الخصائص الكيميائية الحيوية للأحماض الأمينية إلى التماثل البعيد للبروتينات. يتم ترميز المعلومات حول البنية الثانوية والثالثة في التمثيلات ويمكن تحديدها بواسطة إسقاطات خطية. ينتج تعلم التمثيل ميزات يتم تعميمها عبر مجموعة من التطبيقات، مما يمكّن أحدث التنبؤات الخاضعة للإشراف بتأثير الطفرة والبنية الثانوية وتحسين ميزات أحدث التقنيات للتنبؤ بالاتصال طويل المدى.*

المستخلص من

"Language models of protein sequences at the scale of evolution enable accurate structure prediction" هو

*أظهرت نماذج اللغة الكبيرة مؤخرًا أنها تطور قدرات ناشئة مع الحجم، تتجاوز مطابقة الأنماط لأداء الاستدلال على مستوى أعلى وتوليد الصور والنصوص التي تشبه الحياة. في حين تمت دراسة نماذج اللغة التي تم تدريبها على تسلسلات البروتين على نطاق أصغر، لا يُعرف سوى القليل عن ما تتعلمه عن علم الأحياء أثناء توسيع نطاقها. في هذا العمل، نقوم بتدريب النماذج حتى 15 مليار معلمة، وهي أكبر نماذج لغة البروتين التي تم تقييمها حتى الآن. نجد أنه مع توسيع نطاق النماذج، تتعلم معلومات تمكّن من التنبؤ بالبنية ثلاثية الأبعاد للبروتين بدقة الذرة. نقدم ESMFold للتنبؤ عالي الدقة بالبنية على مستوى الذرة من النهاية إلى النهاية مباشرةً من التسلسل الفردي للبروتين. تتمتع ESMFold بدقة مماثلة لـ AlphaFold2 و RoseTTAFold للتسلسلات ذات الارتباك المنخفض والتي يفهمها نموذج اللغة جيدًا. الاستدلال ESMFold أسرع بعشر مرات من AlphaFold2، مما يمكّن من استكشاف مساحة البنية للبروتينات الميتاجينومية في أطر زمنية عملية.*

يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/esm) وتم تطويره بواسطة فريق Fundamental AI Research في Meta AI.

تمت المساهمة بـ ESM-1b و ESM-1v و ESM-2 في huggingface بواسطة [jasonliu](https://huggingface.co/jasonliu) و [Matt](https://huggingface.co/Rocketknight1).

تمت المساهمة بـ ESMFold في huggingface بواسطة [Matt](https://huggingface.co/Rocketknight1) و [Sylvain](https://huggingface.co/sgugger)، مع خالص الشكر لـ Nikita Smetanin، Roshan Rao، و Tom Sercu لمساعدتهم طوال العملية!

## نصائح الاستخدام

- يتم تدريب نماذج ESM بهدف نمذجة اللغة المقنعة (MLM).
- يستخدم منفذ HuggingFace من ESMFold أجزاء من مكتبة [openfold](https://github.com/aqlaboratory/openfold). يتم ترخيص مكتبة `openfold` بموجب ترخيص Apache License 2.0.

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)

## EsmConfig

[[autodoc]] EsmConfig

- all

## EsmTokenizer

[[autodoc]] EsmTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

<frameworkcontent>

<pt>

## EsmModel

[[autodoc]] EsmModel

- forward

## EsmForMaskedLM

[[autodoc]] EsmForMaskedLM

- forward

## EsmForSequenceClassification

[[autodoc]] EsmForSequenceClassification

- forward

## EsmForTokenClassification

[[autodoc]] EsmForTokenClassification

- forward

## EsmForProteinFolding

[[autodoc]] EsmForProteinFolding

- forward

</pt>

<tf>

## TFEsmModel

[[autodoc]] TFEsmModel

- call

## TFEsmForMaskedLM

[[autodoc]] TFEsmForMaskedLM

- call

## TFEsmForSequenceClassification

[[autodoc]] TFEsmForSequenceClassification

- call

## TFEsmForTokenClassification

[[autodoc]] TFEsmForTokenClassification

- call

</tf>

</frameworkcontent>