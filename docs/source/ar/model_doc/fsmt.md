# FSMT

## نظرة عامة

تم تقديم نماذج FSMT (FairSeq Machine Translation) في [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616) بواسطة Nathan Ng و Kyra Yee و Alexei Baevski و Myle Ott و Michael Auli و Sergey Edunov.

ملخص الورقة هو كما يلي:

*تصف هذه الورقة مشاركة Facebook FAIR في مسابقة WMT19 للترجمة الإخبارية. نشارك في زوجين لغويين وأربعة اتجاهات لغوية، الإنجليزية <-> الألمانية والإنجليزية <-> الروسية. وبناءً على مشاركتنا في العام الماضي، فإن أنظمتنا الأساسية هي نماذج محول كبيرة تعتمد على BPE تم تدريبها باستخدام مجموعة أدوات نمذجة Fairseq والتي تعتمد على ترجمات عكسية تمت معاينتها. وفي هذا العام، نجري تجارب على مخططات مختلفة لتصفية بيانات النصوص المتوازية، بالإضافة إلى إضافة بيانات مترجمة عكسياً تمت تصفيتها. كما نقوم بتجميع وتعديل دقيق لنماذجنا على بيانات خاصة بالمجال، ثم نقوم بفك الترميز باستخدام إعادة ترتيب نموذج القناة الصاخبة. وقد احتلت مشاركاتنا المرتبة الأولى في جميع الاتجاهات الأربعة للحملة التقييمية البشرية. وفي اتجاه En->De، تفوق نظامنا بشكل كبير على الأنظمة الأخرى والترجمات البشرية. ويحسن هذا النظام من مشاركتنا في WMT'18 بمقدار 4.5 نقاط BLEU.*

تمت المساهمة بهذا النموذج من قبل [stas](https://huggingface.co/stas). يمكن العثور على الكود الأصلي [هنا](https://github.com/pytorch/fairseq/tree/master/examples/wmt19).

## ملاحظات التنفيذ

- يستخدم FSMT أزواج مفردات المصدر والهدف التي لا يتم دمجها في واحدة. كما أنه لا يشارك رموز التعلم. تشبه طريقة تمييزه إلى حد كبير [`XLMTokenizer`] ويتم اشتقاق النموذج الرئيسي من [`BartModel`].

## FSMTConfig

[[autodoc]] FSMTConfig

## FSMTTokenizer

[[autodoc]] FSMTTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## FSMTModel

[[autodoc]] FSMTModel

- forward

## FSMTForConditionalGeneration

[[autodoc]] FSMTForConditionalGeneration

- forward