# Swin2SR

## نظرة عامة

اقترح نموذج Swin2SR في ورقة "Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration" بواسطة Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte.

يحسن Swin2R نموذج SwinIR من خلال دمج طبقات Swin Transformer v2 التي تخفف من مشكلات مثل عدم استقرار التدريب، والفجوات في الدقة بين التدريب الأولي والضبط الدقيق، والجوع على البيانات.

ملخص الورقة هو كما يلي:

> "يلعب الضغط دورًا مهمًا في نقل وتخزين الصور ومقاطع الفيديو بكفاءة عبر الأنظمة المحدودة النطاق مثل خدمات البث، والواقع الافتراضي، أو ألعاب الفيديو. ومع ذلك، يؤدي الضغط حتمًا إلى ظهور تشوهات وفقدان المعلومات الأصلية، الأمر الذي قد يتسبب في تدهور جودة الصورة بشكل كبير. ولهذه الأسباب، أصبحت تحسين جودة الصور المضغوطة مجالًا بحثيًا شائعًا. في حين أن معظم طرق استعادة الصور المتقدمة تعتمد على الشبكات العصبية التلافيفية، إلا أن هناك طرقًا أخرى تعتمد على المحولات مثل SwinIR، والتي تظهر أداءً مثيرًا للإعجاب في هذه المهام.

> في هذه الورقة، نستكشف محول Swin Transformer V2 الجديد لتحسين نموذج SwinIR لاستعادة الدقة الفائقة للصور، وخاصة سيناريو الإدخال المضغوط. باستخدام هذه الطريقة، يمكننا معالجة المشكلات الرئيسية في تدريب نماذج الرؤية المحولة، مثل عدم استقرار التدريب، والفجوات في الدقة بين التدريب الأولي والضبط الدقيق، والجوع على البيانات. نجري تجارب على ثلاث مهام تمثيلية: إزالة تشوهات ضغط JPEG، واستعادة الدقة الفائقة للصور (الكلاسيكية والخفيفة)، واستعادة الدقة الفائقة للصور المضغوطة. وتظهر النتائج التجريبية أن طريقتنا، Swin2SR، يمكن أن تحسن من تقارب التدريب وأداء نموذج SwinIR، وهي إحدى الحلول الخمسة الأولى في مسابقة AIM 2022 Challenge on Super-Resolution of Compressed Image and Video".

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/swin2sr_architecture.png"
alt="drawing" width="600"/>

<small> بنية Swin2SR. مأخوذة من <a href="https://arxiv.org/abs/2209.11345">الورقة الأصلية.</a> </small>

تمت المساهمة بهذا النموذج بواسطة [nielsr](https://huggingface.co/nielsr).

يمكن العثور على الكود الأصلي [هنا](https://github.com/mv-lab/swin2sr).

## الموارد

يمكن العثور على دفاتر الملاحظات التجريبية لـ Swin2SR [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Swin2SR).

يمكن العثور على مساحة تجريبية لاستعادة الدقة الفائقة للصور باستخدام SwinSR [هنا](https://huggingface.co/spaces/jjourney1125/swin2sr).

## Swin2SRImageProcessor

[[autodoc]] Swin2SRImageProcessor

- preprocess

## Swin2SRConfig

[[autodoc]] Swin2SRConfig

## Swin2SRModel

[[autodoc]] Swin2SRModel

- forward

## Swin2SRForImageSuperResolution

[[autodoc]] Swin2SRForImageSuperResolution

- forward