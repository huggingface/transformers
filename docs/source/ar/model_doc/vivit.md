# محول الفيديو الرؤية (ViViT)

## نظرة عامة
تم اقتراح نموذج Vivit في [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) بواسطة Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid.
تقترح الورقة واحدة من أولى مجموعات النماذج الناجحة القائمة على المحول النقي لفهم الفيديو.

المستخلص من الورقة هو ما يلي:

*نقدم نماذج قائمة على المحول النقي لتصنيف الفيديو، بالاعتماد على النجاح الأخير لمثل هذه النماذج في تصنيف الصور. يستخرج نموذجنا الرموز المكانية-الزمانية من فيديو الإدخال، والتي يتم ترميزها بعد ذلك بواسطة سلسلة من طبقات المحول. وللتعامل مع التسلسلات الطويلة من الرموز التي يتم مواجهتها في الفيديو، نقترح عدة متغيرات فعالة من نموذجنا الذي يحلل الأبعاد المكانية والزمانية للإدخال. على الرغم من أن من المعروف أن النماذج القائمة على المحول تكون فعالة فقط عندما تكون مجموعات البيانات التدريبية الكبيرة متاحة، إلا أننا نوضح كيف يمكننا تنظيم النموذج بشكل فعال أثناء التدريب والاستفادة من نماذج الصور المسبقة التدريب لتتمكن من التدريب على مجموعات البيانات الصغيرة نسبيًا. نجري دراسات شاملة للتحليل، ونحقق نتائج متقدمة في عدة معايير لتصنيف الفيديو بما في ذلك Kinetics 400 و600، وEpic Kitchens، وSomething-Something v2 وMoments in Time، متفوقة على الطرق السابقة القائمة على الشبكات العصبية العميقة ثلاثية الأبعاد.*

تمت المساهمة بهذا النموذج من قبل [jegormeister](https://huggingface.co/jegormeister). يمكن العثور على الكود الأصلي (المكتوب في JAX) [هنا](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit).

## VivitConfig

[[autodoc]] VivitConfig

## VivitImageProcessor

[[autodoc]] VivitImageProcessor

- preprocess

## VivitModel

[[autodoc]] VivitModel

- forward

## VivitForVideoClassification

[[autodoc]] transformers.VivitForVideoClassification

- forward