# VipLlava

## نظرة عامة

اقترح نموذج VipLlava في الورقة البحثية بعنوان "جعل النماذج متعددة الوسائط كبيرة الحجم تفهم الإشارات المرئية التعسفية" من قبل Mu Cai و Haotian Liu و Siva Karthik Mustikovela و Gregory P. Meyer و Yuning Chai و Dennis Park و Yong Jae Lee.

يحسن VipLlava بروتوكول التدريب الخاص بـ Llava من خلال وضع علامات على الصور والتفاعل مع النموذج باستخدام إشارات طبيعية مثل "مربع حد أحمر" أو "سهم يشير" أثناء التدريب.

الملخص من الورقة هو كما يلي:

*في حين أن النماذج متعددة الوسائط الكبيرة الحجم الحالية تركز على فهم الصورة بالكامل، هناك فجوة بارزة في تحقيق الفهم المحدد للمنطقة. غالبًا ما تفشل الأساليب الحالية التي تستخدم الإحداثيات النصية أو الترميزات المكانية في توفير واجهة سهلة الاستخدام للإشارات المرئية. ولمعالجة هذا التحدي، نقدم نموذجًا متعدد الوسائط جديدًا قادرًا على فك تشفير الإشارات المرئية التعسفية. يسمح هذا للمستخدمين بوضع علامات على الصور والتفاعل مع النموذج باستخدام إشارات طبيعية مثل "مربع حد أحمر" أو "سهم يشير". يعتمد تصميمنا البسيط مباشرة العلامات المرئية على صورة RGB، مما يلغي الحاجة إلى ترميزات المنطقة المعقدة، ولكنه يحقق أداءً من مستوى الدولة من الفن في مهام فهم المنطقة مثل Visual7W و PointQA و Visual Commonsense Reasoning benchmark. علاوة على ذلك، نقدم ViP-Bench، وهو معيار شامل لتقييم قدرة النماذج على فهم الإشارات المرئية عبر أبعاد متعددة، مما يمكن الأبحاث المستقبلية في هذا المجال. الكود والبيانات والنماذج متاحة للجمهور.*

نصائح:

- الهندسة المعمارية مشابهة لهندسة llava باستثناء أن العارض متعدد الوسائط يأخذ مجموعة من حالات الإخفاء المرئية المدمجة ولديه طبقة LayerNorm إضافية في ذلك الوحدة.

- ننصح المستخدمين باستخدام `padding_side="left"` عند حساب التوليد الدفعي لأنه يؤدي إلى نتائج أكثر دقة. فقط تأكد من استدعاء `processor.tokenizer.padding_side = "left"` قبل التوليد.

- لاحظ أن النموذج لم يتم تدريبه بشكل صريح لمعالجة صور متعددة في نفس المطالبة، على الرغم من أن هذا ممكن من الناحية الفنية، فقد تواجه نتائج غير دقيقة.

- للحصول على نتائج أفضل، نوصي المستخدمين بتنبيه النموذج بتنسيق المطالبة الصحيح:

```bash
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n<prompt>###Assistant:
```

للحصول على محادثة متعددة الأدوار:

```bash
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n<prompt1>###Assistant: <answer1>###Human: <prompt2>###Assistant:
```

يمكن العثور على الكود الأصلي [هنا](https://github.com/mu-cai/ViP-LLaVA).

تمت المساهمة بهذا النموذج من قبل [Younes Belkada](https://huggingface.co/ybelkada)

## VipLlavaConfig

[[autodoc]] VipLlavaConfig

## VipLlavaForConditionalGeneration

[[autodoc]] VipLlavaForConditionalGeneration

- forward