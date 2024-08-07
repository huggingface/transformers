# BLIP

## نظرة عامة
تم اقتراح نموذج BLIP في [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) بواسطة Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.

BLIP هو نموذج قادر على أداء مهام متعددة الوسائط المختلفة بما في ذلك:

- الإجابة على الأسئلة البصرية
- استرجاع الصور والنصوص (مطابقة الصور والنصوص)
- وضع تعليقات توضيحية على الصور

ملخص الورقة البحثية هو كما يلي:

*أدى التدريب المسبق للرؤية واللغة (VLP) إلى تقدم الأداء في العديد من مهام الرؤية واللغة. ومع ذلك، فإن معظم النماذج التي تم تدريبها مسبقًا لا تتفوق إلا في مهام الفهم أو المهام القائمة على التوليد. علاوة على ذلك، تم تحقيق تحسين الأداء إلى حد كبير عن طريق توسيع نطاق مجموعة البيانات باستخدام أزواج من الصور والنصوص الضخمة التي تم جمعها من الويب، والتي تعد مصدرًا غير مثاليًا للإشراف. في هذه الورقة، نقترح BLIP، وهو إطار عمل جديد لـ VLP ينتقل بمرونة إلى مهام الفهم وتوليد الرؤية واللغة. يستخدم BLIP البيانات الضخمة من الويب بشكل فعال من خلال تعزيز التعليقات التوضيحية، حيث تقوم أداة إنشاء التعليقات التوضيحية بتوليد تعليقات توضيحية اصطناعية وتقوم أداة الترشيح بإزالة غير النظيفة منها. نحقق نتائج متقدمة في مجموعة واسعة من مهام الرؤية واللغة، مثل استرجاع الصور والنصوص (+2.7% في متوسط الاستدعاء@1)، ووضع تعليقات توضيحية على الصور (+2.8% في CIDEr)، والإجابة على الأسئلة البصرية (+1.6% في درجة VQA). كما يظهر BLIP قدرة تعميم قوية عند نقله مباشرة إلى مهام اللغة المرئية بطريقة الصفر. تم إصدار الكود والنماذج ومجموعات البيانات.*

![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif)

تمت المساهمة بهذا النموذج بواسطة [ybelkada](https://huggingface.co/ybelkada). يمكن العثور على الكود الأصلي [هنا](https://github.com/salesforce/BLIP).

## الموارد

- [دفتر Jupyter](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb) حول كيفية ضبط نموذج BLIP لوضع تعليقات توضيحية على الصور باستخدام مجموعة بيانات مخصصة

## BlipConfig

[[autodoc]] BlipConfig

- from_text_vision_configs

## BlipTextConfig

[[autodoc]] BlipTextConfig

## BlipVisionConfig

[[autodoc]] BlipVisionConfig

## BlipProcessor

[[autodoc]] BlipProcessor

## BlipImageProcessor

[[autodoc]] BlipImageProcessor

- preprocess

<frameworkcontent>

<pt>

## BlipModel

سيتم إيقاف استخدام `BlipModel` في الإصدارات المستقبلية، يرجى استخدام `BlipForConditionalGeneration` أو `BlipForImageTextRetrieval` أو `BlipForQuestionAnswering` حسب حالتك الاستخدامية.

[[autodoc]] BlipModel

- forward
- get_text_features
- get_image_features

## BlipTextModel

[[autodoc]] BlipTextModel

- forward

## BlipVisionModel

[[autodoc]] BlipVisionModel

- forward

## BlipForConditionalGeneration

[[autodoc]] BlipForConditionalGeneration

- forward

## BlipForImageTextRetrieval

[[autodoc]] BlipForImageTextRetrieval

- forward

## BlipForQuestionAnswering

[[autodoc]] BlipForQuestionAnswering

- forward

</pt>

<tf>

## TFBlipModel

[[autodoc]] TFBlipModel

- call
- get_text_features
- get_image_features

## TFBlipTextModel

[[autodoc]] TFBlipTextModel

- call

## TFBlipVisionModel

[[autodoc]] TFBlipVisionModel

- call


## TFBlipForConditionalGeneration

[[autodoc]] TFBlipForConditionalGeneration

- call

## TFBlipForImageTextRetrieval

[[autodoc]] TFBlipForImageTextRetrieval

- call

## TFBlipForQuestionAnswering

[[autodoc]] TFBlipForQuestionAnswering

- call

</tf>

</frameworkcontent>