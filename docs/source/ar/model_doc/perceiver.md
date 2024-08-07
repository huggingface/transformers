# Perceiver

## نظرة عامة

اقترح نموذج Perceiver IO في ورقة "Perceiver IO: A General Architecture for Structured Inputs & Outputs" بواسطة Andrew Jaegle وآخرون.

Perceiver IO هو تعميم لنموذج Perceiver للتعامل مع المخرجات الاعتباطية بالإضافة إلى المدخلات الاعتباطية. كان النموذج الأصلي لـ Perceiver ينتج فقط تصنيفًا واحدًا. بالإضافة إلى تصنيفات العلامات، يمكن لـ Perceiver IO إنتاج (على سبيل المثال) اللغة والتدفق البصري ومقاطع الفيديو متعددة الوسائط مع الصوت.

يتم ذلك باستخدام نفس الوحدات البنائية للنموذج الأصلي لـ Perceiver. تعقيد Perceiver IO الحسابي خطي في حجم الإدخال والإخراج، ويحدث الجزء الأكبر من المعالجة في مساحة المخفية، مما يسمح لنا بمعالجة المدخلات والمخرجات التي تكون أكبر بكثير مما يمكن أن تتعامل معه محولات القياسية. وهذا يعني، على سبيل المثال، أن Perceiver IO يمكنه إجراء نمذجة اللغة المقنعة على طريقة BERT مباشرة باستخدام البايت بدلاً من المدخلات المعلمة.

المقتطف من الورقة هو كما يلي:

*يحصل نموذج Perceiver المقترح مؤخرًا على نتائج جيدة في عدة مجالات (الصور والصوت والوسائط المتعددة وسحب النقاط) مع الحفاظ على النطاق الخطي في الحساب والذاكرة مع حجم الإدخال. في حين أن Perceiver يدعم العديد من أنواع المدخلات، إلا أنه يمكنه فقط إنتاج مخرجات بسيطة جدًا مثل درجات الفصل. يتغلب Perceiver IO على هذا القيد دون التضحية بخصائص الأصل من خلال التعلم لاستعلام مساحة المخفية للنموذج بشكل مرن لإنتاج مخرجات ذات حجم ودلالة اعتباطية. لا يزال Perceiver IO يفك ارتباط عمق النموذج من حجم البيانات، ولا يزال يتدرج خطيًا مع حجم البيانات، ولكن الآن فيما يتعلق بحجم الإدخال والمخرجات. يحقق نموذج Perceiver IO الكامل نتائج قوية في المهام ذات مساحات الإخراج المنظمة للغاية، مثل الفهم الطبيعي للغة والرؤية، وStarCraft II، ومجالات المهام المتعددة والوسائط المتعددة. على سبيل المثال، يطابق Perceiver IO خط أساس BERT القائم على المحول على معيار لغة GLUE دون الحاجة إلى تمييز المدخلات ويحقق أداءً متميزًا في تقدير التدفق البصري لـ Sintel.*

هذا هو الشرح المبسط لكيفية عمل Perceiver:

المشكلة الرئيسية مع آلية الاهتمام الذاتي للمحول هي أن متطلبات الوقت والذاكرة تتناسب طرديًا مع طول التسلسل. وبالتالي، فإن النماذج مثل BERT وRoBERTa تقتصر على طول تسلسل أقصاه 512 رمزًا. يهدف Perceiver إلى حل هذه المشكلة من خلال، بدلاً من أداء الاهتمام الذاتي على المدخلات، أدائه على مجموعة من المتغيرات المخفية، واستخدام المدخلات فقط للاهتمام المتقاطع. بهذه الطريقة، لا تعتمد متطلبات الوقت والذاكرة على طول المدخلات بعد الآن، حيث تستخدم عددًا ثابتًا من المتغيرات المخفية، مثل 256 أو 512. يتم تهيئتها بشكل عشوائي، وبعد ذلك يتم تدريبها من النهاية إلى النهاية باستخدام backpropagation.

داخلياً، سينشئ [`PerceiverModel`] المخفية، والتي هي عبارة عن مصفوفة ذات شكل `(batch_size، num_latents، d_latents)`. يجب توفير `inputs` (التي يمكن أن تكون نصًا أو صورًا أو صوتًا، أي شيء!) إلى النموذج، والذي سيستخدمه لأداء الاهتمام المتقاطع مع المخفية. إخراج مشفر Perceiver عبارة عن مصفوفة بنفس الشكل. بعد ذلك، يمكن للمرء، على غرار BERT، تحويل الحالات المخفية الأخيرة للمخفية إلى احتمالات التصنيف عن طريق حساب المتوسط على طول البعد التسلسلي، ووضع طبقة خطية في الأعلى لمشروع `d_latents` إلى `num_labels`.

كانت هذه هي فكرة ورقة Perceiver الأصلية. ومع ذلك، فقد كان بإمكانه فقط إخراج احتمالات التصنيف. في عمل متابعة، PerceiverIO، قاموا بتعميمه للسماح للنموذج أيضًا بإنتاج مخرجات ذات حجم اعتباطي. كيف، قد تتساءل؟ الفكرة بسيطة نسبيًا: يتم تعريف المخرجات ذات الحجم الاعتباطي، ثم يتم تطبيق الاهتمام المتقاطع مع الحالات المخفية الأخيرة، باستخدام المخرجات كاستفسارات، والمخفية كمفاتيح وقيم.

لذلك، لنفترض أن المرء يريد إجراء نمذجة اللغة المقنعة (على طريقة BERT) باستخدام Perceiver. نظرًا لأن طول إدخال Perceiver لن يؤثر على وقت الحساب لطبقات الاهتمام الذاتي، فيمكنك توفير البايتات الخام، وتوفير `inputs` بطول 2048 للنموذج. إذا قمت الآن بقناع بعض هذه الرموز 2048، فيمكنك تعريف `outputs` على أنها ذات شكل: `(batch_size، 2048، 768)`. بعد ذلك، قم بأداء الاهتمام المتقاطع مع الحالات المخفية النهائية لتحديث مصفوفة `outputs`. بعد الاهتمام المتقاطع، لا يزال لديك مصفوفة ذات شكل `(batch_size، 2048، 768)`. يمكنك بعد ذلك وضع رأس نمذجة اللغة العادي في الأعلى، لمشروع البعد الأخير إلى حجم مفردة النموذج، أي إنشاء احتمالات ذات شكل `(batch_size، 2048، 262)` (نظرًا لأن Perceiver يستخدم حجم مفردة 262 معرفات بايت).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perceiver_architecture.jpg" alt="drawing" width="600"/>

<small> تم المساهم بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). يمكن العثور على الكود الأصلي [هنا](https://github.com/deepmind/deepmind-research/tree/master/perceiver). </small>

<Tip warning={true}>

Perceiver لا يعمل مع `torch.nn.DataParallel` بسبب وجود خلل في PyTorch، راجع [القضية #36035](https://github.com/pytorch/pytorch/issues/36035)

</Tip>

## الموارد

- أسرع طريقة للبدء في استخدام Perceiver هي التحقق من [دفاتر الملاحظات التعليمية](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Perceiver).

- راجع [منشور المدونة](https://huggingface.co/blog/perceiver) إذا كنت تريد فهم كيفية عمل النموذج وتنفيذه في المكتبة بشكل كامل. لاحظ أن النماذج المتوفرة في المكتبة تعرض فقط بعض الأمثلة على ما يمكنك فعله باستخدام Perceiver. هناك العديد من حالات الاستخدام الأخرى، بما في ذلك الإجابة على الأسئلة، والتعرف على الكيانات المسماة، وكشف الأشياء، وتصنيف الصوت، وتصنيف الفيديو، وما إلى ذلك.

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)

- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)

- [دليل مهمة تصنيف الصور](../tasks/image_classification)

## مخرجات Perceiver المحددة

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverModelOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverDecoderOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassifierOutput

## PerceiverConfig

[[autodoc]] PerceiverConfig

## PerceiverTokenizer

[[autodoc]] PerceiverTokenizer

- __call__

## PerceiverFeatureExtractor

[[autodoc]] PerceiverFeatureExtractor

- __call__

## PerceiverImageProcessor

[[autodoc]] PerceiverImageProcessor

- preprocess

## PerceiverTextPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverTextPreprocessor

## PerceiverImagePreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverImagePreprocessor

## PerceiverOneHotPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverOneHotPreprocessor

## PerceiverAudioPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverAudioPreprocessor

## PerceiverMultimodalPreprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor

## PerceiverProjectionDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverProjectionDecoder

## PerceiverBasicDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverBasicDecoder

## PerceiverClassificationDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassificationDecoder

## PerceiverOpticalFlowDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder

## PerceiverBasicVideoAutoencodingDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverBasicVideoAutoencodingDecoder

## PerceiverMultimodalDecoder

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder

## PerceiverProjectionPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverProjectionPostprocessor

## PerceiverAudioPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverAudioPostprocessor

## PerceiverClassificationPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverClassificationPostprocessor

## PerceiverMultimodalPostprocessor

[[autodoc]] models.perceiver.modeling_perceiver.PerceiverMultimodalPostprocessor

## PerceiverModel

[[autodoc]] PerceiverModel

- forward

## PerceiverForMaskedLM

[[autodoc]] PerceiverForMaskedLM

- forward

## PerceiverForSequenceClassification

[[autodoc]] PerceiverForSequenceClassification

- forward

## PerceiverForImageClassificationLearned

[[autodoc]] PerceiverForImageClassificationLearned

- forward

## PerceiverForImageClassificationFourier

[[autodoc]] PerceiverForImageClassificationFourier

- forward

## PerceiverForImageClassificationConvProcessing


[[autodoc]] PerceiverForImageClassificationConvProcessing

- forward

## PerceiverForOpticalFlow

[[autodoc]] PerceiverForOpticalFlow

- forward

## PerceiverForMultimodalAutoencoding

[[autodoc]] PerceiverForMultimodalAutoencoding

- forward