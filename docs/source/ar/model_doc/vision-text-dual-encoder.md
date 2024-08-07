# VisionTextDualEncoder

## نظرة عامة
يمكن استخدام [`VisionTextDualEncoderModel`] لتهيئة نموذج مشفر مزدوج للرؤية والنص مع أي نموذج رؤية ذاتي التشفير مسبق التدريب كمشفر رؤية (على سبيل المثال [ViT](vit)، [BEiT](beit)، [DeiT](deit)) وأي نموذج نص ذاتي التشفير مسبق التدريب كمشفر نص (على سبيل المثال [RoBERTa](roberta)، [BERT](bert)). يتم إضافة طبقتين من الطبقات العلوية فوق كل من مشفر الرؤية ومشفر النص لمشروعات المخرجات المضمنة إلى مساحة كامنة مشتركة. يتم تهيئة الطبقات العلوية بشكل عشوائي، لذلك يجب ضبط نموذج دقيق على مهمة أسفل البنية. يمكن استخدام هذا النموذج لمواءمة تضمين الرؤية والنص باستخدام تدريب الصور النصية المتباينة على غرار CLIP، وبعد ذلك يمكن استخدامه لمهمات الرؤية بدون إشراف مثل تصنيف الصور أو الاسترجاع.

في [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) يظهر كيف أن الاستفادة من النموذج المسبق التدريب (المغلق/المجمد) للصورة والنص للتعلم التبايني يحقق تحسنًا كبيرًا في مهام الرؤية الجديدة بدون إشراف مثل تصنيف الصور أو الاسترجاع.

## VisionTextDualEncoderConfig

[[autodoc]] VisionTextDualEncoderConfig

## VisionTextDualEncoderProcessor

[[autodoc]] VisionTextDualEncoderProcessor

<frameworkcontent>
<pt>

## VisionTextDualEncoderModel

[[autodoc]] VisionTextDualEncoderModel

- forward

</pt>
<tf>

## FlaxVisionTextDualEncoderModel

[[autodoc]] FlaxVisionTextDualEncoderModel

- __call__

</tf>
<jax>

## TFVisionTextDualEncoderModel

[[autodoc]] TFVisionTextDualEncoderModel

- call

</jax>
</frameworkcontent>