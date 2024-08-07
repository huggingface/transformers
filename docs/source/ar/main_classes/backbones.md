# Backbone

عمود فقري هو نموذج يستخدم لاستخراج الميزات لمهام رؤية الكمبيوتر ذات المستوى الأعلى مثل اكتشاف الأشياء وتصنيف الصور. يوفر Transformers فئة [`AutoBackbone`] لتهيئة عمود فقري من أوزان النموذج المُدرب مسبقًا، وفئتين مساعدتين:

* [`~utils.BackboneMixin`] تمكن من تهيئة عمود فقري من Transformers أو [timm](https://hf.co/docs/timm/index) وتشمل وظائف لإرجاع ميزات ومؤشرات الإخراج.
* [`~utils.BackboneConfigMixin`] يحدد ميزات ومؤشرات إخراج تكوين العمود الفقري.

يتم تحميل نماذج [timm](https://hf.co/docs/timm/index) باستخدام فئات [`TimmBackbone`] و [`TimmBackboneConfig`].

يتم دعم العمود الفقري للنماذج التالية:

* [BEiT](..model_doc/beit)
* [BiT](../model_doc/bit)
* [ConvNet](../model_doc/convnext)
* [ConvNextV2](../model_doc/convnextv2)
* [DiNAT](..model_doc/dinat)
* [DINOV2](../model_doc/dinov2)
* [FocalNet](../model_doc/focalnet)
* [MaskFormer](../model_doc/maskformer)
* [NAT](../model_doc/nat)
* [ResNet](../model_doc/resnet)
* [Swin Transformer](../model_doc/swin)
* [Swin Transformer v2](../model_doc/swinv2)
* [ViTDet](../model_doc/vitdet)

## AutoBackbone

[[autodoc]] AutoBackbone

## BackboneMixin

[[autodoc]] utils.BackboneMixin

## BackboneConfigMixin

[[autodoc]] utils.BackboneConfigMixin

## TimmBackbone

[[autodoc]] models.timm_backbone.TimmBackbone

## TimmBackboneConfig

[[autodoc]] models.timm_backbone.TimmBackboneConfig