# التوليد 

يحتوي كل إطار عمل على طريقة توليد للنص مُنفذة في فئة `GenerationMixin` الخاصة بها:

- PyTorch [`~generation.GenerationMixin.generate`] مُنفذة في [`~generation.GenerationMixin`].
- TensorFlow [`~generation.TFGenerationMixin.generate`] مُنفذة في [`~generation.TFGenerationMixin`].
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] مُنفذة في [`~generation.FlaxGenerationMixin`].

بغض النظر عن إطار العمل الذي تختاره، يمكنك تحديد طريقة التوليد باستخدام فئة [`~generation.GenerationConfig`]
راجع هذه الفئة للحصول على القائمة الكاملة لمعلمات التوليد، والتي تتحكم في سلوك طريقة التوليد.

لمعرفة كيفية فحص تكوين التوليد الخاص بالنموذج، وما هي القيم الافتراضية، وكيفية تغيير المعلمات حسب الحاجة،
وكيفية إنشاء وحفظ تكوين توليد مخصص، راجع دليل
[استراتيجيات توليد النص](../generation_strategies). كما يشرح الدليل كيفية استخدام الميزات ذات الصلة،
مثل بث الرموز.

## GenerationConfig

[[autodoc]] generation.GenerationConfig

- from_pretrained
- from_model_config
- save_pretrained
- update
- validate
- get_generation_mode

[[autodoc]] generation.WatermarkingConfig

## GenerationMixin

[[autodoc]] generation.GenerationMixin

- generate
- compute_transition_scores

## TFGenerationMixin

[[autodoc]] generation.TFGenerationMixin

- generate
- compute_transition_scores

## FlaxGenerationMixin

[[autodoc]] generation.FlaxGenerationMixin

- generate