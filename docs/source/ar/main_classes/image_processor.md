# معالج الصور

يتولى معالج الصور مهمة إعداد ميزات الإدخال لنماذج الرؤية ومعالجة مخرجاتها. ويشمل ذلك تحويلات مثل تغيير الحجم والتحجيم والتحويل إلى تنسورات PyTorch و TensorFlow و Flax و Numpy. وقد يشمل أيضًا معالجة لاحقة خاصة بالنموذج مثل تحويل logits إلى أقنعة تجزئة.

## ImageProcessingMixin

[[autodoc]] image_processing_utils.ImageProcessingMixin

- from_pretrained

- save_pretrained

## BatchFeature

[[autodoc]] BatchFeature

## BaseImageProcessor

[[autodoc]] image_processing_utils.BaseImageProcessor

## BaseImageProcessorFast

[[autodoc]] image_processing_utils_fast.BaseImageProcessorFast