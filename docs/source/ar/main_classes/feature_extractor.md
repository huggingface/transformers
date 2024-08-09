# مستخرج الخصائص

مستخرج الخصائص هو المسؤول عن إعداد ميزات الإدخال لنماذج الصوت أو الرؤية. ويشمل ذلك استخراج الميزات من التسلسلات، مثل معالجة ملفات الصوت لتوليد ميزات مخطط Mel اللوغاريتمي، واستخراج الميزات من الصور، مثل قص ملفات الصور، وكذلك الحشو والتوحيد، والتحويل إلى نumpy و PyTorch و TensorFlow.

## FeatureExtractionMixin

[[autodoc]] feature_extraction_utils.FeatureExtractionMixin

- from_pretrained

- save_pretrained

## SequenceFeatureExtractor

[[autodoc]] SequenceFeatureExtractor

- pad

## BatchFeature

[[autodoc]] BatchFeature

## ImageFeatureExtractionMixin

[[autodoc]] image_utils.ImageFeatureExtractionMixin