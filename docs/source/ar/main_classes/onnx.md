# تصدير نماذج 🤗 Transformers إلى ONNX

يوفر 🤗 Transformers حزمة transformers.onnx التي تتيح لك تحويل نقاط تفتيش النموذج إلى رسم ONNX باستخدام كائنات التكوين. راجع الدليل حول تصدير نماذج 🤗 Transformers لمزيد من التفاصيل.

## تكوينات ONNX

نقدم ثلاث فئات مجردة يجب أن ترث منها، اعتمادًا على نوع بنية النموذج الذي تريد تصديره:

* ترث النماذج المستندة إلى الترميز من [`~onnx.config.OnnxConfig`]
* ترث النماذج المستندة إلى فك الترميز من [`~onnx.config.OnnxConfigWithPast`]
* ترث نماذج الترميز-فك الترميز من [`~onnx.config.OnnxSeq2SeqConfigWithPast`]

### OnnxConfig

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ميزات ONNX

يرتبط كل تكوين ONNX بمجموعة من الميزات التي تتيح لك تصدير النماذج لأنواع مختلفة من التوبولوجيات أو المهام.

### FeaturesManager

[[autodoc]] onnx.features.FeaturesManager