# التهيئة 

تنفذ الفئة الأساسية [PretrainedConfig] الأساليب الشائعة لتحميل/حفظ تهيئة إما من ملف أو دليل محلي، أو من تهيئة نموذج مُدرب مسبقًا يوفرها المكتبة (تم تنزيلها من مستودع HuggingFace AWS S3).

تنفذ كل فئة تهيئة مشتقة سمات خاصة بالنموذج. السمات الشائعة الموجودة في جميع فئات التهيئة هي: hidden_size، وnum_attention_heads، وnum_hidden_layers. وتنفذ النماذج النصية كذلك: vocab_size.

## PretrainedConfig

[[autodoc]] PretrainedConfig

- push_to_hub

- all