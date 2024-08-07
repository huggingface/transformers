# SpeechT5

## نظرة عامة

اقترح نموذج SpeechT5 في بحث "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing" من قبل Junyi Ao وآخرون.

ملخص البحث هو كما يلي:

*انطلاقًا من نجاح نموذج T5 (Text-To-Text Transfer Transformer) في معالجة اللغات الطبيعية المدَربة مسبقًا، نقترح إطار عمل SpeechT5 الموحد للطريقة الذي يستكشف التدريب المسبق للترميز فك الترميز للتعلم الذاتي التمثيل اللغوي المنطوق/المكتوب. ويتكون إطار عمل SpeechT5 من شبكة ترميز وفك ترميز مشتركة وست شبكات محددة للطريقة (منطوقة/مكتوبة) للمراحل السابقة/اللاحقة. وبعد معالجة المدخلات المنطوقة/المكتوبة من خلال المراحل السابقة، تقوم شبكة الترميز وفك الترميز المشتركة بنمذجة التحويل من سلسلة إلى أخرى، ثم تقوم المراحل اللاحقة بتوليد المخرجات في طريقة الكلام/النص بناءً على مخرجات فك الترميز. وباستخدام البيانات الكبيرة غير الموسومة للكلام والنص، نقوم بتدريب SpeechT5 مسبقًا لتعلم تمثيل موحد للطريقة، على أمل تحسين قدرة النمذجة لكل من الكلام والنص. ولمواءمة المعلومات النصية والمنطوقة في هذه المساحة الدلالية الموحدة، نقترح نهجًا للتحليل الكمي متعدد الطرق الذي يخلط عشوائيًا حالات الكلام/النص مع الوحدات الكامنة كوسيط بين الترميز وفك الترميز. وتظهر التقييمات الشاملة تفوق إطار عمل SpeechT5 المقترح في مجموعة واسعة من مهام معالجة اللغة المنطوقة، بما في ذلك التعرف التلقائي على الكلام، وتوليف الكلام، وترجمة الكلام، وتحويل الصوت، وتحسين الكلام، والتعرف على المتحدث.*

تمت المساهمة بهذا النموذج من قبل [Matthijs](https://huggingface.co/Matthijs). ويمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/SpeechT5).

## SpeechT5Config

[[autodoc]] SpeechT5Config

## SpeechT5HifiGanConfig

[[autodoc]] SpeechT5HifiGanConfig

## SpeechT5Tokenizer

[[autodoc]] SpeechT5Tokenizer

- __call__

- save_vocabulary

- decode

- batch_decode

## SpeechT5FeatureExtractor

[[autodoc]] SpeechT5FeatureExtractor

- __call__

## SpeechT5Processor

[[autodoc]] SpeechT5Processor

- __call__

- pad

- from_pretrained

- save_pretrained

- batch_decode

- decode

## SpeechT5Model

[[autodoc]] SpeechT5Model

- forward

## SpeechT5ForSpeechToText

[[autodoc]] SpeechT5ForSpeechToText

- forward

## SpeechT5ForTextToSpeech

[[autodoc]] SpeechT5ForTextToSpeech

- forward

- generate

## SpeechT5ForSpeechToSpeech

[[autodoc]] SpeechT5ForSpeechToSpeech

- forward

- generate_speech

## SpeechT5HifiGan

[[autodoc]] SpeechT5HifiGan

- forward