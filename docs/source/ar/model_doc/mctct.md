# M-CTC-T

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code. If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0. You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## نظرة عامة

اقترح نموذج M-CTC-T في [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161) بواسطة Loren Lugosch وTatiana Likhomanenko وGabriel Synnaeve وRonan Collobert. النموذج عبارة عن محول Encoder بحجم 1 بليون معلمة، مع رأس CTC فوق 8065 تسمية حرفية ورأس تعريف لغة فوق 60 تسمية معرف لغة. تم تدريبه على Common Voice (الإصدار 6.1، إصدار ديسمبر 2020) وVoxPopuli. بعد التدريب على Common Voice وVoxPopuli، يتم تدريب النموذج على Common Voice فقط. التسميات هي نصوص حرفية غير معيارية (لم تتم إزالة علامات الترقيم والتهجئة). يأخذ النموذج كإدخال ميزات Mel filterbank من إشارة صوتية بتردد 16 كيلو هرتز.

الملخص من الورقة هو كما يلي:

> "أصبح التعلم شبه المُشرف من خلال وضع العلامات الوهمية ركنًا أساسيًا في أنظمة التعرف على الكلام أحادية اللغة المتقدمة. في هذا العمل، نقوم بتوسيع وضع العلامات الوهمية للتعرف على الكلام متعدد اللغات بشكل كبير مع 60 لغة. نقترح وصفة وضع علامات وهمية بسيطة تعمل بشكل جيد حتى مع اللغات منخفضة الموارد: تدريب نموذج متعدد اللغات مُشرف، وضبط دقته باستخدام التعلم شبه المُشرف على لغة مستهدفة، وإنشاء علامات وهمية لتلك اللغة، وتدريب نموذج نهائي باستخدام العلامات الوهمية لجميع اللغات، إما من الصفر أو عن طريق الضبط الدقيق. تُظهر التجارب على مجموعات بيانات Common Voice المُعنونة وVoxPopuli غير المُعنونة أن وصفتنا يمكن أن تنتج نموذجًا بأداء أفضل للعديد من اللغات التي تنتقل أيضًا بشكل جيد إلى LibriSpeech."

تمت المساهمة بهذا النموذج بواسطة [cwkeam](https://huggingface.co/cwkeam). يمكن العثور على الكود الأصلي [هنا](https://github.com/flashlight/wav2letter/tree/main/recipes/mling_pl).

## نصائح الاستخدام

تتوفر نسخة PyTorch من هذا النموذج فقط في الإصدار 1.9 والإصدارات الأحدث.

## الموارد

- [دليل مهام التعرف التلقائي على الكلام](../tasks/asr)

## MCTCTConfig

[[autodoc]] MCTCTConfig

## MCTCTFeatureExtractor

[[autodoc]] MCTCTFeatureExtractor

- __call__

## MCTCTProcessor

[[autodoc]] MCTCTProcessor

- __call__
- from_pretrained
- save_pretrained
- batch_decode
- decode

## MCTCTModel

[[autodoc]] MCTCTModel

- forward

## MCTCTForCTC

[[autodoc]] MCTCTForCTC

- forward