# Wav2Vec2-BERT

## نظرة عامة
يقترح فريق Seamless Communication من Meta AI نموذج Wav2Vec2-BERT في [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) .

تم تدريب هذا النموذج مسبقًا على 4.5 مليون ساعة من بيانات الصوت غير المعلمة التي تغطي أكثر من 143 لغة. يجب ضبط دقة النموذج لاستخدامه في المهام النهائية مثل التعرف التلقائي على الكلام (ASR) أو التصنيف الصوتي.

يمكن العثور على النتائج الرسمية للنموذج في القسم 3.2.1 من الورقة.

المستخلص من الورقة هو ما يلي:

*حققت التطورات الأخيرة في الترجمة الكلامية التلقائية توسعًا كبيرًا في تغطية اللغة، وتحسين القدرات متعددة الوسائط، وتمكين مجموعة واسعة من المهام والوظائف. ومع ذلك، تفتقر أنظمة الترجمة الكلامية واسعة النطاق اليوم إلى ميزات رئيسية تساعد في جعل التواصل بوساطة الآلة سلسًا عند مقارنته بالحوار بين البشر. في هذا العمل، نقدم عائلة من النماذج التي تمكن الترجمات التعبيرية والمتعددة اللغات بطريقة متواصلة. أولاً، نساهم في تقديم نسخة محسنة من نموذج SeamlessM4T متعدد اللغات والوسائط بشكل كبير - SeamlessM4T v2. تم تدريب هذا النموذج الأحدث، الذي يتضمن إطار عمل UnitY2 المحدث، على المزيد من بيانات اللغات منخفضة الموارد. تضيف النسخة الموسعة من SeamlessAlign 114,800 ساعة من البيانات المحاذاة تلقائيًا لما مجموعه 76 لغة. يوفر SeamlessM4T v2 الأساس الذي تم بناء أحدث نموذجين لدينا، SeamlessExpressive و SeamlessStreaming، عليه. يتيح SeamlessExpressive الترجمة التي تحافظ على الأساليب الصوتية والتنغيم. مقارنة بالجهود السابقة في أبحاث الكلام التعبيري، يعالج عملنا بعض الجوانب الأقل استكشافًا في علم العروض، مثل معدل الكلام والتوقفات، مع الحفاظ على أسلوب صوت الشخص في نفس الوقت. أما بالنسبة لـ SeamlessStreaming، فإن نموذجنا يستفيد من آلية Efficient Monotonic Multihead Attention (EMMA) لتوليد ترجمات مستهدفة منخفضة الكمون دون انتظار عبارات المصدر الكاملة. يعد SeamlessStreaming الأول من نوعه، حيث يمكّن الترجمة المتزامنة من الكلام إلى الكلام/النص لعدة لغات مصدر وهدف. لفهم أداء هذه النماذج، قمنا بدمج الإصدارات الجديدة والمعدلة من المقاييس التلقائية الموجودة لتقييم علم العروض والكمون والمتانة. بالنسبة للتقييمات البشرية، قمنا بتكييف البروتوكولات الموجودة المصممة لقياس أكثر السمات ملاءمة في الحفاظ على المعنى والطبيعية والتعبير. ولضمان إمكانية استخدام نماذجنا بشكل آمن ومسؤول، قمنا بتنفيذ أول جهد معروف للفريق الأحمر للترجمة الآلية متعددة الوسائط، ونظام للكشف عن السمية المضافة والتخفيف منها، وتقييم منهجي للتحيز بين الجنسين، وآلية ترميز علامات مائية محلية غير مسموعة مصممة للتخفيف من تأثيرات deepfakes. وبالتالي، فإننا نجمع بين المكونات الرئيسية من SeamlessExpressive و SeamlessStreaming لتشكيل Seamless، وهو أول نظام متاح للجمهور يفتح التواصل التعبيري عبر اللغات في الوقت الفعلي. في الختام، يمنحنا Seamless نظرة محورية على الأساس التقني اللازم لتحويل المترجم الكلامي العالمي من مفهوم الخيال العلمي إلى تقنية واقعية. وأخيرًا، يتم إطلاق المساهمات في هذا العمل - بما في ذلك النماذج والشفرة وكاشف العلامات المائية - علنًا ويمكن الوصول إليها من خلال الرابط أدناه.*

تمت المساهمة بهذا النموذج من قبل [ylacombe](https://huggingface.co/ylacombe). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/seamless_communication) .

## نصائح الاستخدام

- يتبع Wav2Vec2-BERT نفس بنية Wav2Vec2-Conformer، ولكنه يستخدم طبقة تعميق محورية ومدخلات تمثيل mel-spectrogram للصوت بدلاً من شكل الموجة الخام.

- يمكن أن يستخدم Wav2Vec2-BERT إما عدم وجود أي تضمين موضع نسبي، أو تضمينات موضع مثل Shaw، أو تضمينات موضع مثل Transformer-XL، أو تضمينات موضع دوارة عن طريق تعيين `config.position_embeddings_type` الصحيح.

- يقدم Wav2Vec2-BERT أيضًا شبكة محول تكيّف قائمة على Conformer بدلاً من شبكة Convolutional بسيطة.

## الموارد

- [`Wav2Vec2BertForCTC`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition) هذا.

- يمكنك أيضًا تكييف دفاتر الملاحظات هذه حول [كيفية ضبط دقة نموذج التعرف على الكلام باللغة الإنجليزية](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb)، و [كيفية ضبط دقة نموذج التعرف على الكلام بأي لغة](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).

- يمكن استخدام [`Wav2Vec2BertForSequenceClassification`] عن طريق تكييف [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) هذا.

- راجع أيضًا: [دليل مهام التصنيف الصوتي](../tasks/audio_classification)

## Wav2Vec2BertConfig

[[autodoc]] Wav2Vec2BertConfig

## Wav2Vec2BertProcessor

[[autodoc]] Wav2Vec2BertProcessor

- __call__
- pad
- from_pretrained
- save_pretrained
- batch_decode
- decode

## Wav2Vec2BertModel

[[autodoc]] Wav2Vec2BertModel

- forward

## Wav2Vec2BertForCTC

[[autodoc]] Wav2Vec2BertForCTC

- forward

## Wav2Vec2BertForSequenceClassification

[[autodoc]] Wav2Vec2BertForSequenceClassification

- forward

## Wav2Vec2BertForAudioFrameClassification

[[autodoc]] Wav2Vec2BertForAudioFrameClassification

- forward

## Wav2Vec2BertForXVector

[[autodoc]] Wav2Vec2BertForXVector

- forward