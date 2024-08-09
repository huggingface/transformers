# CLAP

## نظرة عامة
تم اقتراح نموذج CLAP في [Large Scale Contrastive Language-Audio pretraining with feature fusion and keyword-to-caption augmentation](https://arxiv.org/pdf/2211.06687.pdf) بواسطة Yusong Wu و Ke Chen و Tianyu Zhang و Yuchen Hui و Taylor Berg-Kirkpatrick و Shlomo Dubnov.

CLAP (Contrastive Language-Audio Pretraining) هي شبكة عصبية تم تدريبها على مجموعة متنوعة من أزواج (الصوت، النص). يمكن توجيهه للتنبؤ بمقطع النص الأكثر ملاءمة، بالنظر إلى الصوت، دون تحسين المهمة مباشرةً. يستخدم نموذج CLAP محول SWINTransformer لاستخراج ميزات الصوت من مدخلات مخطط Mel اللوغاريتمي، ونموذج RoBERTa للحصول على ميزات النص. بعد ذلك، يتم إسقاط كل من ميزات النص والصوت إلى مساحة كامنة ذات أبعاد متطابقة. ثم يتم استخدام حاصل الضرب النقطي بين ميزات الصوت والنص المسقط كدرجة تشابه.

الملخص من الورقة هو كما يلي:

> أظهر التعلم التبايني نجاحًا ملحوظًا في مجال تعلم التمثيل متعدد الوسائط. في هذه الورقة، نقترح خط أنابيب للتعلم التبايني للغة الصوت لتطوير تمثيل صوتي من خلال دمج بيانات الصوت مع أوصاف اللغة الطبيعية. لتحقيق هذا الهدف، نقوم أولاً بإصدار LAION-Audio-630K، وهو مجموعة كبيرة من 633,526 زوجًا من الصوت والنص من مصادر بيانات مختلفة. ثانيًا، نقوم ببناء نموذج للتعلم التبايني للغة الصوت من خلال مراعاة مختلف برامج ترميز الصوت ومشفرات النص. ندمج آلية دمج الميزات وتعزيز الكلمات الرئيسية إلى العناوين في تصميم النموذج لتمكين النموذج بشكل أكبر من معالجة إدخالات الصوت ذات الأطوال المتغيرة وتعزيز الأداء. ثالثًا، نقوم بإجراء تجارب شاملة لتقييم نموذجنا عبر ثلاث مهام: استرجاع النص إلى الصوت، والتصنيف الصوتي باستخدام الصفر، والتصنيف الصوتي الخاضع للإشراف. تُظهر النتائج أن نموذجنا يحقق أداءً متفوقًا في مهمة استرجاع النص إلى الصوت. في مهام التصنيف الصوتي، يحقق النموذج أداءً متميزًا في إعداد الصفر ويتمكن من الحصول على أداء قابل للمقارنة مع نتائج النماذج في الإعداد غير الصفري. LAION-Audio-6

تمت المساهمة بهذا النموذج من قبل [Younes Belkada](https://huggingface.co/ybelkada) و [Arthur Zucker](https://huggingface.co/ArthurZ).
يمكن العثور على الكود الأصلي [هنا](https://github.com/LAION-AI/Clap).

## ClapConfig

[[autodoc]] ClapConfig

- from_text_audio_configs

## ClapTextConfig

[[autodoc]] ClapTextConfig

## ClapAudioConfig

[[autodoc]] ClapAudioConfig

## ClapFeatureExtractor

[[autodoc]] ClapFeatureExtractor

## ClapProcessor

[[autodoc]] ClapProcessor

## ClapModel

[[autodoc]] ClapModel

- forward

- get_text_features

- get_audio_features

## ClapTextModel

[[autodoc]] ClapTextModel

- forward

## ClapTextModelWithProjection


[[autodoc]] ClapTextModelWithProjection

- forward

## ClapAudioModel

[[autodoc]] ClapAudioModel

- forward

## ClapAudioModelWithProjection


[[autodoc]] ClapAudioModelWithProjection

- forward