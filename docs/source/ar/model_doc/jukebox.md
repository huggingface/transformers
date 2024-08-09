# Jukebox

**ملاحظة:** هذا النموذج في وضع الصيانة فقط، ولا نقبل أي طلبات سحب (Pull Requests) جديدة لتغيير شفرته البرمجية. في حالة مواجهة أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2. يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.

## نظرة عامة

اقترح نموذج Jukebox في الورقة البحثية [Jukebox: A generative model for music](https://arxiv.org/pdf/2005.00341.pdf) بواسطة Prafulla Dhariwal وHeewoo Jun وChristine Payne وJong Wook Kim وAlec Radford وIlya Sutskever. ويقدم نموذجًا موسيقيًا توليديًا يمكنه إنتاج عينات طويلة تصل إلى دقيقة، ويمكن تكييفها مع فنان أو نوع موسيقي أو كلمات أغنية.

الملخص من الورقة البحثية هو كما يلي:

> "نقدم نموذج Jukebox، وهو نموذج لتوليد الموسيقى مع الغناء في مجال الصوت الخام. نعالج السياق الطويل للصوت الخام باستخدام VQ-VAE متعدد المقاييس لضغطه إلى رموز منفصلة، ونقوم بنمذجة تلك الرموز باستخدام محولات Transformers ذاتية الارتباط. نُظهر أن النموذج المدمج على نطاق واسع يمكنه توليد أغانٍ عالية الدقة ومتنوعة مع اتساق يصل إلى عدة دقائق. يمكننا تكييف النموذج حسب الفنان والنوع الموسيقي لتوجيه الأسلوب الموسيقي والغناء، وحسب كلمات الأغنية غير المترابطة لجعل الغناء أكثر قابلية للتحكم. نقوم بإطلاق آلاف العينات غير المختارة يدويًا، إلى جانب أوزان النموذج والشيفرة البرمجية."

كما هو موضح في الشكل التالي، يتكون نموذج Jukebox من 3 "مولدات" (Priors) وهي عبارة عن نماذج فك تشفير فقط. وتتبع هذه المولدات البنية الموضحة في الورقة البحثية [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)، مع تعديلها لدعم طول سياق أطول.

أولاً، يتم استخدام ترميز فك تشفير ذاتي (Autoencoder) لتشفير كلمات الأغنية النصية. بعد ذلك، يقوم المولد الأول (يسمى أيضًا "top_prior") بالاهتمام بحالات الإخفاء الأخيرة المستخرجة من ترميز كلمات الأغنية. يتم ربط المولدات بالنماذج السابقة على التوالي عبر وحدة "AudioConditioner". وتقوم وحدة "AudioConditioner" بزيادة دقة إخراج النموذج السابق إلى رموز خام بمعدل إطارات صوتية معينة في الثانية.

تتم تمرير البيانات الوصفية مثل اسم الفنان والنوع الموسيقي والتوقيت إلى كل مولد، في شكل رمز بداية (Start Token) ومؤشر موضعي (Positional Embedding) لبيانات التوقيت. يتم رسم حالات الإخفاء إلى أقرب متجه رمزي من VQVAE لتحويلها إلى صوت خام.

![JukeboxModel](https://gist.githubusercontent.com/ArthurZucker/92c1acaae62ebf1b6a951710bdd8b6af/raw/c9c517bf4eff61393f6c7dec9366ef02bdd059a3/jukebox.svg)

تمت المساهمة بهذا النموذج من قبل [Arthur Zucker](https://huggingface.co/ArthurZ). ويمكن إيجاد الشيفرة البرمجية الأصلية [هنا](https://github.com/openai/jukebox).

## نصائح الاستخدام

- يدعم هذا النموذج الاستنتاج فقط. هناك عدة أسباب لذلك، معظمها يرجع إلى أن تدريب النموذج يتطلب كمية كبيرة من الذاكرة. يمكنك فتح طلب سحب (Pull Request) وإضافة ما ينقص لإتاحة الدمج الكامل مع مدرب Hugging Face!

- هذا النموذج بطيء للغاية، ويستغرق 8 ساعات لتوليد دقيقة واحدة من الصوت باستخدام المولد الأعلى (top prior) على وحدة معالجة الرسوميات (GPU) من نوع V100. وللمعالجة التلقائية للجهاز الذي يجب أن ينفذ عليه النموذج، استخدم "accelerate".

- على عكس الورقة البحثية، فإن ترتيب المولدات يبدأ من "0" إلى "1" لأنه بدا أكثر بديهية: نبدأ بالنمذجة من "0".

- تتطلب عملية توليد العينات الموجهة (Primed sampling) (تكييف عملية التوليد على صوت خام) ذاكرة أكبر من عملية توليد العينات الأسية (Ancestral sampling) ويجب استخدامها مع تعيين "fp16" إلى "True".

تمت المساهمة بهذا النموذج من قبل [Arthur Zucker](https://huggingface.co/ArthurZ). ويمكن إيجاد الشيفرة البرمجية الأصلية [هنا](https://github.com/openai/jukebox).

## JukeboxConfig

[[autodoc]] JukeboxConfig

## JukeboxPriorConfig

[[autodoc]] JukeboxPriorConfig

## JukeboxVQVAEConfig

[[autodoc]] JukeboxVQVAEConfig

## JukeboxTokenizer

[[autodoc]] JukeboxTokenizer

- save_vocabulary

## JukeboxModel

[[autodoc]] JukeboxModel

- ancestral_sample

- primed_sample

- continue_sample

- upsample

- _sample

## JukeboxPrior

[[autodoc]] JukeboxPrior

- sample

- forward

## JukeboxVQVAE

[[autodoc]] JukeboxVQVAE

- forward

- encode

- decode