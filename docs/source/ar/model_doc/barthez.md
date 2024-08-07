# BARThez
## نظرة عامة
اقترح نموذج BARThez في ورقة بحثية بعنوان "BARThez: a Skilled Pretrained French Sequence-to-Sequence Model" بواسطة Moussa Kamal Eddine و Antoine J.-P. Tixier و Michalis Vazirgiannis في 23 أكتوبر 2020.

ملخص الورقة البحثية:

*أحدث التعلم التحويلي الاستقرائي، الذي مكنه التعلم الذاتي الإشراف، ثورة في مجال معالجة اللغات الطبيعية بالكامل، مع نماذج مثل BERT و BART التي حطمت الأرقام القياسية في العديد من مهام فهم اللغة الطبيعية. وعلى الرغم من وجود بعض الاستثناءات الملحوظة، فإن معظم النماذج والبحوث المتاحة أجريت باللغة الإنجليزية. وفي هذا العمل، نقدم BARThez، أول نموذج BART للغة الفرنسية (على حد علمنا). تم تدريب نموذج BARThez مسبقًا على مجموعة بيانات ضخمة أحادية اللغة باللغة الفرنسية من الأبحاث السابقة التي قمنا بتكييفها لتناسب مخططات الإزعاج الخاصة بنموذج BART. وعلى عكس نماذج اللغة الفرنسية القائمة على BERT مثل CamemBERT و FlauBERT، فإن نموذج BARThez مناسب بشكل خاص للمهام التوليدية، حيث تم تدريب كلا من المشفر وفك المشفر مسبقًا. بالإضافة إلى المهام التمييزية من معيار FLUE، نقيم نموذج BARThez على مجموعة بيانات تلخيص جديدة، OrangeSum، والتي نقوم بإصدارها مع هذه الورقة. كما نواصل أيضًا التدريب المسبق لنموذج BART متعدد اللغات الذي تم تدريبه مسبقًا بالفعل على مجموعة بيانات BARThez، ونظهر أن النموذج الناتج، والذي نسميه mBARTHez، يوفر تحسنًا كبيرًا عن نموذج BARThez الأساسي، وهو على قدم المساواة مع نماذج CamemBERT و FlauBERT أو يتفوق عليها.*

تمت المساهمة بهذا النموذج بواسطة [moussakam](https://huggingface.co/moussakam). يمكن العثور على كود المؤلفين [هنا](https://github.com/moussaKam/BARThez).

## الموارد
- يمكن ضبط نموذج BARThez الدقيق على مهام التسلسل إلى تسلسل بطريقة مماثلة لنموذج BART، راجع: [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).

## BarthezTokenizer
[[autodoc]] BarthezTokenizer

## BarthezTokenizerFast
[[autodoc]] BarthezTokenizerFast