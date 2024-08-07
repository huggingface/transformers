# REALM

<Tip warning={true}>
تم وضع هذا النموذج في وضع الصيانة فقط، ولن نقبل أي طلبات سحب (Pull Requests) جديدة لتغيير شفرته البرمجية. إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2. يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.
</Tip>

## نظرة عامة

تم اقتراح نموذج REALM في ورقة بحثية بعنوان [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) بواسطة Kelvin Guu وKenton Lee وZora Tung وPanupong Pasupat وMing-Wei Chang. وهو نموذج لغة معزز بالاسترجاع يقوم أولاً باسترجاع المستندات من مجموعة بيانات نصية ثم يستخدم المستندات المستردة لمعالجة مهام الإجابة على الأسئلة.

ملخص الورقة البحثية هو كما يلي:

> "أظهرت عملية التدريب المسبق لنموذج اللغة أنها تلتقط قدرًا مذهلاً من المعرفة بالعالم، وهو أمر بالغ الأهمية لمهام معالجة اللغة الطبيعية مثل الإجابة على الأسئلة. ومع ذلك، يتم تخزين هذه المعرفة ضمنيًا في معلمات شبكة عصبية، مما يتطلب شبكات أكبر باستمرار لتغطية المزيد من الحقائق. وللتقاط المعرفة بطريقة أكثر قابلية للتطوير وأكثر قابلية للتفسير، نقوم بتعزيز التدريب المسبق لنموذج اللغة باستخدام مسترد معرفي خفي، والذي يسمح للنموذج باسترداد والاهتمام بالمستندات من مجموعة بيانات كبيرة مثل ويكيبيديا، والتي يتم استخدامها أثناء التدريب المسبق والضبط الدقيق والاستنتاج. ولأول مرة، نُظهر كيفية تدريب مسترد المعرفة هذا بطريقة غير خاضعة للإشراف، باستخدام نمذجة اللغة المقنعة كإشارة تعلم، والانتشار الخلفي من خلال خطوة الاسترجاع التي تأخذ في الاعتبار ملايين المستندات. ونحن نثبت فعالية التدريب المسبق لنموذج اللغة المعزز بالاسترجاع (REALM) من خلال الضبط الدقيق على المهمة الصعبة للإجابة على الأسئلة المفتوحة المجال. نقوم بالمقارنة مع أحدث الطرازات لطرق تخزين المعرفة الصريحة والضمنية على ثلاثة معايير شائعة للإجابة على الأسئلة المفتوحة، ونجد أننا نتفوق على جميع الطرق السابقة بهامش كبير (4-16% دقة مطلقة)، مع تقديم فوائد نوعية مثل القابلية للتفسير والقابلية للتطوير."

تمت المساهمة بهذا النموذج من قبل [qqaatw](https://huggingface.co/qqaatw). يمكن العثور على الشفرة البرمجية الأصلية [هنا](https://github.com/google-research/language/tree/master/language/realm).

## RealmConfig

[[autodoc]] RealmConfig

## RealmTokenizer

[[autodoc]] RealmTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary
- batch_encode_candidates

## RealmTokenizerFast

[[autodoc]] RealmTokenizerFast

- batch_encode_candidates

## RealmRetriever

[[autodoc]] RealmRetriever

## RealmEmbedder

[[autodoc]] RealmEmbedder

- forward

## RealmScorer

[[autodoc]] RealmScorer

- forward

## RealmKnowledgeAugEncoder

[[autodoc]] RealmKnowledgeAugEncoder

- forward

## RealmReader

[[autodoc]] RealmReader

- forward

## RealmForOpenQA

[[autodoc]] RealmForOpenQA

- block_embedding_to
- forward