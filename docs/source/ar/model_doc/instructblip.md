# InstructBLIP

## نظرة عامة
تم اقتراح نموذج InstructBLIP في ورقة بحثية بعنوان [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) بواسطة Wenliang Dai و Junnan Li و Dongxu Li و Anthony Meng Huat Tiong و Junqi Zhao و Weisheng Wang و Boyang Li و Pascale Fung و Steven Hoi.

يستفيد InstructBLIP من بنية BLIP-2 في الضبط البصري للتعليمات.

هذا ملخص الورقة:

> "ظهرت نماذج اللغة متعددة الأغراض القادرة على حل مهام مختلفة في مجال اللغة، وذلك بفضل خط أنابيب ما قبل التدريب والضبط التعليمي. ومع ذلك، فإن بناء نماذج متعددة الوسائط متعددة الأغراض يمثل تحديًا بسبب زيادة التباين في المهام الذي تسببه المدخلات البصرية الإضافية. على الرغم من أن التدريب المسبق متعدد الوسائط قد تمت دراسته على نطاق واسع، إلا أن الضبط التعليمي متعدد الوسائط لم يتم استكشافه نسبيًا. في هذه الورقة، نجري دراسة منهجية وشاملة حول الضبط التعليمي متعدد الوسائط بناءً على النماذج المُدربة مسبقًا BLIP-2. نقوم بتجميع مجموعة متنوعة من 26 مجموعة بيانات متاحة للجمهور، وتحويلها إلى تنسيق الضبط التعليمي، وتصنيفها إلى مجموعتين للضبط التعليمي المُحتجز والتقييم الصفري المحتجز. بالإضافة إلى ذلك، نقدم طريقة استخراج الميزات البصرية الواعية بالتعليمات، وهي طريقة حاسمة تمكن النموذج من استخراج ميزات مفيدة مصممة خصيصًا للتعليمات المعطاة. تحقق نماذج InstructBLIP الناتجة أداءً متميزًا في جميع مجموعات البيانات المحتجزة البالغ عددها 13، متفوقة بشكل كبير على BLIP-2 و Flamingo الأكبر حجمًا. كما أن نماذجنا تؤدي إلى أداء متميز عند ضبط دقتها على مهام منفصلة للأسفل (على سبيل المثال، دقة 90.7% على ScienceQA IMG). علاوة على ذلك، نثبت نوعيًا مزايا InstructBLIP على النماذج متعددة الوسائط المتزامنة."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/instructblip_architecture.jpg"
alt="drawing" width="600"/>

<small> بنية InstructBLIP. مأخوذة من <a href="https://arxiv.org/abs/2305.06500">الورقة الأصلية.</a> </small>

تمت المساهمة بهذا النموذج بواسطة [nielsr](https://huggingface.co/nielsr). يمكن العثور على الكود الأصلي [هنا](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip).

## نصائح الاستخدام
يستخدم InstructBLIP نفس البنية مثل [BLIP-2](blip2) مع اختلاف بسيط ولكنه مهم: فهو أيضًا يغذي موجه النص (التعليمات) إلى Q-Former.

## InstructBlipConfig

[[autodoc]] InstructBlipConfig

- from_vision_qformer_text_configs

## InstructBlipVisionConfig

[[autodoc]] InstructBlipVisionConfig

## InstructBlipQFormerConfig

[[autodoc]] InstructBlipQFormerConfig

## InstructBlipProcessor

[[autodoc]] InstructBlipProcessor

## InstructBlipVisionModel

[[autodoc]] InstructBlipVisionModel

- forward

## InstructBlipQFormerModel

[[autodoc]] InstructBlipQFormerModel

- forward

## InstructBlipForConditionalGeneration

[[autodoc]] InstructBlipForConditionalGeneration

- forward

- generate