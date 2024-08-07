# LayoutLMv3

## نظرة عامة

اقتُرح نموذج LayoutLMv3 في ورقة بحثية بعنوان "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" من قبل يوبان هوانغ، وتينغتشاو ليو، وليي كوي، ويوتونغ لو، وفورو وي.

LayoutLMv3 هو نسخة مبسطة من [LayoutLMv2](layoutlmv2) حيث يستخدم embeddings رقعة (كما في [ViT](vit)) بدلاً من الاستفادة من الشبكة العصبية التلافيفية CNN، ويتم التدريب المسبق للنموذج على 3 مهام: نمذجة اللغة المقنعة (MLM)، ونمذجة الصور المقنعة (MIM)، ومواءمة الكلمات-الرقعة (WPA).

ملخص الورقة البحثية هو كما يلي:

"حققت تقنيات التدريب الذاتي المسبق تقدمًا ملحوظًا في مجال الذكاء الاصطناعي للوثائق. تستخدم معظم النماذج متعددة الوسائط التي تم تدريبها مسبقًا هدف نمذجة اللغة المقنعة لتعلم التمثيلات ثنائية الاتجاه على طريقة اللغة، ولكنها تختلف في أهداف التدريب المسبق لطريقة الصورة. يضيف هذا الاختلاف صعوبة في تعلم التمثيل متعدد الوسائط. في هذه الورقة، نقترح LayoutLMv3 لتدريب المحولات متعددة الوسائط مسبقًا من أجل الذكاء الاصطناعي للوثائق مع القناع الموحد للنص والصورة. بالإضافة إلى ذلك، يتم تدريب LayoutLMv3 مسبقًا بهدف مواءمة الكلمات-الرقعة لتعلم المحاذاة متعددة الوسائط عن طريق التنبؤ بما إذا كانت رقعة الصورة المقابلة لكلمة نصية مقنعة. يجعل هذا التصميم المعماري الموحد البسيط وأهداف التدريب LayoutLMv3 نموذجًا مسبقًا للتدريب العام لكل من مهام الذكاء الاصطناعي للوثائق التي تركز على النص وتلك التي تركز على الصور. وتظهر النتائج التجريبية أن LayoutLMv3 يحقق أداءً متميزًا ليس فقط في المهام التي تركز على النص، بما في ذلك فهم النماذج والفهم الإيصالات، والأسئلة البصرية للوثائق، ولكن أيضًا في المهام التي تركز على الصور مثل تصنيف صور المستندات وتحليل تخطيط المستندات."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png"
alt="drawing" width="600"/>

<small> تصميم LayoutLMv3. مأخوذ من <a href="https://arxiv.org/abs/2204.08387">الورقة البحثية الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). تمت إضافة إصدار TensorFlow من هذا النموذج بواسطة [chriskoo](https://huggingface.co/chriskoo)، و [tokec](https://huggingface.co/tokec)، و [lre](https://huggingface.co/lre). يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/unilm/tree/master/layoutlmv3).

## نصائح الاستخدام

- من حيث معالجة البيانات، فإن LayoutLMv3 مطابق لسلفه [LayoutLMv2](layoutlmv2)، باستثناء ما يلي:

- يجب تغيير حجم الصور وتطبيعها مع القنوات بتنسيق RGB العادي. من ناحية أخرى، يقوم LayoutLMv2 بتطبيع الصور داخليًا ويتوقع القنوات بتنسيق BGR.

- يتم توكين النص باستخدام الترميز ثنائي البايت (BPE)، على عكس WordPiece.

بسبب هذه الاختلافات في معالجة البيانات الأولية، يمكن استخدام [`LayoutLMv3Processor`] الذي يجمع داخليًا بين [`LayoutLMv3ImageProcessor`] (لطريقة الصورة) و [`LayoutLMv3Tokenizer`]/[`LayoutLMv3TokenizerFast`] (لطريقة النص) لإعداد جميع البيانات للنموذج.

- فيما يتعلق باستخدام [`LayoutLMv3Processor`]]، نشير إلى [دليل الاستخدام](layoutlmv2#usage-layoutlmv2processor) لسلفه.

## الموارد

فيما يلي قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام LayoutLMv3. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب Pull Request وسنقوم بمراجعته! يجب أن يوضح المورد المثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

<Tip>

LayoutLMv3 متطابق تقريبًا مع LayoutLMv2، لذلك أدرجنا أيضًا موارد LayoutLMv2 التي يمكنك تكييفها لمهام LayoutLMv3. بالنسبة إلى دفاتر الملاحظات هذه، احرص على استخدام [`LayoutLMv2Processor`] بدلاً من ذلك عند إعداد البيانات للنموذج!

</Tip>

- يمكن العثور على دفاتر الملاحظات التجريبية لـ LayoutLMv3 [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3).

- يمكن العثور على النصوص التجريبية [هنا](https://github.com/huggingface/transformers/tree/main/examples/research_projects/layoutlmv3).

<PipelineTag pipeline="text-classification"/>

- يتم دعم [`LayoutLMv2ForSequenceClassification`] بواسطة هذا [دفتر الملاحظات](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb).

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)

<PipelineTag pipeline="token-classification"/>

- يتم دعم [`LayoutLMv3ForTokenClassification`] بواسطة [نص النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/research_projects/layoutlmv3) و [دفتر الملاحظات](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb).

- [دفتر الملاحظات](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Inference_with_LayoutLMv2ForTokenClassification.ipynb) حول كيفية إجراء الاستدلال مع [`LayoutLMv2ForTokenClassification`] و [دفتر الملاحظات](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/True_inference_with_LayoutLMv2ForTokenClassification_%2B_Gradio_demo.ipynb) حول كيفية إجراء الاستدلال عند عدم توفر التسميات مع [`LayoutLMv2ForTokenClassification`].

- [دفتر الملاحظات](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb) حول كيفية ضبط نموذج [`LayoutLMv2ForTokenClassification`] باستخدام مدرب 🤗 Trainer.

- [دليل مهام تصنيف الرموز](../tasks/token_classification)

<PipelineTag pipeline="question-answering"/>

- يتم دعم [`LayoutLMv2ForQuestionAnswering`] بواسطة هذا [دفتر الملاحظات](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb).

- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)

**الإجابة على أسئلة الوثائق**

- [دليل مهام الإجابة على أسئلة الوثائق](../tasks/document_question_answering)

## LayoutLMv3Config

[[autodoc]] LayoutLMv3Config

## LayoutLMv3FeatureExtractor

[[autodoc]] LayoutLMv3FeatureExtractor

- __call__

## LayoutLMv3ImageProcessor

[[autodoc]] LayoutLMv3ImageProcessor

- preprocess

## LayoutLMv3Tokenizer

[[autodoc]] LayoutLMv3Tokenizer

- __call__

- save_vocabulary

## LayoutLMv3TokenizerFast

[[autodoc]] LayoutLMv3TokenizerFast

- __call__

## LayoutLMv3Processor

[[autodoc]] LayoutLMv3Processor

- __call__

<frameworkcontent>
<pt>

## LayoutLMv3Model

[[autodoc]] LayoutLMv3Model

- forward

## LayoutLMv3ForSequenceClassification

[[autodoc]] LayoutLMv3ForSequenceClassification

- forward

## LayoutLMv3ForTokenClassification

[[autodoc]] LayoutLMv3ForTokenClassification

- forward

## LayoutLMv3ForQuestionAnswering

[[autodoc]] LayoutLMv3ForQuestionAnswering

- forward

</pt>
<tf>

## TFLayoutLMv3Model

[[autodoc]] TFLayoutLMv3Model

- call


## TFLayoutLMv3ForSequenceClassification

[[autodoc]] TFLayoutLMv3ForSequenceClassification

- call

## TFLayoutLMv3ForTokenClassification

[[autodoc]] TFLayoutLMv3ForTokenClassification

- call

## TFLayoutLMv3ForQuestionAnswering

[[autodoc]] TFLayoutLMv3ForQuestionAnswering

- call

</tf>
</frameworkcontent>