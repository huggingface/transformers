# BLOOM

## نظرة عامة
تم اقتراح نموذج BLOOM بإصداراته المختلفة من خلال [ورشة BigScience](https://bigscience.huggingface.co/). ويستلهم BigScience الإلهام من مبادرات العلوم المفتوحة الأخرى حيث يجمع الباحثون وقتهم ومواردهم لتحقيق تأثير أكبر بشكل جماعي.

يتشابه تصميم BLOOM بشكل أساسي مع تصميم GPT3 (نموذج التنبؤ بالرمز التلقائي التراجعي)، ولكنه تم تدريبه على 46 لغة مختلفة و13 لغة برمجة.

تم تدريب عدة إصدارات أصغر من النماذج على نفس مجموعة البيانات. BLOOM متاح في الإصدارات التالية:

- [bloom-560m](https://huggingface.co/bigscience/bloom-560m)
- [bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- [bloom-3b](https://huggingface.co/bigscience/bloom-3b)
- [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [bloom](https://huggingface.co/bigscience/bloom) (176B معلمات)

## الموارد

قائمة بموارد Hugging Face الرسمية والمجتمعية (مشار إليها بـ 🌎) لمساعدتك في البدء مع BLOOM. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنقوم بمراجعته! يجب أن يوضح المورد في الوضع المثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="text-generation"/>

- [`BloomForCausalLM`] مدعوم بواسطة [مثال على النص البرمجي للنمذجة اللغوية السببية](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

انظر أيضًا:

- [دليل مهام النمذجة اللغوية السببية](../tasks/language_modeling)
- [دليل مهام التصنيف النصي](../tasks/sequence_classification)
- [دليل مهام التصنيف الرمزي](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)

⚡️ الاستنتاج

- مدونة حول [قصة التحسين: استنتاج Bloom](https://huggingface.co/blog/bloom-inference-optimization).
- مدونة حول [سرعة استنتاج BLOOM بشكل لا يصدق باستخدام DeepSpeed و Accelerate](https://huggingface.co/blog/bloom-inference-pytorch-scripts).

⚙️ التدريب

- مدونة حول [التكنولوجيا وراء تدريب BLOOM](https://huggingface.co/blog/bloom-megatron-deepspeed).

## BloomConfig

[[autodoc]] BloomConfig

- all

## BloomTokenizerFast

[[autodoc]] BloomTokenizerFast

- all

<frameworkcontent>
<pt>

## BloomModel

[[autodoc]] BloomModel

- forward

## BloomForCausalLM

[[autodoc]] BloomForCausalLM

- forward

## BloomForSequenceClassification

[[autodoc]] BloomForSequenceClassification

- forward

## BloomForTokenClassification

[[autodoc]] BloomForTokenClassification

- forward

## BloomForQuestionAnswering

[[autodoc]] BloomForQuestionAnswering

- forward

</pt>
<jax>

## FlaxBloomModel

[[autodoc]] FlaxBloomModel

- __call__

## FlaxBloomForCausalLM

[[autodoc]] FlaxBloomForCausalLM

- __call__

</jax>
</frameworkcontent>