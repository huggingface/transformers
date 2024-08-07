# XGLM

## نظرة عامة

اقترح نموذج XGLM في [التعلم القائم على القليل من اللقطات باستخدام نماذج اللغة متعددة اللغات](https://arxiv.org/abs/2112.10668) من قبل Xi Victoria Lin، Todor Mihaylov، Mikel Artetxe، Tianlu Wang، Shuohui Chen، Daniel Simig، Myle Ott، Naman Goyal، Shruti Bhosale، Jingfei Du، Ramakanth Pasunuru، Sam Shleifer، Punit Singh Koura، Vishrav Chaudhary، Brian O'Horo، Jeff Wang، Luke Zettlemoyer، Zornitsa Kozareva، Mona Diab، Veselin Stoyanov، Xian Li.

الملخص من الورقة هو ما يلي:

*نماذج اللغة التلقائية كبيرة الحجم مثل GPT-3 هي برامج تعليمية قليلة اللقطات يمكنها أداء مجموعة واسعة من المهام اللغوية دون ضبط دقيق. في حين أن من المعروف أن هذه النماذج قادرة على تمثيل العديد من اللغات المختلفة بشكل مشترك، إلا أن بيانات التدريب الخاصة بها تهيمن عليها اللغة الإنجليزية، مما قد يحد من تعميمها عبر اللغات. في هذا العمل، نقوم بتدريب نماذج اللغة التلقائية متعددة اللغات على مجموعة بيانات متوازنة تغطي مجموعة متنوعة من اللغات، وندرس قدراتها على التعلم القائم على القليل من اللقطات والتعلم الصفري في مجموعة واسعة من المهام. ويحدد نموذجنا الأكبر الذي يحتوي على 7.5 مليار معلمة الحالة الجديدة للفن في التعلم القائم على القليل من اللقطات في أكثر من 20 لغة ممثلة، متفوقًا على GPT-3 من الحجم المماثل في الاستدلال متعدد اللغات (بتحسن دقة مطلق بنسبة 7.4% في إعدادات اللقطة الصفرية و +9.4% في إعدادات 4-shot) والاستدلال اللغوي (+5.4% في كل من إعدادات اللقطة الصفرية و 4-shot). في معيار الترجمة الآلية FLORES-101، يتفوق نموذجنا على GPT-3 في 171 من أصل 182 اتجاهات الترجمة باستخدام 32 مثالًا تدريبيًا، بينما يتفوق على خط الأساس الخاضع للإشراف الرسمي في 45 اتجاهًا. نقدم تحليلًا مفصلاً لنجاحات النموذج وإخفاقاته، مما يظهر على وجه الخصوص أنه يمكّن التعلم السياقي متعدد اللغات لبعض المهام، في حين لا يزال هناك مجال للتحسين فيما يتعلق بقوة الشكل السطحي والتكيف مع المهام التي ليس لها شكل استكمال طبيعي. وأخيرًا، نقيم نماذجنا في مهام القيمة الاجتماعية مثل الكشف عن خطاب الكراهية في خمس لغات ونجد أنها تعاني من قيود مماثلة لنماذج GPT-3 ذات الحجم المماثل.*

تمت المساهمة بهذا النموذج من قبل [Suraj](https://huggingface.co/valhalla). يمكن العثور على الكود الأصلي [هنا](https://github.com/pytorch/fairseq/tree/main/examples/xglm).

## الموارد

- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

## XGLMConfig

[[autodoc]] XGLMConfig

## XGLMTokenizer

[[autodoc]] XGLMTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## XGLMTokenizerFast

[[autodoc]] XGLMTokenizerFast

<frameworkcontent>

<pt>

## XGLMModel

[[autodoc]] XGLMModel

- forward

## XGLMForCausalLM

[[autodoc]] XGLMForCausalLM

- forward

</pt>

<tf>

## TFXGLMModel

[[autodoc]] TFXGLMModel

- call

## TFXGLMForCausalLM

[[autodoc]] TFXGLMForCausalLM

- call

</tf>

<jax>

## FlaxXGLMModel

[[autodoc]] FlaxXGLMModel

- __call__

## FlaxXGLMForCausalLM

[[autodoc]] FlaxXGLMForCausalLM

- __call__

</jax>

</frameworkcontent>