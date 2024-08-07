# MegatronBERT

## نظرة عامة
تم اقتراح نموذج MegatronBERT في ورقة بحثية بعنوان "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" بواسطة Mohammad Shoeybi وآخرين.

ملخص الورقة البحثية هو كما يلي:

أظهرت الأعمال الحديثة في نمذجة اللغة أن تدريب نماذج المحول الكبيرة تطور حالة الفن في تطبيقات معالجة اللغة الطبيعية. ومع ذلك، يمكن أن تكون النماذج الكبيرة جداً صعبة التدريب بسبب قيود الذاكرة. في هذا العمل، نقدم تقنياتنا لتدريب نماذج المحول الكبيرة للغاية وننفذ نهجًا موازيًا داخلي الطبقة بسيطًا وفعالًا يمكّن من تدريب نماذج المحول مع مليارات من المعلمات. لا يتطلب نهجنا مترجمًا أو تغييرات في المكتبة الجديدة، وهو متعامد ومتكامل مع موازاة نموذج الأنابيب، ويمكن تنفيذه بالكامل من خلال إدراج بعض عمليات الاتصال في PyTorch الأصلي. نوضح هذا النهج من خلال تقارب النماذج القائمة على المحول حتى 8.3 مليار معلمة باستخدام 512 وحدة معالجة رسومية (GPU). نحافظ على 15.1 PetaFLOPs عبر التطبيق بالكامل بكفاءة توسيع 76% عند مقارنته بخط أساس GPU واحد قوي يحافظ على 39 TeraFLOPs، والتي هي 30% من ذروة FLOPs. ولإثبات أن نماذج اللغة الكبيرة يمكن أن تواصل تطوير حالة الفن، نقوم بتدريب نموذج لغة محول 8.3 مليار معلمة مشابه لـ GPT-2 ونموذج 3.9 مليار معلمة مشابه لـ BERT. ونظهر أن الاهتمام الدقيق بوضع التطبيع الطبقي في النماذج الشبيهة بـ BERT أمر بالغ الأهمية لتحقيق أداء متزايد مع نمو حجم النموذج. باستخدام نموذج GPT-2، نحقق نتائج حالة أفضل على مجموعات بيانات WikiText103 (10.8 مقارنة بـ 15.8 من حالة أفضل) و LAMBADA (66.5% مقارنة بدقة 63.2% من حالة أفضل). ويحقق نموذج BERT الخاص بنا نتائج حالة أفضل على مجموعة بيانات RACE (90.9% مقارنة بدقة 89.4% من حالة أفضل).

تمت المساهمة بهذا النموذج من قبل [jdemouth]. يمكن العثور على الكود الأصلي [هنا]. يحتوي هذا المستودع على تنفيذ متعدد وحدات معالجة الرسومات (GPU) ومتعدد العقد لنماذج Megatron Language. على وجه التحديد، يحتوي على نهج موازاة نموذج هجين باستخدام تقنيات "tensor parallel" و "pipeline parallel".

## نصائح الاستخدام

قمنا بتوفير نقاط تفتيش [BERT-345M] مسبقة التدريب للاستخدام في تقييم أو الضبط الدقيق لمهام المصب.

للوصول إلى هذه النقاط المرجعية، قم أولاً بالتسجيل في NVIDIA GPU Cloud (NGC) وإعداد واجهة سطر الأوامر الخاصة بسجل NGC. يمكن العثور على مزيد من الوثائق حول تنزيل النماذج في [وثائق NGC].

أو، يمكنك تنزيل نقاط التفتيش مباشرة باستخدام ما يلي:

BERT-345M-uncased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip
-O megatron_bert_345m_v0_1_uncased.zip
```

BERT-345M-cased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O
megatron_bert_345m_v0_1_cased.zip
```

بمجرد الحصول على نقاط التفتيش من NVIDIA GPU Cloud (NGC)، يجب تحويلها إلى تنسيق يمكن تحميله بسهولة بواسطة Hugging Face Transformers ومنفذنا لرمز BERT.

تسمح الأوامر التالية بإجراء التحويل. نفترض أن المجلد 'models/megatron_bert' يحتوي على 'megatron_bert_345m_v0_1_ {cased، uncased}.zip' وأن الأوامر يتم تشغيلها من داخل ذلك المجلد:

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_uncased.zip
```

```bash
python3 $PATH_TO_TRANSFORMtakan/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_cased.zip
```

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## MegatronBertConfig

[[autodoc]] MegatronBertConfig

## MegatronBertModel

[[autodoc]] MegatronBertModel

- forward

## MegatronBertForMaskedLM

[[autodoc]] MegatronBertForMaskedLM

- forward

## MegatronBertForCausalLM

[[autodoc]] MegatronBertForCausalLM

- forward

## MegatronBertForNextSentencePrediction

[[autodoc]] MegatronBertForNextSentencePrediction

- forward

## MegatronBertForPreTraining

[[autodoc]] MegatronBertForPreTraining

- forward

## MegatronBertForSequenceClassification

[[autodoc]] MegatronBertForSequenceClassification

- forward

## MegatronBertForMultipleChoice

[[autodoc]] MegatronBertForMultipleChoice

- forward

## MegatronBertForTokenClassification

[[autodoc]] MegatronBertForTokenClassification

- forward

## MegatronBertForQuestionAnswering

[[autodoc]] MegatronBertForQuestionAnswering

- forward