# Llama2

## نظرة عامة
اقترح نموذج Llama2 في [LLaMA: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) بواسطة Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel، Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushka rMishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing EllenTan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom. وهي مجموعة من نماذج اللغة الأساسية التي تتراوح من 7 بليون إلى 70 بليون معلمة، مع نقاط تفتيش تمت معايرتها لتطبيق الدردشة!

الملخص من الورقة هو ما يلي:

> في هذا العمل، نقوم بتطوير وإصدار Llama 2، وهي مجموعة من نماذج اللغة الكبيرة المسبقة التدريب والمعايرة التي تتراوح في النطاق من 7 مليارات إلى 70 مليار معلمة. تم تحسين نماذجنا المعايرة للغة، والتي يطلق عليها اسم Llama 2-Chat، لاستخدامات الحوار. تتفوق نماذجنا على نماذج الدردشة مفتوحة المصدر في معظم المعايير التي اختبرناها، وقد تكون، بناءً على تقييماتنا البشرية للفائدة والسلامة، بديلاً مناسباً لنماذج المصدر المغلق. نقدم وصفًا مفصلاً لنهجنا في المعايرة وتحسينات السلامة لنموذج Llama 2-Chat من أجل تمكين المجتمع من البناء على عملنا والمساهمة في التطوير المسؤول لنماذج اللغة الكبيرة.

تحقق من جميع نقاط تفتيش نموذج Llama2 [هنا](https://huggingface.co/models?search=llama2).

تمت المساهمة بهذا النموذج من قبل [Arthur Zucker](https://huggingface.co/ArthurZ) بمساهمات من [Lysandre Debut](https://huggingface.co/lysandre). يعتمد كود التنفيذ في Hugging Face على GPT-NeoX [هنا](https://github.com/EleutherAI/gpt-neox). يمكن العثور على الكود الأصلي للمؤلفين [هنا](https://github.com/facebookresearch/llama).

## نصائح الاستخدام

<Tip warning={true}>

تم تدريب نماذج "Llama2" باستخدام "bfloat16"، ولكن الاستدلال الأصلي يستخدم "float16". تستخدم نقاط التفتيش التي تم تحميلها على المحاور `torch_dtype = 'float16'`، والتي سيتم استخدامها بواسطة واجهة برمجة تطبيقات `AutoModel` لتحويل نقاط التفتيش من `torch.float32` إلى `torch.float16`.

`dtype` للأوزان عبر الإنترنت غير ذي صلة في الغالب ما لم تكن تستخدم `torch_dtype="auto"` عند تهيئة نموذج باستخدام `model = AutoModelForCausalLM.from_pretrained("path"، torch_dtype = "auto")`. والسبب هو أن النموذج سيتم تنزيله أولاً (باستخدام `dtype` لنقاط التفتيش عبر الإنترنت)، ثم سيتم تحويله إلى `dtype` الافتراضي لـ `torch` (يصبح `torch.float32`)، وأخيراً، إذا كان هناك `torch_dtype` مقدم في التكوين، فسيتم استخدامه.

لا يوصى بالتدريب على النموذج في `float16` وهو معروف بإنتاج `nan`؛ لذلك، يجب تدريب النموذج في `bfloat16`.

</Tip>

نصائح:

- يمكن الحصول على الأوزان لنماذج Llama2 عن طريق ملء [هذا النموذج](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- الهندسة المعمارية مشابهة جدًا لـ Llama الأول، مع إضافة Grouped Query Attention (GQA) بعد هذه [الورقة](https://arxiv.org/pdf/2305.13245.pdf)
- يؤدي تعيين `config.pretraining_tp` إلى قيمة مختلفة عن 1 إلى تنشيط الحساب الأكثر دقة ولكن الأبطأ للطبقات الخطية، والتي يجب أن تتطابق بشكل أفضل مع اللوغاريتمات الأصلية.
- يستخدم النموذج الأصلي `pad_id = -1` مما يعني أنه لا يوجد رمز خاص للتعبئة. لا يمكننا أن يكون لدينا نفس المنطق، تأكد من إضافة رمز خاص للتعبئة باستخدام `tokenizer.add_special_tokens({"pad_token": "<pad>"})` وقم بتغيير حجم تضمين الرمز وفقًا لذلك. يجب عليك أيضًا تعيين `model.config.pad_token_id`. يتم تهيئة طبقة `embed_tokens` للنموذج باستخدام `self.embed_tokens = nn.Embedding(config.vocab_size، config.hidden_size، self.config.padding_idx)`، والذي يتأكد من أن ترميز الرمز الخاص بالتعبئة سيخرج أصفارًا، لذا يوصى بتمريره عند التهيئة.
- بعد ملء النموذج والحصول على حق الوصول إلى نقاط تفتيش النموذج، يجب أن تتمكن من استخدام نقاط التفتيش المحولة بالفعل. وإلا، إذا كنت تقوم بتحويل نموذجك الخاص، فلا تتردد في استخدام [سكريبت التحويل](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). يمكن استدعاء البرنامج النصي باستخدام الأمر التالي (كمثال):

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
--input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- بعد التحويل، يمكن تحميل النموذج والمحلل اللغوي عبر:

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

ملاحظة: يتطلب تنفيذ البرنامج النصي مساحة كافية من ذاكرة الوصول العشوائي (RAM) لاستضافة النموذج بالكامل في دقة float16 (حتى إذا كانت أكبر الإصدارات تأتي في عدة نقاط تفتيش، يحتوي كل منها على جزء من كل وزن للنموذج، لذا نحتاج إلى تحميلها جميعًا في ذاكرة الوصول العشوائي). بالنسبة للنموذج 75B، هناك حاجة إلى 145 جيجابايت من ذاكرة الوصول العشوائي.

- LLaMA tokenizer عبارة عن نموذج BPE يعتمد على [sentencepiece](https://github.com/google/sentencepiece). إحدى غرائب sentencepiece هي أنه عند فك تشفير تسلسل، إذا كان الرمز الأول هو بداية الكلمة (على سبيل المثال "Banana")، فإن المحلل اللغوي لا يسبق المسافة البادئة إلى السلسلة.
- عند استخدام Flash Attention 2 عبر `attn_implementation="flash_attention_2"`، لا تمرر `torch_dtype` إلى طريقة `from_pretrained` واستخدم التدريب Mixed-Precision التلقائي. عند استخدام `Trainer`، يكون الأمر ببساطة بتحديد إما `fp16` أو `bf16` إلى `True`. وإلا، تأكد من استخدامك لـ `torch.autocast`. هذا مطلوب لأن Flash Attention يدعم فقط أنواع البيانات `fp16` و`bf16`.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام LLaMA2. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب ومراجعته! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- [Llama 2 is here - get it on Hugging Face](https://huggingface.co/blog/llama2)، منشور مدونة حول Llama 2 وكيفية استخدامه مع 🤗 Transformers و🤗 PEFT.
- [LLaMA 2 - Every Resource you need](https://www.philschmid.de/llama-2)، مجموعة من الموارد ذات الصلة للتعرف على LLaMA 2 وكيفية البدء بسرعة.
<PipelineTag pipeline="text-generation"/>
- [دفتر الملاحظات](https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing) حول كيفية معايرة Llama 2 في Google Colab باستخدام QLoRA ودقة 4 بت. 🌎
- [دفتر الملاحظات](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing) حول كيفية معايرة نموذج "Llama-v2-7b-guanaco" باستخدام QLoRA بدقة 4 بت وتوليد مجموعات بيانات الأسئلة والأجوبة من ملفات PDF. 🌎
<PipelineTag pipeline="text-classification"/>
- [دفتر الملاحظات](https://colab.research.google.com/drive/1ggaa2oRFphdBmqIjSEbnb_HGkcIRC2ZB?usp=sharing) حول كيفية معايرة نموذج Llama 2 باستخدام QLoRa وTRL ومجموعة بيانات التصنيف النصي الكورية. 🌎🇰🇷
⚗️ Optimization
- [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl)، دليل لاستخدام طريقة DPO لمكتبة TRL لمعايرة Llama 2 على مجموعة بيانات محددة.
- [Extended Guide: Instruction-tune Llama 2](https://www.philschmid.de/instruction-tune-llama-2)، دليل لتدريب Llama 2 لتوليد التعليمات من المدخلات، وتحويل النموذج من اتباع التعليمات إلى إعطاء التعليمات.
- [دفتر الملاحظات](https://colab.research.google.com/drive/1SYpgFpcmtIUzdE7pxqknrM4ArCASfkFQ?usp=sharing) حول كيفية معايرة نموذج Llama 2 على جهاز كمبيوتر شخصي باستخدام QLoRa وTRL. 🌎
⚡️ Inference
- [دفتر الملاحظات](https://colab.research.google.com/drive/1TC56ArKerXUpbgRy5vM3woRsbTEVNq7h?usp=sharing) حول كيفية تحديد كمية نموذج Llama 2 باستخدام GPTQ من مكتبة AutoGPTQ. 🌎
- [دفتر الملاحظات](https://colab.research.google.com/drive/1X1z9Q6domMKl2CnEM0QGHNwidLfR4dW2?usp=sharing) حول كيفية تشغيل نموذج Llama 2 Chat Model بدقة 4 بت على جهاز كمبيوتر محلي أو Google Colab. 🌎
🚀 Deploy
- [Fine-tune LLaMA 2 (7-70B) on Amazon SageMaker](https://www.philschmid.de/sagemaker-llama2-qlora)، دليل كامل من الإعداد إلى معايرة QLoRA والنشر على Amazon SageMaker.
- [Deploy Llama 2 7B/13B/70B on Amazon SageMaker](https://www.philschmid.de/sagemaker-llama-llm)، دليل لاستخدام حاوية DLC LLM من Hugging Face للنشر الآمن والقابل للتطوير.


## LlamaConfig

[[autodoc]] LlamaConfig
## LlamaTokenizer

[[autodoc]] LlamaTokenizer
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary
## LlamaTokenizerFast

[[autodoc]] LlamaTokenizerFast
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- update_post_processor
- save_vocabulary
## LlamaModel

[[autodoc]] LlamaModel
- forward
## LlamaForCausalLM

[[autodoc]] LlamaForCausalLM
- forward
## LlamaForSequenceClassification

[[autodoc]] LlamaForSequenceClassification
- forward