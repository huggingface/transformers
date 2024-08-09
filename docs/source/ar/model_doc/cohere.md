# Cohere

## نظرة عامة
اقترح فريق Cohere نموذج Cohere Command-R في المنشور على المدونة [Command-R: Retrieval Augmented Generation at Production Scale](https://txt.cohere.com/command-r/) بواسطة فريق Cohere.

المقتطف من الورقة هو ما يلي:

> "Command-R هو نموذج توليدي قابل للتطوير يستهدف RAG واستخدام الأدوات لتمكين الذكاء الاصطناعي على نطاق الإنتاج للمؤسسات. واليوم، نقدم Command-R، وهو نموذج LLM جديد يستهدف أعباء العمل الإنتاجية واسعة النطاق. ويستهدف Command-R فئة "قابلة للتطوير" الناشئة من النماذج التي توازن بين الكفاءة العالية والدقة القوية، مما يمكّن الشركات من تجاوز مفهوم الإثبات، والانتقال إلى الإنتاج."

> "Command-R هو نموذج توليدي مُحُسّن لمهام السياق الطويلة مثل استرجاع التوليد المعزز (RAG) واستخدام واجهات برمجة التطبيقات والأدوات الخارجية. وقد صُمم للعمل بالتنسيق مع نماذج Embed وRerank الرائدة في الصناعة لتوفير أفضل تكامل لتطبيقات RAG والتميز في حالات استخدام المؤسسات. وباعتباره نموذجًا مُصممًا للشركات لتنفيذه على نطاق واسع، يتفاخر Command-R بما يلي:

- دقة قوية في RAG واستخدام الأدوات
- انخفاض زمن الوصول، وسرعة عالية في معالجة البيانات
- سياق أطول يبلغ 128 كيلو بايت وتكلفة أقل
- قدرات قوية عبر 10 لغات رئيسية
- أوزان النموذج متاحة على HuggingFace للبحث والتقييم

تفقد نقاط تفتيش النموذج [هنا](https://huggingface.co/CohereForAI/c4ai-command-r-v01).

ساهم بهذا النموذج [Saurabh Dash](https://huggingface.co/saurabhdash) و [Ahmet Üstün](https://huggingface.co/ahmetustun). وتستند شفرة التنفيذ في Hugging Face إلى GPT-NeoX [هنا](https://github.com/EleutherAI/gpt-neox).

## نصائح الاستخدام

<Tip warning={true}>

تستخدم نقاط التفتيش المرفوعة على المحاور `torch_dtype = 'float16'`، والتي سيتم استخدامها بواسطة واجهة برمجة التطبيقات `AutoModel` لتحويل نقاط التفتيش من `torch.float32` إلى `torch.float16`.

إن نوع بيانات الأوزان عبر الإنترنت غير ذي صلة إلى حد كبير ما لم تكن تستخدم `torch_dtype="auto"` عند تهيئة نموذج باستخدام `model = AutoModelForCausalLM.from_pretrained("path"، torch_dtype = "auto")`. والسبب هو أن النموذج سيتم تنزيله أولاً (باستخدام نوع بيانات نقاط التفتيش عبر الإنترنت)، ثم سيتم تحويله إلى نوع بيانات الافتراضي لـ `torch` (يصبح `torch.float32`)، وأخيراً، إذا كان هناك `torch_dtype` مقدم في التكوين، فسيتم استخدامه.

لا يُنصح بالتدريب على النموذج في `float16` ومن المعروف أنه ينتج عنه `nan`؛ لذلك، يجب تدريب النموذج في `bfloat16`.

</Tip>

يمكن تحميل النموذج والمحلل اللغوي باستخدام ما يلي:

```python
# pip install transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# تنسيق الرسالة باستخدام قالب الدردشة command-r
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## \<BOS_TOKEN\>\<\|START_OF_TURN_TOKEN\|\>\<\|USER_TOKEN\|\>Hello, how are you?\<\|END_OF_TURN_TOKEN\|\>\<\|START_OF_TURN_TOKEN\|\>\<\|CHATBOT_TOKEN\|\>

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

- عند استخدام Flash Attention 2 عبر `attn_implementation="flash_attention_2"`، لا تمرر `torch_dtype` إلى طريقة الفصل `from_pretrained` واستخدم التدريب على الدقة المختلطة التلقائية. عند استخدام `Trainer`، فهو ببساطة تحديد إما `fp16` أو `bf16` إلى `True`. وإلا، تأكد من استخدامك لـ `torch.autocast`. وهذا مطلوب لأن Flash Attention يدعم فقط نوعي البيانات `fp16` و`bf16`.

## الموارد

فيما يلي قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بالعالم 🌎) لمساعدتك في البدء باستخدام Command-R. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب وسنقوم بمراجعته! ويفضل أن يُظهر المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="text-generation"/>

تحميل نموذج FP16

```python
# pip install transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# تنسيق الرسالة باستخدام قالب الدردشة command-r
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## \<BOS_TOKEN\>\<\|START_OF_TURN_TOKEN\|\>\<\|USER_TOKEN\|\>Hello, how are you?\<\|END_OF_TURN_TOKEN\|\>\<\|START_OF_TURN_TOKEN\|\>\<\|CHATBOT_TOKEN\|\>

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

تحميل نموذج bitsnbytes 4bit الكمي

```python
# pip install transformers bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
)

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

## CohereConfig

[[autodoc]] CohereConfig

## CohereTokenizerFast

[[autodoc]] CohereTokenizerFast

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- update_post_processor
- save_vocabulary

## CohereModel

[[autodoc]] CohereModel

- forward

## CohereForCausalLM

[[autodoc]] CohereForCausalLM

- forward