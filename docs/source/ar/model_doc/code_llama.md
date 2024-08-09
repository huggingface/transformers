# CodeLlama

## نظرة عامة
اقترح نموذج Code Llama في [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) بواسطة Baptiste Rozière، Jonas Gehring، Fabian Gloeckle، Sten Sootla، Itai Gat، Xiaoqing Ellen Tan، Yossi Adi، Jingyu Liu، Tal Remez، Jérémy Rapin، Artyom Kozhevnikov، Ivan Evtimov، Joanna Bitton، Manish Bhatt، Cristian Canton Ferrer، Aaron Grattafiori، Wenhan Xiong، Alexandre Défossez، Jade Copet، Faisal Azhar، Hugo Touvron، Louis Martin، Nicolas Usunier، Thomas Scialom، Gabriel Synnaeve.

ملخص الورقة هو ما يلي:

*نحن نطلق Code Llama، وهي عائلة من نماذج اللغة الكبيرة للكود المستندة إلى Llama 2 والتي توفر أداءً متميزًا بين النماذج المفتوحة، وقدرات infilling، ودعم سياقات الإدخال الكبيرة، والقدرة على اتباع التعليمات من الصفر لمهام البرمجة. نوفر نكهات متعددة لتغطية مجموعة واسعة من التطبيقات: نماذج الأساس (Code Llama)، والتخصصات في Python (Code Llama - Python)، ونماذج اتباع التعليمات (Code Llama - Instruct) مع 7B و13B و34B من المعلمات لكل منها. تم تدريب جميع النماذج على تسلسلات من 16000 رمز، وتظهر تحسينات على الإدخالات التي يصل طولها إلى 100000 رمز. تدعم متغيرات Code Llama وCode Llama - Instruct بسعة 7B و13B infilling بناءً على المحتوى المحيط. يحقق Code Llama أداءً متميزًا بين النماذج المفتوحة على العديد من معايير الكود، مع تحقيق نتائج تصل إلى 53% و55% على HumanEval وMBPP على التوالي. ومن الجدير بالذكر أن Code Llama - Python 7B يتفوق على Llama 2 70B في HumanEval وMBPP، وتتفوق جميع نماذجنا على أي نموذج متاح للجمهور في MultiPL-E. نطلق Code Llama بموجب ترخيص متساهل يسمح باستخدامه في كل من الأبحاث والتطبيقات التجارية.*

اطلع على جميع نقاط تفتيش نموذج Code Llama [هنا](https://huggingface.co/models?search=code_llama) والنماذج التي تم إصدارها رسميًا في [Meta Llama org](https://huggingface.co/meta-llama).

تمت المساهمة بهذا النموذج بواسطة [ArthurZucker](https://huggingface.co/ArthurZ). يمكن العثور على الكود الأصلي للمؤلفين [هنا](https://github.com/facebookresearch/llama).

## نصائح الاستخدام وأمثلة

<Tip warning={true}>
تم تدريب نماذج عائلة `Llama2`، التي يعتمد عليها Code Llama، باستخدام `bfloat16`، ولكن الاستدلال الأصلي يستخدم `float16`. دعونا نلقي نظرة على الدقة المختلفة:

* `float32`: اتفاقية PyTorch عند تهيئة النموذج هي تحميل النماذج في `float32`، بغض النظر عن `dtype` الذي تم تخزين أوزان النموذج به. يتبع `transformers` أيضًا هذه الاتفاقية للاتساق مع PyTorch. سيتم اختياره افتراضيًا. إذا كنت تريد أن يقوم واجهة برمجة التطبيقات (API) لـ `AutoModel` بتحويل نقاط تفتيش التحميل إلى نوع أوزان التخزين، فيجب عليك تحديد `torch_dtype="auto"`، على سبيل المثال `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`.

* `bfloat16`: تم تدريب Code Llama بهذه الدقة، لذلك نوصي باستخدامها للمزيد من التدريب أو الضبط الدقيق.

* `float16`: نوصي بتشغيل الاستدلال باستخدام هذه الدقة، حيث أنها عادةً ما تكون أسرع من `bfloat16`، ولا تظهر مقاييس التقييم أي تدهور ملحوظ فيما يتعلق بـ `bfloat16`. يمكنك أيضًا تشغيل الاستدلال باستخدام `bfloat16`، ونوصيك بالتحقق من نتائج الاستدلال باستخدام كل من `float16` و`bfloat16` بعد الضبط الدقيق.

كما ذكرنا سابقًا، فإن `dtype` لأوزان التخزين غير مهم إلى حد كبير ما لم تكن تستخدم `torch_dtype="auto"` عند تهيئة نموذج باستخدام. والسبب هو أن النموذج سيتم تنزيله أولاً (باستخدام `dtype` لنقاط التفتيش عبر الإنترنت) ثم سيتم تحويله إلى `dtype` الافتراضي لـ `torch` (يصبح `torch.float32`). إذا كان هناك `torch_dtype` محدد، فسيتم استخدامه بدلاً من ذلك.
</Tip>

نصائح:

- يتم دعم مهمة infilling بشكل افتراضي. يجب عليك استخدام `tokenizer.fill_token` حيث تريد ملء الإدخال الخاص بك.

- نص البرنامج النصي لتحويل النموذج هو نفسه لعائلة `Llama2`:

فيما يلي مثال على الاستخدام:

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
--input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

ملاحظة: يتطلب تنفيذ البرنامج النصي مساحة كافية من ذاكرة الوصول العشوائي (RAM) في وحدة المعالجة المركزية (CPU) لاستضافة النموذج بالكامل في دقة float16 (حتى إذا كانت أكبر الإصدارات
تأتي في عدة نقاط تفتيش، يحتوي كل منها على جزء من كل وزن للنموذج، لذلك نحتاج إلى تحميلها جميعًا في ذاكرة الوصول العشوائي).

بعد التحويل، يمكن تحميل النموذج والمحلل اللغوي باستخدام ما يلي:

```python
>>> from transformers import LlamaForCausalLM, CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
...     """ <FILL_ME>
...     return result
... '''
>>> input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
>>> generated_ids = model.generate(input_ids, max_new_tokens=128)

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.
<BLANKLINE>
    Args:
        s: The string to remove non-ASCII characters from.
<BLANKLINE>
    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
<BLANKLINE>
```

إذا كنت تريد فقط الجزء المملوء:

```python
>>> from transformers import pipeline
>>> import torch

>>> generator = pipeline("text-generation",model="meta-llama/CodeLlama-7b-hf",torch_dtype=torch.float16, device_map="auto")
>>> generator('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens = 128)
[{'generated_text': 'def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return resultRemove non-ASCII characters from a string. """\n    result = ""\n    for c in s:\n        if ord(c) < 128:\n            result += c'}]
```

في الخلفية، يقوم المحلل اللغوي [بالتقسيم تلقائيًا بواسطة `<FILL_ME>`](https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.fill_token) لإنشاء سلسلة إدخال بتنسيق يتبع [نمط التدريب الأصلي](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402). هذه الطريقة أكثر متانة من إعداد النمط بنفسك: فهي تتجنب المشكلات، مثل لصق الرموز، والتي يصعب تصحيحها. لمعرفة مقدار ذاكرة وحدة المعالجة المركزية (CPU) وذاكرة وحدة معالجة الرسومات (GPU) التي تحتاجها لهذا النموذج أو غيره، جرب [هذه الآلة الحاسبة](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) والتي يمكن أن تساعد في تحديد تلك القيمة.

محلل LLaMA اللغوي هو نموذج BPE يعتمد على [sentencepiece](https://github.com/google/sentencepiece). إحدى ميزات sentencepiece هي أنه عند فك تشفير تسلسل، إذا كان الرمز الأول هو بداية الكلمة (مثل "Banana")، فإن المحلل اللغوي لا يسبق المسافة البادئة للسلسلة.

<Tip>
يحتوي Code Llama على نفس بنية نماذج `Llama2`، راجع [صفحة وثائق Llama2](llama2) للحصول على مرجع واجهة برمجة التطبيقات (API).
يمكنك العثور على مرجع محلل Llama اللغوي أدناه.
</Tip>

## CodeLlamaTokenizer

[[autodoc]] CodeLlamaTokenizer
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## CodeLlamaTokenizerFast

[[autodoc]] CodeLlamaTokenizerFast
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- update_post_processor
- save_vocabulary