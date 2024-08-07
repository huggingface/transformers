# Phi-3

## نظرة عامة

اقترحت مايكروسوفت نموذج Phi-3 في [التقرير الفني لـ Phi-3: نموذج لغة عالي القدرة محليًا على هاتفك](https://arxiv.org/abs/2404.14219).

### ملخص

فيما يلي ملخص ورقة Phi-3:

نقدم phi-3-mini، وهو نموذج لغة يتكون من 3.8 مليار معلمة تم تدريبه على 3.3 تريليون رمز، ويضاهي أداؤه العام، وفقًا للمقاييس الأكاديمية والاختبارات الداخلية، أداء نماذج مثل Mixtral 8x7B و GPT-3.5 (على سبيل المثال، يحقق phi-3-mini نسبة 69% في MMLU و 8.38 في MT-bench)، على الرغم من صغر حجمه بما يكفي لنشره على الهاتف. تكمن الابتكار بالكامل في مجموعة البيانات الخاصة بنا للتدريب، وهي نسخة موسعة من تلك المستخدمة في phi-2، وتتكون من بيانات ويب تمت تصفيتها بشكل كبير وبيانات اصطناعية. كما يتم محاذاة النموذج بشكل أكبر من أجل المتانة والسلامة وتنسيق الدردشة. نقدم أيضًا بعض النتائج الأولية لقياس المعلمات باستخدام نماذج 7B و 14B التي تم تدريبها على 4.8 تريليون رمز، تسمى phi-3-small و phi-3-medium، والتي تتفوق بشكل كبير على phi-3-mini (على سبيل المثال، 75% و 78% على التوالي في MMLU، و 8.7 و 8.9 على MT-bench).

يمكن العثور على الكود الأصلي لـ Phi-3 [هنا](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

## نصائح الاستخدام

- يشبه هذا النموذج كثيرًا نموذج "Llama" مع الاختلاف الرئيسي في [`Phi3SuScaledRotaryEmbedding`] و [`Phi3YarnScaledRotaryEmbedding`]، حيث يتم استخدامها لتمديد سياق التراكيب الدوارة. يتم دمج الاستعلام والمفتاح والقيم، كما يتم دمج طبقات MLP لأعلى ولأسفل في طبقات الإسقاط.
- محلل المفردات المستخدم لهذا النموذج مطابق لـ [`LlamaTokenizer`]، باستثناء الرموز الإضافية.

## كيفية استخدام Phi-3

<Tip warning={true}>

تم دمج Phi-3 في الإصدار التطويري (4.40.0.dev) من `transformers`. حتى يتم إصدار الإصدار الرسمي عبر `pip`، تأكد من قيامك بواحد مما يلي:

* عند تحميل النموذج، تأكد من تمرير `trust_remote_code=True` كحجة لدالة `from_pretrained()`.
* قم بتحديث إصدار "المحولات" المحلية لديك إلى الإصدار التطويري: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`. الأمر السابق هو بديل عن الاستنساخ والتثبيت من المصدر.

</Tip>

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

>>> messages = [{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}]
>>> inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

>>> outputs = model.generate(inputs, max_new_tokens=32)
>>> text = tokenizer.batch_decode(outputs)[0]
>>> print(text)
<s><|user|>
Can you provide ways to eat combinations of bananas and dragonfruits?<|end|>
<|assistant|>
Certainly! Bananas and dragonfruits can be combined in various delicious ways. Here are some ideas for eating combinations of bananas and
```

## Phi3Config

[[autodoc]] Phi3Config

<frameworkcontent>
<pt>

## Phi3Model

[[autodoc]] Phi3Model

- forward

## Phi3ForCausalLM

[[autodoc]] Phi3ForCausalLM

- forward

- generate

## Phi3ForSequenceClassification

[[autodoc]] Phi3ForSequenceClassification

- forward

## Phi3ForTokenClassification

[[autodoc]] Phi3ForTokenClassification

- forward

</pt>
</frameworkcontent>