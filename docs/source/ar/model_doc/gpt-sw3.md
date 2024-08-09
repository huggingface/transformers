# GPT-Sw3

## نظرة عامة

اقترح نموذج GPT-Sw3 لأول مرة في ورقة "الدروس المستفادة من GPT-SW3: بناء أول نموذج لغوي توليدي واسع النطاق للغة السويدية" من قبل أرييل إكغرين، وأمارو كوبا جيلينستن، وإيفانجيليا جوجولو، وأليس هايمان، وسيفيرين فيرليندن، وجوي أوهمان، وفريدريك كارلسون، وماغنوس ساهلجرين.

ومنذ تلك الورقة الأولى، وسع المؤلفون عملهم ودربوا نماذج جديدة على مجموعات بياناتهم الجديدة التي تبلغ 1.2 تيرابايت والتي تسمى The Nordic Pile.

GPT-Sw3 عبارة عن مجموعة من نماذج المحول اللغوي فك التشفير فقط كبيرة الحجم التي تم تطويرها بواسطة AI Sweden بالتعاون مع RISE و WASP WARA for Media and Language. تم تدريب GPT-Sw3 على مجموعة بيانات تحتوي على 320 مليار رمز في السويدية والنرويجية والدانماركية والإيسلندية والإنجليزية، ورموز البرمجة. تم تدريب النموذج المسبق باستخدام هدف نمذجة اللغة السببية (CLM) باستخدام تنفيذ NeMo Megatron GPT.

تمت المساهمة بهذا النموذج من قبل [نماذج AI Sweden](https://huggingface.co/AI-Sweden-Models).

## مثال على الاستخدام

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-356m")
>>> model = AutoModelForCausalLM.from_pretrained("AI-Sweden-Models/gpt-sw3-352m")

>>> input_ids = tokenizer("Träd är fina för att", return_tensors="pt")["input_ids"]

>>> generated_token_ids = model.generate(inputs=input_ids, max_new_tokens=10, do_sample=True)[0]

>>> print(tokenizer.decode(generated_token_ids))
Träd är fina för att de är färgstarka. Men ibland är det fint
```

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل نمذجة اللغة السببية](../tasks/language_modeling)

<Tip>

يستخدم التنفيذ `GPT2Model` المقترن بـ `GPTSw3Tokenizer`. راجع [وثائق GPT2Model](gpt2) للحصول على المراجع والأمثلة الخاصة بواجهة برمجة التطبيقات.

ملاحظة: مطلوب تثبيت Sentencepiece لاستخدام محولنا البرمجي ويمكن تثبيته باستخدام `pip install transformers[sentencepiece]` أو `pip install sentencepiece`.

</Tip>

## GPTSw3Tokenizer

[[autodoc]] GPTSw3Tokenizer

- save_vocabulary