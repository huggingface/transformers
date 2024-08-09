# GPTSAN-japanese

<Tip warning={true}>
تم وضع هذا النموذج في وضع الصيانة فقط، ولا نقبل أي طلبات سحب (Pull Requests) جديدة لتغيير شفرته.
إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2.
يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.
</Tip>

## نظرة عامة
تم إصدار نموذج GPTSAN-japanese في المستودع بواسطة Toshiyuki Sakamoto (tanreinama).
GPTSAN هو نموذج لغة ياباني يستخدم Switch Transformer. لديه نفس بنية النموذج الذي تم تقديمه على أنه Prefix LM في ورقة T5، ويدعم كل من مهام Text Generation و Masked Language Modeling. يمكن أيضًا ضبط هذه المهام الأساسية بشكل دقيق للترجمة أو الملخص.

### مثال على الاستخدام
يمكن استخدام طريقة `generate()` لتوليد نص باستخدام نموذج GPTSAN-Japanese.

```python
>>> from transformers import AutoModel, AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
>>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").cuda()
>>> x_tok = tokenizer("は、", prefix_text="織田信長", return_tensors="pt")
>>> torch.manual_seed(0)
>>> gen_tok = model.generate(x_tok.input_ids.cuda(), token_type_ids=x_tok.token_type_ids.cuda(), max_new_tokens=20)
>>> tokenizer.decode(gen_tok[0])
'織田信長は、2004年に『戦国BASARA』のために、豊臣秀吉'
```

## ميزات GPTSAN
يتمتع GPTSAN ببعض الميزات الفريدة. فهو يتمتع ببنية نموذج Prefix-LM. يعمل كنموذج Masked Language Model منزاح للرموز المميزة للبادئة (Prefix Input tokens). وتتصرف المدخلات غير المميزة كنماذج توليدية عادية.

Spout vector هو مدخل خاص بنموذج GPTSAN. تم تدريب Spout مسبقًا باستخدام مدخلات عشوائية، ولكن يمكنك تحديد فئة من النص أو متجهًا عشوائيًا أثناء الضبط الدقيق. يسمح لك ذلك بالإشارة إلى اتجاه النص المولد.

يحتوي GPTSAN على طبقة توصيل متفرقة (Feed Forward) تعتمد على Switch-Transformer. يمكنك أيضًا إضافة طبقات أخرى وتدريبها بشكل جزئي. راجع مستودع GPTSAN الأصلي للحصول على التفاصيل.

### نموذج Prefix-LM
يتمتع GPTSAN ببنية النموذج المسمى Prefix-LM في ورقة "T5". (يطلق عليه المستودع الأصلي لـ GPTSAN اسم "hybrid")

في GPTSAN، يمكن تحديد جزء "Prefix" من Prefix-LM، أي موضع الإدخال الذي يمكن أن يرجع إليه كلا الرمزين، بأي طول.

يمكن أيضًا تحديد أطوال مختلفة لكل دفعة.

ينطبق هذا الطول على النص الذي تم إدخاله في `prefix_text` لـ tokenizer.

تعيد tokenizer قناع جزء "Prefix" من Prefix-LM على أنه `token_type_ids`.

يعامل النموذج الجزء الذي يكون فيه `token_type_ids` 1 كجزء "Prefix"، أي أن الإدخال يمكن أن يشير إلى كلا الرمزين قبل وبعد.

## نصائح الاستخدام
يتم تحديد جزء البادئة باستخدام قناع يتم تمريره إلى self-attention.
عندما يكون token_type_ids=None أو كله أصفار، فهو مكافئ لقناع سببي منتظم

على سبيل المثال:

```
>>> x_token = tokenizer("ｱｲｳｴ")
input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |
prefix_lm_mask:
SOT | 1 0 0 0 0 0 |
SEG | 1 1 0 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |

>>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |
prefix_lm_mask:
SOT | 1 1 1 1 1 0 |
ｱ   | 1 1 1 1 1 0 |
ｲ   | 1 1 1 1 1 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 0 |
SEG | 1 1 1 1 1 1 |

>>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
prefix_lm_mask:
SOT | 1 1 1 0 0 0 |
ｱ   | 1 1 1 0 0 0 |
ｲ   | 1 1 1 0 0 0 |
SEG | 1 1 1 1 0 0 |
ｳ   | 1 1 1 1 1 0 |
ｴ   | 1 1 1 1 1 1 |
```

### Spout Vector
Spout Vector هو متجه خاص للتحكم في توليد النص.
يتم التعامل مع هذا المتجه على أنه أول تضمين في self-attention لجلب الانتباه الخارجي إلى الرموز المولدة.

في النموذج المُدرب مسبقًا المنشور من `Tanrei/GPTSAN-japanese`، يكون Spout Vector عبارة عن متجه 128-بعدي يمر عبر 8 طبقات متصلة بالكامل في النموذج ويتم إسقاطه في الفضاء الذي يعمل كاهتمام خارجي.

يتم تقسيم Spout Vector الذي تم إسقاطه بواسطة الطبقة المتصلة بالكامل لتمريره إلى جميع عمليات self-attention.

## GPTSanJapaneseConfig
[[autodoc]] GPTSanJapaneseConfig

## GPTSanJapaneseTokenizer
[[autodoc]] GPTSanJapaneseTokenizer

## GPTSanJapaneseModel
[[autodoc]] GPTSanJapaneseModel

## GPTSanJapaneseForConditionalGeneration
[[autodoc]] GPTSanJapaneseForConditionalGeneration

- forward