# استخدام المحللون اللغويون من 🤗 Tokenizers

يعتمد [`PreTrainedTokenizerFast`] على مكتبة [🤗 Tokenizers](https://huggingface.co/docs/tokenizers). يمكن تحميل المحللين اللغويين الذين تم الحصول عليهم من مكتبة 🤗 Tokenizers ببساطة شديدة في 🤗 Transformers.

قبل الدخول في التفاصيل، دعونا نبدأ أولاً بإنشاء محلل لغوي وهمي في بضع سطور:

```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> from tokenizers.pre_tokenizers import Whitespace

>>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
>>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

>>> tokenizer.pre_tokenizer = Whitespace()
>>> files = [...]
>>> tokenizer.train(files, trainer)
```

الآن لدينا محلل لغوي مدرب على الملفات التي حددناها. يمكننا إما الاستمرار في استخدامه في وقت التشغيل هذا، أو حفظه في ملف JSON لإعادة استخدامه لاحقًا.

## التحميل مباشرة من كائن المحلل اللغوي

دعونا نرى كيف يمكننا الاستفادة من كائن المحلل اللغوي هذا في مكتبة 🤗 Transformers. تسمح فئة [`PreTrainedTokenizerFast`] بالتشغيل الفوري، من خلال قبول كائن *المحلل اللغوي* الذي تم إنشاؤه كحجة:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

يمكن الآن استخدام هذا الكائن مع جميع الطرق التي تشترك فيها المحللات اللغوية لـ 🤗 Transformers! انتقل إلى [صفحة المحلل اللغوي](main_classes/tokenizer) لمزيد من المعلومات.

## التحميل من ملف JSON

لتحميل محلل لغوي من ملف JSON، دعونا نبدأ أولاً بحفظ محللنا اللغوي:

```python
>>> tokenizer.save("tokenizer.json")
```

يمكن تمرير المسار الذي حفظنا به هذا الملف إلى طريقة تهيئة [`PreTrainedTokenizerFast`] باستخدام معلمة `tokenizer_file`:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

يمكن الآن استخدام هذا الكائن مع جميع الطرق التي تشترك فيها المحللات اللغوية لـ 🤗 Transformers! انتقل إلى [صفحة المحلل اللغوي](main_classes/tokenizer) لمزيد من المعلومات.