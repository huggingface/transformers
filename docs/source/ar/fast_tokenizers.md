# استخدام مجزئيات النصوص من 🤗 Tokenizers

يعتمد [`PreTrainedTokenizerFast`] على مكتبة [🤗 Tokenizers](https://huggingface.co/docs/tokenizers). يمكن تحميل المجزئات اللغويين الذين تم الحصول عليهم من مكتبة 🤗 Tokenizers ببساطة شديدة في 🤗 Transformers.

قبل الدخول في التفاصيل، دعونا نبدأ أولاً بإنشاء مُجزىء لغوي تجريبي في بضع سطور:

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

الآن لدينا مُجزىء لغوي مدرب على الملفات التي حددناها. يمكننا إما الاستمرار في استخدامه في وقت التشغيل هذا، أو حفظه في ملف JSON لإعادة استخدامه لاحقًا.

## تحميل مُجزئ  النّصوص  مُباشرةً

دعونا نرى كيف يمكننا الاستفادة من كائن (مُجزئ النصوص) في مكتبة 🤗 Transformers. تسمح فئة [`PreTrainedTokenizerFast`] سهولة إنشاء *tokenizer*، من خلال قبول كائن *المُجزئ النصوص*  مُهيّأ مُسبقًا كمعامل:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

يمكن الآن استخدام هذا الكائن مع جميع الطرق المُشتركة بين مُجزّئي النّصوص  لـ 🤗 Transformers! انتقل إلى [صفحة مُجزّئ  النّصوص](main_classes/tokenizer) لمزيد من المعلومات.

## التحميل من ملف JSON

لتحميل مُجزّئ النص من ملف JSON، دعونا نبدأ أولاً بحفظ مُجزّئ النّصوص:

```python
>>> tokenizer.save("tokenizer.json")
```

يمكن تمرير المسار الذي حفظنا به هذا الملف إلى طريقة تهيئة [`PreTrainedTokenizerFast`] باستخدام المُعامل  `tokenizer_file`:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

يمكن الآن استخدام هذا الكائن مع جميع الطرق التي تشترك فيها مُجزّئي  النّصوص لـ 🤗 Transformers! انتقل إلى [صفحة مُجزّئ النص](main_classes/tokenizer) لمزيد من المعلومات.
