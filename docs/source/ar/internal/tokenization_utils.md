# أدوات مساعدة لتقسيم النص إلى رموز

تدرج هذه الصفحة جميع دالات الأدوات المساعدة التي يستخدمها برامج التقسيم، وبشكل أساسي الفئة [~tokenization_utils_base.PreTrainedTokenizerBase] التي تنفذ الدالات المشتركة بين [PreTrainedTokenizer] و [PreTrainedTokenizerFast] و [~tokenization_utils_base.SpecialTokensMixin].

معظم هذه الدالات مفيدة فقط إذا كنت تدرس شيفرة برامج التقسيم في المكتبة.

## PreTrainedTokenizerBase

[[autodoc]] tokenization_utils_base.PreTrainedTokenizerBase
- __call__
- all

## SpecialTokensMixin

[[autodoc]] tokenization_utils_base.SpecialTokensMixin

## Enums و namedtuples

[[autodoc]] tokenization_utils_base.TruncationStrategy

[[autodoc]] tokenization_utils_base.CharSpan

[[autodoc]] tokenization_utils_base.TokenSpan