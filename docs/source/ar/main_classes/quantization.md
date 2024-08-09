# الضبط الكمي 

تقنيات الضبط الكمي تقلل من تكاليف الذاكرة والحوسبة من خلال تمثيل الأوزان والتنشيطات باستخدام أنواع بيانات أقل دقة مثل الأعداد الصحيحة 8-بت (int8). يسمح ذلك بتحميل نماذج أكبر عادةً لا يمكنك تحميلها في الذاكرة، وتسريع الاستدلال. تدعم مكتبة Transformers خوارزميات الضبط الكمي AWQ وGPTQ، كما تدعم الضبط الكمي 8-بت و4-بت مع bitsandbytes.

يمكن إضافة تقنيات الضبط الكمي التي لا تدعمها مكتبة Transformers باستخدام فئة [`HfQuantizer`].

تعلم كيفية ضبط نماذج في الدليل [الضبط الكمي](../الضبط-الكمي).

## QuantoConfig
[[autodoc]] QuantoConfig

## AqlmConfig
[[autodoc]] AqlmConfig

## AwqConfig
[[autodoc]] AwqConfig

## EetqConfig
[[autodoc]] EetqConfig

## GPTQConfig
[[autodoc]] GPTQConfig

## BitsAndBytesConfig
[[autodoc]] BitsAndBytesConfig

## HfQuantizer
[[autodoc]] quantizers.base.HfQuantizer

## HqqConfig
[[autodoc]] HqqConfig