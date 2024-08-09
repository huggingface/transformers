# FLAVA

## نظرة عامة

اقترح نموذج FLAVA في [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482) بواسطة Amanpreet Singh، Ronghang Hu، Vedanuj Goswami، Guillaume Couairon، Wojciech Galuba، Marcus Rohrbach، و Douwe Kiela وتم قبوله في مؤتمر CVPR 2022.

تهدف هذه الورقة إلى إنشاء نموذج أساس موحد واحد يمكنه العمل عبر الرؤية واللغة، بالإضافة إلى المهام متعددة الوسائط للرؤية واللغة.

الملخص من الورقة هو ما يلي:

*تعتمد أحدث نماذج الرؤية والرؤية واللغة على التدريب اللغوي على نطاق واسع للحصول على أداء جيد في مجموعة متنوعة من المهام اللاحقة. بشكل عام، تكون هذه النماذج غالبًا إما متعددة الوسائط (اندماج سابق) أو متعددة الوسائط ولكن ليس كلاهما؛ وغالبًا ما تستهدف طرائق أو مهام محددة فقط. سيكون الاتجاه الواعد هو استخدام نموذج شامل عالمي واحد، باعتباره "أساسًا"، يستهدف جميع الطرائق في نفس الوقت - يجب أن يكون النموذج الأساسي الحقيقي للرؤية واللغة جيدًا في مهام الرؤية ومهام اللغة، والمهام متعددة الوسائط للرؤية واللغة. نقدم FLAVA كنموذج من هذا القبيل ونظهر أداءً رائعًا في مجموعة واسعة من 35 مهمة تشمل هذه الطرائق المستهدفة.*

تمت المساهمة بهذا النموذج من قبل [aps](https://huggingface.co/aps). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/multimodal/tree/main/examples/flava).

## FlavaConfig

[[autodoc]] FlavaConfig

## FlavaTextConfig

[[autodoc]] FlavaTextConfig

## FlavaImageConfig

[[autodoc]] FlavaImageConfig

## FlavaMultimodalConfig

[[autodoc]] FlavaMultimodalConfig

## FlavaImageCodebookConfig

[[autodoc]] FlavaImageCodebookConfig

## FlavaProcessor

[[autodoc]] FlavaProcessor

## FlavaFeatureExtractor

[[autodoc]] FlavaFeatureExtractor

## FlavaImageProcessor

- preprocess

## FlavaForPreTraining

[[autodoc]] FlavaForPreTraining

- forward

## FlavaModel

[[autodoc]] FlavaModel

- forward

- get_text_features

- get_image_features

## FlavaImageCodebook

[[autodoc]] FlavaImageCodebook

- forward

- get_codebook_indices

- get_codebook_probs

## FlavaTextModel

[[autodoc]] FlavaTextModel

- forward

## FlavaImageModel

[[autodoc]] FlavaImageModel


- forward

## FlavaMultimodalModel

[[autodoc]] FlavaMultimodalModel

- forward