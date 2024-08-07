# Falcon

## نظرة عامة

Falcon هي فئة من فك تشفير النماذج السببية التي بناها [TII](https://www.tii.ae/). تم تدريب أكبر نقاط تفتيش Falcon على >=1T من الرموز النصية، مع التركيز بشكل خاص على [RefinedWeb](https://arxiv.org/abs/2306.01116) corpus. وهي متاحة بموجب ترخيص Apache 2.0.

تم تصميم بنية Falcon بشكل حديث وتمت تهيئتها للتنفيذ، مع دعم الاهتمام متعدد الاستعلامات واهتمام فعال للمتغيرات مثل `FlashAttention`. تتوفر نماذج "base" المدربة فقط كنماذج لغة سببية، بالإضافة إلى نماذج "instruct" التي خضعت لمزيد من الضبط الدقيق.

تعد نماذج Falcon (اعتبارًا من عام 2023) من أكبر نماذج اللغة مفتوحة المصدر وأكثرها قوة، وتحتل باستمرار مرتبة عالية في [لوحة قيادة OpenLLM](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

## تحويل نقاط التفتيش المخصصة

<Tip>

تمت إضافة نماذج Falcon في البداية إلى Hugging Face Hub كنقاط تفتيش للرمز المخصص. ومع ذلك، يتم الآن دعم Falcon بالكامل في مكتبة المحولات. إذا قمت بضبط دقيق لنموذج من نقطة تفتيش للرمز المخصص، فإننا نوصي بتحويل نقطة تفتيشك إلى التنسيق الجديد داخل المكتبة، حيث يجب أن يوفر ذلك تحسينات كبيرة في الاستقرار والأداء، خاصة للجيل، بالإضافة إلى إزالة الحاجة إلى استخدام `trust_remote_code=True`!

</Tip>

يمكنك تحويل نقاط تفتيش الرمز المخصص إلى نقاط تفتيش Transformers كاملة باستخدام `convert_custom_code_checkpoint.py` النصي الموجود في [دليل نموذج Falcon](https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon) في مكتبة المحولات. لاستخدام هذا البرنامج النصي، ما عليك سوى استدعائه باستخدام `python convert_custom_code_checkpoint.py --checkpoint_dir my_model`. سيؤدي هذا إلى تحويل نقطة التفتيش الخاصة بك في المكان، ويمكنك تحميلها على الفور من الدليل بعد ذلك باستخدام `from_pretrained()`. إذا لم يتم تحميل نموذجك إلى Hub، فنحن نوصي بعمل نسخة احتياطية قبل محاولة التحويل، تحسبا لأي طارئ!

## FalconConfig

[[autodoc]] FalconConfig

- all

## FalconModel

[[autodoc]] FalconModel

- forword

## FalconForCausalLM

[[autodoc]] FalconForCausalLM

- forword

## FalconForSequenceClassification

[[autodoc]] FalconForSequenceClassification

- forword

## FalconForTokenClassification

[[autodoc]] FalconForTokenClassification

- forword

## FalconForQuestionAnswering

[[autodoc]] FalconForQuestionAnswering

- forword