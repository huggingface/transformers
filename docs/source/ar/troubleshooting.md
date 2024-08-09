# استكشاف الأخطاء وإصلاحها

في بعض الأحيان تحدث الأخطاء، لكننا هنا للمساعدة! يغطي هذا الدليل بعضًا من أكثر المشكلات شيوعًا التي رأيناها وكيفية حلها. ومع ذلك، لا يُقصد بهذا الدليل أن يكون مجموعة شاملة من كل مشكلة في مكتبة Hugging Face Transformers. للحصول على مزيد من المساعدة في استكشاف أخطائك وإصلاحها، جرّب ما يلي:

1. اطلب المساعدة على [المنتديات](https://discuss.huggingface.co/). هناك فئات محددة يمكنك نشر سؤالك فيها، مثل [المبتدئين](https://discuss.huggingface.co/c/beginners/5) أو [Hugging Face Transformers](https://discuss.huggingface.co/c/transformers/9). تأكد من كتابة منشور جيد وواضح على المنتدى مع بعض التعليمات البرمجية القابلة للتكرار لزيادة احتمالية حل مشكلتك!

2. قم بإنشاء [قضية](https://github.com/huggingface/transformers/issues/new/choose) في مستودع Hugging Face Transformers إذا كانت هناك مشكلة متعلقة بالمكتبة. حاول تضمين أكبر قدر ممكن من المعلومات التي تصف المشكلة لمساعدتنا في معرفة ما هو الخطأ وكيفية إصلاحه.

3. تحقق من دليل [الهجرة](migration) إذا كنت تستخدم إصدارًا أقدم من مكتبة Hugging Face Transformers حيث تم تقديم بعض التغييرات المهمة بين الإصدارات.

للحصول على مزيد من التفاصيل حول استكشاف الأخطاء وإصلاحها والحصول على المساعدة، راجع [الفصل 8](https://huggingface.co/course/chapter8/1?fw=pt) من دورة Hugging Face.

## بيئات جدار الحماية

تكون بعض مثيلات GPU في السحابة وإعدادات الشبكة الداخلية محمية بجدار حماية من الاتصالات الخارجية، مما يؤدي إلى حدوث خطأ في الاتصال. عندما تحاول تعليمات البرنامج النصي تنزيل أوزان النموذج أو مجموعات البيانات، سيتوقف التنزيل ثم ينتهي بخطأ مثل:

```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

في هذه الحالة، يجب عليك تشغيل مكتبة Hugging Face Transformers في [وضع عدم الاتصال](installation#offline-mode) لتجنب خطأ الاتصال.

## CUDA نفاد الذاكرة

يمكن أن يكون تدريب النماذج الكبيرة التي تحتوي على ملايين المعلمات أمرًا صعبًا بدون الأجهزة المناسبة. أحد الأخطاء الشائعة التي قد تواجهها عند نفاد ذاكرة GPU هو:

```
CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 11.17 GiB total capacity; 9.70 GiB already allocated; 179.81 MiB free; 9.85 GiB reserved in total by PyTorch)
```

فيما يلي بعض الحلول المحتملة التي يمكنك تجربتها لتقليل استخدام الذاكرة:

- قلل من قيمة [`per_device_train_batch_size`](main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size) في [`TrainingArguments`].

- حاول استخدام [`gradient_accumulation_steps`](main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) في [`TrainingArguments`] لزيادة حجم الدُفعة بشكل فعال.

<Tip>
راجع دليل [الأداء](performance) لمزيد من التفاصيل حول تقنيات توفير الذاكرة.
</Tip>

## عدم القدرة على تحميل نموذج TensorFlow محفوظ

تقوم طريقة TensorFlow [model.save](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) بحفظ النموذج بالكامل - الهندسة المعمارية، الأوزان، تكوين التدريب - في ملف واحد. ومع ذلك، عند تحميل ملف النموذج مرة أخرى، قد تواجه خطأ لأن مكتبة Hugging Face Transformers قد لا تقوم بتحميل جميع الكائنات المتعلقة بـ TensorFlow في ملف النموذج. لتجنب المشكلات عند حفظ نماذج TensorFlow وتحميلها، نوصي بما يلي:

- احفظ أوزان النموذج كملف `h5` باستخدام [`model.save_weights`](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) ثم أعد تحميل النموذج باستخدام [`~TFPreTrainedModel.from_pretrained`]:

```بايثون
>>> from transformers import TFPreTrainedModel
>>> from tensorflow import keras

>>> model.save_weights("some_folder/tf_model.h5")
>>> model = TFPreTrainedModel.from_pretrained("some_folder")
```

- احفظ النموذج باستخدام [`~TFPretrainedModel.save_pretrained`] وقم بتحميله مرة أخرى باستخدام [`~TFPreTrainedModel.from_pretrained`]:

```بايثون
>>> from transformers import TFPreTrainedModel

>>> model.save_pretrained("path_to/model")
>>> model = TFPreTrainedModel.from_pretrained("path_to/model")
```

## ImportError

خطأ شائع آخر قد تواجهه، خاصة إذا كان نموذجًا تم إصداره حديثًا، هو `ImportError`:

```
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

بالنسبة لأنواع الأخطاء هذه، تحقق من أن لديك أحدث إصدار من مكتبة Hugging Face Transformers مثبتًا للوصول إلى أحدث النماذج:

```bash
pip install transformers --upgrade
```

## خطأ CUDA: تم تشغيل التأكيد على جانب الجهاز

في بعض الأحيان، قد تواجه خطأ CUDA عامًا حول خطأ في كود الجهاز.

```
RuntimeError: CUDA error: device-side assert triggered
```

يجب عليك محاولة تشغيل الكود على وحدة المعالجة المركزية (CPU) أولاً للحصول على رسالة خطأ أكثر دقة. أضف متغير البيئة التالي في بداية كودك للتبديل إلى وحدة المعالجة المركزية:

```بايثون
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

الخيار الآخر هو الحصول على تتبع مكدس أفضل من GPU. أضف متغير البيئة التالي في بداية كودك للحصول على تتبع المكدس للإشارة إلى مصدر الخطأ:

```بايثون
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## إخراج غير صحيح عند عدم إخفاء رموز الحشو

في بعض الحالات، قد يكون الإخراج `hidden_state` غير صحيح إذا تضمنت `input_ids` رموز حشو. ولإثبات ذلك، قم بتحميل نموذج ومصنف الرموز. يمكنك الوصول إلى `pad_token_id` للنموذج لمعرفة قيمته. قد تكون `pad_token_id` `None` لبعض النماذج، ولكن يمكنك دائمًا تعيينها يدويًا.

```بايثون
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
>>> model.config.pad_token_id
0
```

يوضح المثال التالي الإخراج بدون إخفاء رموز الحشو:

```بايثون
>>> input_ids = torch.tensor([[7592, 2057, 2097, 2393, 9611, 2115], [7592, 0, 0, 0, 0, 0]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
[ 0.1317, -0.1683]], grad_fn=<AddmmBackward0>)
```

هذا هو الإخراج الفعلي للتسلسل الثاني:

```بايثون
>>> input_ids = torch.tensor([[7592]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

يجب عليك في معظم الوقت توفير `attention_mask` للنموذج لتجاهل رموز الحشو لتجنب هذا الخطأ الصامت. الآن يتطابق إخراج التسلسل الثاني مع الإخراج الفعلي:

<Tip>
يقوم المصنف الرموز افتراضيًا بإنشاء `attention_mask` لك بناءً على إعدادات المصنف الرموز الافتراضية.
</Tip>

```بايثون
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

لا تقوم مكتبة Hugging Face Transformers تلقائيًا بإنشاء `attention_mask` لإخفاء رمز الحشو إذا كان موجودًا لأن:

- بعض النماذج ليس لها رمز حشو.

- بالنسبة لبعض الاستخدامات، يريد المستخدمون أن ينتبه النموذج إلى رمز الحشو.
## ValueError: فئة التكوين غير المعترف بها XYZ لهذا النوع من AutoModel

بشكل عام، نوصي باستخدام فئة [`AutoModel`] لتحميل مثيلات مُدربة مسبقًا من النماذج. يمكن لهذه الفئة أن تستنتج وتُحمل تلقائيًا البنية الصحيحة من نقطة تفتيش معينة بناءً على التكوين. إذا رأيت هذا الخطأ `ValueError` عند تحميل نموذج من نقطة تفتيش، فهذا يعني أن الفئة التلقائية لم تتمكن من العثور على خريطة من التكوين في نقطة التفتيش المعطاة إلى نوع النموذج الذي تحاول تحميله. وغالبًا ما يحدث هذا عندما لا تدعم نقطة التفتيش مهمة معينة.

على سبيل المثال، سترى هذا الخطأ في المثال التالي لأنه لا يوجد GPT2 للإجابة على الأسئلة:

```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("openai-community/gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("openai-community/gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```