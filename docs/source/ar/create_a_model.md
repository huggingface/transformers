# إنشاء بنية مخصصة

تستنتج فئة [`AutoClass`](model_doc/auto) تلقائيًا بنية النموذج وتقوم بتنزيل تكوين ووزن مسبقين. بشكل عام، نوصي باستخدام `AutoClass` لإنتاج كود غير مرتبط بنقطة تفتيش معينة. ولكن يمكن للمستخدمين الذين يريدون مزيدًا من التحكم في معلمات النموذج المحددة إنشاء نموذج مخصص من 🤗 Transformers من مجرد بضع فئات أساسية. قد يكون هذا مفيدًا بشكل خاص لأي شخص مهتم بدراسة نموذج 🤗 Transformers أو تدريبه أو إجراء تجارب عليه. في هذا الدليل، سنغوص بشكل أعمق في إنشاء نموذج مخصص بدون `AutoClass`. تعرف على كيفية:

- تحميل تكوين النموذج وتخصيصه.
- إنشاء بنية نموذج.
- إنشاء محلل نحوي سريع وبطيء للنص.
- إنشاء معالج صور لمهام الرؤية.
- إنشاء مستخرج ميزات لمهام الصوت.
- إنشاء معالج للمهام متعددة الوسائط.

## التكوين

يشير [التكوين](main_classes/configuration) إلى السمات المحددة للنموذج. لكل تكوين نموذج سمات مختلفة؛ على سبيل المثال، تمتلك جميع نماذج NLP سمات `hidden_size` و`num_attention_heads` و`num_hidden_layers` و`vocab_size` المشتركة. تحدد هذه السمات عدد رؤوس الاهتمام أو الطبقات المخفية لبناء نموذج بها.

الق نظرة فاحصة على [DistilBERT](model_doc/distilbert) عن طريق الوصول إلى [`DistilBertConfig`] لفحص سماته:

```py
>>> from transformers import DistilBertConfig

>>> config = DistilBertConfig()
>>> print(config)
DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

يعرض [`DistilBertConfig`] جميع السمات الافتراضية المستخدمة لبناء [`DistilBertModel`] أساسي. جميع السمات قابلة للتخصيص، مما يخلق مساحة للتجريب. على سبيل المثال، يمكنك تخصيص نموذج افتراضي لـ:

- تجربة دالة تنشيط مختلفة مع معلمة `activation`.
- استخدام نسبة إسقاط أعلى لاحتمالات الاهتمام مع معلمة `attention_dropout`.

```py
>>> my_config = DistilBertConfig(activation="relu", attention_dropout=0.4)
>>> print(my_config)
DistilBertConfig {
  "activation": "relu",
  "attention_dropout": 0.4,
 
```

يمكن تعديل سمات النموذج المدرب مسبقًا في دالة [`~PretrainedConfig.from_pretrained`] :

```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

بمجرد أن تصبح راضيًا عن تكوين نموذجك، يمكنك حفظه باستخدام [`~PretrainedConfig.save_pretrained`]. يتم تخزين ملف التكوين الخاص بك على أنه ملف JSON في دليل الحفظ المحدد:
بمجرد أن تصبح راضيًا عن تكوين نموذجك، يمكنك حفظه باستخدام [`~PretrainedConfig.save_pretrained`]. يتم تخزين ملف التكوين الخاص بك على أنه ملف JSON في دليل الحفظ المحدد:

```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

لإعادة استخدام ملف التكوين، قم بتحميله باستخدام [`~PretrainedConfig.from_pretrained`]:

```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
```

<Tip>

يمكنك أيضًا حفظ ملف التكوين الخاص بك كقاموس أو حتى كفرق بين سمات التكوين المخصص وسمات التكوين الافتراضي! راجع وثائق [التكوين](main_classes/configuration) لمزيد من التفاصيل.

</Tip>


## النموذج

الخطوة التالية هي إنشاء [نموذج](main_classes/models). النموذج - الذي يشار إليه أيضًا بشكل فضفاض باسم الهندسة المعمارية - يحدد ما تفعله كل طبقة وما هي العمليات التي تحدث. تُستخدم سمات مثل `num_hidden_layers` من التكوين لتحديد الهندسة المعمارية. تشترك جميع النماذج في فئة الأساس [`PreTrainedModel`] وبعض الطرق الشائعة مثل تغيير حجم تضمين الإدخال وتشذيب رؤوس الاهتمام الذاتي. بالإضافة إلى ذلك، فإن جميع النماذج هي أيضًا إما [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)، [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) أو [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) فئة فرعية. وهذا يعني أن النماذج متوافقة مع كل استخدام إطار عملها الخاص.

<frameworkcontent>
<pt>
قم بتحميل سمات التكوين المخصصة الخاصة بك في النموذج:

```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
>>> model = DistilBertModel(my_config)
```

هذا ينشئ نموذجًا بقيم عشوائية بدلاً من الأوزان المسبقة التدريب. لن تتمكن من استخدام هذا النموذج لأي شيء مفيد حتى الآن حتى تقوم بتدريبه. التدريب عملية مكلفة وتستغرق وقتًا طويلاً. من الأفضل بشكل عام استخدام نموذج مُدرب مسبقًا للحصول على نتائج أفضل بشكل أسرع، مع استخدام جزء بسيط فقط من الموارد المطلوبة للتدريب.

قم بإنشاء نموذج مُدرب مسبقًا باستخدام [`~PreTrainedModel.from_pretrained`]:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

عندما تقوم بتحميل الأوزان المُدربة مسبقًا، يتم تحميل تكوين النموذج الافتراضي تلقائيًا إذا كان النموذج مقدمًا من قبل 🤗 Transformers. ومع ذلك، يمكنك أيضًا استبدال - بعض أو كل - سمات تكوين النموذج الافتراضي بسماتك الخاصة إذا أردت ذلك:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased"، config=my_config)
```
</pt>
<tf>
قم بتحميل سمات التكوين المخصصة الخاصة بك في النموذج:

```py
>>> from transformers import TFDistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
>>> tf_model = TFDistilBertModel(my_config)
```

هذا ينشئ نموذجًا بقيم عشوائية بدلاً من الأوزان المسبقة التدريب. لن تتمكن من استخدام هذا النموذج لأي شيء مفيد حتى الآن حتى تقوم بتدريبه. التدريب عملية مكلفة وتستغرق وقتًا طويلاً. من الأفضل بشكل عام استخدام نموذج مُدرب مسبقًا للحصول على نتائج أفضل بشكل أسرع، مع استخدام جزء بسيط فقط من الموارد المطلوبة للتدريب.

قم بإنشاء نموذج مُدرب مسبقًا باستخدام [`~TFPreTrainedModel.from_pretrained`]:

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

عندما تقوم بتحميل الأوزان المُدربة مسبقًا، يتم تحميل تكوين النموذج الافتراضي تلقائيًا إذا كان النموذج مقدمًا من قبل 🤗 Transformers. ومع ذلك، يمكنك أيضًا استبدال - بعض أو كل - سمات تكوين النموذج الافتراضي بسماتك الخاصة إذا أردت ذلك:

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased"، config=my_config)
```
</tf>
</frameworkcontent>

### رؤوس النموذج

في هذه المرحلة، لديك نموذج DistilBERT الأساسي الذي يخرج *حالات مخفية*. يتم تمرير الحالات المخفية كإدخالات لرأس النموذج لإنتاج الإخراج النهائي. يوفر 🤗 Transformers رأس نموذج مختلف لكل مهمة طالما أن النموذج يدعم المهمة (أي لا يمكنك استخدام DistilBERT لمهمة تسلسل إلى تسلسل مثل الترجمة).

<frameworkcontent>
<pt>
على سبيل المثال، [`DistilBertForSequenceClassification`] هو نموذج DistilBERT الأساسي برأس تصنيف تسلسل. رأس التصنيف التسلسلي هو طبقة خطية أعلى المخرجات المجمعة.

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

أعد استخدام هذا نقطة التحقق لمهمة أخرى عن طريق التبديل إلى رأس نموذج مختلف. لمهمة الإجابة على الأسئلة، ستستخدم رأس النموذج [`DistilBertForQuestionAnswering`]. رأس الإجابة على الأسئلة مشابه لرأس التصنيف التسلسلي باستثناء أنه طبقة خطية أعلى حالات الإخراج المخفية.

```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```
</pt>
<tf>
على سبيل المثال، [`TFDistilBertForSequenceClassification`] هو نموذج DistilBERT الأساسي برأس تصنيف تسلسل. رأس التصنيف التسلسلي هو طبقة خطية أعلى المخرجات المجمعة.

```py
>>> from transformers import TFDistilBertForSequenceClassification

>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

أعد استخدام هذا نقطة التحقق لمهمة أخرى عن طريق التبديل إلى رأس نموذج مختلف. لمهمة الإجابة على الأسئلة، ستستخدم رأس النموذج [`TFDistilBertForQuestionAnswering`]. رأس الإجابة على الأسئلة مشابه لرأس التصنيف التسلسلي باستثناء أنه طبقة خطية أعلى حالات الإخراج المخفية.

```py
>>> from transformers import TFDistilBertForQuestionAnswering

>>> tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```
</tf>
</frameworkcontent>

## محول الرموز

الفئة الأساسية الأخيرة التي تحتاجها قبل استخدام نموذج للبيانات النصية هي [محول الرموز](main_classes/tokenizer) لتحويل النص الخام إلى تنسورات. هناك نوعان من المحولات التي يمكنك استخدامها مع 🤗 Transformers:

- [`PreTrainedTokenizer`]: تنفيذ Python لمحول الرموز.
- [`PreTrainedTokenizerFast`]: محول رموز من مكتبة [🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/) المستندة إلى Rust الخاصة بنا. هذا النوع من المحول أسرع بشكل ملحوظ - خاصة أثناء توكين الدُفعات - بسبب تنفيذه في Rust. يوفر محول الرموز السريع أيضًا طرقًا إضافية مثل *مخطط الإزاحة* الذي يقوم بتعيين الرموز إلى كلماتها أو أحرفها الأصلية.

يدعم كلا النوعين من المحولات طرقًا شائعة مثل الترميز وفك الترميز وإضافة رموز جديدة وإدارة الرموز الخاصة.

<Tip warning={true}>

لا يدعم كل نموذج محول رموز سريع. الق نظرة على هذا [جدول](index#supported-frameworks) للتحقق مما إذا كان النموذج يحتوي على دعم محول رموز سريع.

</Tip>

إذا قمت بتدريب محول رموز مخصص، فيمكنك إنشاء واحد من ملف *المفردات* الخاص بك:

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt"، do_lower_case=False، padding_side="left")
```

من المهم أن تتذكر أن القاموس من محلل نحوي مخصص سيكون مختلفًا عن القاموس الذي تم إنشاؤه بواسطة محلل نحوي لنموذج مدرب مسبقًا. تحتاج إلى استخدام قاموس نموذج مدرب مسبقًا إذا كنت تستخدم نموذجًا مدربًا مسبقًا، وإلا فلن يكون للإدخالات معنى. قم بإنشاء محلل نحوي باستخدام قاموس نموذج مدرب مسبقًا باستخدام فئة [`DistilBertTokenizer`] :

```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

قم بإنشاء محلل نحوي سريع باستخدام فئة [`DistilBertTokenizerFast`] :

```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip>

افتراضيًا، سيحاول [`AutoTokenizer`] تحميل محلل نحوي سريع. يمكنك تعطيل هذا السلوك عن طريق تعيين `use_fast=False` في `from_pretrained`.

</Tip>

## معالج الصور

يقوم معالج الصور بمعالجة إدخالات الرؤية. إنه يرث من فئة الأساس [`~image_processing_utils.ImageProcessingMixin`].

للاستخدام، قم بإنشاء معالج صور مرتبط بالنموذج الذي تستخدمه. على سبيل المثال، قم بإنشاء [`ViTImageProcessor`] افتراضي إذا كنت تستخدم [ViT](model_doc/vit) لتصنيف الصور:

```py
>>> from transformers import ViTImageProcessor

>>> vit_extractor = ViTImageProcessor()
>>> print(vit_extractor)
ViTImageProcessor {
  "do_normalize": true,
  "do_resize": true,
  "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

<Tip>

إذا كنت لا تبحث عن أي تخصيص، فما عليك سوى استخدام طريقة `from_pretrained` لتحميل معلمات معالج الصور الافتراضية للنموذج.

</Tip>

عدل أيًا من معلمات [`ViTImageProcessor`] لإنشاء معالج الصور المخصص الخاص بك:

```py
>>> from transformers import ViTImageProcessor

>>> my_vit_extractor = ViTImageProcessor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTImageProcessor {
  "do_normalize": false,
  "do_resize": true,
 "image_processor_type": "ViTImageProcessor",
  "image_mean": [
    0.3,
    0.3,
    0.3
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": "PIL.Image.BOX",
  "size": 224
}
```
## العمود الفقري

<div style="text-align: center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Backbone.png">
</div>

تتكون نماذج رؤية الكمبيوتر من عمود فقري وعنق ورأس. يستخرج العمود الفقري الميزات من صورة الإدخال، ويجمع العنق الميزات المستخرجة ويعززها، ويتم استخدام الرأس للمهمة الرئيسية (مثل اكتشاف الكائنات). ابدأ عن طريق تهيئة عمود فقري في تكوين النموذج وحدد ما إذا كنت تريد تحميل أوزان مدربة مسبقًا أو تحميل أوزان مُهيأة بشكل عشوائي. بعد ذلك، يمكنك تمرير تكوين النموذج إلى رأس النموذج.

على سبيل المثال، لتحميل [ResNet](../model_doc/resnet) backbone في نموذج [MaskFormer](../model_doc/maskformer) مع رأس تجزئة مثيل:

<hfoptions id="backbone">
<hfoption id="pretrained weights">

قم بتعيين `use_pretrained_backbone=True` لتحميل الأوزان المسبقة التدريب لـ ResNet للعمود الفقري.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="microsoft/resnet-50", use_pretrained_backbone=True) # تكوين العمود الفقري والعنق
model = MaskFormerForInstanceSegmentation(config) # الرأس
```

</hfoption>
<hfoption id="random weights">

قم بتعيين `use_pretrained_backbone=False` لتهيئة عمود فقري ResNet بشكل عشوائي.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="microsoft/resnet-50", use_pretrained_backbone=False) # تكوين العمود الفقري والعنق
model = MaskFormerForInstanceSegmentation(config) # الرأس
```

يمكنك أيضًا تحميل تكوين العمود الفقري بشكل منفصل ثم تمريره إلى تكوين النموذج.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig

backbone_config = ResNetConfig()
config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

</hfoption>
</hfoptions id="timm backbone">

يتم تحميل نماذج [timm](https://hf.co/docs/timm/index) داخل نموذج باستخدام `use_timm_backbone=True` أو باستخدام [`TimmBackbone`] و [`TimmBackboneConfig`].

استخدم `use_timm_backbone=True` و `use_pretrained_backbone=True` لتحميل الأوزان المسبقة التدريب لـ timm للعمود الفقري.

```python
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="resnet50", use_pretrained_backbone=True, use_timm_backbone=True) # تكوين العمود الفقري والعنق
model = MaskFormerForInstanceSegmentation(config) # الرأس
```

قم بتعيين `use_timm_backbone=True` و `use_pretrained_backbone=False` لتحميل عمود فقري timm مبدئي عشوائي.

```python
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone="resnet50", use_pretrained_backbone=False, use_timm_backbone=True) # تكوين العمود الفقري والعنق
model = MaskFormerForInstanceSegmentation(config) # الرأس
```

يمكنك أيضًا تحميل تكوين العمود الفقري واستخدامه لإنشاء `TimmBackbone` أو تمريره إلى تكوين النموذج. سيتم تحميل العمود الفقري لـ Timm الأوزان المسبقة التدريب بشكل افتراضي. قم بتعيين `use_pretrained_backbone=False` لتحميل الأوزان المبدئية العشوائية.

```python
from transformers import TimmBackboneConfig, TimmBackbone

backbone_config = TimmBackboneConfig("resnet50", use_pretrained_backbone=False)

# قم بإنشاء مثيل من العمود الفقري
backbone = TimmBackbone(config=backbone_config)

# قم بإنشاء نموذج باستخدام عمود فقري timm
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

## مستخرج الميزات

يقوم مستخرج الميزات بمعالجة المدخلات الصوتية. يرث من فئة الأساس [`~feature_extraction_utils.FeatureExtractionMixin`]، وقد يرث أيضًا من فئة [`SequenceFeatureExtractor`] لمعالجة المدخلات الصوتية.

للاستخدام، قم بإنشاء مستخرج ميزات مرتبط بالنموذج الذي تستخدمه. على سبيل المثال، قم بإنشاء مستخرج ميزات Wav2Vec2 الافتراضي إذا كنت تستخدم [Wav2Vec2](model_doc/wav2vec2) لتصنيف الصوت:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor()
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 16000
}
```

<Tip>

إذا كنت لا تبحث عن أي تخصيص، فما عليك سوى استخدام طريقة `from_pretrained` لتحميل معلمات مستخرج الميزات الافتراضية للنموذج.

</Tip>

قم بتعديل أي من معلمات [`Wav2Vec2FeatureExtractor`] لإنشاء مستخرج ميزات مخصص:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor(sampling_rate=8000، do_normalize=False)
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": false,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor"،
  "feature_size": 1،
  "padding_side": "right"،
  "padding_value": 0.0،
  "return_attention_mask": false،
  "sampling_rate": 8000
}
```

## المعالج

بالنسبة للنماذج التي تدعم مهام متعددة الوسائط، توفر مكتبة 🤗 Transformers فئة معالج تدمج بشكل ملائم فئات المعالجة مثل مستخرج الميزات ومقسّم الرموز في كائن واحد. على سبيل المثال، دعنا نستخدم [`Wav2Vec2Processor`] لمهمة التعرف التلقائي على الكلام (ASR). يقوم ASR بنقل الصوت إلى نص، لذلك ستحتاج إلى مستخرج ميزات ومقسّم رموز.

قم بإنشاء مستخرج ميزات لمعالجة المدخلات الصوتية:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

قم بإنشاء مقسّم رموز لمعالجة المدخلات النصية:

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

قم بدمج مستخرج الميزات ومقسّم الرموز في [`Wav2Vec2Processor`]:

```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

مع فئتين أساسيتين - التكوين والنموذج - وفئة معالجة مسبقة إضافية (مقسّم رموز أو معالج صورة أو مستخرج ميزات أو معالج)، يمكنك إنشاء أي من النماذج التي تدعمها مكتبة 🤗 Transformers. يمكن تكوين كل من هذه الفئات الأساسية، مما يسمح لك باستخدام السمات المحددة التي تريدها. يمكنك بسهولة إعداد نموذج للتدريب أو تعديل نموذج مسبق التدريب موجود للضبط الدقيق.