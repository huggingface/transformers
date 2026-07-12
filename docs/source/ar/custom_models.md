# بناء نماذج مخصصة

تم تصميم مكتبة 🤗 Transformers لتكون قابلة للتوسيع بسهولة. كل نموذج مُشفّر بالكامل في مجلد فرعي معين بالمستودع، دون أي تجريد، لذلك يمكنك بسهولة نسخ ملف النمذجة وتعديله وفقًا لاحتياجاتك.

إذا كنت تُنشئ نموذجًا جديدًا تمامًا، فقد يكون من الأسهل البدء من الصفر. في هذا البرنامج التعليمي، سنُرِيك كيفية كتابة نموذج مخصص وتكوينه ليُستخدم داخل Transformers، وكيفية مشاركته مع المجتمع (مع الكود الذي يعتمد عليه) بحيث يمكن لأي شخص استخدامه، حتى إذا لم يكن موجودًا في مكتبة 🤗 Transformers. سنرى كيفية البناء على المحولات ونوسّع الإطار  باستخدام الأدوات التي يمكن استخدامها لتعديل سلوك الإطار (hooks) والتعليمات البرمجية المخصصة.

سنوضح كل هذا من خلال نموذج ResNet، بتغليف فئة ResNet من
[مكتبة timm](https://github.com/rwightman/pytorch-image-models) داخل [`PreTrainedModel`].

## كتابة إعدادات مخصصة

لنبدأ بكتابة إعدادات النموذج. إعدادات النموذج هو كائنٌ يحتوي على جميع المعلومات اللازمة لبنائه. كما سنرى لاحقًا، يتطلب النموذج كائن `config` لتهيئته، لذا يجب أن يكون هذا الكائن كاملاً.

<Tip>

تتبع النماذج في مكتبة `transformers` اتفاقية قبول كائن `config` في دالة  `__init__` الخاصة بها، ثم تمرر كائن `config` بالكامل إلى الطبقات الفرعية في النموذج، بدلاً من تقسيمه إلى معامﻻت متعددة. يؤدي كتابة نموذجك بهذا الأسلوب إلى كود أبسط مع "مصدر حقيقة" واضح لأي فرط معلمات، كما يسهل إعادة استخدام الكود من نماذج أخرى في `transformers`.

</Tip>

في مثالنا، سنعدّل بعض الوسائط في فئة ResNet التي قد نرغب في ضبطها. ستعطينا التكوينات المختلفة أنواع ResNets المختلفة الممكنة. سنقوم بتخزين هذه الوسائط بعد التحقق من صحته.

```python
from transformers import PreTrainedConfig
from typing import List


class ResnetConfig(PreTrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: list[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```
الأشياء الثلاثة المهمة التي يجب تذكرها عند كتابة تكوينك الخاص هي:

- يجب أن ترث من `PreTrainedConfig`،
- يجب أن تقبل دالة  `__init__` الخاصة بـ `PreTrainedConfig` أي معامﻻت إضافية kwargs،
- يجب تمرير هذه المعامﻻت الإضافية إلى دالة `__init__` فى الفئة الأساسية الاعلى.

يضمن الإرث حصولك على جميع الوظائف من مكتبة 🤗 Transformers، في حين أن القيدين التانى والثالث يأتيان من حقيقة أن `PreTrainedConfig` لديه المزيد من الحقول أكثر من تلك التي تقوم بتعيينها. عند إعادة تحميل تكوين باستخدام طريقة `from_pretrained`، يجب أن يقبل تكوينك هذه الحقول ثم إرسالها إلى الفئة الأساسية الأعلى.

تحديد `model_type` لتكوينك (هنا `model_type="resnet"`) ليس إلزاميًا، ما لم ترغب في
تسجيل نموذجك باستخدام الفئات التلقائية (راجع القسم الأخير).

مع القيام بذلك، يمكنك بسهولة إنشاء تكوينك وحفظه مثلما تفعل مع أي تكوين نموذج آخر في
المكتبة. إليك كيفية إنشاء تكوين resnet50d وحفظه:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

سيؤدي هذا إلى حفظ ملف باسم `config.json` داخل مجلد `custom-resnet`. يمكنك بعد ذلك إعادة تحميل تكوينك باستخدام
طريقة `from_pretrained`:

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

يمكنك أيضًا استخدام أي طريقة أخرى من فئة [`PreTrainedConfig`]، مثل [`~PreTrainedConfig.push_to_hub`] لتحميل تكوينك مباشرة إلى Hub.

## كتابة نموذج مخصص

الآن بعد أن أصبح لدينا تكوين ResNet، يمكننا المتابعة لإنشاء نموذجين: الأول يستخرج الميزات المخفية  من دفعة من الصور (مثل [`BertModel`]) والآخر مناسب لتصنيف الصور (مثل [`BertForSequenceClassification`]).

 كما ذكرنا سابقًا، سنقوم ببناء نموذج مبسط لتسهيل الفهم في هذا المثال. الخطوة الوحيدة المطلوبة قبل كتابة هذه الفئة هي لربط أنواع وحدات البناء بفئات ذات وحدات بناء فعلية. بعد ذلك، يُعرّف النموذج من خلال التكوين عبر تمرير كل شيء إلى فئة `ResNet`:

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

بالنسبة للنموذج الذي سيصنف الصور، فإننا نغير فقط طريقة التقديم:

```py
import torch


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```
في كلتا الحالتين، لاحظ كيف نرث من `PreTrainedModel` ونستدعي مُهيئ الفئة الرئيسية باستخدام `config` (كما تفعل عند إنشاء وحدة `torch.nn.Module` عادية). ليس من الضروري تعريف `config_class` إلا إذا كنت ترغب في تسجيل نموذجك مع الفئات التلقائية (راجع القسم الأخير).

<Tip>

إذا كان نموذجك مشابهًا جدًا لنموذج داخل المكتبة، فيمكنك إعادة استخدام نفس التكوين مثل هذا النموذج.

</Tip>

يمكن لنموذجك أن يعيد أي شيء تريده، ولكن إعادة قاموس مثلما فعلنا لـ
`ResnetModelForImageClassification`، مع تضمين الخسارة عند تمرير العلامات، سيجعل نموذجك قابلًا للاستخدام مباشرة داخل فئة [`Trainer`]. يعد استخدام تنسيق إخراج آخر أمرًا جيدًا طالما أنك تخطط لاستخدام حلقة تدريب خاصة بك أو مكتبة أخرى للتدريب.

الآن بعد أن أصبح لدينا فئة النموذج، دعنا ننشئ واحدة:

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

يمكنك استخدام أي من طرق فئة [`PreTrainedModel`]، مثل [`~PreTrainedModel.save_pretrained`] أو
[`~PreTrainedModel.push_to_hub`]. سنستخدم الثاني في القسم التالي، وسنرى كيفية دفع أوزان النموذج مع كود نموذجنا. ولكن أولاً، دعنا نحمل بعض الأوزان المُعلمة مسبقًا داخل نموذجنا.

في حالة الاستخدام الخاصة بك، فمن المحتمل أن تقوم بتدريب نموذجك المخصص على بياناتك الخاصة. للانتقال بسرعة خلال هذا البرنامج التعليمي،
سنستخدم الإصدار المُعلم مسبقًا من resnet50d. نظرًا لأن نموذجنا هو مجرد غلاف حوله، فمن السهل نقل هذه الأوزان:

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

الآن دعونا نرى كيفية التأكد من أنه عند قيامنا بـ [`~PreTrainedModel.save_pretrained`] أو [`~PreTrainedModel.push_to_hub`]، يتم حفظ كود النموذج.

## تسجيل نموذج مع كود مخصص للفئات التلقائية

إذا كنت تكتب مكتبة توسع 🤗 Transformers، فقد ترغب في توسيع الفئات التلقائية لتشمل نموذجك الخاص. يختلف هذا عن نشر الكود إلى Hub بمعنى أن المستخدمين سيحتاجون إلى استيراد مكتبتك للحصول على النماذج المخصصة (على عكس تنزيل كود النموذج تلقائيًا من Hub).

ما دام تكوينك يحتوي على معامل  `model_type` مختلفة عن أنواع النماذج الحالية، وأن فئات نماذجك لديك لديها الخصائص الصحيحة `config_class`، فيمكنك ببساطة إضافتها إلى الفئات التلقائية مثل هذا:

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

لاحظ أن الحجة الأولى المستخدمة عند تسجيل تكوينك المخصص لـ [`AutoConfig`] يجب أن تتطابق مع `model_type`
من تكوينك المخصص، والحجة الأولى المستخدمة عند تسجيل نماذجك المخصصة لأي فئة نموذج تلقائي يجب
أن تتطابق مع `config_class` من تلك النماذج.

## إرسال الكود إلى Hub

<Tip warning={true}>

هذا API تجريبي وقد يكون له بعض التغييرات الطفيفة في الإصدارات القادمة.

</Tip>

أولاً، تأكد من تعريف نموذجك بالكامل في ملف `.py`. يمكن أن يعتمد على الاستيراد النسبي لملفات أخرى طالما أن جميع الملفات موجودة في نفس الدليل (لا ندعم الوحدات الفرعية لهذه الميزة حتى الآن). في مثالنا، سنحدد ملف `modeling_resnet.py` وملف `configuration_resnet.py` في مجلد باسم "resnet_model" في دليل العمل الحالي. يحتوي ملف التكوين على كود لـ `ResnetConfig` ويحتوي ملف النمذجة على كود لـ `ResnetModel` و`ResnetModelForImageClassification`.

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

يمكن أن يكون ملف `__init__.py` فارغًا، فهو موجود فقط حتى يتمكن Python من اكتشاف أن `resnet_model` يمكن استخدامه كموديل.

<Tip warning={true}>

إذا كنت تقوم بنسخ ملفات النمذجة من المكتبة، فسوف تحتاج إلى استبدال جميع الواردات النسبية في أعلى الملف
لاستيرادها من حزمة `transformers`.

</Tip>

لاحظ أنه يمكنك إعادة استخدام (أو توسيع) تكوين/نموذج موجود.

لمشاركة نموذجك مع المجتمع، اتبع الخطوات التالية: أولاً، قم باستيراد نموذج ResNet والتكوين من الملفات التي تم إنشاؤها حديثًا:

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

بعد ذلك، يجب عليك إخبار المكتبة بأنك تريد نسخ ملفات الكود الخاصة بهذه الكائنات عند استخدام طريقة `save_pretrained`
وتسجيلها بشكل صحيح باستخدام فئة تلقائية (خاصة للنماذج)، ما عليك سوى تشغيل:

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

لاحظ أنه لا توجد حاجة لتحديد فئة تلقائية للتكوين (هناك فئة تلقائية واحدة فقط لها،
[`AutoConfig`]) ولكن الأمر يختلف بالنسبة للنماذج. قد يكون نموذجك المخصص مناسبًا للعديد من المهام المختلفة، لذلك يجب
تحديد أي من الفئات التلقائية هو الصحيح لنموذجك.

<Tip>

استخدم `register_for_auto_class()` إذا كنت تريد نسخ ملفات الكود. إذا كنت تفضل استخدام الكود على Hub من مستودع آخر،
فلا تحتاج إلى استدعائه. في الحالات التي يوجد فيها أكثر من فئة تلقائية واحدة، يمكنك تعديل ملف `config.json` مباشرة باستخدام
الهيكل التالي:

```json
"auto_map": {
	"AutoConfig": "<your-repo-name>--<config-name>",
	"AutoModel": "<your-repo-name>--<config-name>",
	"AutoModelFor<Task>": "<your-repo-name>--<config-name>",
},
```

</Tip>

بعد ذلك، دعنا نقوم بإنشاء التكوين والنماذج كما فعلنا من قبل:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

الآن لإرسال النموذج إلى Hub، تأكد من تسجيل الدخول. إما تشغيل في المحطة الأوامر الطرفية الخاصة بك:

```bash
hf auth login
```

أو من دفتر ملاحظات:

```py
from huggingface_hub import notebook_login

notebook_login()
```

يمكنك بعد ذلك الضغط على مساحة الاسم الخاصة بك (أو منظمة أنت عضو فيها) مثل هذا:

```py
resnet50d.push_to_hub("custom-resnet50d")
```

بالإضافة إلى أوزان النمذجة والتكوين بتنسيق json، فقد قام هذا أيضًا بنسخ ملفات النمذجة والتكوين `.py` في مجلد `custom-resnet50d` وتحميل النتيجة إلى Hub. يمكنك التحقق من النتيجة في هذا [مستودع النموذج](https://huggingface.co/sgugger/custom-resnet50d).

راجع [البرنامج التعليمي للمشاركة](model_sharing) لمزيد من المعلومات حول طريقة الدفع إلى المحور.

### استخدام نموذج مع كود مخصص

يمكنك استخدام أي تكوين أو نموذج أو مقسم لغوي مع ملفات برمجة مخصصة في مستودعه باستخدام الفئات التلقائية و دالة `from_pretrained`.تُفحص جميع الملفات والرموز المرفوع إلى Hub بحثًا عن البرامج الضارة (راجع وثائق [أمان Hub](https://huggingface.co/docs/hub/security#malware-scanning) لمزيد من المعلومات)، ولكن يجب عليك مراجعة كود النموذج والمؤلف لتجنب تنفيذ التعليمات البرمجية الضارة على جهازك. لتفعيل نموذج يحتوي على شفرة برمجية مخصصة،  عيّن `trust_remote_code=True`:

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

يُنصح بشدة بتحديد رقم إصدار (commit hash) كـ `revision`  للتأكد من عدم تعديل مؤلف النموذج للشفرة لاحقًابإضافة أسطر ضارة  (إلا إذا كنت تثق تمامًا بمؤلفي النموذج):

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d"، trust_remote_code=True، revision=commit_hash
)
```

لاحظ وجود زرّ لنسخ رقم إصدار بسهولة عند تصفح سجل التزامات مستودع النموذج على منصة Hugging Face.
