# Trainer

تُتيح وحدة [`Trainer`] حلقة تدريب وتقييم متكاملة لنماذج PyTorch المطبقة في مكتبة Transformers. تحتاج فقط إلى تمرير المكونات الضرورية للتدريب (النموذج، والمجزىء النصى، ومجموعة البيانات، دالة التقييم، معلمات التدريب الفائقة، إلخ)، وستتولى فئة [`Trainer`] الباقي. هذا يُسهّل بدء التدريب بشكل أسرع دون كتابة حلقة التدريب الخاصة بك يدويًا. ولكن في الوقت نفسه، فإن [`Trainer`] قابل للتخصيص بدرجة كبيرة ويوفر العديد من خيارات التدريب حتى تتمكن من تخصيصه وفقًا لاحتياجات التدريب الخاصة بك بدقة.

<Tip>

بالإضافة إلى فئة [`Trainer`], توفر مكتبة Transformers أيضًا فئة [`Seq2SeqTrainer`] للمهام التسلسلية مثل الترجمة أو التلخيص. هناك أيضًا فئة [`~trl.SFTTrainer`] من مكتبة [TRL](https://hf.co/docs/trl) التي تغلّف فئة [`Trainer`] وهي مُحُسَّنة لتدريب نماذج اللغة مثل Llama-2 وMistral باستخدام تقنيات التوليد اللغوي. كما يدعم [`~trl.SFTTrainer`] ميزات مثل حزم التسلسلات، وLoRA، والقياس الكمي، وDeepSpeed مما يُمكّن من التدريب بكفاءة على نماذج ضخمة الحجم.

<br>

لا تتردد في الاطلاع على [مرجع API](./main_classes/trainer) لهذه الفئات الأخرى من النوع [`Trainer`] لمعرفة المزيد حول متى يتم استخدام كل منها. بشكل عام، [`Trainer`] هو الخيار الأكثر تنوعًا ومناسبًا لمجموعة واسعة من المهام. تم تصميم [`Seq2SeqTrainer`] للمهام التسلسلية ، و [`~trl.SFTTrainer`] مُصمم لتدريب نماذج اللغة الكبيرة.

</Tip>

قبل البدء، تأكد من تثبيت مكتبة [Accelerate](https://hf.co/docs/accelerate) - وهي مكتبة تُمكّن تشغيل تدريب PyTorch في بيئات مُوزعة.

```bash
pip install accelerate

# upgrade
pip install accelerate --upgrade
```

يوفر هذا الدليل نظرة عامة على فئة [`Trainer`].

## الاستخدام الأساسي

يتضمن [`Trainer`] جميع التعليمات البرمجية التي ستجدها في حلقة التدريب الأساسية:

1. قم بتنفيذ خطوة تدريب لحساب الخسارة
2. احسب المشتقات باستخدام طريقة [`~accelerate.Accelerator.backward`]
3. تحديث الأوزان بناءً على المشتقات
4. كرر هذه العملية حتى تصل إلى عدد محدد مسبقًا من الدورات (epochs).

تُجرد فئة [`Trainer`] كل هذه التعليمات البرمجية حتى لا تضطر إلى القلق بشأن كتابة حلقة تدريب يدويًا في كل مرة أما إذا كنت بدأت للتو في PyTorch والتدريب. كل ما عليك فعله هو توفير المكونات الأساسية اللازمة للتدريب، مثل النموذج ومجموعة بيانات، وتتعامل فئة [`Trainer`] مع كل شيء آخر.

إذا كنت تُريد تحديد أي خيارات تدريب أو معلمات فائقة، فيمكنك العثور عليها في فئة [`TrainingArguments`]. على سبيل المثال، دعنا نحدد أين يتم حفظ النموذج في `output_dir` ورفع النموذج إلى Hub بعد التدريب باستخدام `push_to_hub=True`.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model"،
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch"،
    save_strategy="epoch"،
    load_best_model_at_end=True,
    push_to_hub=True,
)
```
مرر `training_args` إلى [`Trainer`] جنبًا إلى جنب مع النموذج، ومجموعة بيانات، وشئ لمعالجة مجموعة البيانات مسبقًا (حسب نوع البيانات، فقد يكون محللًا رمزيًا أو مستخرج ميزات أو معالج صور)، وجامع بيانات، ودالة لحساب المقاييس التي تُريد تتبعها أثناء التدريب.

أخيرًا، استدعِ [`~Trainer.train`] لبدء التدريب!

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]،
    eval_dataset=dataset["test"]،
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### نقاط الحفظ

تحفظ فئة [`Trainer`] نقاط الحفظ النموذج في الدليل المحدد في معامل `output_dir` من [`TrainingArguments`]. ستجد نقاط الحفظ في مجلد فرعي يسمى `checkpoint-000` حيث تتوافق الأرقام في النهاية مع خطوة التدريب. إن حفظ نقاط الحفظ مفيد لاستئناف التدريب لاحقًا.

```py
# استأنف من أحدث نقطة حفظ
trainer.train(resume_from_checkpoint=True)

# استأنف من نقطة حفظ محددة محفوظة في دليل الإخراج
trainer.train(resume_from_checkpoint="your-model/checkpoint-1000")
```

يمكنك حفظ نقاط الحفظ الخاصة بك (لا يتم حفظ حالة المُجزىء اللغوى تقائيًا)  إلى Hub عن طريق تعيين `push_to_hub=True` في [`TrainingArguments`] لرفعها. الخيارات الأخرى لاتخاذ القرار بشأن كيفية حفظ هذة النقاط الخاصة بك هي الإعداد في معامل [`hub_strategy`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_strategy):

* `hub_strategy="checkpoint"` يدفع أحدث نقطة حفظ إلى مجلد فرعي يسمى "last-checkpoint" يمكنك استئناف التدريب منه
* `hub_strategy="all_checkpoints"` يدفع جميع نقاط الحفظ إلى الدليل المحدد في `output_dir` (سترى نقطة حفظ واحدة لكل مجلد في مستودع النموذج الخاص بك)

عند استئناف التدريب من نقطة حفظ، تُحاول [`Trainer`] الحفاظ على حالات RNG Python وNumPy وPyTorch كما كانت عندما تم حفظ نقطة الحفظ. ولكن لأن PyTorch لديها العديد من الإعدادات الافتراضية غير الحتمية مُتنوعة، فإن حالات RNG ليست مضمونة لتكون هي نفسها. إذا كنت تريد تمكين الحتمية الكاملة، فراجع دليل [التحكم في مصادر العشوائية](https://pytorch.org/docs/stable/notes/randomness#controlling-sources-of-randomness) لمعرفة ما يُمكنك تمكينه لجعل تدريبك حتميًا تمامًا. ضع في اعتبارك أنه من خلال جعل إعدادات معينة حتمية، فقد يكون التدريب أبطأ.

## تخصيص المدرب

في حين أن فئة [`Trainer`] مُصممة لتكون سهلة الوصول وسهلة الاستخدام، فإنها توفر أيضًا الكثير من قابلية التخصيص للمستخدمين المغامرين.  يُمكن إنشاء فئات فرعية من العديد من أساليب [`Trainer`] وتجاوزها لدعم الوظائف التي تُريدها، دون الحاجة إلى إعادة كتابة حلقة التدريب بأكملها من البداية لاستيعابها. تتضمن هذه الأساليب:

* [`~Trainer.get_train_dataloader`] ينشئ DataLoader للتدريب
* [`~Trainer.get_eval_dataloader`] ينشئ DataLoader للتقييم
* [`~Trainer.get_test_dataloader`] ينشئ DataLoader للاختبار
* [`~Trainer.log`] يسجل معلومات حول مختلف الكائنات التي تراقب التدريب
* [`~Trainer.create_optimizer_and_scheduler`] ينشئ محسنًا ومخططًا لمُعدل التعلم إذا لم يتم تمريرهما في `__init__`؛ يمكن أيضًا تخصيص هذه الوظائف بشكل منفصل باستخدام [`~Trainer.create_optimizer`] و [`~Trainer.create_scheduler`] على التوالي
* [`~Trainer.compute_loss`] يحسب دالة الخسارة على دفعة من مُدخلات التدريب
* [`~Trainer.training_step`] يُنفذ خطوة التدريب
* [`~Trainer.prediction_step`] يُنفذ خطوة التنبؤ والاختبار
* [`~Trainer.evaluate`] يُقيّم النموذج ويعيد مقاييس التقييم
* [`~Trainer.predict`] يُجري التنبؤات (مع المقاييس إذا كانت العلامات متاحة) على مجموعة الاختبار

على سبيل المثال، إذا كنت تريد تخصيص طريقة [`~Trainer.compute_loss`] لاستخدام دالة خسارة ذات ترجيح بدلاً من ذلك.


```py
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

### دوال الاستدعاء Callbacks

خيار آخر لتخصيص [`Trainer`] هو استخدام [دوال الاستدعاء](callbacks). لا *تغير* دوال الاستدعاء أي شيء في حلقة التدريب. إنهم تفحص حالة حلقة التدريب ثم تُنفذ بعض الإجراءات (مثل الإيقاف المبكر أو تسجيل النتائج، إلخ) اعتمادًا على الحالة. وبعبارة أخرى، لا يمكن استخدام دالة الاستدعاء لتنفيذ شيء مثل دالة خسارة مخصصة، ويجب عليك تجاوز دالة [`~Trainer.compute_loss`] لذلك.

على سبيل المثال، إذا كنت تريد إضافة دالة استدعاء إيقاف مبكر إلى حلقة التدريب بعد 10 خطوات.

```py
from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}
```

ثم مرره إلى معامل `callback` في [`Trainer`].

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]،
    eval_dataset=dataset["test"]،
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callback=[EarlyStoppingCallback()],
)
```

## تسجيل الأحداث (Logging)

<Tip>

راجع مرجع [API](./main_classes/logging) للتسجيل للحصول على مزيد من المعلومات حول مستويات التسجيل المختلفة للأحداث.

</Tip>

يتم تعيين [`Trainer`] إلى `logging.INFO` افتراضيًا والذي يُبلغ عن الأخطاء والتحذيرات ومعلومات أساسية أخرى. يتم تعيين نسخة [`Trainer`] - في البيئات الموزعة - إلى `logging.WARNING` والتي يُبلغ فقط عن الأخطاء والتحذيرات. يمكنك تغيير مستوى تسجيل الأحداث باستخدام معاملي [`log_level`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level) و [`log_level_replica`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level_replica) في [`TrainingArguments`].

لتهيئة إعداد مُستوى تسجيل  اﻷحداث لكل عقدة، استخدم معامل [`log_on_each_node`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.log_on_each_node) لتحديد ما إذا كان سيتم استخدام مُستوى السجل على كل عقدة أو فقط على العقدة الرئيسية.

<Tip>

يحدد [`Trainer`] مُستوى التسجيل بشكل مُنفصل لكل عقدة في طريقة [`Trainer.__init__`]، لذا فقد ترغب في التفكير في تعيين هذا الإعداد في وقت سابق إذا كنت تستخدم وظائف Transformers الأخرى قبل إنشاء كائن [`Trainer`].

</Tip>

على سبيل المثال، لتعيين التعليمات البرمجية والوحدات النمطية الرئيسية الخاصة بك لاستخدام نفس مُستوى التسجيل وفقًا لكل عقدة:

```py
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"،
    datefmt="%m/%d/%Y %H:%M:%S"،
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

استخدم تركيبات مختلفة من `log_level` و `log_level_replica` لتهيئة ما يتم تسجيله على كل من العقد.


<hfoptions id="logging">
<hfoption id="single node">

```bash
my_app.py ... --log_level warning --log_level_replica error
```

</hfoption>
<hfoption id="multi-node">

أضف معلمة `log_on_each_node 0` لبيئات متعددة العقد.

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0

# set to only report errors
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

</hfoption>
</hfoptions>

## NEFTune

[NEFTune](https://hf.co/papers/2310.05914) هي تقنية يمكن أن تحسن الأداء عن طريق إضافة ضوضاء إلى مُتجهات التعلم أثناء التدريب. لتمكينه في [`Trainer`], قم بتعيين معامل `neftune_noise_alpha` في [`TrainingArguments`] للتحكم في مقدار الضوضاء المُضافة.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(..., neftune_noise_alpha=0.1)
trainer = Trainer(..., args=training_args)
```

يتم تعطيل NEFTune بعد التدريب لاستعادة طبقة التعلم الأصلية لتجنب أي سلوك غير متوقع.

## نواة Liger
[Liger-Kernel](https://github.com/linkedin/Liger-Kernel) Kernel هي مجموعة من نوى Triton التي طورتها Linkedin مُصممة خصيصًا لتدريب نماذج اللغة الكبيرة (LLM). لقد قمنا بتنفيذ RMSNorm و RoPE و SwiGLU و CrossEntropy و FusedLinearCrossEntropy مُتوافقة مع Hugging Face، والمزيد قادم. يُمكنها زيادة إنتاجية التدريب متعدد وحدات معالجة الرسومات (GPU) بنسبة 20٪ وتقليل استخدام الذاكرة بنسبة 60٪. تعمل النواة بشكل تلقائي مع flash attention و PyTorch FSDP و Microsoft DeepSpeed.

احصل على زيادة في الإنتاجية بنسبة 20٪ وتقليل استخدام الذاكرة بنسبة 60٪ على تدريب نماذج LLaMA 3-8B. حقق أطوال سياق أكبر وأحجام دفعات أكبر. كما أنها مُفيدة إذا كنت تُريد زيادة حجم نموذجك إلى تدريب بنماذج متعددة الرؤوس أو أحجام مُفردات ضخمة. أطلق العنان للتدريب بنماذج متعددة الرؤوس (medusa) والمزيد. راجع التفاصيل والأمثلة في [Liger](https://github.com/linkedin/Liger-Kernel/tree/main/examples)
تأكد أولاً من تثبيت مستودع Liger الرسمي:
```bash
pip install liger-kernel
```
يجب عليك تمرير `use_liger_kernel=True` لتطبيق نواة `liger` على نموذجك، على سبيل المثال:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    use_liger_kernel=True
)
```

تدعم النواة معماريات نماذج Llama و Gemma و Mistral و Mixtral. يُمكن العثور على أحدث قائمة بالنمائج المدعومة [هنا](https://github.com/linkedin/Liger-Kernel). عندما يتم تعيين `use_liger_kernel` إلى `True`، سيتم تصحيح الطبقات المُقابلة في النموذج الأصلي باستخدام تطبيق Liger الفعال، لذلك لا تحتاج إلى فعل أي شيء إضافي بخلاف تعيين قيمة المعامل.

## المُحسِّنات
يمكنك اختيار مُحسِّن مدمج للتدريب باستخدام:
```python
from transformers import TrainingArguments
training_args = TrainingArguments(..., optim="adamw_torch")
```
اطلع على [`OptimizerNames`](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) للاطلاع على القائمة الكاملة للخيارات. نُدرج أمثلة مُتقدمة في الأقسام أدناه.

يمكنك أيضًا استخدام مُحسِّن PyTorch عشوائي عبر:
```python
import torch

optimizer_cls = torch.optim.AdamW
optimizer_kwargs = {
    "lr": 4e-3,
    "betas": (0.9, 0.999),
    "weight_decay": 0.05,
}

from transformers import Trainer
trainer = Trainer(..., optimizer_cls_and_kwargs=(optimizer_cls, optimizer_kwargs))
```




### GaLore

إسقاط التدرج ذو الرتبة المنخفضة (GaLore) هو إستراتيجية تدريب ذات رتبة منخفضة فعّالة من حيث الذاكرة، تسمح بتعلم المعلمات الكاملة ولكنها أكثر كفاءة من حيث الذاكرة من أساليب التكيّف الشائعة ذات الرتبة المنخفضة، مثل LoRA.

أولاً، تأكد من تثبيت المستودع الرسمي لـ GaLore:

```bash
pip install galore-torch
```

ثم أضف ببساطة أحد `["galore_adamw"، "galore_adafactor"، "galore_adamw_8bit"]` في `optim` جنبًا إلى جنب مع `optim_target_modules`، والتي يمكن أن تكون قائمة من السلاسل أو التعبيرات النمطية regex أو المسار الكامل المطابق لأسماء الوحدات المستهدفة التي تريد تكييفها. فيما يلي مثال على النص البرمجي كامل(تأكد من `pip install trl datasets`):

```python
import torch
import datasets
import trl

from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-galore"،
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw"،
    optim_target_modules=[r".*.attn.*"، r".*.mlp.*"]
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)

trainer.train()
```

لتمرير معامﻻت إضافية يدعمها  GaLore، يجب عليك تمرير `optim_args` بشكل صحيح، على سبيل المثال:

```python
import torch
import datasets
import trl

from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-galore",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="rank=64, update_proj_gap=100, scale=0.10",
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)

trainer.train()
```
يمكنك قراءة المزيد حول الطريقة في [المستودع الأصلي](https://github.com/jiaweizzhao/GaLore) أو [الورقة البحثية](https://arxiv.org/abs/2403.03507).

حاليًا، يمكنك فقط تدريب الطبقات الخطية التي تعتبر طبقات GaLore وستستخدم التحلل  ذو الرتبة المنخفضة للتدريب بينما سيتم تحسين الطبقات المتبقية بالطريقة التقليدية.

لاحظ أنه سيستغرق الأمر بعض الوقت قبل بدء التدريب (~3 دقائق لنموذج 2B على NVIDIA A100)، ولكن يجب أن يسير التدريب بسلاسة بعد ذلك.

يمكنك أيضًا إجراء تحسين طبقة تلو الأخرى عن طريق إضافة `layerwise` إلى اسم المُحسِّن كما هو موضح أدناه:

```python
import torch
import datasets
import trl

from transformers import TrainingArguments، AutoConfig، AutoTokenizer، AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb'، split='train')

args = TrainingArguments(
    output_dir="./test-galore"،
    max_steps=100،
    per_device_train_batch_size=2،
    optim="galore_adamw_layerwise"،
    optim_target_modules=[r".*.attn.*"، r".*.mlp.*"]
)

model_id = "google/gemma-2b"

config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config).to(0)

trainer = trl.SFTTrainer(
    model=model،
    args=args،
    train_dataset=train_dataset،
    dataset_text_field='text'،
    max_seq_length=512،
)

trainer.train()
```

لاحظ أن تحسين الطبقة تجريبي إلى حد ما ولا يدعم DDP (Distributed Data Parallel)، وبالتالي يمكنك تشغيل التعليمات البرمجية  للتدريب على وحدة معالجة الرسومات (GPU) واحدة فقط. يرجى الاطلاع على [هذا القسم المناسب](https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#train-7b-model-with-a-single-gpu-with-24gb-memory) لمزيد من التفاصيل. قد لا تدعم الميزات الأخرى مثل تقليم التدرجات أو DeepSpeed، إلخ. من الصندوق. يرجى [تقديم تقرير عن المشكلة على GitHub](https://github.com/huggingface/transformers/issues) إذا واجهتك مثل هذه المشكلة.

### محسنات LOMO

تم تقديم مُحسِّنات LOMO في [التدريب على المعلمات الكاملة لنماذج اللغة الكبيرة باستخدام موارد محدودة](https://hf.co/papers/2306.09782) و [AdaLomo: تحسين ذاكرة منخفضة بمعدل تعلم متكيف](https://hf.co/papers/2310.10195).
يتكون كلاهما من طريقة فعالة لضبط المعلمات الكاملة. تدمج محسنات LOMO حساب الاشتقاق وتحديث المعلمات في خطوة واحدة لتقليل استخدام الذاكرة. محسنات LOMO المدعومة هي `"lomo"` و `"adalomo"`. أولاً قم بتثبيت LOMO من pypi `pip install lomo-optim` أو قم بتثبيته من المصدر باستخدام `pip install git+https://github.com/OpenLMLab/LOMO.git`.

<Tip>

وفقًا للمؤلفين، يوصى باستخدام `AdaLomo` بدون `grad_norm` للحصول على أداء أفضل وسرعة أعلى.

</Tip>

فيما يلي نص برمجي بسيط يوضح كيفية ضبط نموذج [google/gemma-2b](https://huggingface.co/google/gemma-2b) على مجموعة بيانات IMDB في الدقة الكاملة:

```python
import torch
import datasets
from transformers import TrainingArguments، AutoTokenizer، AutoModelForCausalLM
import trl

train_dataset = datasets.load_dataset('imdb'، split='train')

args = TrainingArguments(
    output_dir="./test-lomo"،
    max_steps=100،
    per_device_train_batch_size=4،
    optim="adalomo"،
    gradient_checkpointing=True،
    logging_strategy="steps"،
    logging_steps=1،
    learning_rate=2e-6،
    save_strategy="no"،
    run_name="lomo-imdb"،
)

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id، low_cpu_mem_usage=True).to(0)

trainer = trl.SFTTrainer(
    model=model،
    args=args،
    train_dataset=train_dataset،
    dataset_text_field='text'،
    max_seq_length=1024،
)

trainer.train()
```

### مُحسِّن GrokAdamW
تم تصميم مُحسِّن GrokAdamW لتعزيز أداء التدريب واستقراره، خاصةً للنماذج التي تستفيد من دوال إشارة `grokking`. لاستخدام `GrokAdamW`، قم أولاً بتثبيت حزمة المُحسِّن باستخدام `pip install grokadamw`.
<Tip>
يُعد GrokAdamW مفيدًا بشكل خاص للنماذج التي تتطلب تقنيات تحسين مُتقدمة لتحقيق أداء واستقرار أفضل.
</Tip>

فيما يلي نص برمجى بسيط لشرح كيفية ضبط [google/gemma-2b](https://huggingface.co/google/gemma-2b) بدقة على مجموعة بيانات IMDB باستخدام مُحسِّن GrokAdamW:
```python
import torch
import datasets
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer

# تحميل مجموعة البيانات IMDB
train_dataset = datasets.load_dataset('imdb', split='train')

# تعريف معامﻻت التدريب
args = TrainingArguments(
    output_dir="./test-grokadamw",
    max_steps=1000,
    per_device_train_batch_size=4,
    optim="grokadamw",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="grokadamw-imdb",
)

# تحميل النموذج والمجزىء اللغوي
model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(0)

# تهيئة المدرب
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

# تدريب النموذج
trainer.train()
```
يوضح هذا النص البرمجى كيفية ضبط نموذج google/gemma-2b بدقة على مجموعة بيانات IMDB باستخدام مُحسِّن GrokAdamW. يتم تكوين TrainingArguments لاستخدام GrokAdamW، ويتم تمرير مجموعة البيانات إلى Trainer للتدريب.

### مُحسِّن بدون جدوله (Schedule Free Optimizer)
تم تقديم مُحسِّنات بدون جدوله في [The Road Less Scheduled](https://hf.co/papers/2405.15682).
يستبدل التعلم بدون جدوله زخم المُحسِّن الأساسي بمزيج من المتوسط ​​والتداخل، لإزالة الحاجة تمامًا إلى تخفيف مُعدل التعلم باستخدام جدوله تقليديه.
المُحسِّنات المدعومة لـ SFO هي "schedule_free_adamw" و "schedule_free_sgd". قم أولاً بتثبيت `schedulefree` من pypi باستخدام الأمر  `pip install schedulefree`.

فيما يلي نص برمجى بسيط لشرح كيفية ضبط [google/gemma-2b](https://huggingface.co/google/gemma-2b) بدقة على مجموعة بيانات IMDB بدقة كاملة:
```python
import torch
import datasets
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import trl

train_dataset = datasets.load_dataset('imdb', split='train')

args = TrainingArguments(
    output_dir="./test-schedulefree",
    max_steps=1000,
    per_device_train_batch_size=4,
    optim="schedule_free_adamw",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="sfo-imdb",
)

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(0)

trainer = trl.SFTTrainer(
    model=model, 
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=1024,
)

trainer.train()
```
## تسريع ومدرب

يتم تشغيل فئة [`Trainer`] بواسطة [تسريع](https://hf.co/docs/accelerate)، وهي مكتبة لتدريب نماذج PyTorch بسهولة في بيئات موزعة مع دعم عمليات التكامل مثل [FullyShardedDataParallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) و [DeepSpeed](https://www.deepspeed.ai/).

<Tip>

تعرف على المزيد حول استراتيجيات تجزئة FSDP، وتفريغ وحدة المعالجة المركزية (CPU)، والمزيد مع [`Trainer`] في [دليل Fully Sharded Data Parallel](fsdp).

</Tip>

لاستخدام Accelerate مع [`Trainer`]]، قم بتشغيل الأمر [`accelerate.config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) لإعداد التدريب لبيئة التدريب الخاصة بك. نشئ هذا الأمر ملف `config_file.yaml` الذي سيتم استخدامه عند تشغيل نص للتدريب البرمجى. على سبيل المثال، بعض تكوينات المثال التي يمكنك إعدادها هي:

<hfoptions id="config">
<hfoption id="DistributedDataParallel">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
distributed_type: MULTI_GPU                                                                                                    
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0 #change rank as per the node
main_process_ip: 192.168.20.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="FSDP">

```yml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed">

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/user/configs/ds_zero3_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed with Accelerate plugin">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
deepspeed_config:                                                                                                              
  gradient_accumulation_steps: 1
  gradient_clipping: 0.7
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="Tensor Parallelism with PyTorch 2">

```yml
compute_environment: LOCAL_MACHINE
tp_config:
  tp_size: 4
distributed_type: TP
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

```

</hfoption>
</hfoptions>
يُعد أمر  [`accelerate_launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) هو الطريقة المُوصى بها لتشغيل نص البرمجى للتدريب على نظام موزع باستخدام Accelerate و [`Trainer`] مع المعلمات المحددة في `config_file.yaml`. يتم حفظ هذا الملف في مجلد ذاكرة التخزين المؤقت لـ Accelerate ويتم تحميله تلقائيًا عند تشغيل `accelerate_launch`.

على سبيل المثال، لتشغيل النص البرنامجي للتدريب [run_glue.py](https://github.com/huggingface/transformers/blob/f4db565b695582891e43a5e042e5d318e28f20b8/examples/pytorch/text-classification/run_glue.py#L4) مع تكوين FSDP:

```bash
accelerate launch \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

يمكنك أيضًا تحديد المعلمات من ملف `config_file.yaml` مباشرة في سطر الأوامر:

```bash
accelerate launch --num_processes=2 \
    --use_fsdp \
    --mixed_precision=bf16 \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
    --fsdp_transformer_layer_cls_to_wrap="BertLayer" \
    --fsdp_sharding_strategy=1 \
    --fsdp_state_dict_type=FULL_STATE_DICT \
    ./examples/pytorch/text-classification/run_glue.py
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

اطلع على برنامج تعليمي [Launching your Accelerate scripts](https://huggingface.co/docs/accelerate/basic_tutorials/launch) لمعرفة المزيد حول `accelerate_launch` والتكوينات المخصصة.
