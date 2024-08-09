# استدعاءات الرجوع 

استدعاءات الرجوع هي كائنات يمكنها تخصيص سلوك حلقة التدريب في PyTorch [`Trainer`] (لم يتم تنفيذ هذه الميزة بعد في TensorFlow) التي يمكنها فحص حالة حلقة التدريب (لإعداد التقارير حول التقدم، وتسجيل الدخول إلى TensorBoard أو منصات ML الأخرى...) واتخاذ القرارات (مثل التوقف المبكر).

استدعاءات الرجوع هي قطع من التعليمات البرمجية "للقراءة فقط"، وبصرف النظر عن كائن [`TrainerControl`] الذي تقوم بإرجاعه، لا يمكنها تغيير أي شيء في حلقة التدريب. بالنسبة للتخصيصات التي تتطلب إجراء تغييرات في حلقة التدريب، يجب عليك إنشاء فئة فرعية من [`Trainer`] والكتابة فوق الطرق التي تحتاجها (راجع [trainer](trainer) للحصول على أمثلة).

بشكل افتراضي، يتم تعيين `TrainingArguments.report_to` إلى `"all"`، لذا سيستخدم [`Trainer`] الاستدعاءات التالية.

- [`DefaultFlowCallback`] الذي يتعامل مع السلوك الافتراضي للتسجيل وحفظ وتقييم.
- [`PrinterCallback`] أو [`ProgressCallback`] لعرض التقدم وطباعة السجلات (يتم استخدام الأول إذا قمت بتنشيط tqdm من خلال [`TrainingArguments`]، وإلا فهو الثاني).
- [`~integrations.TensorBoardCallback`] إذا كان TensorBoard متاحًا (إما من خلال PyTorch >= 1.4 أو tensorboardX).
- [`~integrations.WandbCallback`] إذا تم تثبيت [wandb](https://www.wandb.com/).
- [`~integrations.CometCallback`] إذا تم تثبيت [comet_ml](https://www.comet.ml/site/).
- [`~integrations.MLflowCallback`] إذا تم تثبيت [mlflow](https://www.mlflow.org/).
- [`~integrations.NeptuneCallback`] إذا تم تثبيت [neptune](https://neptune.ai/).
- [`~integrations.AzureMLCallback`] إذا تم تثبيت [azureml-sdk](https://pypi.org/project/azureml-sdk/).
- [`~integrations.CodeCarbonCallback`] إذا تم تثبيت [codecarbon](https://pypi.org/project/codecarbon/).
- [`~integrations.ClearMLCallback`] إذا تم تثبيت [clearml](https://github.com/allegroai/clearml).
- [`~integrations.DagsHubCallback`] إذا تم تثبيت [dagshub](https://dagshub.com/).
- [`~integrations.FlyteCallback`] إذا تم تثبيت [flyte](https://flyte.org/).
- [`~integrations.DVCLiveCallback`] إذا تم تثبيت [dvclive](https://dvc.org/doc/dvclive).

إذا تم تثبيت حزمة ولكنك لا تريد استخدام التكامل المصاحب، فيمكنك تغيير `TrainingArguments.report_to` إلى قائمة بالتكاملات التي تريد استخدامها فقط (على سبيل المثال `["azure_ml"، "wandb"]`).

الفئة الرئيسية التي تنفذ استدعاءات الرجوع هي [`TrainerCallback`]. فهو يحصل على [`TrainingArguments`] المستخدم لإنشاء مثيل لـ [`Trainer`]، ويمكنه الوصول إلى حالة المدرب الداخلية عبر [`TrainerState`]، ويمكنه اتخاذ بعض الإجراءات على حلقة التدريب عبر [`TrainerControl`].

## استدعاءات الرجوع المتاحة

فيما يلي قائمة باستدعاءات الرجوع [`TrainerCallback`] المتوفرة في المكتبة:

[[autodoc]] integrations.CometCallback
- setup

[[autodoc]] DefaultFlowCallback

[[autodoc]] PrinterCallback

[[autodoc]] ProgressCallback

[[autodoc]] EarlyStoppingCallback

[[autodoc]] integrations.TensorBoardCallback

[[autodoc]] integrations.WandbCallback
- setup

[[autodoc]] integrations.MLflowCallback
- setup

[[autodoc]] integrations.AzureMLCallback

[[autodoc]] integrations.CodeCarbonCallback

[[autodoc]] integrations.NeptuneCallback

[[autodoc]] integrations.ClearMLCallback

[[autodoc]] integrations.DagsHubCallback

[[autodoc]] integrations.FlyteCallback

[[autodoc]] integrations.DVCLiveCallback
- setup

## TrainerCallback

[[autodoc]] TrainerCallback

فيما يلي مثال على كيفية تسجيل استدعاء رجوع مخصص مع PyTorch [`Trainer`]:

```python
class MyCallback(TrainerCallback):
    "استدعاء رجوع يقوم بطباعة رسالة في بداية التدريب"

    def on_train_begin(self, args، state، control، **kwargs):
        print("بدء التدريب")

trainer = Trainer(
    model،
    args،
    train_dataset=train_dataset،
    eval_dataset=eval_dataset،
    callbacks=[MyCallback]،  # يمكننا إما تمرير فئة استدعاء الرجوع بهذه الطريقة أو مثيل منها (MyCallback())
    )
```

هناك طريقة أخرى لتسجيل استدعاء الرجوع وهي استدعاء `trainer.add_callback()` كما يلي:

```python
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# أو، يمكننا تمرير مثيل من فئة استدعاء الرجوع
trainer.add_callback(MyCallback())
```

## TrainerState

[[autodoc]] TrainerState

## TrainerControl

[[autodoc]] TrainerControl