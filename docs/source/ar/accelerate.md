لمساعدة المستخدمين على تدريب نماذج 🤗 Transformers بسهولة على أي نوع من الإعدادات الموزعة، سواء كان ذلك باستخدام وحدات GPU متعددة على جهاز واحد أو وحدات GPU متعددة عبر أجهزة متعددة، أنشأنا في Hugging Face مكتبة [🤗 Accelerate](https://huggingface.co/docs/accelerate). في هذا البرنامج التعليمي، تعلم كيفية تخصيص حلقة التدريب الأصلية في PyTorch لتمكين التدريب في بيئة موزعة.

## الإعداد

ابدأ بتثبيت 🤗 Accelerate:

```bash
pip install accelerate
```

ثم قم باستيراد وإنشاء كائن [`~accelerate.Accelerator`]. سيقوم [`~accelerate.Accelerator`] تلقائيًا بالكشف عن نوع الإعداد الموزع الخاص بك وتهيئة جميع المكونات اللازمة للتدريب. لا تحتاج إلى وضع نموذجك على جهاز بشكل صريح.

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## الاستعداد للتسريع

الخطوة التالية هي تمرير جميع كائنات التدريب ذات الصلة إلى طريقة [`~accelerate.Accelerator.prepare`]. ويشمل ذلك DataLoaders للتدريب والتقييم، ونموذجًا ومُحَسِّنًا:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## الخلفي

الإضافة الأخيرة هي استبدال `loss.backward()` النموذجي في حلقة التدريب الخاصة بك بأسلوب [`~accelerate.Accelerator.backward`] في 🤗 Accelerate:

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

كما يمكنك أن ترى في الكود التالي، فأنت بحاجة فقط إلى إضافة أربعة أسطر من الكود إلى حلقة التدريب الخاصة بك لتمكين التدريب الموزع!

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)
optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## تدريب

بمجرد إضافة أسطر الكود ذات الصلة، قم بتشغيل التدريب الخاص بك في أحد النصوص أو الدفاتر مثل Colaboratory.

### التدريب باستخدام نص برمجي

إذا كنت تشغل التدريب الخاص بك من نص برمجي، فقم بتشغيل الأمر التالي لإنشاء وحفظ ملف تكوين:

```bash
accelerate config
```

ثم قم بتشغيل التدريب الخاص بك باستخدام:

```bash
accelerate launch train.py
```

### التدريب باستخدام دفتر ملاحظات

يمكن أيضًا تشغيل 🤗 Accelerate في دفتر ملاحظات إذا كنت تخطط لاستخدام وحدات معالجة الرسوميات (TPUs) في Colaboratory. قم بتغليف كل الكود المسؤول عن التدريب في دالة، ومررها إلى [`~accelerate.notebook_launcher`]:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

للحصول على مزيد من المعلومات حول 🤗 Accelerate وميزاته الغنية، يرجى الرجوع إلى [الوثائق](https://huggingface.co/docs/accelerate).