<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Verteiltes Training mit 🤗 Accelerate

Da die Modelle immer größer werden, hat sich die Parallelität als Strategie zum Trainieren größerer Modelle auf begrenzter Hardware und zur Beschleunigung der Trainingsgeschwindigkeit um mehrere Größenordnungen erwiesen. Bei Hugging Face haben wir die Bibliothek [🤗 Accelerate](https://huggingface.co/docs/accelerate) entwickelt, um Nutzern zu helfen, ein 🤗 Transformers-Modell auf jeder Art von verteiltem Setup zu trainieren, egal ob es sich um mehrere GPUs auf einer Maschine oder mehrere GPUs auf mehreren Maschinen handelt. In diesem Tutorial lernen Sie, wie Sie Ihre native PyTorch-Trainingsschleife anpassen, um das Training in einer verteilten Umgebung zu ermöglichen.

## Einrichtung

Beginnen Sie mit der Installation von 🤗 Accelerate:

```bash
pip install accelerate
```

Dann importieren und erstellen Sie ein [`~accelerate.Accelerator`]-Objekt. Der [`~accelerate.Accelerator`] wird automatisch Ihre Art der verteilten Einrichtung erkennen und alle notwendigen Komponenten für das Training initialisieren. Sie müssen Ihr Modell nicht explizit auf einem Gerät platzieren.

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## Vorbereiten auf die Beschleunigung

Der nächste Schritt ist die Übergabe aller relevanten Trainingsobjekte an die Methode [`~accelerate.Accelerator.prepare`]. Dazu gehören Ihre Trainings- und Evaluierungs-DataLoader, ein Modell und ein Optimierer:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## Rückwärts

Die letzte Ergänzung besteht darin, das typische `loss.backward()` in der Trainingsschleife durch die 🤗 Accelerate-Methode [`~accelerate.Accelerator.backward`] zu ersetzen:

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

Wie Sie im folgenden Code sehen können, müssen Sie nur vier zusätzliche Codezeilen zu Ihrer Trainingsschleife hinzufügen, um verteiltes Training zu ermöglichen!

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

## Trainieren

Sobald Sie die entsprechenden Codezeilen hinzugefügt haben, starten Sie Ihr Training in einem Skript oder einem Notebook wie Colaboratory.

### Trainieren mit einem Skript

Wenn Sie Ihr Training mit einem Skript durchführen, führen Sie den folgenden Befehl aus, um eine Konfigurationsdatei zu erstellen und zu speichern:

```bash
accelerate config
```

Dann starten Sie Ihr Training mit:

```bash
accelerate launch train.py
```

### Trainieren mit einem Notebook

🤗 Accelerate kann auch in einem Notebook laufen, wenn Sie planen, die TPUs von Colaboratory zu verwenden. Verpacken Sie den gesamten Code, der für das Training verantwortlich ist, in eine Funktion und übergeben Sie diese an [`~accelerate.notebook_launcher`]:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

Weitere Informationen über 🤗 Accelerate und seine umfangreichen Funktionen finden Sie in der [Dokumentation](https://huggingface.co/docs/accelerate).
