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

# Entraînement distribué avec 🤗 Accelerate

Comme les modèles deviennent plus gros, le parallélisme est devenu une stratégie pour entraîner des modèles plus grands sur du matériel aux capacités limitées et accélérer la vitesse d'entraînement de plusieurs ordres de grandeur. Hugging Face fournit la librairie [🤗 Accelerate](https://huggingface.co/docs/accelerate) pour aider les utilisateurs à entraîner facilement un modèle 🤗 Transformers sur n'importe quel type de configuration distribuée, qu'il s'agisse de plusieurs GPU sur une machine ou de plusieurs GPU sur plusieurs machines. Dans ce tutoriel, vous apprenez à personnaliser votre boucle d'entraînement avec PyTorch pour permettre l'entraînement dans un environnement distribué.

## Configuration

Commencez par installer 🤗 Accelerate :

```bash
pip install accelerate
```

Ensuite, importez et créez un objet [`~accelerate.Accelerator`]. L'objet [`~accelerate.Accelerator`] détectera automatiquement votre type de configuration distribuée et initialisera tous les composants nécessaires à l'entraînement. Vous n'avez pas besoin de placer explicitement votre modèle sur une carte graphique ou CPU.

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## Préparation pour l'accélération

L'étape suivante consiste à passer tous les objets d'entraînement pertinents à la méthode [`~accelerate.Accelerator.prepare`]. Cela inclut les DataLoaders pour l'entraînement et l'évaluation, un modèle et un optimiseur :

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## Retropropagation

La dernière étape consiste à remplacer `loss.backward()` dans votre boucle d'entraînement par la fonction [`~accelerate.Accelerator.backward`] de 🤗 Accelerate :

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

Comme vous pouvez le voir dans le code dessous, vous avez seulement besoin d'ajouter quatre lignes de code à votre boucle d'entraînement pour activer l'entraînement distribué !

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

## Entraînement

Une fois que vous avez ajouté les lignes de code nécessaires, vous pouvez lancer votre entraînement avec un script ou un notebook comme Colaboratory.

### Entraînement avec un script

Si votre entraînement est lancé avec un script, vous devez exécuter la commande suivante pour créer et enregistrer un fichier de configuration :

```bash
accelerate config
```

Puis, vous lancez l'entraînement avec la commande suivante :

```bash
accelerate launch train.py
```

### Entraînement avec un notebook

🤗 Accelerate peut aussi etre utilisé dans un notebook si vous prévoyez d'utiliser les TPUs de Colaboratory. Créez une fonction contenant le code responsable de l'entraînement, et passez-la à [`~accelerate.notebook_launcher`]:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

Pour plus d'informations sur 🤗 Accelerate et ses fonctionnalités, consultez la [documentation](https://huggingface.co/docs/accelerate).
