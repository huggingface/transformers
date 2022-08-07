<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Entrenamiento distribuido con 🤗 Accelerate

El paralelismo ha emergido como una estrategia para entrenar modelos grandes en hardware limitado e incrementar la velocidad de entrenamiento en varios órdenes de magnitud. En Hugging Face creamos la biblioteca [🤗 Accelerate](https://huggingface.co/docs/accelerate) para ayudar a los usuarios a entrenar modelos 🤗 Transformers en cualquier tipo de configuración distribuida, ya sea en una máquina con múltiples GPUs o en múltiples GPUs distribuidas entre muchas máquinas. En este tutorial aprenderás cómo personalizar tu bucle de entrenamiento de PyTorch nativo para poder entrenar en entornos distribuidos.

## Configuración

Empecemos por instalar 🤗 Accelerate:

```bash
pip install accelerate
```

Luego, importamos y creamos un objeto [`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator). `Accelerator` detectará automáticamente el tipo de configuración distribuida que tengas disponible e inicializará todos los componentes necesarios para el entrenamiento. No necesitas especificar el dispositivo en donde se debe colocar tu modelo.

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## Prepárate para acelerar

Pasa todos los objetos relevantes para el entrenamiento al método [`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare). Esto incluye los DataLoaders de entrenamiento y evaluación, un modelo y un optimizador:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## Backward

Por último, reemplaza el típico `loss.backward()` en tu bucle de entrenamiento con el método [`backward`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.backward) de 🤗 Accelerate:

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

Como se puede ver en el siguiente código, ¡solo necesitas adicionar cuatro líneas de código a tu bucle de entrenamiento para habilitar el entrenamiento distribuido!

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

## Entrenamiento

Una vez que hayas añadido las líneas de código relevantes, inicia el entrenamiento desde un script o notebook como Colaboratory.

### Entrenar con un script

Si estás corriendo tu entrenamiento desde un script ejecuta el siguiente comando para crear y guardar un archivo de configuración:

```bash
accelerate config
```

Comienza el entrenamiento con:

```bash
accelerate launch train.py
```

### Entrenar con un notebook

🤗 Accelerate puede correr en un notebook si, por ejemplo, estás planeando utilizar las TPUs de Colaboratory. Encierra el código responsable del entrenamiento en una función y pásalo a `notebook_launcher`:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

Para obtener más información sobre 🤗 Accelerate y sus numerosas funciones, consulta la [documentación](https://huggingface.co/docs/accelerate).
