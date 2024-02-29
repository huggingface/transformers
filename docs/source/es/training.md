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

# Fine-tuning a un modelo pre-entrenado

[[open-in-colab]]

El uso de un modelo pre-entrenado tiene importantes ventajas. Reduce los costos de computación, la huella de carbono y te permite utilizar modelos de última generación sin tener que entrenar uno desde cero.

* Fine-tuning a un modelo pre-entrenado con 🤗 Transformers [`Trainer`].
* Fine-tuning a un modelo pre-entrenado en TensorFlow con Keras.
* Fine-tuning a un modelo pre-entrenado en PyTorch nativo.

<a id='data-processing'></a>

## Prepara un dataset

<Youtube id="_BZearw7f0w"/>

Antes de aplicar fine-tuning a un modelo pre-entrenado, descarga un dataset y prepáralo para el entrenamiento. El tutorial anterior nos enseñó cómo procesar los datos para el entrenamiento, y ahora es la oportunidad de poner a prueba estas habilidades.

Comienza cargando el dataset de [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full):

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset[100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

Como ya sabes, necesitas un tokenizador para procesar el texto e incluir una estrategia para el padding y el truncamiento para manejar cualquier longitud de secuencia variable. Para procesar tu dataset en un solo paso, utiliza el método de 🤗 Datasets map para aplicar una función de preprocesamiento sobre todo el dataset:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

Si lo deseas, puedes crear un subconjunto más pequeño del dataset completo para aplicarle fine-tuning y así reducir el tiempo.

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Fine-tuning con `Trainer`

<Youtube id="nvBXf7s7vTI"/>

🤗 Transformers proporciona una clase [`Trainer`] optimizada para el entrenamiento de modelos de 🤗 Transformers, haciendo más fácil el inicio del entrenamiento sin necesidad de escribir manualmente tu propio ciclo. La API del [`Trainer`] soporta una amplia gama de opciones de entrenamiento y características como el logging, el gradient accumulation y el mixed precision.

Comienza cargando tu modelo y especifica el número de labels previstas. A partir del [Card Dataset](https://huggingface.co/datasets/yelp_review_full#data-fields) de Yelp Review, que como ya sabemos tiene 5 labels:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

<Tip>

Verás una advertencia acerca de que algunos de los pesos pre-entrenados no están siendo utilizados y que algunos pesos están siendo inicializados al azar. No te preocupes, esto es completamente normal.
El head/cabezal pre-entrenado del modelo BERT se descarta y se sustituye por un head de clasificación inicializado aleatoriamente. Puedes aplicar fine-tuning a este nuevo head del modelo en tu tarea de clasificación de secuencias haciendo transfer learning del modelo pre-entrenado.

</Tip>

### Hiperparámetros de entrenamiento

A continuación, crea una clase [`TrainingArguments`] que contenga todos los hiperparámetros que puedes ajustar así como los indicadores para activar las diferentes opciones de entrenamiento. Para este tutorial puedes empezar con los [hiperparámetros](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) de entrenamiento por defecto, pero siéntete libre de experimentar con ellos para encontrar tu configuración óptima.

Especifica dónde vas a guardar los checkpoints de tu entrenamiento:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### Métricas

El [`Trainer`] no evalúa automáticamente el rendimiento del modelo durante el entrenamiento. Tendrás que pasarle a [`Trainer`] una función para calcular y hacer un reporte de las métricas. La biblioteca de 🤗 Datasets proporciona una función de [`accuracy`](https://huggingface.co/metrics/accuracy) simple que puedes cargar con la función `load_metric` (ver este [tutorial](https://huggingface.co/docs/datasets/metrics) para más información):

```py
>>> import numpy as np
>>> from datasets import load_metric

>>> metric = load_metric("accuracy")
```

Define la función `compute` en `metric` para calcular el accuracy de tus predicciones. Antes de pasar tus predicciones a `compute`, necesitas convertir las predicciones a logits (recuerda que todos los modelos de 🤗 Transformers devuelven logits).

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

Si quieres controlar tus métricas de evaluación durante el fine-tuning, especifica el parámetro `evaluation_strategy` en tus argumentos de entrenamiento para que el modelo tenga en cuenta la métrica de evaluación al final de cada época:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```

### Trainer

Crea un objeto [`Trainer`] con tu modelo, argumentos de entrenamiento, datasets de entrenamiento y de prueba, y tu función de evaluación:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

A continuación, aplica fine-tuning a tu modelo llamando [`~transformers.Trainer.train`]:

```py
>>> trainer.train()
```

<a id='keras'></a>

## Fine-tuning con Keras

<Youtube id="rnTGBy2ax1c"/>

Los modelos de 🤗 Transformers también permiten realizar el entrenamiento en TensorFlow con la API de Keras. Solo es necesario hacer algunos cambios antes de hacer fine-tuning.

### Convierte el dataset al formato de TensorFlow

El [`DefaultDataCollator`] junta los tensores en un batch para que el modelo se entrene en él. Asegúrate de especificar `return_tensors` para devolver los tensores de TensorFlow:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```

<Tip>

[`Trainer`] utiliza [`DataCollatorWithPadding`] por defecto por lo que no es necesario especificar explícitamente un intercalador de datos (data collator, en inglés).

</Tip>

A continuación, convierte los datasets tokenizados en datasets de TensorFlow con el método [`to_tf_dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.to_tf_dataset). Especifica tus entradas en `columns` y tu etiqueta en `label_cols`:

```py
>>> tf_train_dataset = small_train_dataset.to_tf_dataset(
...     columns=["attention_mask", "input_ids", "token_type_ids"],
...     label_cols="labels",
...     shuffle=True,
...     collate_fn=data_collator,
...     batch_size=8,
... )

>>> tf_validation_dataset = small_eval_dataset.to_tf_dataset(
...     columns=["attention_mask", "input_ids", "token_type_ids"],
...     label_cols="labels",
...     shuffle=False,
...     collate_fn=data_collator,
...     batch_size=8,
... )
```

### Compila y ajusta

Carguemos un modelo TensorFlow con el número esperado de labels:

```py
>>> import tensorflow as tf
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

A continuación, compila y aplica fine-tuning a tu modelo con [`fit`](https://keras.io/api/models/model_training_apis/) como lo harías con cualquier otro modelo de Keras:

```py
>>> model.compile(
...     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
...     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
...     metrics=tf.metrics.SparseCategoricalAccuracy(),
... )

>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)
```

<a id='pytorch_native'></a>

## Fine-tune en PyTorch nativo

<Youtube id="Dh9CL8fyG80"/>

El [`Trainer`] se encarga del ciclo de entrenamiento y permite aplicar fine-tuning a un modelo en una sola línea de código. Para los que prefieran escribir su propio ciclo de entrenamiento, también pueden aplicar fine-tuning a un modelo de 🤗 Transformers en PyTorch nativo.

En este punto, es posible que necesites reiniciar tu notebook o ejecutar el siguiente código para liberar algo de memoria:

```py
del model
del pytorch_model
del trainer
torch.cuda.empty_cache()
```

A continuación, haremos un post-procesamiento manual al `tokenized_dataset` y así prepararlo para el entrenamiento.

1. Elimina la columna de `text` porque el modelo no acepta texto en crudo como entrada:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. Cambia el nombre de la columna de `label` a `labels` porque el modelo espera que el argumento se llame `labels`:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. Establece el formato del dataset para devolver tensores PyTorch en lugar de listas:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

A continuación, crea un subconjunto más pequeño del dataset como se ha mostrado anteriormente para acelerar el fine-tuning:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

Crea un `DataLoader` para tus datasets de entrenamiento y de prueba para poder iterar sobre batches de datos:

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

Carga tu modelo con el número de labels previstas:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

### Optimiza y programa el learning rate

Crea un optimizador y el learning rate para aplicar fine-tuning al modelo. Vamos a utilizar el optimizador [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) de PyTorch:

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

Crea el learning rate desde el [`Trainer`]:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

Por último, especifica el `device` o entorno de ejecución para utilizar una GPU si tienes acceso a una. De lo contrario, el entrenamiento en una CPU puede llevarte varias horas en lugar de un par de minutos.

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

Consigue acceso gratuito a una GPU en la nube si es que no tienes este recurso de forma local con un notebook alojado en [Colaboratory](https://colab.research.google.com/) o [SageMaker StudioLab](https://studiolab.sagemaker.aws/).

</Tip>

Genial, ¡ahora podemos entrenar! 🥳

### Ciclo de entrenamiento

Para hacer un seguimiento al progreso del entrenamiento, utiliza la biblioteca [tqdm](https://tqdm.github.io/) para añadir una barra de progreso sobre el número de pasos de entrenamiento:

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### Métricas

De la misma manera que necesitas añadir una función de evaluación al [`Trainer`], necesitas hacer lo mismo cuando escribas tu propio ciclo de entrenamiento. Pero en lugar de calcular y reportar la métrica al final de cada época, esta vez acumularás todos los batches con [`add_batch`](https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=add_batch#datasets.Metric.add_batch) y calcularás la métrica al final.

```py
>>> metric = load_metric("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```

<a id='additional-resources'></a>

## Recursos adicionales

Para más ejemplos de fine-tuning consulta:

- [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) incluye scripts
  para entrenar tareas comunes de NLP en PyTorch y TensorFlow.

- [🤗 Transformers Notebooks](notebooks) contiene varios notebooks sobre cómo aplicar fine-tuning a un modelo para tareas específicas en PyTorch y TensorFlow.
