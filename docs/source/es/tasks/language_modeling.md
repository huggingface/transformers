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

# Modelado de lenguaje

El modelado de lenguaje predice palabras en un enunciado. Hay dos formas de modelado de lenguaje.

<Youtube id="Vpjb1lu0MDk"/>

El modelado de lenguaje causal predice el siguiente token en una secuencia de tokens, y el modelo solo puede considerar los tokens a la izquierda.

<Youtube id="mqElG5QJWUg"/>

El modelado de lenguaje por enmascaramiento predice un token enmascarado en una secuencia, y el modelo puede considerar los tokens bidireccionalmente.

Esta guía te mostrará cómo realizar fine-tuning [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) para modelos de lenguaje causales y [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) para modelos de lenguaje por enmascaramiento en el [r/askscience](https://www.reddit.com/r/askscience/) subdataset [ELI5](https://huggingface.co/datasets/eli5). 

<Tip>

Puedes realizar fine-tuning a otras arquitecturas para modelos de lenguaje como [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-125M), [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B) y [BERT](https://huggingface.co/google-bert/bert-base-uncased) siguiendo los mismos pasos presentados en esta guía!

Mira la [página de tarea](https://huggingface.co/tasks/text-generation) para generación de texto y la [página de tarea](https://huggingface.co/tasks/fill-mask) para modelos de lenguajes por enmascaramiento para obtener más información sobre los modelos, datasets, y métricas asociadas.

</Tip>

## Carga el dataset ELI5

Carga solo los primeros 5000 registros desde la biblioteca 🤗 Datasets, dado que es bastante grande:

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5", split="train_asks[:5000]")
```

Divide este dataset en subdatasets para el entrenamiento y el test:

```py
eli5 = eli5.train_test_split(test_size=0.2)
```

Luego observa un ejemplo:

```py
>>> eli5["train"][0]
{'answers': {'a_id': ['c3d1aib', 'c3d4lya'],
  'score': [6, 3],
  'text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
   "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"]},
 'answers_urls': {'url': []},
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls': {'url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg']},
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls': {'url': []}}
```

Observa que `text` es un subcampo anidado dentro del diccionario `answers`. Cuando preproceses el dataset, deberás extraer el subcampo `text` en una columna aparte.

## Preprocesamiento

<Youtube id="ma1TrR7gE7I"/>

Para modelados de lenguaje causales carga el tokenizador DistilGPT2 para procesar el subcampo `text`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
```

<Youtube id="8PmhEIXhBvI"/>

Para modelados de lenguaje por enmascaramiento carga el tokenizador DistilRoBERTa, en lugar de DistilGPT2:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
```

Extrae el subcampo `text` desde su estructura anidado con el método [`flatten`](https://huggingface.co/docs/datasets/process#flatten):

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'answers.a_id': ['c3d1aib', 'c3d4lya'],
 'answers.score': [6, 3],
 'answers.text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
  "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"],
 'answers_urls.url': [],
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls.url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg'],
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls.url': []}
```

Cada subcampo es ahora una columna separada, como lo indica el prefijo `answers`. Observa que `answers.text` es una lista. En lugar de tokenizar cada enunciado por separado, convierte la lista en un string para tokenizarlos conjuntamente.

Así es como puedes crear una función de preprocesamiento para convertir la lista en una cadena y truncar las secuencias para que no superen la longitud máxima de input de DistilGPT2:

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]], truncation=True)
```

Usa de 🤗 Datasets la función [`map`](https://huggingface.co/docs/datasets/process#map) para aplicar la función de preprocesamiento sobre el dataset en su totalidad. Puedes acelerar la función `map` configurando el argumento `batched=True` para procesar múltiples elementos del dataset a la vez y aumentar la cantidad de procesos con `num_proc`. Elimina las columnas que no necesitas:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

Ahora necesitas una segunda función de preprocesamiento para capturar el texto truncado de cualquier ejemplo demasiado largo para evitar cualquier pérdida de información. Esta función de preprocesamiento debería:

- Concatenar todo el texto.
- Dividir el texto concatenado en trozos más pequeños definidos por un `block_size`.

```py
>>> block_size = 128


>>> def group_texts(examples):
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     total_length = (total_length // block_size) * block_size
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     result["labels"] = result["input_ids"].copy()
...     return result
```

Aplica la función `group_texts` sobre todo el dataset:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

Para modelados de lenguaje causales, usa [`DataCollatorForLanguageModeling`] para crear un lote de ejemplos. Esto también *rellenará dinámicamente* tu texto a la dimensión del elemento más largo del lote para que de esta manera tengan largo uniforme. Si bien es posible rellenar tu texto en la función `tokenizer` mediante el argumento `padding=True`, el rellenado dinámico es más eficiente. 

<frameworkcontent>
<pt>
Puedes usar el token de final de secuencia como el token de relleno y asignar `mlm=False`. Esto usará los inputs como etiquetas movidas un elemento hacia la derecha:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

Para modelados de lenguaje por enmascaramiento usa el mismo [`DataCollatorForLanguageModeling`] excepto que deberás especificar `mlm_probability` para enmascarar tokens aleatoriamente cada vez que iteras sobre los datos.

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```
</pt>
<tf>
Puedes usar el token de final de secuencia como el token de relleno y asignar `mlm=False`. Esto usará los inputs como etiquetas movidas un elemento hacia la derecha:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")
```

Para modelados de lenguajes por enmascaramiento usa el mismo [`DataCollatorForLanguageModeling`] excepto que deberás especificar `mlm_probability` para enmascarar tokens aleatoriamente cada vez que iteras sobre los datos.

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")
```
</tf>
</frameworkcontent>

## Modelado de lenguaje causal

El modelado de lenguaje causal es frecuentemente utilizado para generación de texto. Esta sección te muestra cómo realizar fine-tuning a [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) para generar nuevo texto.

### Entrenamiento

<frameworkcontent>
<pt>
Carga DistilGPT2 con [`AutoModelForCausalLM`]:

```py
>>> from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

<Tip>

Si no estás familiarizado con el proceso de realizar fine-tuning sobre un modelo con [`Trainer`], considera el tutorial básico [aquí](../training#finetune-with-trainer)!

</Tip>

A este punto, solo faltan tres pasos:

1. Definir tus hiperparámetros de entrenamiento en [`TrainingArguments`].
2. Pasarle los argumentos de entrenamiento a [`Trainer`] junto con el modelo, dataset, y el data collator.
3. Realiza la llamada [`~Trainer.train`] para realizar el fine-tuning sobre tu modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="./results",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     weight_decay=0.01,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
... )

>>> trainer.train()
```
</pt>
<tf>
Para realizar el fine-tuning de un modelo en TensorFlow, comienza por convertir tus datasets al formato `tf.data.Dataset` con [`to_tf_dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.to_tf_dataset). Especifica los inputs y etiquetas en `columns`, ya sea para mezclar el dataset, tamaño de lote, y el data collator:

```py
>>> tf_train_set = lm_dataset["train"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     dummy_labels=True,
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = lm_dataset["test"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     dummy_labels=True,
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

<Tip>

Si no estás familiarizado con realizar fine-tuning de tus modelos con Keras, considera el tutorial básico [aquí](training#finetune-with-keras)!

</Tip>

Crea la función optimizadora, la tasa de aprendizaje, y algunos hiperparámetros de entrenamiento:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

Carga DistilGPT2 con [`TFAutoModelForCausalLM`]:

```py
>>> from transformers import TFAutoModelForCausalLM

>>> model = TFAutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

Configura el modelo para entrenamiento con [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

Llama a [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) para realizar el fine-tuning del modelo:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3)
```
</tf>
</frameworkcontent>

## Modelado de lenguaje por enmascaramiento

El modelado de lenguaje por enmascaramiento es también conocido como una tarea de rellenar la máscara, pues predice un token enmascarado dada una secuencia. Los modelos de lenguaje por enmascaramiento requieren una buena comprensión del contexto de una secuencia entera, en lugar de solo el contexto a la izquierda. Esta sección te enseña como realizar el fine-tuning de [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) para predecir una palabra enmascarada.

### Entrenamiento

<frameworkcontent>
<pt>
Carga DistilRoBERTa con [`AutoModelForMaskedlM`]:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

<Tip>

Si no estás familiarizado con el proceso de realizar fine-tuning sobre un modelo con [`Trainer`], considera el tutorial básico [aquí](../training#finetune-with-trainer)!

</Tip>

A este punto, solo faltan tres pasos:

1. Definir tus hiperparámetros de entrenamiento en [`TrainingArguments`].
2. Pasarle los argumentos de entrenamiento a [`Trainer`] junto con el modelo, dataset, y el data collator.
3. Realiza la llamada [`~Trainer.train`] para realizar el fine-tuning de tu modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="./results",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     num_train_epochs=3,
...     weight_decay=0.01,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
... )

>>> trainer.train()
```
</pt>
<tf>
Para realizar el fine-tuning de un modelo en TensorFlow, comienza por convertir tus datasets al formato `tf.data.Dataset` con [`to_tf_dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.to_tf_dataset). Especifica los inputs y etiquetas en `columns`, ya sea para mezclar el dataset, tamaño de lote, y el data collator:

```py
>>> tf_train_set = lm_dataset["train"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     dummy_labels=True,
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = lm_dataset["test"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     dummy_labels=True,
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

<Tip>

Si no estás familiarizado con realizar fine-tuning de tus modelos con Keras, considera el tutorial básico [aquí](training#finetune-with-keras)!

</Tip>

Crea la función optimizadora, la tasa de aprendizaje, y algunos hiperparámetros de entrenamiento:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

Carga DistilRoBERTa con [`TFAutoModelForMaskedLM`]:

```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForCausalLM.from_pretrained("distilbert/distilroberta-base")
```

Configura el modelo para entrenamiento con [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

Llama a [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) para realizar el fine-tuning del modelo:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3)
```
</tf>
</frameworkcontent>

<Tip>

Para un ejemplo más profundo sobre cómo realizar el fine-tuning sobre un modelo de lenguaje causal, considera
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
o [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

</Tip>