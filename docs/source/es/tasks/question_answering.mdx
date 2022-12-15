<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Respuesta a preguntas

<Youtube id="ajPx5LwJD-I"/>

La respuesta a preguntas devuelve una respuesta a partir de una pregunta dada. Existen dos formas comunes de responder preguntas:

- Extractiva: extraer la respuesta a partir del contexto dado.
- Abstractiva: generar una respuesta que responda correctamente la pregunta a partir del contexto dado.

Esta guía te mostrará como hacer fine-tuning de [DistilBERT](https://huggingface.co/distilbert-base-uncased) en el dataset [SQuAD](https://huggingface.co/datasets/squad) para responder preguntas de forma extractiva.

<Tip>

Revisa la [página de la tarea](https://huggingface.co/tasks/question-answering) de responder preguntas para tener más información sobre otras formas de responder preguntas y los modelos, datasets y métricas asociadas.

</Tip>

## Carga el dataset SQuAD

Carga el dataset SQuAD con la biblioteca 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> squad = load_dataset("squad")
```

Ahora, échale un vistazo a una muestra:

```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

El campo `answers` es un diccionario que contiene la posición inicial de la respuesta y el `texto` de la respuesta.

## Preprocesamiento

<Youtube id="qgaM0weJHpA"/>

Carga el tokenizer de DistilBERT para procesar los campos `question` (pregunta) y `context` (contexto):

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

Hay algunos pasos de preprocesamiento específicos para la tarea de respuesta a preguntas que debes tener en cuenta:

1. Algunos ejemplos en un dataset pueden tener un contexto que supera la longitud máxima de entrada de un modelo. Trunca solamente el contexto asignándole el valor `"only_second"` al parámetro `truncation`.
2. A continuación, mapea las posiciones de inicio y fin de la respuesta al contexto original asignándole el valor `True` al parámetro `return_offsets_mapping`.
3. Una vez tengas el mapeo, puedes encontrar los tokens de inicio y fin de la respuesta. Usa el método [`sequence_ids`](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.Encoding.sequence_ids)
para encontrar qué parte de la lista de tokens desplazados corresponde a la pregunta y cuál corresponde al contexto.

A continuación puedes ver como se crea una función para truncar y mapear los tokens de inicio y fin de la respuesta al `context`:

```py
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding="max_length",
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         # Encuentra el inicio y el fin del contexto
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # Si la respuesta entera no está dentro del contexto, etiquétala como (0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # De lo contrario, esta es la posición de los tokens de inicio y fin
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

Usa la función [`~datasets.Dataset.map`] de 🤗 Datasets para aplicarle la función de preprocesamiento al dataset entero. Puedes acelerar la función `map` haciendo `batched=True` para procesar varios elementos del dataset a la vez.
Quita las columnas que no necesites:

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

Usa el [`DefaultDataCollator`] para crear un lote de ejemplos. A diferencia de los otros collators de datos en 🤗 Transformers, el `DefaultDataCollator` no aplica ningún procesamiento adicional (como el rellenado).

<frameworkcontent>
<pt>
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt>
<tf>
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>

## Entrenamiento

<frameworkcontent>
<pt>
Carga el modelo DistilBERT con [`AutoModelForQuestionAnswering`]:

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
```

<Tip>

Para familiarizarte con el fine-tuning con [`Trainer`], ¡mira el tutorial básico [aquí](../training#finetune-with-trainer)!

</Tip>

En este punto, solo quedan tres pasos:

1. Definir tus hiperparámetros de entrenamiento en [`TrainingArguments`].
2. Pasarle los argumentos del entrenamiento al [`Trainer`] junto con el modelo, el dataset, el tokenizer y el collator de datos.
3. Invocar el método [`~Trainer.train`] para realizar el fine-tuning del modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="./results",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["validation"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```
</pt>
<tf>
Para realizar el fine-tuning de un modelo en TensorFlow, primero convierte tus datasets al formato `tf.data.Dataset` con el método [`~TFPreTrainedModel.prepare_tf_dataset`].

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_squad["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_squad["validation"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

<Tip>

Para familiarizarte con el fine-tuning con Keras, ¡mira el tutorial básico [aquí](training#finetune-with-keras)!

</Tip>

Prepara una función de optimización, un programa para la tasa de aprendizaje y algunos hiperparámetros de entrenamiento:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 2
>>> total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
>>> optimizer, schedule = create_optimizer(
...     init_lr=2e-5,
...     num_warmup_steps=0,
...     num_train_steps=total_train_steps,
... )
```

Carga el modelo DistilBERT con [`TFAutoModelForQuestionAnswering`]:

```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering("distilbert-base-uncased")
```

Configura el modelo para entrenarlo con [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

Invoca el método [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) para realizar el fine-tuning del modelo:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3)
```
</tf>
</frameworkcontent>

<Tip>

Para un ejemplo con mayor profundidad de cómo hacer fine-tuning a un modelo para responder preguntas, échale un vistazo al
[cuaderno de PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) o al
[cuaderno de TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb) correspondiente.

</Tip>
