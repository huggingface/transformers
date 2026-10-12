<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Clasificación de tokens

[[open-in-colab]]

<Youtube id="wVHdVlPScxA"/>

La clasificación de tokens asigna una etiqueta a cada token de un enunciado. Una de las tareas más comunes de clasificación de tokens es NER (Named Entity Recognition). NER intenta encontrar una etiqueta para cada entidad de un enunciado, como una persona, un lugar o una organización.

Esta guía te mostrará cómo:

1. Realizar fine-tuning de [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) en el dataset [WNUT 17](https://huggingface.co/datasets/wnut_17) para detectar entidades nuevas.
2. Usar tu modelo con fine-tuning para inferencia.

<Tip>

Para ver todas las arquitecturas y checkpoints compatibles con esta tarea, te recomendamos revisar la [página de la tarea](https://huggingface.co/tasks/token-classification).

</Tip>

Antes de empezar, asegúrate de tener instaladas todas las bibliotecas necesarias:

```bash
pip install transformers datasets evaluate seqeval
```

Te recomendamos que inicies sesión en tu cuenta de Hugging Face para que puedas subir y compartir tu modelo con la comunidad. Cuando te lo pida, ingresa tu token para iniciar sesión:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Carga el dataset WNUT 17

Empieza cargando el dataset WNUT 17 desde la biblioteca 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

Luego observa un ejemplo:

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

Cada número en `ner_tags` representa una entidad. Convierte los números a sus nombres de etiqueta para saber cuáles son las entidades:

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
```

La letra que precede a cada `ner_tag` indica la posición del token dentro de la entidad:

- `B-` indica el comienzo de una entidad.
- `I-` indica que un token está contenido dentro de la misma entidad (por ejemplo, el token `State` forma parte de una entidad como
  `Empire State Building`).
- `0` indica que el token no corresponde a ninguna entidad.

## Preprocesamiento

<Youtube id="iY2AZYdZAr0"/>

El siguiente paso es cargar un tokenizador DistilBERT para preprocesar el campo `tokens`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

Como viste en el campo `tokens` del ejemplo de arriba, parece que el input ya está tokenizado. En realidad el input todavía no está tokenizado y tendrás que configurar `is_split_into_words=True` para tokenizar las palabras en subpalabras. Por ejemplo:

```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

Sin embargo, esto añade algunos tokens especiales `[CLS]` y `[SEP]` y la tokenización en subpalabras crea un desajuste entre el input y las etiquetas. Una sola palabra que corresponde a una sola etiqueta ahora puede dividirse en dos subpalabras. Tendrás que realinear los tokens y las etiquetas:

1. Mapeando todos los tokens a su palabra correspondiente con el método [`~BatchEncoding#word_ids`].
2. Asignando la etiqueta `-100` a los tokens especiales `[CLS]` y `[SEP]` para que los ignore la función de pérdida de PyTorch (consulta [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
3. Etiquetando solo el primer token de una palabra dada. Asigna `-100` a los demás subtokens de la misma palabra.

Así es como puedes crear una función para realinear los tokens y las etiquetas, y truncar las secuencias para que no superen la longitud máxima de input de DistilBERT:

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # Set the special tokens to -100.
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

Para aplicar la función de preprocesamiento sobre el dataset en su totalidad, usa la función [`~datasets.Dataset.map`] de 🤗 Datasets. Puedes acelerar la función `map` configurando `batched=True` para procesar múltiples elementos del dataset a la vez:

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

Ahora crea un lote de ejemplos con [`DataCollatorWithPadding`]. Es más eficiente *rellenar dinámicamente* los enunciados a la longitud más larga del lote al agruparlos, en lugar de rellenar todo el dataset a la longitud máxima.

```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

## Evaluación

Incluir una métrica durante el entrenamiento suele ser útil para evaluar el rendimiento de tu modelo. Puedes cargar rápidamente un método de evaluación con la biblioteca 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). Para esta tarea, carga el framework [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) (consulta el [tour rápido](https://huggingface.co/docs/evaluate/a_quick_tour) de 🤗 Evaluate para saber más sobre cómo cargar y calcular una métrica). Seqeval produce varias puntuaciones: precision, recall, F1 y accuracy.

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

Primero obtén las etiquetas NER y luego crea una función que pase tus predicciones verdaderas y tus etiquetas verdaderas a [`~evaluate.EvaluationModule.compute`] para calcular las puntuaciones:

```py
>>> import numpy as np

>>> labels = [label_list[i] for i in example[f"ner_tags"]]


>>> def compute_metrics(p):
...     predictions, labels = p
...     predictions = np.argmax(predictions, axis=2)

...     true_predictions = [
...         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]
...     true_labels = [
...         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]

...     results = seqeval.compute(predictions=true_predictions, references=true_labels)
...     return {
...         "precision": results["overall_precision"],
...         "recall": results["overall_recall"],
...         "f1": results["overall_f1"],
...         "accuracy": results["overall_accuracy"],
...     }
```

Tu función `compute_metrics` ya está lista, y volverás a ella cuando configures el entrenamiento.

## Entrenamiento

Antes de empezar a entrenar tu modelo, crea un mapeo de los ids esperados a sus etiquetas con `id2label` y `label2id`:

```py
>>> id2label = {
...     0: "O",
...     1: "B-corporation",
...     2: "I-corporation",
...     3: "B-creative-work",
...     4: "I-creative-work",
...     5: "B-group",
...     6: "I-group",
...     7: "B-location",
...     8: "I-location",
...     9: "B-person",
...     10: "I-person",
...     11: "B-product",
...     12: "I-product",
... }
>>> label2id = {
...     "O": 0,
...     "B-corporation": 1,
...     "I-corporation": 2,
...     "B-creative-work": 3,
...     "I-creative-work": 4,
...     "B-group": 5,
...     "I-group": 6,
...     "B-location": 7,
...     "I-location": 8,
...     "B-person": 9,
...     "I-person": 10,
...     "B-product": 11,
...     "I-product": 12,
... }
```

<Tip>

Para familiarizarte con el proceso de realizar fine-tuning sobre un modelo con el [`Trainer`], ¡mira el tutorial básico [aquí](../training#train-with-pytorch-trainer)!

</Tip>

¡Ya puedes empezar a entrenar tu modelo! Carga DistilBERT con [`AutoModelForTokenClassification`] junto con el número de etiquetas esperadas y los mapeos de etiquetas:

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

En este punto, solo quedan tres pasos:

1. Define tus hiperparámetros de entrenamiento en [`TrainingArguments`]. El único parámetro obligatorio es `output_dir`, que indica dónde guardar tu modelo. Subirás este modelo al Hub configurando `push_to_hub=True` (necesitas haber iniciado sesión en Hugging Face para subir tu modelo). Al final de cada epoch, el [`Trainer`] evaluará las puntuaciones de seqeval y guardará el checkpoint de entrenamiento.
2. Pásale los argumentos de entrenamiento a [`Trainer`] junto con el modelo, el dataset, el tokenizador, el data collator y la función `compute_metrics`.
3. Llama a [`~Trainer.train`] para realizar el fine-tuning de tu modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_wnut["train"],
...     eval_dataset=tokenized_wnut["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Cuando termine el entrenamiento, comparte tu modelo en el Hub con el método [`~transformers.Trainer.push_to_hub`] para que cualquiera pueda usarlo:

```py
>>> trainer.push_to_hub()
```

<Tip>

Para un ejemplo con mayor profundidad de cómo realizar fine-tuning de un modelo para clasificación de tokens, échale un vistazo al
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb) correspondiente.

</Tip>

## Inferencia

Genial, ahora que ya realizaste fine-tuning de un modelo, ¡puedes usarlo para inferencia!

Toma un texto sobre el que quieras hacer inferencia:

```py
>>> text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
```

La forma más simple de probar tu modelo con fine-tuning para inferencia es usarlo en un [`pipeline`]. Instancia un `pipeline` para NER con tu modelo y pásale tu texto:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
>>> classifier(text)
[{'entity': 'B-location',
  'score': 0.42658573,
  'index': 2,
  'word': 'golden',
  'start': 4,
  'end': 10},
 {'entity': 'I-location',
  'score': 0.35856336,
  'index': 3,
  'word': 'state',
  'start': 11,
  'end': 16},
 {'entity': 'B-group',
  'score': 0.3064001,
  'index': 4,
  'word': 'warriors',
  'start': 17,
  'end': 25},
 {'entity': 'B-location',
  'score': 0.65523505,
  'index': 13,
  'word': 'san',
  'start': 80,
  'end': 83},
 {'entity': 'B-location',
  'score': 0.4668663,
  'index': 14,
  'word': 'francisco',
  'start': 84,
  'end': 93}]
```

También puedes replicar a mano los resultados del `pipeline` si quieres:

Tokeniza el texto y devuelve tensores de PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

Pasa tus inputs al modelo y obtén los `logits`:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

Obtén la clase con la probabilidad más alta y usa el mapeo `id2label` del modelo para convertirla en una etiqueta de texto:

```py
>>> predictions = torch.argmax(logits, dim=2)
>>> predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
