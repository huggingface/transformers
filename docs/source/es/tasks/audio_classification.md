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

# Clasificación de audio

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>

Clasificación de audio - al igual que con texto — asigna una etiqueta de clase como salida desde las entradas de datos. La diferencia única es en vez de entrada de texto, tiene formas de onda de audio. Algunas aplicaciones prácticas de clasificación incluye identificar la intención del hablante, identificación del idioma, y la clasificación de animales por sus sonidos.

En esta guía te mostraremos como: 

1. Hacer fine-tuning al modelo [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) en el dataset [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) para clasificar la intención del hablante.
2. Usar tu modelo ajustado para tareas de inferencia.


<Tip>

Consulta la [página de la tarea](https://huggingface.co/tasks/audio-classification) de clasificación de audio para acceder a más información sobre los modelos, datasets, y métricas asociados.

</Tip>

Antes de comenzar, asegúrate de haber instalado todas las librerías necesarias:

```bash
pip install transformers datasets evaluate
```

Te aconsejamos iniciar sesión con tu cuenta de Hugging Face para que puedas subir tu modelo y compartirlo con la comunidad. Cuando se te solicite, ingresa tu token para iniciar sesión:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Carga el dataset MInDS-14

Comencemos cargando el dataset MInDS-14 con la biblioteca de 🤗 Datasets:

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

Divide el conjunto de `train` (entrenamiento) en un conjunto de entrenamiento y prueba mas pequeño con el método [`~datasets.Dataset.train_test_split`]. De esta forma, tendrás la oportunidad para experimentar y asegúrate de que todo funcióne antes de invertir más tiempo entrenando con el dataset entero.

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

Ahora échale un vistazo al dataset:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 450
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 113
    })
})
```

Aunque el dataset contiene mucha información útil, como los campos `land_id` (identificador del lenguaje) y `english_transcription` (transcripción al inglés), en esta guía nos enfocaremos en los campos `audio` y `intent_class` (clase de intención). Puedes quitar las otras columnas con cel método [`~datasets.Dataset.remove_columns`]:

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

Aquí está un ejemplo:

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

Hay dos campos:

- `audio`: un `array` (arreglo) unidimensional de la señal de voz que se obtiene al cargar y volver a muestrear el archivo de audio.
- `intent_class`: representa el identificador de la clase de la intención del hablante.

Crea un diccionario que asigne el nombre de la etiqueta a un número entero y viceversa para facilitar la obtención del nombre de la etiqueta a partir de su identificador.

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

Ahora puedes convertir el identificador de la etiqueta a un nombre de etiqueta:

```py
>>> id2label[str(2)]
'app_error'
```

## Preprocesamiento

Seguidamente carga el feature extractor (función de extracción de características) de Wav2Vec para procesar la señal de audio:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

El dataset MInDS-14 tiene una tasa de muestreo de 8kHz (puedes encontrar esta información en su [tarjeta de dataset](https://huggingface.co/datasets/PolyAI/minds14)), lo que significa que tendrás que volver a muestrear el dataset a 16kHZ para poder usar el modelo Wav2Vec2 preentranado:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

Ahora vamos a crear una función de preprocesamiento:

1. Invoque la columna `audio` para cargar, y si es necesario, volver a muestrear al archivo de audio.
2. Comprueba si la frecuencia de muestreo del archivo de audio coincide con la frecuencia de muestreo de los datos de audio con los que se entrenó previamente el modelo. Puedes encontrar esta información en la [tarjeta de modelo](https://huggingface.co/facebook/wav2vec2-base) de Wav2Vec2.
3. Establece una longitud de entrada máxima para agrupar entradas más largas sin truncarlas.

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

Para aplicar la función de preprocesamiento a todo el dataset, puedes usar la función [`~datasets.Dataset.map`] de 🤗 Datasets. Acelera la función `map` haciendo `batched=True` para procesar varios elementos del dataset a la vez. Quitas las columnas que no necesites con el método `[~datasets.Dataset.remove_columns]` y cambia el nombre de `intent_class` a `label`, como requiere el modelo.

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

## Evaluación
A menudo es útil incluir una métrica durante el entrenamiento para evaluar el rendimiento de tu modelo. Puedes cargar un método de evaluación rapidamente con la biblioteca de 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). Para esta tarea, puedes usar la métrica de [exactitud](https://huggingface.co/spaces/evaluate-metric/accuracy) (accuracy). Puedes ver la [guía rápida](https://huggingface.co/docs/evaluate/a_quick_tour) de 🤗 Evaluate para aprender más de cómo cargar y computar una métrica:

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

Ahora crea una función que le pase tus predicciones y etiquetas a [`~evaluate.EvaluationModule.compute`] para calcular la exactitud:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

Ahora tu función `compute_metrics` (computar métricas) está lista y podrás usarla cuando estés preparando tu entrenamiento.

## Entrenamiento

<Tip>

¡Si no tienes experiencia haciéndo *fine-tuning* a un modelo con el [`Trainer`], échale un vistazo al tutorial básico [aquí](../training#train-with-pytorch-trainer)!

</Tip>

¡Ya puedes empezar a entrenar tu modelo! Carga Wav2Vec2 con [`AutoModelForAudioClassification`] junto con el especifica el número de etiquetas, y pasa al modelo los *mappings* entre el número entero de etiqueta y la clase de etiqueta.

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

Al llegar a este punto, solo quedan tres pasos:

1. Define tus hiperparámetros de entrenamiento en [`TrainingArguments`]. El único parámetro obligatorio es `output_dir` (carpeta de salida), el cual especifica dónde guardar tu modelo. Puedes subir este modelo al Hub haciendo `push_to_hub=True` (debes haber iniciado sesión en Hugging Face para subir tu modelo). Al final de cada época, el [`Trainer`] evaluará la exactitud y guardará el punto de control del entrenamiento.
2. Pásale los argumentos del entrenamiento al [`Trainer`] junto con el modelo, el dataset, el tokenizer, el data collator y la función `compute_metrics`.
3. Llama el método [`~Trainer.train`] para hacerle fine-tuning a tu modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_steps=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Una vez que el entrenamiento haya sido completado, comparte tu modelo en el Hub con el método [`~transformers.Trainer.push_to_hub`] para que todo el mundo puede usar tu modelo.

```py
>>> trainer.push_to_hub()
```

<Tip>

Para ver un ejemplo más detallado de comó hacerle fine-tuning a un modelo para clasificación, échale un vistazo al correspondiente [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).

</Tip>

## Inference

¡Genial, ahora que le has hecho *fine-tuned* a un modelo, puedes usarlo para hacer inferencia!

Carga el archivo de audio para hacer inferencia. Recuerda volver a muestrear la tasa de muestreo del archivo de audio para que sea la misma del modelo si es necesario.

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

La manera más simple de probar tu modelo para hacer inferencia es usarlo en un [`pipeline`]. Puedes instanciar un `pipeline` para clasificación de audio con tu modelo y pasarle tu archivo de audio:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model")
>>> classifier(audio_file)
[
    {'score': 0.09766869246959686, 'label': 'cash_deposit'},
    {'score': 0.07998877018690109, 'label': 'app_error'},
    {'score': 0.0781070664525032, 'label': 'joint_account'},
    {'score': 0.07667109370231628, 'label': 'pay_bill'},
    {'score': 0.0755252093076706, 'label': 'balance'}
]
```

También puedes replicar de forma manual los resultados del `pipeline` si lo deseas:

Carga el feature extractor para preprocesar el archivo de audio y devuelve el `input` como un tensor de PyTorch:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

Pásale tus entradas al modelo y devuelve los logits:

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

Obtén los identificadores de los clases con mayor probabilidad y usa el *mapping* `id2label` del modelo para convertirle a una etiqueta:

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
