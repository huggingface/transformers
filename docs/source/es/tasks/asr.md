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

# Reconocimiento automático del habla

<Youtube id="TksaY_FDgnk"/>

El reconocimiento automático del habla (ASR, por sus siglas en inglés) convierte una señal de habla en texto y mapea una secuencia de entradas de audio en salidas en forma de texto. Los asistentes virtuales como Siri y Alexa usan modelos de ASR para ayudar a sus usuarios todos los días. De igual forma, hay muchas otras aplicaciones, como la transcripción de contenidos en vivo y la toma automática de notas durante reuniones.

En esta guía te mostraremos como:

1. Hacer fine-tuning al modelo [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) con el dataset [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) para transcribir audio a texto.
2. Usar tu modelo ajustado para tareas de inferencia.

<Tip>

Revisa la [página de la tarea](https://huggingface.co/tasks/automatic-speech-recognition) de reconocimiento automático del habla para acceder a más información sobre los modelos, datasets y métricas asociados.

</Tip>

Antes de comenzar, asegúrate de haber instalado todas las librerías necesarias:

```bash
pip install transformers datasets evaluate jiwer
```

Te aconsejamos iniciar sesión con tu cuenta de Hugging Face para que puedas subir tu modelo y comartirlo con la comunidad. Cuando te sea solicitado, ingresa tu token para iniciar sesión:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Cargar el dataset MInDS-14

Comencemos cargando un subconjunto más pequeño del dataset [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) desde la biblioteca 🤗 Datasets. De esta forma, tendrás la oportunidad de experimentar y asegurarte de que todo funcione antes de invertir más tiempo entrenando con el dataset entero.

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```
Divide la partición `train` (entrenamiento) en una partición de entrenamiento y una de prueba usando el método [`~Dataset.train_test_split`]:

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

Ahora échale un vistazo al dataset:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

Aunque el dataset contiene mucha información útil, como los campos `lang_id` (identificador del lenguaje) y `english_transcription` (transcripción al inglés), en esta guía nos enfocaremos en los campos `audio` y `transcription`. Puedes quitar las otras columnas con el método [`~datasets.Dataset.remove_columns`]:

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

Vuelve a echarle un vistazo al ejemplo:

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

Hay dos campos:

- `audio`: un `array` (arreglo) unidimensional de la señal de habla que debe ser invocado para cargar y re-muestrear el archivo de audio.
- `transcription`: el texto objetivo.

## Preprocesamiento

El siguiente paso es cargar un procesador Wav2Vec2 para procesar la señal de audio:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```
El dataset MInDS-14 tiene una tasa de muestreo de 8000kHz (puedes encontrar esta información en su [tarjeta de dataset](https://huggingface.co/datasets/PolyAI/minds14)), lo que significa que tendrás que re-muestrear el dataset a 16000kHz para poder usar el modelo Wav2Vec2 pre-entrenado:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

Como puedes ver en el campo `transcription`, el texto contiene una mezcla de carácteres en mayúsculas y en minúsculas. El tokenizer Wav2Vec2 fue entrenado únicamente con carácteres en mayúsculas, así que tendrás que asegurarte de que el texto se ajuste al vocabulario del tokenizer:

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

Ahora vamos a crear una función de preprocesamiento que:

1. Invoque la columna `audio` para cargar y re-muestrear el archivo de audio.
2. Extraiga el campo `input_values` (valores de entrada) del archivo de audio y haga la tokenización de la columna `transcription` con el procesador.

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

Para aplicar la función de preprocesamiento a todo el dataset, puedes usar la función [`~datasets.Dataset.map`] de 🤗 Datasets. Para acelerar la función `map` puedes incrementar el número de procesos con el parámetro `num_proc`. Quita las columnas que no necesites con el método [`~datasets.Dataset.remove_columns`]:

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

🤗 Transformers no tiene un collator de datos para la tarea de ASR, así que tendrás que adaptar el [`DataCollatorWithPadding`] para crear un lote de ejemplos. El collator también le aplicará padding dinámico a tu texto y etiquetas para que tengan la longitud del elemento más largo en su lote (en vez de la mayor longitud en el dataset entero), de forma que todas las muestras tengan una longitud uniforme. Aunque es posible hacerle padding a tu texto con el `tokenizer` haciendo `padding=True`, el padding dinámico es más eficiente.

A diferencia de otros collators de datos, este tiene que aplicarle un método de padding distinto a los campos `input_values` (valores de entrada) y `labels` (etiquetas):

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         # particiona las entradas y las etiquetas ya que tienen que tener longitudes distintas y
...         # requieren métodos de padding diferentes
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # remplaza el padding con -100 para ignorar la pérdida de forma correcta
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

Ahora puedes instanciar tu `DataCollatorForCTCWithPadding`:

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## Evaluación

A menudo es útil incluir una métrica durante el entrenamiento para evaluar el rendimiento de tu modelo. Puedes cargar un método de evaluación rápidamente con la biblioteca 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). Para esta tarea, puedes usar la métrica de [tasa de error por palabra](https://huggingface.co/spaces/evaluate-metric/wer) (WER, por sus siglas en inglés). Puedes ver la [guía rápida](https://huggingface.co/docs/evaluate/a_quick_tour) de 🤗 Evaluate para aprender más acerca de cómo cargar y computar una métrica.

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

Ahora crea una función que le pase tus predicciones y etiquetas a [`~evaluate.EvaluationModule.compute`] para calcular la WER:

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

Ahora tu función `compute_metrics` (computar métricas) está lista y podrás usarla cuando estés preparando tu entrenamiento.

## Entrenamiento

<frameworkcontent>
<pt>
<Tip>

Si no tienes experiencia haciéndole fine-tuning a un modelo con el [`Trainer`], ¡échale un vistazo al tutorial básico [aquí](../training#train-with-pytorch-trainer)!

</Tip>

¡Ya puedes empezar a entrenar tu modelo! Para ello, carga Wav2Vec2 con [`AutoModelForCTC`]. Especifica la reducción que quieres aplicar con el parámetro `ctc_loss_reduction`. A menudo, es mejor usar el promedio en lugar de la sumatoria que se hace por defecto.

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```
En este punto, solo quedan tres pasos:

1. Define tus hiperparámetros de entrenamiento en [`TrainingArguments`]. El único parámetro obligatorio es `output_dir` (carpeta de salida), el cual especifica dónde guardar tu modelo. Puedes subir este modelo al Hub haciendo `push_to_hub=True` (debes haber iniciado sesión en Hugging Face para subir tu modelo). Al final de cada época, el [`Trainer`] evaluará la WER y guardará el punto de control del entrenamiento.
2. Pásale los argumentos del entrenamiento al [`Trainer`] junto con el modelo, el dataset, el tokenizer, el collator de datos y la función `compute_metrics`.
3. Llama el método [`~Trainer.train`] para hacerle fine-tuning a tu modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     group_by_length=True,
...     evaluation_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     tokenizer=processor.feature_extractor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Una vez que el entrenamiento haya sido completado, comparte tu modelo en el Hub con el método [`~transformers.Trainer.push_to_hub`] para que todo el mundo pueda usar tu modelo:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

Para ver un ejemplo más detallado de cómo hacerle fine-tuning a un modelo para reconocimiento automático del habla, échale un vistazo a esta [entrada de blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) para ASR en inglés y a esta [entrada](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) para ASR multilingüe.

</Tip>

## Inferencia

¡Genial, ahora que le has hecho fine-tuning a un modelo, puedes usarlo para inferencia!

Carga el archivo de audio sobre el cual quieras correr la inferencia. ¡Recuerda re-muestrar la tasa de muestreo del archivo de audio para que sea la misma del modelo si es necesario!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

La manera más simple de probar tu modelo para hacer inferencia es usarlo en un [`pipeline`]. Puedes instanciar un `pipeline` para reconocimiento automático del habla con tu modelo y pasarle tu archivo de audio:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

<Tip>

La transcripción es decente, pero podría ser mejor. ¡Intenta hacerle fine-tuning a tu modelo con más ejemplos para obtener resultados aún mejores!

</Tip>

También puedes replicar de forma manual los resultados del `pipeline` si lo deseas:

<frameworkcontent>
<pt>
Carga un procesador para preprocesar el archivo de audio y la transcripción y devuelve el `input` como un tensor de PyTorch:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

Pásale tus entradas al modelo y devuelve los logits:

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

Obtén los identificadores de los tokens con mayor probabilidad en las predicciones y usa el procesador para decodificarlos y transformarlos en texto:

```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```
</pt>
</frameworkcontent>
