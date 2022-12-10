<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
-->

# ¿Cómo puedo crear un pipeline personalizado?

En esta guía, veremos cómo crear un pipeline personalizado y cómo compartirlo en el [Hub](hf.co/models) o añadirlo
a la biblioteca 🤗 Transformers.

En primer lugar, debes decidir las entradas que tu pipeline podrá recibir. Pueden ser strings, bytes,
diccionarios o lo que te parezca que vaya a ser la entrada más apropiada. Intenta mantener estas entradas en un
formato que sea tan Python puro como sea posible, puesto que esto facilita la compatibilidad (incluso con otros
lenguajes de programación por medio de JSON). Estos serán los `inputs` (entradas) del pipeline (`preprocess`).

Ahora debes definir los `outputs` (salidas). Al igual que con los `inputs`, entre más simple el formato, mejor.
Estas serán las salidas del método `postprocess` (posprocesamiento).

Empieza heredando la clase base `Pipeline` con los 4 métodos que debemos implementar: `preprocess` (preprocesamiento),
`_forward` (ejecución), `postprocess` (posprocesamiento) y `_sanitize_parameters` (verificar parámetros).

```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Quizá {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

La estructura de este desglose es así para garantizar una compatibilidad más o menos transparente con el uso de
CPU/GPU y el pre/posprocesamiento en CPU en varios hilos.

`preprocess` tomará las entradas definidas originalmente y las convertirá en algo que se le pueda pasar al modelo.
Podría contener más información y a menudo es un objeto `Dict` (diccionario).

`_forward` contiene los detalles de la implementación y no debería ser invocado de forma directa. `forward` es el
método preferido a utilizar pues contiene verificaciones para asegurar que todo funcione en el dispositivo correcto.
Cualquier cosa que esté relacionada con un modelo real debería ir en el método `_forward`, todo lo demás va en
los métodos de preprocesamiento y posprocesamiento.

Los métodos `postprocess` reciben la salida `_forward` y la convierten en la salida final que decidimos
anteriormente.

`_sanitize_parameters` existe para permitir a los usuarios pasar cualesquiera parámetros cuando lo deseen, ya
sea al momento de inicializar el pipeline `pipeline(...., maybe_arg=4)` o al momento de invocarlo
`pipe = pipeline(...); output = pipe(...., maybe_arg=4)`.


El método `_sanitize_parameters` devuelve 3 diccionarios de kwargs que serán pasados directamente a `preprocess`,
`_forward` y `postprocess`. No ingreses nada si el caller no se va a invocar con parámetros adicionales.
Esto permite mantener los parámetros por defecto de la definición de la función, lo que es más "natural".

Un ejemplo clásico sería un argumento `top_k` en el posprocesamiento de una tarea de clasificación.

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

Para lograrlo, actualizaremos nuestro método `postprocess` con un valor por defecto de `5` y  modificaremos
`_sanitize_parameters` para permitir este nuevo parámetro.


```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Añade la lógica para manejar el top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

Intenta que las entradas y salidas sean muy simples e, idealmente, que puedan serializarse como JSON, pues esto
hace el uso del pipeline muy sencillo sin que el usuario tenga que preocuparse por conocer nuevos tipos de objetos.
También es relativamente común tener compatibilidad con muchos tipos diferentes de argumentos por facilidad de uso
(por ejemplo, los archivos de audio pueden ser nombres de archivo, URLs o bytes).


## Añadirlo a la lista de tareas

Para registrar tu `new-task` (nueva tarea) en la lista de tareas, debes añadirla al
`PIPELINE_REGISTRY` (registro de pipelines):

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

Puedes especificar un modelo por defecto si lo deseas, en cuyo caso debe venir con una versión específica (que puede ser el nombre de un branch o hash de commit, en este caso usamos `"abcdef"`), así como el tipo:

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # tipo de datos que maneja: texto, audio, imagen, multi-modalidad
)
```

## Comparte tu pipeline en el Hub

Para compartir tu pipeline personalizado en el Hub, solo tienes que guardar el código personalizado de tu sub-clase
`Pipeline` en un archivo de Python. Por ejemplo, digamos que queremos usar un pipeline personalizado para la
clasificación de duplas de oraciones de esta forma:

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

La implementación es independiente del framework y funcionará con modelos de PyTorch y TensorFlow. Si guardamos
esto en un archivo llamado `pair_classification.py`, podemos importarlo y registrarlo de la siguiente manera:

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

Una vez hecho esto, podemos usarlo con un modelo pre-entrenado. Por ejemplo, al modelo `sgugger/finetuned-bert-mrpc`
se le hizo fine-tuning con el dataset MRPC, en el cual se clasifican duplas de oraciones como paráfrasis o no.

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

Ahora podemos compartirlo en el Hub usando el método `save_pretrained` (guardar pre-entrenado) en un `Repository`:

```py
from huggingface_hub import Repository

repo = Repository("test-dynamic-pipeline", clone_from="{your_username}/test-dynamic-pipeline")
classifier.save_pretrained("test-dynamic-pipeline")
repo.push_to_hub()
```

Esto copiará el archivo donde definiste `PairClassificationPipeline` dentro de la carpeta `"test-dynamic-pipeline"`,
y además guardará el modelo y el tokenizer del pipeline, antes de enviar todo al repositorio
`{your_username}/test-dynamic-pipeline`. Después de esto, cualquier persona puede usarlo siempre que usen la opción
`trust_remote_code=True` (confiar en código remoto):

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## Añadir el pipeline a 🤗 Transformers

Si quieres contribuir tu pipeline a la biblioteca 🤗 Transformers, tendrás que añadirlo a un nuevo módulo en el
sub-módulo `pipelines` con el código de tu pipeline. Luego, debes añadirlo a la lista de tareas definidas en
`pipelines/__init__.py`.

A continuación tienes que añadir las pruebas. Crea un nuevo archivo llamado `tests/test_pipelines_MY_PIPELINE.py`
basándote en las pruebas existentes.

La función `run_pipeline_test` será muy genérica y se correrá sobre modelos pequeños escogidos al azar sobre todas las
arquitecturas posibles definidas en `model_mapping` y `tf_model_mapping`.

Esto es muy importante para probar compatibilidades a futuro, lo que significa que si alguien añade un nuevo modelo
para `XXXForQuestionAnswering` entonces el pipeline intentará ejecutarse con ese modelo. Ya que los modelos son aleatorios,
es imposible verificar los valores como tales, y es por eso que hay un helper `ANY` que simplemente intentará que la
salida tenga el mismo tipo que la salida esperada del pipeline.

También *debes* implementar 2 (preferiblemente 4) pruebas:

- `test_small_model_pt` : Define un (1) modelo pequeño para este pipeline (no importa si los resultados no tienen sentido)
y prueba las salidas del pipeline. Los resultados deberían ser los mismos que en `test_small_model_tf`.
- `test_small_model_tf` : Define un (1) modelo pequeño para este pipeline (no importa si los resultados no tienen sentido)
y prueba las salidas del pipeline. Los resultados deberían ser los mismos que en `test_small_model_pt`.
- `test_large_model_pt` (`optional`): Prueba el pipeline en una tarea real en la que los resultados deben tener sentido.
Estas pruebas son lentas y deben marcarse como tales. El objetivo de esto es ejemplificar el pipeline y asegurarse de que
no haya divergencias en versiones futuras.
- `test_large_model_tf` (`optional`): Prueba el pipeline en una tarea real en la que los resultados deben tener sentido.
Estas pruebas son lentas y deben marcarse como tales. El objetivo de esto es ejemplificar el pipeline y asegurarse de que
no haya divergencias en versiones futuras.
