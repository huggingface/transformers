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

# Pipelines para inferencia

Un [`pipeline`] simplifica el uso de cualquier modelo del [Model Hub](https://huggingface.co/models) para la inferencia en una variedad de tareas como la generación de texto, la segmentación de imágenes y la clasificación de audio. Incluso si no tienes experiencia con una modalidad específica o no comprendes el código que alimenta los modelos, ¡aún puedes usarlos con el [`pipeline`]! Este tutorial te enseñará a:

* Utilizar un [`pipeline`] para inferencia.
* Utilizar un tokenizador o modelo específico.
* Utilizar un [`pipeline`] para tareas de audio y visión.

<Tip>

Echa un vistazo a la documentación de [`pipeline`] para obtener una lista completa de tareas admitidas.

</Tip>

## Uso del pipeline

Si bien cada tarea tiene un [`pipeline`] asociado, es más sencillo usar la abstracción general [`pipeline`] que contiene todos los pipelines de tareas específicas. El [`pipeline`] carga automáticamente un modelo predeterminado y un tokenizador con capacidad de inferencia para tu tarea.

1. Comienza creando un [`pipeline`] y específica una tarea de inferencia:

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="text-generation")
```

2. Pasa tu texto de entrada al [`pipeline`]:

```py
>>> generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Iron-priests at the door to the east, and thirteen for the Lord Kings at the end of the mountain'}]
```

Si tienes más de una entrada, pásala como una lista:

```py
>>> generator(
...     [
...         "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
...         "Nine for Mortal Men, doomed to die, One for the Dark Lord on his dark throne",
...     ]
... )
```

Cualquier parámetro adicional para tu tarea también se puede incluir en el [`pipeline`]. La tarea `text-generation` tiene un método [`~generation.GenerationMixin.generate`] con varios parámetros para controlar la salida. Por ejemplo, si deseas generar más de una salida, defínelo en el parámetro `num_return_sequences`:

```py
>>> generator(
...     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
...     num_return_sequences=2,
... )
```

### Selecciona un modelo y un tokenizador

El [`pipeline`] acepta cualquier modelo del [Model Hub](https://huggingface.co/models). Hay etiquetas en el Model Hub que te permiten filtrar por el modelo que te gustaría utilizar para tu tarea. Una vez que hayas elegido un modelo apropiado, cárgalo con la clase `AutoModelFor` y [`AutoTokenizer`] correspondientes. Por ejemplo, carga la clase [`AutoModelForCausalLM`] para una tarea de modelado de lenguaje causal:

```py
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

Crea un [`pipeline`] para tu tarea y específica el modelo y el tokenizador que cargaste:

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
```

Pasa tu texto de entrada a [`pipeline`] para generar algo de texto:

```py
>>> generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Dragon-lords (for them to rule in a world ruled by their rulers, and all who live within the realm'}]
```

## Pipeline de audio

La flexibilidad de [`pipeline`] significa que también se puede extender a tareas de audio.

Por ejemplo, clasifiquemos la emoción de un breve fragmento del famoso discurso de John F. Kennedy ["We choose to go to the Moon"](https://en.wikipedia.org/wiki/We_choose_to_go_to_the_Moon). Encuentra un modelo de [audio classification](https://huggingface.co/models?pipeline_tag=audio-classification) para reconocimiento de emociones en el Model Hub y cárgalo en el [`pipeline`]:

```py
>>> from transformers import pipeline

>>> audio_classifier = pipeline(
...     task="audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

Pasa el archivo de audio al [`pipeline`]:

```py
>>> audio_classifier("jfk_moon_speech.wav")
[{'label': 'calm', 'score': 0.13856211304664612},
 {'label': 'disgust', 'score': 0.13148026168346405},
 {'label': 'happy', 'score': 0.12635163962841034},
 {'label': 'angry', 'score': 0.12439591437578201},
 {'label': 'fearful', 'score': 0.12404385954141617}]
```

## Pipeline de visión

Finalmente, utilizar un [`pipeline`] para tareas de visión es prácticamente igual.

Específica tu tarea de visión y pasa tu imagen al clasificador. La imagen puede ser un enlace o una ruta local a la imagen. Por ejemplo, ¿qué especie de gato se muestra a continuación?

![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

```py
>>> from transformers import pipeline

>>> vision_classifier = pipeline(task="image-classification")
>>> vision_classifier(
...     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
[{'label': 'lynx, catamount', 'score': 0.4403027892112732},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.03433405980467796},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.032148055732250214},
 {'label': 'Egyptian cat', 'score': 0.02353910356760025},
 {'label': 'tiger cat', 'score': 0.023034192621707916}]
```
