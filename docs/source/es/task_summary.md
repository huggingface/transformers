<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Lo que 🤗 Transformers puede hacer

🤗 Transformers es una biblioteca de modelos preentrenados de última generación para procesamiento del lenguaje natural (NLP, por sus siglas en inglés), visión por computadora y tareas de procesamiento de audio y voz. No solo contiene modelos Transformer, sino también modelos no Transformer como redes convolucionales modernas para tareas de visión por computadora. Si observas algunos de los productos de consumo más populares hoy en día, como teléfonos inteligentes, aplicaciones y televisores, es probable que haya alguna tecnología de aprendizaje profundo detrás. ¿Quieres quitar un objeto de fondo de una foto tomada por tu teléfono inteligente? Este es un ejemplo de una tarea de segmentación panóptica (no te preocupes si aún no sabes qué significa, ¡lo describiremos en las siguientes secciones!).

Esta página proporciona una descripción general de las diferentes tareas de procesamiento de audio y voz, visión por computadora y NLP que se pueden resolver con la biblioteca 🤗 Transformers en solo tres líneas de código.

## Audio

Las tareas de procesamiento de audio y voz son un poco diferentes de las otras modalidades principalmente porque el audio como entrada es una señal continua. A diferencia del texto, una forma de onda de audio cruda no se puede dividir ordenadamente en fragmentos discretos de la misma manera en que una oración puede dividirse en palabras. Para superar esto, la señal de audio cruda generalmente se muestrea a intervalos regulares. Si tomas más muestras dentro de un intervalo, la tasa de muestreo es mayor y el audio se asemeja más a la fuente de audio original.

Enfoques anteriores preprocesaban el audio para extraer características útiles. Ahora es más común comenzar las tareas de procesamiento de audio y voz alimentando directamente la forma de onda de audio cruda a un codificador de características para extraer una representación de audio. Esto simplifica el paso de preprocesamiento y permite que el modelo aprenda las características más esenciales.

### Clasificación de audio

La clasificación de audio es una tarea que etiqueta datos de audio con un conjunto predefinido de clases. Es una categoría amplia con muchas aplicaciones específicas, algunas de las cuales incluyen:

* clasificación de escena acústica: etiquetar audio con una etiqueta de escena ("oficina", "playa", "estadio")
* detección de eventos acústicos: etiquetar audio con una etiqueta de evento de sonido ("bocina de automóvil", "llamada de ballena", "cristal rompiéndose")
* etiquetado: etiquetar audio que contiene varios sonidos (canto de pájaros, identificación de altavoces en una reunión)
* clasificación de música: etiquetar música con una etiqueta de género ("metal", "hip-hop", "country")

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### Reconocimiento automático del habla

El reconocimiento automático del habla (ASR, por sus siglas en inglés) transcribe el habla a texto. Es una de las tareas de audio más comunes, en parte debido a que el habla es una forma natural de comunicación humana. Hoy en día, los sistemas ASR están integrados en productos de tecnología "inteligente" como altavoces, teléfonos y automóviles. Podemos pedirle a nuestros asistentes virtuales que reproduzcan música, establezcan recordatorios y nos informen sobre el clima.

Pero uno de los desafíos clave que las arquitecturas Transformer han ayudado a superar es en los idiomas con recursos limitados. Al preentrenar con grandes cantidades de datos de habla, afinar el modelo solo con una hora de datos de habla etiquetados en un idioma con recursos limitados aún puede producir resultados de alta calidad en comparación con los sistemas ASR anteriores entrenados con 100 veces más datos etiquetados.

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## Visión por computadora

Una de las primeras y exitosas tareas de visión por computadora fue reconocer imágenes de números de código postal utilizando una [red neuronal convolucional](glossary#convolution) (CNN, por sus siglas en inglés). Una imagen está compuesta por píxeles, y cada píxel tiene un valor numérico. Esto facilita representar una imagen como una matriz de valores de píxeles. Cada combinación particular de valores de píxeles describe los colores de una imagen.

Dos formas generales en las que se pueden resolver las tareas de visión por computadora son:

1. Utilizar convoluciones para aprender las características jerárquicas de una imagen, desde características de bajo nivel hasta cosas abstractas de alto nivel.
2. Dividir una imagen en parches y utilizar un Transformer para aprender gradualmente cómo cada parche de imagen se relaciona entre sí para formar una imagen. A diferencia del enfoque ascendente preferido por una CNN, esto es como comenzar con una imagen borrosa y luego enfocarla gradualmente.

### Clasificación de imágenes

La clasificación de imágenes etiqueta una imagen completa con un conjunto predefinido de clases. Como la mayoría de las tareas de clasificación, hay muchos casos prácticos para la clasificación de imágenes, algunos de los cuales incluyen:

* salud: etiquetar imágenes médicas para detectar enfermedades o monitorear la salud del paciente
* medio ambiente: etiquetar imágenes de satélite para monitorear la deforestación, informar la gestión de áreas silvestres o detectar incendios forestales
* agricultura: etiquetar imágenes de cultivos para monitorear la salud de las plantas o imágenes de satélite para el monitoreo del uso del suelo
* ecología: etiquetar imágenes de especies animales o vegetales para monitorear poblaciones de vida silvestre o rastrear especies en peligro de extinción

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### Detección de objetos

A diferencia de la clasificación de imágenes, la detección de objetos identifica múltiples objetos dentro de una imagen y las posiciones de los objetos en la imagen (definidas por el cuadro delimitador). Algunas aplicaciones ejemplares de la detección de objetos incluyen:

* vehículos autónomos: detectar objetos de tráfico cotidianos como otros vehículos, peatones y semáforos
* teledetección: monitoreo de desastres, planificación urbana y pronóstico del tiempo
* detección de defectos: detectar grietas o daños estructurales en edificios y defectos de fabricación

```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### Segmentación de imágenes

La segmentación de imágenes es una tarea a nivel de píxeles que asigna cada píxel en una imagen a una clase. A diferencia de la detección de objetos, que utiliza cuadros delimitadores para etiquetar y predecir objetos en una imagen, la segmentación es más granular. La segmentación puede detectar objetos a nivel de píxeles. Hay varios tipos de segmentación de imágenes:

* segmentación de instancias: además de etiquetar la clase de un objeto, también etiqueta cada instancia distinta de un objeto ("perro-1", "perro-2")
* segmentación panóptica: una combinación de segmentación semántica y de instancias; etiqueta cada píxel con una clase semántica **y** cada instancia distinta de un objeto

Las tareas de segmentación son útiles en vehículos autónomos para crear un mapa a nivel de píxeles del mundo que los rodea para que puedan navegar de manera segura alrededor de peatones y otros vehículos. También es útil en imágenes médicas, donde la mayor granularidad de la tarea puede ayudar a identificar células anormales o características de órganos. La segmentación de imágenes también se puede utilizar en comercio electrónico para probar virtualmente la ropa o crear experiencias de realidad aumentada superponiendo objetos en el mundo real a través de tu cámara.

```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### Estimación de profundidad

La estimación de profundidad predice la distancia de cada píxel en una imagen desde la cámara. Esta tarea de visión por computadora es especialmente importante para la comprensión y reconstrucción de escenas. Por ejemplo, en los vehículos autónomos, es necesario entender qué tan lejos están los objetos como peatones, señales de tráfico y otros vehículos para evitar obstáculos y colisiones. La información de profundidad también es útil para construir representaciones 3D a partir de imágenes 2D y se puede utilizar para crear representaciones 3D de alta calidad de estructuras biológicas o edificios.

Hay dos enfoques para la estimación de profundidad:

* estéreo: las profundidades se estiman comparando dos imágenes de la misma escena desde ángulos ligeramente diferentes
* monocular: las profundidades se estiman a partir de una sola imagen

```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```

## Procesamiento del lenguaje natural

Las tareas de procesamiento del lenguaje natural (NLP, por sus siglas en inglés) están entre los tipos de tareas más comunes porque el texto es una forma natural de comunicación para nosotros. Para convertir el texto en un formato reconocido por un modelo, es necesario tokenizarlo. Esto significa dividir una secuencia de texto en palabras o subpalabras separadas (tokens) y luego convertir estos tokens en números. Como resultado, puedes representar una secuencia de texto como una secuencia de números, y una vez que tienes una secuencia de números, se puede ingresar a un modelo para resolver todo tipo de tareas de NLP.

### Clasificación de texto

Al igual que las tareas de clasificación en cualquier modalidad, la clasificación de texto etiqueta una secuencia de texto (puede ser a nivel de oración, párrafo o documento) de un conjunto predefinido de clases. Hay muchas aplicaciones prácticas para la clasificación de texto, algunas de las cuales incluyen:

* análisis de sentimientos: etiquetar texto según alguna polaridad como `positivo` o `negativo`, lo que puede informar y respaldar la toma de decisiones en campos como política, finanzas y marketing
* clasificación de contenido: etiquetar texto según algún tema para ayudar a organizar y filtrar información en noticias y feeds de redes sociales (`clima`, `deportes`, `finanzas`, etc.)

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### Clasificación de tokens

En cualquier tarea de NLP, el texto se procesa separando la secuencia de texto en palabras o subpalabras individuales. Estas se conocen como [tokens](glossary#token). La clasificación de tokens asigna a cada token una etiqueta de un conjunto predefinido de clases.

Dos tipos comunes de clasificación de tokens son:

* reconocimiento de entidades nombradas (NER, por sus siglas en inglés): etiquetar un token según una categoría de entidad como organización, persona, ubicación o fecha. NER es especialmente popular en entornos biomédicos, donde puede etiquetar genes, proteínas y nombres de medicamentos
* etiquetado de partes del discurso (POS, por sus siglas en inglés): etiquetar un token según su parte del discurso, como sustantivo, verbo o adjetivo. POS es útil para ayudar a los sistemas de traducción a comprender cómo dos palabras idénticas son gramaticalmente diferentes (por ejemplo, "corte" como sustantivo versus "corte" como verbo)

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

### Respuestas a preguntas

Responder preguntas es otra tarea a nivel de tokens que devuelve una respuesta a una pregunta, a veces con contexto (dominio abierto) y otras veces sin contexto (dominio cerrado). Esta tarea ocurre cuando le preguntamos algo a un asistente virtual, como si un restaurante está abierto. También puede proporcionar soporte al cliente o técnico y ayudar a los motores de búsqueda a recuperar la información relevante que estás buscando.

Hay dos tipos comunes de respuestas a preguntas:

* extractivas: dada una pregunta y algún contexto, la respuesta es un fragmento de texto del contexto que el modelo debe extraer
* abstractivas: dada una pregunta y algún contexto, la respuesta se genera a partir del contexto; este enfoque lo maneja la [`Text2TextGenerationPipeline`] en lugar del [`QuestionAnsweringPipeline`] que se muestra a continuación

```py
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="What is the name of the repository?",
...     context="The name of the repository is huggingface/transformers",
... )
>>> print(
...     f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
... )
score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

### Resumir

Al resumir se crea una versión más corta de un texto más largo mientras intenta preservar la mayor parte del significado del documento original. Resumir es una tarea de secuencia a secuencia; produce una secuencia de texto más corta que la entrada. Hay muchos documentos de formato largo que se pueden resumir para ayudar a los lectores a comprender rápidamente los puntos principales. Proyectos de ley legislativos, documentos legales y financieros, patentes y artículos científicos son algunos ejemplos de documentos que podrían resumirse para ahorrar tiempo a los lectores y servir como ayuda para la lectura.

Al igual que en las respuestas a preguntas, hay dos tipos de resumen:

* extractiva: identifica y extrae las oraciones más importantes del texto original
* abstractiva: genera el resumen objetivo (que puede incluir nuevas palabras no presentes en el documento de entrada) a partir del texto original; el [`SummarizationPipeline`] utiliza el enfoque abstractivo

```py
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
... )
[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

### Traducción

La traducción convierte una secuencia de texto en un idioma a otro. Es importante para ayudar a personas de diferentes orígenes a comunicarse entre sí, traducir contenido para llegar a audiencias más amplias e incluso ser una herramienta de aprendizaje para ayudar a las personas a aprender un nuevo idioma. Al igual que resumir, la traducción es una tarea de secuencia a secuencia, lo que significa que el modelo recibe una secuencia de entrada y devuelve una secuencia de salida objetivo.

En sus primeros días, los modelos de traducción eran principalmente monolingües, pero recientemente ha habido un creciente interés en modelos multilingües que pueden traducir entre muchas combinaciones de idiomas.

```py
>>> from transformers import pipeline

>>> text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
>>> translator = pipeline(task="translation", model="t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

### Modelado de lenguaje

El modelado de lenguaje es una tarea que predice una palabra en una secuencia de texto. Se ha vuelto una tarea de NLP muy popular porque un modelo de lenguaje preentrenado puede ser afinado para muchas otras tareas secundarias. Últimamente, ha habido mucho interés en modelos de lenguaje grandes (LLM, por sus siglas en inglés) que demuestran aprendizaje de cero o con pocas muestras (zero- or few-shot learning). ¡Esto significa que el modelo puede resolver tareas para las cuales no fue entrenado explícitamente! Los modelos de lenguaje se pueden utilizar para generar texto fluido y convincente, aunque debes tener cuidado, ya que el texto no siempre puede ser preciso.

Hay dos tipos de modelado de lenguaje:

* causal: el objetivo del modelo es predecir el próximo token en una secuencia, y los tokens futuros están enmascarados

    ```py
    >>> from transformers import pipeline

    >>> prompt = "Hugging Face is a community-based open-source platform for machine learning."
    >>> generator = pipeline(task="text-generation")
    >>> generator(prompt)  # doctest: +SKIP
    ```

* enmascarado: el objetivo del modelo es predecir un token enmascarado en una secuencia con acceso completo a los tokens en la secuencia

    ```py
    >>> text = "Hugging Face is a community-based open-source <mask> for machine learning."
    >>> fill_mask = pipeline(task="fill-mask")
    >>> preds = fill_mask(text, top_k=1)
    >>> preds = [
    ...     {
    ...         "score": round(pred["score"], 4),
    ...         "token": pred["token"],
    ...         "token_str": pred["token_str"],
    ...         "sequence": pred["sequence"],
    ...     }
    ...     for pred in preds
    ... ]
    >>> preds
    [{'score': 0.2236,
      'token': 1761,
      'token_str': ' platform',
      'sequence': 'Hugging Face is a community-based open-source platform for machine learning.'}]
    ```

## Multimodal

Las tareas multimodales requieren que un modelo procese múltiples modalidades de datos (texto, imagen, audio, video) para resolver un problema particular. La descripción de imágenes es un ejemplo de una tarea multimodal en la que el modelo toma una imagen como entrada y produce una secuencia de texto que describe la imagen o algunas propiedades de la imagen.

Aunque los modelos multimodales trabajan con diferentes tipos de datos o modalidades, internamente, los pasos de preprocesamiento ayudan al modelo a convertir todos los tipos de datos en embeddings (vectores o listas de números que contienen información significativa sobre los datos). Para una tarea como la descripción de imágenes, el modelo aprende las relaciones entre los embeddings de imágenes y los embeddings de texto.

### Respuestas a preguntas de documentos

Las respuestas a preguntas de documentos es una tarea que responde preguntas en lenguaje natural a partir de un documento. A diferencia de una tarea de respuestas a preguntas a nivel de token que toma texto como entrada, las respuestas a preguntas de documentos toman una imagen de un documento como entrada junto con una pregunta sobre el documento y devuelven una respuesta. Las respuestas a preguntas de documentos pueden usarse para analizar documentos estructurados y extraer información clave de ellos. En el ejemplo a continuación, el monto total y el cambio debido se pueden extraer de un recibo.

```py
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/2/image/image.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
>>> preds = doc_question_answerer(
...     question="What is the total amount?",
...     image=image,
... )
>>> preds
[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

Con suerte, esta página te ha proporcionado más información de fondo sobre todos los tipos de tareas en cada modalidad y la importancia práctica de cada una. En la próxima [sección](tasks_explained), aprenderás **cómo** 🤗 Transformers trabaja para resolver estas tareas.
