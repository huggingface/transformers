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

# Glosario

Este glosario define términos generales de aprendizaje automático y términos relacionados con 🤗 Transformers para ayudarte a comprender mejor la documentación.

## A

### attention mask

La máscara de atención es un argumento opcional utilizado al agrupar secuencias.

<Youtube id="M6adb1j2jPI"/>

Este argumento indica al modelo qué tokens deben recibir atención y cuáles no.

Por ejemplo, considera estas dos secuencias:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

>>> sequence_a = "This is a short sequence."
>>> sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

>>> encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
>>> encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```

Las versiones codificadas tienen longitudes diferentes:

```python
>>> len(encoded_sequence_a), len(encoded_sequence_b)
(8, 19)
```

Por lo tanto, no podemos colocarlas juntas en el mismo tensor tal cual. La primera secuencia necesita ser rellenada hasta la longitud de la segunda, o la segunda necesita ser truncada hasta la longitud de la primera.

En el primer caso, la lista de IDs se extenderá con los índices de relleno. Podemos pasar una lista al tokenizador y pedirle que realice el relleno de esta manera:

```python
>>> padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
```

Podemos ver que se han agregado ceros a la derecha de la primera oración para que tenga la misma longitud que la segunda:

```python
>>> padded_sequences["input_ids"]
[[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]
```

Esto luego se puede convertir en un tensor en PyTorch o TensorFlow. La máscara de atención es un tensor binario que indica la posición de los índices de relleno para que el modelo no los tenga en cuenta. Para el [`BertTokenizer`], `1` indica un valor al que se debe prestar atención, mientras que `0` indica un valor de relleno. Esta máscara de atención está en el diccionario devuelto por el tokenizador bajo la clave "attention_mask":

```python
>>> padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

### autoencoding models

Consulta [modelos de codificación](#encoder-models) y [modelado de lenguaje enmascarado](#masked-language-modeling-mlm)

### autoregressive models

Consulta [modelado de lenguaje causal](#causal-language-modeling) y [modelos de decodificación](#decoder-models)

## B

### backbone

La columna vertebral, backbone en inglés, es la red (embeddings y layers) que produce los estados ocultos o características crudas. Normalmente, está conectado a una [cabecera](#head), que acepta las características como entrada para hacer una predicción. Por ejemplo, [`ViTModel`] es una columna vertebral sin una cabecera específica encima. Otros modelos también pueden usar [`VitModel`] como columna vertebral, como por ejemplo [DPT](model_doc/dpt).

## C

### causal language modeling

Una tarea de preentrenamiento donde el modelo lee los textos en orden y tiene que predecir la siguiente palabra. Generalmente, se realiza leyendo toda la oración, pero utilizando una máscara dentro del modelo para ocultar los tokens futuros en un cierto paso de tiempo.

### channel

Las imágenes a color están compuestas por alguna combinación de valores en tres canales: rojo, verde y azul (RGB), y las imágenes en escala de grises solo tienen un canal. En 🤗 Transformers, el canal puede ser la primera o última dimensión del tensor de una imagen: [`n_channels`, `height`, `width`] o [`height`, `width`, `n_channels`].

### connectionist temporal classification (CTC)

Un algoritmo que permite que un modelo aprenda sin saber exactamente cómo están alineadas la entrada y la salida; CTC calcula la distribución de todas las salidas posibles para una entrada dada y elige la salida más probable de ella. CTC se utiliza comúnmente en tareas de reconocimiento de voz porque el habla no siempre se alinea perfectamente con la transcripción debido a diversas razones, como las diferentes velocidades de habla de los oradores.

### convolution

Un tipo de capa en una red neuronal donde la matriz de entrada se multiplica elemento por elemento por una matriz más pequeña (núcleo o filtro) y los valores se suman en una nueva matriz. Esto se conoce como una operación de convolución que se repite sobre toda la matriz de entrada. Cada operación se aplica a un segmento diferente de la matriz de entrada. Las redes neuronales convolucionales (CNN) se utilizan comúnmente en visión por computadora.

## D

### DataParallel (DP)

Técnica de paralelismo para entrenamiento en múltiples GPUs donde se replica la misma configuración varias veces, con cada instancia recibiendo una porción de datos única. El procesamiento se realiza en paralelo y todas las configuraciones se sincronizan al final de cada paso de entrenamiento.

Obtén más información sobre cómo funciona el DataParallel [aquí](perf_train_gpu_many#dataparallel-vs-distributeddataparallel).

### decoder input IDs

Esta entrada es específica para modelos codificador-decodificador y contiene los IDs de entrada que se enviarán al decodificador. Estas entradas deben usarse para tareas de secuencia a secuencia, como traducción o resumen, y generalmente se construyen de una manera específica para cada modelo.

La mayoría de los modelos codificador-decodificador (BART, T5) crean sus `decoder_input_ids` por sí mismos a partir de las `labels`. En tales modelos, pasar las `labels` es la forma preferida de manejar el entrenamiento.

Consulta la documentación de cada modelo para ver cómo manejan estos IDs de entrada para el entrenamiento de secuencia a secuencia.

### decoder models

También conocidos como modelos autorregresivos, los modelos decodificadores involucran una tarea de preentrenamiento (llamada modelado de lenguaje causal) donde el modelo lee los textos en orden y tiene que predecir la siguiente palabra. Generalmente, se realiza leyendo la oración completa con una máscara para ocultar los tokens futuros en un cierto paso de tiempo.

<Youtube id="d_ixlCubqQw"/>

### deep learning (DL)

Algoritmos de aprendizaje automático que utilizan redes neuronales con varias capas.

## E

### encoder models

También conocidos como modelos de codificación automática (autoencoding models), los modelos codificadores toman una entrada (como texto o imágenes) y las transforman en una representación numérica condensada llamada embedding. A menudo, los modelos codificadores se entrenan previamente utilizando técnicas como el [modelado de lenguaje enmascarado](#masked-language-modeling-mlm), que enmascara partes de la secuencia de entrada y obliga al modelo a crear representaciones más significativas.

<Youtube id="H39Z_720T5s"/>

## F

### feature extraction

El proceso de seleccionar y transformar datos crudos en un conjunto de características más informativas y útiles para algoritmos de aprendizaje automático. Algunos ejemplos de extracción de características incluyen transformar texto crudo en embeddings de palabras y extraer características importantes como bordes o formas de datos de imágenes/videos.

### feed forward chunking

En cada bloque de atención residual en los transformadores, la capa de autoatención suele ir seguida de 2 capas de avance. El tamaño de embedding intermedio de las capas de avance suele ser mayor que el tamaño oculto del modelo (por ejemplo, para `google-bert/bert-base-uncased`).

Para una entrada de tamaño `[batch_size, sequence_length]`, la memoria requerida para almacenar los embeddings intermedios de avance `[batch_size, sequence_length, config.intermediate_size]` puede representar una gran fracción del uso de memoria. Los autores de [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) observaron que, dado que el cálculo es independiente de la dimensión `sequence_length`, es matemáticamente equivalente calcular los embeddings de salida de ambas capas de avance  `[batch_size, config.hidden_size]_0, ..., [batch_size, config.hidden_size]_n` individualmente y concatenarlos después a `[batch_size, sequence_length, config.hidden_size]` con `n = sequence_length`, lo que intercambia el aumento del tiempo de cálculo por una reducción en el uso de memoria, pero produce un resultado matemáticamente **equivalente**.

Para modelos que utilizan la función [`apply_chunking_to_forward`], el `chunk_size` define el número de embeddings de salida que se calculan en paralelo y, por lo tanto, define el equilibrio entre la complejidad de memoria y tiempo. Si `chunk_size` se establece en 0, no se realiza ninguna fragmentación de avance.

### finetuned models

El ajuste fino es una forma de transferencia de aprendizaje que implica tomar un modelo entrenado previamente, congelar sus pesos y reemplazar la capa de salida con una nueva [cabecera de modelo](#head) recién añadida. La cabecera del modelo se entrena en tu conjunto de datos objetivo.

Consulta el tutorial [Ajustar finamente un modelo pre-entrenado](https://huggingface.co/docs/transformers/training) para obtener más detalles y aprende cómo ajustar finamente modelos con 🤗 Transformers.

## H

### head

La cabecera del modelo se refiere a la última capa de una red neuronal que acepta los estados ocultos crudos y los proyecta en una dimensión diferente. Hay una cabecera de modelo diferente para cada tarea. Por ejemplo:

  * [`GPT2ForSequenceClassification`] es una cabecera de clasificación de secuencias, es decir, una capa lineal, encima del modelo base [`GPT2Model`].
  * [`ViTForImageClassification`] es una cabecera de clasificación de imágenes, es decir, una capa lineal encima del estado oculto final del token `CLS`, encima del modelo base [`ViTModel`].
  * [`Wav2Vec2ForCTC`] es una cabecera de modelado de lenguaje con [CTC](#connectionist-temporal-classification-ctc) encima del modelo base [`Wav2Vec2Model`].

## I

### image patch

Los modelos de Transformers basados en visión dividen una imagen en parches más pequeños que se incorporan linealmente y luego se pasan como una secuencia al modelo. Puedes encontrar el `patch_size` (o resolución del modelo) en su configuración.

### inference

La inferencia es el proceso de evaluar un modelo en nuevos datos después de completar el entrenamiento. Consulta el tutorial [Pipeline for inference](https://huggingface.co/docs/transformers/pipeline_tutorial) para aprender cómo realizar inferencias con 🤗 Transformers.

### input IDs

Los IDs de entrada a menudo son los únicos parámetros necesarios que se deben pasar al modelo como entrada. Son índices de tokens, representaciones numéricas de tokens que construyen las secuencias que se utilizarán como entrada por el modelo.

<Youtube id="VFp38yj8h3A"/>

Cada tokenizador funciona de manera diferente, pero el mecanismo subyacente sigue siendo el mismo. Aquí tienes un ejemplo utilizando el tokenizador BERT, que es un tokenizador [WordPiece](https://arxiv.org/pdf/1609.08144.pdf):

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

>>> sequence = "A Titan RTX has 24GB of VRAM"
```

El tokenizador se encarga de dividir la secuencia en tokens disponibles en el vocabulario del tokenizador.

```python
>>> tokenized_sequence = tokenizer.tokenize(sequence)
```

Los tokens son palabras o sub palabras. Por ejemplo, "VRAM" no estaba en el vocabulario del modelo, así que se dividió
en "V", "RA" y "M". Para indicar que estos tokens no son palabras separadas sino partes de la misma palabra, se añade un prefijo de doble almohadilla para "RA" y "M":

```python
>>> print(tokenized_sequence)
['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```

Estos tokens luego se pueden convertir en IDs que son comprensibles por el modelo. Esto se puede hacer alimentando directamente la oración al tokenizador, que aprovecha la implementación en Rust de [🤗 Tokenizers](https://github.com/huggingface/tokenizers) para obtener un rendimiento óptimo.

```python
>>> inputs = tokenizer(sequence)
```

El tokenizador devuelve un diccionario con todos los argumentos necesarios para que su modelo correspondiente funcione correctamente. Los índices de los tokens están bajo la clave `input_ids`:

```python
>>> encoded_sequence = inputs["input_ids"]
>>> print(encoded_sequence)
[101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
```

Ten en cuenta que el tokenizador añade automáticamente "tokens especiales" (si el modelo asociado depende de ellos), que son IDs especiales que el modelo utiliza en ocasiones.

Si descodificamos la secuencia anterior de IDs,

```python
>>> decoded_sequence = tokenizer.decode(encoded_sequence)
```

Veremos

```python
>>> print(decoded_sequence)
[CLS] A Titan RTX has 24GB of VRAM [SEP]
```

Porque esta es la forma en que un [`BertModel`] espera sus entradas.

## L

### labels

Las etiquetas son un argumento opcional que se puede pasar para que el modelo calcule la pérdida por sí mismo. Estas etiquetas deberían ser la predicción esperada del modelo: usará la pérdida estándar para calcular la pérdida entre sus
predicciones y el valor esperado (la etiqueta).

Estas etiquetas son diferentes según la cabecera del modelo, por ejemplo:

- Para modelos de clasificación de secuencias ([`BertForSequenceClassification`]), el modelo espera un tensor de dimensión
  `(batch_size)` con cada valor del lote correspondiente a la etiqueta esperada de toda la secuencia.
- Para modelos de clasificación de tokens ([`BertForTokenClassification`]), el modelo espera un tensor de dimensión
  `(batch_size, seq_length)` con cada valor correspondiente a la etiqueta esperada de cada token individual.
- Para el modelado de lenguaje enmascarado ([`BertForMaskedLM`]), el modelo espera un tensor de dimensión `(batch_size, seq_length)` con cada valor correspondiente a la etiqueta esperada de cada token individual: las etiquetas son el ID del token enmascarado y los valores deben ignorarse para el resto (generalmente -100).
- Para tareas de secuencia a secuencia ([`BartForConditionalGeneration`], [`MBartForConditionalGeneration`]), el modelo
  espera un tensor de dimensión `(batch_size, tgt_seq_length)` con cada valor correspondiente a las secuencias objetivo asociadas con cada secuencia de entrada. Durante el entrenamiento, tanto BART como T5 generarán internamente los `decoder_input_ids` y las máscaras de atención del decodificador. Por lo general, no es necesario suministrarlos. Esto no se aplica a los modelos que aprovechan el marco codificador-decodificador.
- Para modelos de clasificación de imágenes ([`ViTForImageClassification`]), el modelo espera un tensor de dimensión
  `(batch_size)` con cada valor del lote correspondiente a la etiqueta esperada de cada imagen individual.
- Para modelos de segmentación semántica ([`SegformerForSemanticSegmentation`]), el modelo espera un tensor de dimensión
  `(batch_size, height, width)` con cada valor del lote correspondiente a la etiqueta esperada de cada píxel individual.
- Para modelos de detección de objetos ([`DetrForObjectDetection`]), el modelo espera una lista de diccionarios con claves `class_labels` y `boxes` donde cada valor del lote corresponde a la etiqueta esperada y el número de cajas delimitadoras de cada imagen individual.
- Para modelos de reconocimiento automático de voz ([`Wav2Vec2ForCTC`]), el modelo espera un tensor de dimensión `(batch_size, target_length)` con cada valor correspondiente a la etiqueta esperada de cada token individual.
  
<Tip>

Las etiquetas de cada modelo pueden ser diferentes, así que asegúrate siempre de revisar la documentación de cada modelo para obtener más información sobre sus etiquetas específicas.

</Tip>

Los modelos base ([`BertModel`]) no aceptan etiquetas, ya que estos son los modelos base de transformadores, que simplemente generan características.

### large language models (LLM)

Un término genérico que se refiere a modelos de lenguaje de transformadores (GPT-3, BLOOM, OPT) que fueron entrenados con una gran cantidad de datos. Estos modelos también tienden a tener un gran número de parámetros que se pueden aprender (por ejemplo, 175 mil millones para GPT-3).

## M

### masked language modeling (MLM)

Una tarea de preentrenamiento en la que el modelo ve una versión corrupta de los textos, generalmente hecha
al enmascarar algunos tokens al azar, y tiene que predecir el texto original.

### multimodal

Una tarea que combina textos con otro tipo de entradas (por ejemplo: imágenes).

## N

### Natural language generation (NLG)

Todas las tareas relacionadas con la generación de texto (por ejemplo: [Escribe con Transformers](https://transformer.huggingface.co/) o traducción).

### Natural language processing (NLP)

Una forma genérica de decir "trabajar con textos".

### Natural language understanding (NLU)

Todas las tareas relacionadas con entender lo que hay en un texto (por ejemplo: clasificar el
texto completo o palabras individuales).

## P

### Pipeline

Un pipeline en 🤗 Transformers es una abstracción que se refiere a una serie de pasos que se ejecutan en un orden específico para preprocesar y transformar datos y devolver una predicción de un modelo. Algunas etapas de ejemplo que se encuentran en un pipeline pueden ser el preprocesamiento de datos, la extracción de características y la normalización.

Para obtener más detalles, consulta [Pipelines para inferencia](https://huggingface.co/docs/transformers/pipeline_tutorial).

### PipelineParallel (PP)

Técnica de paralelismo en la que el modelo se divide verticalmente (a nivel de capa) en varios GPU, de modo que solo una o varias capas del modelo se colocan en un solo GPU. Cada GPU procesa en paralelo diferentes etapas del pipeline y trabaja en un pequeño fragmento del lote. Obtén más información sobre cómo funciona PipelineParallel [aquí](perf_train_gpu_many#from-naive-model-parallelism-to-pipeline-parallelism).

### pixel values

Un tensor de las representaciones numéricas de una imagen que se pasa a un modelo. Los valores de píxeles tienen una forma de [`batch_size`, `num_channels`, `height`, `width`], y se generan a partir de un procesador de imágenes.

### pooling

Una operación que reduce una matriz a una matriz más pequeña, ya sea tomando el máximo o el promedio de la dimensión (o dimensiones) agrupada(s). Las capas de agrupación se encuentran comúnmente entre capas convolucionales para reducir la representación de características.

### position IDs

A diferencia de las RNN que tienen la posición de cada token incrustada en ellas, los transformers no son conscientes de la posición de cada token. Por lo tanto, se utilizan los IDs de posición (`position_ids`) para que el modelo identifique la posición de cada token en la lista de tokens.

Son un parámetro opcional. Si no se pasan `position_ids` al modelo, los IDs se crean automáticamente como embeddings de posición absolutas.

Los embeddings de posición absolutas se seleccionan en el rango `[0, config.max_position_embeddings - 1]`. Algunos modelos utilizan otros tipos de embeddings de posición, como embeddings de posición sinusoidales o embeddings de posición relativas.

### preprocessing

La tarea de preparar datos crudos en un formato que pueda ser fácilmente consumido por modelos de aprendizaje automático. Por ejemplo, el texto se preprocesa típicamente mediante la tokenización. Para tener una mejor idea de cómo es el preprocesamiento para otros tipos de entrada, consulta el tutorial [Pre-procesar](https://huggingface.co/docs/transformers/preprocessing).

### pretrained model

Un modelo que ha sido pre-entrenado en algunos datos (por ejemplo, toda Wikipedia). Los métodos de preentrenamiento involucran un objetivo auto-supervisado, que puede ser leer el texto e intentar predecir la siguiente palabra (ver [modelado de lenguaje causal](#causal-language-modeling)) o enmascarar algunas palabras e intentar predecirlas (ver [modelado de lenguaje enmascarado](#masked-language-modeling-mlm)).

Los modelos de habla y visión tienen sus propios objetivos de pre-entrenamiento. Por ejemplo, Wav2Vec2 es un modelo de habla pre-entrenado en una tarea contrastiva que requiere que el modelo identifique la representación de habla "verdadera" de un conjunto de representaciones de habla "falsas". Por otro lado, BEiT es un modelo de visión pre-entrenado en una tarea de modelado de imágenes enmascaradas que enmascara algunos de los parches de la imagen y requiere que el modelo prediga los parches enmascarados (similar al objetivo de modelado de lenguaje enmascarado).

## R

### recurrent neural network (RNN)

Un tipo de modelo que utiliza un bucle sobre una capa para procesar textos.

### representation learning

Un subcampo del aprendizaje automático que se centra en aprender representaciones significativas de datos en bruto. Algunos ejemplos de técnicas de aprendizaje de representaciones incluyen embeddings de palabras, auto-encoders y Redes Generativas Adversarias (Generative Adversarial Networks, GANs).

## S

### sampling rate

Una medida en hercios del número de muestras (la señal de audio) tomadas por segundo. La tasa de muestreo es el resultado de aproximar una señal continua como el habla.

### self-attention

Cada elemento de la entrada averigua a cuáles otros elementos de la entrada debe prestar atención.

### self-supervised learning

Una categoría de técnicas de aprendizaje automático en la que un modelo crea su propio objetivo de aprendizaje a partir de datos no etiquetados. Difiere del [aprendizaje no supervisado](#unsupervised-learning) y del [aprendizaje supervisado](#supervised-learning) en que el proceso de aprendizaje está supervisado, pero no explícitamente por el usuario.

Un ejemplo de aprendizaje auto-supervisado es el [modelado de lenguaje enmascarado](#masked-language-modeling-mlm), donde un modelo recibe oraciones con una proporción de sus tokens eliminados y aprende a predecir los tokens faltantes.

### semi-supervised learning

Una amplia categoría de técnicas de entrenamiento de aprendizaje automático que aprovecha una pequeña cantidad de datos etiquetados con una mayor cantidad de datos no etiquetados para mejorar la precisión de un modelo, a diferencia del [aprendizaje supervisado](#supervised-learning) y del [aprendizaje no supervisado](#unsupervised-learning).

Un ejemplo de un enfoque de aprendizaje semi-supervisado es "auto-entrenamiento", en el que un modelo se entrena con datos etiquetados y luego se utiliza para hacer predicciones sobre los datos no etiquetados. La porción de datos no etiquetados que el modelo predice con mayor confianza se agrega al conjunto de datos etiquetados y se utiliza para volver a entrenar el modelo.

### sequence-to-sequence (seq2seq)

Modelos que generan una nueva secuencia a partir de una entrada, como modelos de traducción o modelos de resumen (como
[Bart](model_doc/bart) o [T5](model_doc/t5)).

### Sharded DDP

Otro nombre para el concepto fundamental de [ZeRO](#zero-redundancy-optimizer-zero) utilizado por varias otras implementaciones de ZeRO.

### stride

En [convolución](#convolution) o [agrupación](#pooling), el paso (stride) se refiere a la distancia que recorre el núcleo sobre una matriz. Un paso de 1 significa que el núcleo se mueve un píxel a la vez, y un paso de 2 significa que el núcleo se mueve dos píxeles a la vez.

### supervised learning

Una forma de entrenamiento de modelos que utiliza directamente datos etiquetados para corregir y dirigir el rendimiento del modelo. Los datos se introducen en el modelo en entrenamiento, y sus predicciones se comparan con las etiquetas conocidas. El modelo actualiza sus pesos en función de cuán incorrectas fueron sus predicciones, y el proceso se repite para optimizar el rendimiento del modelo.

## T

### Tensor Parallelism (TP)

Técnica de paralelismo para entrenamiento en múltiples GPU en la que cada tensor se divide en múltiples fragmentos, de modo que en lugar de tener todo el tensor en una sola GPU, cada fragmento del tensor reside en su GPU designada. Los fragmentos se procesan por separado y en paralelo en diferentes GPU y los resultados se sincronizan al final del paso de procesamiento.Esto es lo que a veces se llama paralelismo horizontal, ya que la división ocurre a nivel horizontal.
Obtén más información sobre el Paralelismo de Tensores [aquí](perf_train_gpu_many#tensor-parallelism).

### token

Parte de una oración, generalmente una palabra, pero también puede ser una sub-palabra (las palabras no comunes a menudo se dividen en sub-palabras) o un símbolo de puntuación.

### token Type IDs

Algunos modelos tienen como objetivo realizar clasificación en pares de oraciones o responder preguntas.

<Youtube id="0u3ioSwev3s"/>

Estos requieren que dos secuencias diferentes se unan en una única entrada "input_ids", lo cual generalmente se realiza con
la ayuda de tokens especiales, como el token de clasificación (`[CLS]`) y el token separador (`[SEP]`). Por ejemplo, el modelo BERT construye sus dos secuencias de entrada de la siguiente manera:

```python
>>> # [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
```

Podemos utilizar nuestro tokenizador para generar automáticamente una oración de este tipo al pasar las dos secuencias a `tokenizer` como dos argumentos (y no como una lista, como antes) de la siguiente manera:

```python
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
>>> sequence_a = "HuggingFace is based in NYC"
>>> sequence_b = "Where is HuggingFace based?"

>>> encoded_dict = tokenizer(sequence_a, sequence_b)
>>> decoded = tokenizer.decode(encoded_dict["input_ids"])
```

Que devolverá:

```python
>>> print(decoded)
[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```

Esto es suficiente para que algunos modelos comprendan dónde termina una secuencia y comienza otra. Sin embargo, otros modelos, como BERT, también utilizan identificadores de tipo de token (también llamados identificadores de segmento). Se representan como una máscara binaria que identifica los dos tipos de secuencia en el modelo.

El tokenizador devuelve esta máscara como la entrada "token_type_ids":

```python
>>> encoded_dict["token_type_ids"]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

La primera secuencia, el "contexto" utilizado para la pregunta, tiene todos sus tokens representados por un `0`, mientras que la segunda secuencia, correspondiente a la "pregunta", tiene todos sus tokens representados por un `1`.

Algunos modelos, como [`XLNetModel`], utilizan un token adicional representado por un `2`.

### transfer learning

Una técnica que implica tomar un modelo pre-entrenado y adaptarlo a un conjunto de datos específico para tu tarea. En lugar de entrenar un modelo desde cero, puedes aprovechar el conocimiento obtenido de un modelo existente como punto de partida. Esto acelera el proceso de aprendizaje y reduce la cantidad de datos de entrenamiento necesarios.

### transformer

Arquitectura de modelo de aprendizaje profundo basada en auto-atención (Self-attention).

## U

### unsupervised learning

Una forma de entrenamiento de modelos en la que los datos proporcionados al modelo no están etiquetados. Las técnicas de aprendizaje no supervisado aprovechan la información estadística de la distribución de datos para encontrar patrones útiles para la tarea en cuestión.

## Z

### Zero Redundancy Optimizer (ZeRO)

Técnica de paralelismo que realiza la fragmentación de los tensores de manera algo similar a [TensorParallel](#tensor-parallelism-tp), excepto que todo el tensor se reconstruye a tiempo para una computación hacia adelante o hacia atrás, por lo tanto, el modelo no necesita ser modificado. Este método también admite diversas técnicas de descarga para compensar la memoria limitada de la GPU. Obtén más información sobre ZeRO [aquí](perf_train_gpu_many#zero-data-parallelism).