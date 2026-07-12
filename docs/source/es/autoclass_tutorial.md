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

# Carga instancias preentrenadas con un AutoClass

Con tantas arquitecturas diferentes de Transformer puede ser retador crear una para tu checkpoint. Como parte de la filosofía central de 🤗 Transformers para hacer que la biblioteca sea fácil, simple y flexible de usar; una `AutoClass` automáticamente infiere y carga la arquitectura correcta desde un checkpoint dado. El método `from_pretrained` te permite cargar rápidamente un modelo preentrenado para cualquier arquitectura, por lo que no tendrás que dedicar tiempo y recursos para entrenar uno desde cero. Producir este tipo de código con checkpoint implica que si funciona con uno, funcionará también con otro (siempre que haya sido entrenado para una tarea similar) incluso si la arquitectura es distinta.

<Tip>

Recuerda, la arquitectura se refiere al esqueleto del modelo y los checkpoints son los pesos para una arquitectura dada. Por ejemplo, [BERT](https://huggingface.co/google-bert/bert-base-uncased) es una arquitectura, mientras que `google-bert/bert-base-uncased` es un checkpoint. Modelo es un término general que puede significar una arquitectura o un checkpoint.

</Tip>

En este tutorial, aprenderás a:

* Cargar un tokenizador pre-entrenado.
* Cargar un extractor de características (feature extractor en inglés) pre-entrenado.
* Cargar un procesador pre-entrenado.
* Cargar un modelo pre-entrenado.

## AutoTokenizer

Casi cualquier tarea de Procesamiento de Lenguaje Natural comienza con un tokenizador. Un tokenizador convierte tu input a un formato que puede ser procesado por el modelo.

Carga un tokenizador con [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

Luego tokeniza tu input como lo mostrado a continuación:

```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoFeatureExtractor

Para tareas de audio y visión, un extractor de características procesa la señal de audio o imagen al formato de input correcto.

Carga un extractor de características con [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

Las tareas multimodales requieren un procesador que combine dos tipos de herramientas de preprocesamiento. Por ejemplo, el modelo [LayoutLMV2](model_doc/layoutlmv2) requiere que un extractor de características maneje las imágenes y que un tokenizador maneje el texto; un procesador combina ambas.

Carga un procesador con [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

Finalmente, las clases `AutoModelFor` te permiten cargar un modelo preentrenado para una tarea dada (revisa [aquí](model_doc/auto) para conocer la lista completa de tareas disponibles). Por ejemplo, cargue un modelo para clasificación de secuencias con [`AutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

Reutiliza fácilmente el mismo checkpoint para cargar una aquitectura para alguna tarea diferente:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

Generalmente recomendamos utilizar las clases `AutoTokenizer` y `AutoModelFor` para cargar instancias pre-entrenadas de modelos. Ésto asegurará que cargues la arquitectura correcta en cada ocasión. En el siguiente [tutorial](preprocessing), aprende a usar tu tokenizador recién cargado, el extractor de características y el procesador para preprocesar un dataset para fine-tuning.
