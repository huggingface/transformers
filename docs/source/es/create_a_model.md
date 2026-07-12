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

# Crea una arquitectura personalizada

Una [`AutoClass`](model_doc/auto) infiere, automáticamente, la arquitectura del modelo y descarga la configuración y los pesos del modelo preentrenado. Normalmente, recomendamos usar una `AutoClass` para producir un código agnóstico a puntos de guardado o checkpoints. Sin embargo, los usuarios que quieran más control sobre los parámetros específicos de los modelos pueden crear su propio modelo 🤗 Transformers personalizado a partir de varias clases base. Esto puede ser particularmente útil para alguien que esté interesado en estudiar, entrenar o experimentar con modelos 🤗 Transformers. En esta guía vamos a profundizar en la creación de modelos personalizados sin usar `AutoClass`. Aprenderemos a:

- Cargar y personalizar una configuración para un modelo.
- Crear una arquitectura para un modelo.
- Crear tokenizadores rápidos y lentos para textos.
- Crear un extractor de propiedades para tareas de audio o imágenes.
- Crear un procesador para tareas multimodales.

## Configuración

Una [configuración](main_classes/configuration) es un conjunto de atributos específicos de un modelo. Cada configuración de modelo tiene atributos diferentes. Por ejemplo, todos los modelos de PLN tienen los atributos `hidden_size`, `num_attention_heads`, `num_hidden_layers` y `vocab_size` en común. Estos atributos especifican el número de cabezas de atención o de capas ocultas con las que se construyen los modelos.

Puedes echarle un vistazo a [DistilBERT](model_doc/distilbert) y sus atributos accediendo a [`DistilBertConfig`]:

```py
>>> from transformers import DistilBertConfig

>>> config = DistilBertConfig()
>>> print(config)
DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

[`DistilBertConfig`] muestra todos los atributos por defecto que se han usado para construir un modelo [`DistilBertModel`] base. Todos ellos son personalizables, lo que deja espacio para poder experimentar. Por ejemplo, puedes personalizar un modelo predeterminado para:

- Probar una función de activación diferente, usando el parámetro `activation`.
- Usar un valor de abandono (también conocido como _dropout_) más alto para las probabilidades de las capas de atención, usando el parámetro `attention_dropout`.

```py
>>> my_config = DistilBertConfig(activation="relu", attention_dropout=0.4)
>>> print(my_config)
DistilBertConfig {
  "activation": "relu",
  "attention_dropout": 0.4,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

Los atributos de los modelos preentrenados pueden ser modificados con la función [`~PreTrainedConfig.from_pretrained`]:

```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

Cuando estés satisfecho con la configuración de tu modelo, puedes guardarlo con la función [`~PreTrainedConfig.save_pretrained`]. Tu configuración se guardará en un archivo JSON dentro del directorio que le especifiques como parámetro.

```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

Para volver a usar el archivo de configuración, puedes cargarlo usando [`~PreTrainedConfig.from_pretrained`]:

```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
```

<Tip>

También puedes guardar los archivos de configuración como un diccionario; o incluso guardar solo la diferencia entre tu archivo personalizado y la configuración por defecto. Consulta la [documentación sobre configuración](main_classes/configuration) para ver más detalles.

</Tip>

## Modelo

El siguiente paso será crear un [modelo](main_classes/models). El modelo, al que a veces también nos referimos como arquitectura, es el encargado de definir cada capa y qué operaciones se realizan. Los atributos como `num_hidden_layers` de la configuración se usan para definir la arquitectura. Todos los modelos comparten una clase base, [`PreTrainedModel`], y algunos métodos comunes que se pueden usar para redimensionar los _embeddings_ o para recortar cabezas de auto-atención (también llamadas _self-attention heads_). Además, todos los modelos son subclases de [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) o [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html), lo que significa que son compatibles con su respectivo framework.


Carga los atributos de tu configuración personalizada en el modelo de la siguiente forma:

```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
>>> model = DistilBertModel(my_config)
```

Esto crea un modelo con valores aleatorios, en lugar de crearlo con los pesos del preentrenamiento, por lo que no serás capaz de usar este modelo para nada útil hasta que no lo entrenes. El entrenamiento es un proceso costoso, tanto en cuestión de recursos como de tiempo, por lo que generalmente es mejor usar un modelo preentrenado para obtener mejores resultados más rápido, consumiendo una fracción de los recursos que un entrenamiento completo hubiera requerido.

Puedes crear un modelo preentrenado con [`~PreTrainedModel.from_pretrained`]:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

Cuando cargues tus pesos del preentrenamiento, el modelo por defecto se carga automáticamente si nos lo proporciona 🤗 Transformers. Sin embargo, siempre puedes reemplazar (todos o algunos de) los atributos del modelo por defecto por los tuyos:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", config=my_config)
```

### Cabezas de modelo

En este punto del tutorial, tenemos un modelo DistilBERT base que devuelve los *hidden states* o estados ocultos. Los *hidden states* se pasan como parámetros de entrada a la cabeza del modelo para producir la salida. 🤗 Transformers ofrece una cabeza de modelo diferente para cada tarea, siempre y cuando el modelo sea compatible para la tarea (por ejemplo, no puedes usar DistilBERT para una tarea secuencia a secuencia como la traducción).



Por ejemplo,  [`DistilBertForSequenceClassification`] es un modelo DistilBERT base con una cabeza de clasificación de secuencias. La cabeza de clasificación de secuencias es una capa superior que precede a la recolección de las salidas.

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

Puedes reutilizar este punto de guardado o *checkpoint* para otra tarea fácilmente cambiando a una cabeza de un modelo diferente. Para una tarea de respuesta a preguntas, puedes usar la cabeza del modelo [`DistilBertForQuestionAnswering`]. La cabeza de respuesta a preguntas es similar a la de clasificación de secuencias, excepto porque consta de una capa lineal delante de la salida de los *hidden states*.


```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

## Tokenizer

La ultima clase base que debes conocer antes de usar un modelo con datos textuales es la clase [tokenizer](main_classes/tokenizer), que convierte el texto bruto en tensores. Hay dos tipos de *tokenizers* que puedes usar con 🤗 Transformers:

- [`PreTrainedTokenizer`]:  una implementación de un *tokenizer* hecha en Python.
- [`PreTrainedTokenizerFast`]: un *tokenizer* de nuestra librería [🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/), basada en Rust. Este tipo de *tokenizer* es bastante más rápido, especialmente durante la tokenización por lotes, gracias a estar implementado en Rust. Esta rápida tokenización también ofrece métodos adicionales como el *offset mapping*, que relaciona los tokens con sus palabras o caracteres originales.

Ambos *tokenizers* son compatibles con los métodos comunes, como los de encodificación y decodificación, los métodos para añadir tokens y aquellos que manejan tokens especiales.

<Tip warning={true}>

No todos los modelos son compatibles con un *tokenizer* rápido. Échale un vistazo a esta [tabla](index#supported-frameworks) para comprobar si un modelo específico es compatible con un *tokenizer* rápido.

</Tip>

Si has entrenado tu propio *tokenizer*, puedes crear uno desde tu archivo de “vocabulario”:

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt", do_lower_case=False, padding_side="left")
```

Es importante recordar que los vocabularios que provienen de un *tokenizer* personalizado serán diferentes a los vocabularios generados por el *tokenizer* de un modelo preentrenado. Debes usar el vocabulario de un *tokenizer* preentrenado si vas a usar un modelo preentrenado, de lo contrario las entradas no tendrán sentido. Crea un *tokenizer* con el vocabulario de un modelo preentrenado usando la clase [`DistilBertTokenizer`]:


```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

Crea un *tokenizer* rápido con la clase [`DistilBertTokenizerFast`]:


```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip>

Por defecto, el [`AutoTokenizer`] intentará cargar un *tokenizer* rápido. Puedes desactivar este comportamiento cambiando el parámetro `use_fast=False` de `from_pretrained`.


</Tip>

## Extractor de Características

Un extractor de características procesa entradas de audio e imagen. Hereda de la clase base [`~feature_extraction_utils.FeatureExtractionMixin`] y también puede heredar de la clase [`ImageFeatureExtractionMixin`] para el procesamiento de características de las imágenes o de la clase [`SequenceFeatureExtractor`] para el procesamiento de entradas de audio.

Dependiendo de si trabajas en una tarea de audio o de video, puedes crear un extractor de características asociado al modelo que estés usando. Por ejemplo, podrías crear un [`ViTFeatureExtractor`] por defecto si estás usando [ViT](model_doc/vit) para clasificación de imágenes:

```py
>>> from transformers import ViTFeatureExtractor

>>> vit_extractor = ViTFeatureExtractor()
>>> print(vit_extractor)
ViTFeatureExtractor {
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

<Tip>

Si no estás buscando ninguna personalización en específico, usa el método `from_pretrained` para cargar los parámetros del extractor de características por defecto del modelo.

</Tip>

Puedes modificar cualquier parámetro de [`ViTFeatureExtractor`] para crear tu extractor de características personalizado:

```py
>>> from transformers import ViTFeatureExtractor

>>> my_vit_extractor = ViTFeatureExtractor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTFeatureExtractor {
  "do_normalize": false,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.3,
    0.3,
    0.3
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": "PIL.Image.BOX",
  "size": 224
}
```

Para las entradas de audio, puedes crear un [`Wav2Vec2FeatureExtractor`] y personalizar los parámetros de una forma similar:


```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor()
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 16000
}
```

## Procesador

Para modelos que son compatibles con tareas multimodales, 🤗 Transformers ofrece una clase *procesador* que agrupa un extractor de características y un *tokenizer* en el mismo objeto. Por ejemplo, probemos a usar el procesador [`Wav2Vec2Processor`] para una tarea de reconocimiento de voz (ASR). Un ASR transcribe el audio a texto, por lo que necesitaremos un extractor de características y un *tokenizer*.

Crea un extractor de características para manejar la entrada de audio:


```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

Crea un *tokenizer* para manejar la entrada de texto:

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

Puedes combinar el extractor de características y el *tokenizer* en el [`Wav2Vec2Processor`]:


```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```
Con dos clases base (la configuración y el modelo) y una clase de preprocesamiento adicional (*tokenizer*, extractor de características o procesador), puedes crear cualquiera de los modelos compatibles con 🤗 Transformers. Cada una de estas clases son configurables, permitiéndote usar sus atributos específicos. Puedes crear un modelo para entrenarlo de una forma fácil, o modificar un modelo preentrenado disponible para especializarlo.
