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

# Preprocesamiento

[[open-in-colab]]

Antes de que puedas utilizar los datos en un modelo, debes procesarlos en un formato aceptable para el modelo. Un modelo no entiende el texto en bruto, las imágenes o el audio. Estas entradas necesitan ser convertidas en números y ensambladas en tensores. En este tutorial, podrás:

* Preprocesar los datos textuales con un tokenizador.
* Preprocesar datos de imagen o audio con un extractor de características.
* Preprocesar datos para una tarea multimodal con un procesador.

## NLP

<Youtube id="Yffk5aydLzg"/>

La principal herramienta para procesar datos textuales es un [tokenizador](main_classes/tokenizer). Un tokenizador comienza dividiendo el texto en *tokens* según un conjunto de reglas. Los tokens se convierten en números, que se utilizan para construir tensores como entrada a un modelo. El tokenizador también añade cualquier entrada adicional que requiera el modelo.

<Tip>

Si tienes previsto utilizar un modelo pre-entrenado, es importante que utilices el tokenizador pre-entrenado asociado. Esto te asegura que el texto se divide de la misma manera que el corpus de pre-entrenamiento y utiliza el mismo índice de tokens correspondiente (usualmente referido como el *vocab*) durante el pre-entrenamiento.

</Tip>

Comienza rápidamente cargando un tokenizador pre-entrenado con la clase [`AutoTokenizer`]. Esto descarga el *vocab* utilizado cuando un modelo es pre-entrenado.

### Tokenizar

Carga un tokenizador pre-entrenado con [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

A continuación, pasa tu frase al tokenizador:

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

El tokenizador devuelve un diccionario con tres ítems importantes:

* [input_ids](glossary#input-ids) son los índices correspondientes a cada token de la frase.
* [attention_mask](glossary#attention-mask) indica si un token debe ser atendido o no.
* [token_type_ids](glossary#token-type-ids) identifica a qué secuencia pertenece un token cuando hay más de una secuencia.

Tu puedes decodificar el `input_ids` para devolver la entrada original:

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

Como puedes ver, el tokenizador ha añadido dos tokens especiales - `CLS` y `SEP` (clasificador y separador) - a la frase. No todos los modelos necesitan
tokens especiales, pero si lo llegas a necesitar,  el tokenizador los añadirá automáticamente.

Si hay varias frases que quieres preprocesar, pasa las frases como una lista al tokenizador:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### Pad

Esto nos lleva a un tema importante. Cuando se procesa un batch de frases, no siempre tienen la misma longitud. Esto es un problema porque los tensores que se introducen en el modelo deben tener una forma uniforme. El pad es una estrategia para asegurar que los tensores sean rectangulares añadiendo un "padding token" especial a las oraciones con menos tokens.

Establece el parámetro `padding` en `True` aplicando el pad a las secuencias más cortas del batch para que coincidan con la secuencia más larga:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

Observa que el tokenizador ha aplicado el pad a la primera y la tercera frase con un "0" porque son más cortas.

### Truncamiento

En el otro extremo del espectro, a veces una secuencia puede ser demasiado larga para un modelo. En este caso, tendrás que truncar la secuencia a una longitud más corta.

Establece el parámetro `truncation` a `True` para truncar una secuencia a la longitud máxima aceptada por el modelo:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

### Construye tensores

Finalmente, si quieres que el tokenizador devuelva los tensores reales que se introducen en el modelo.

Establece el parámetro `return_tensors` como `pt` para PyTorch, o `tf` para TensorFlow:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[  101,   153,  7719, 21490,  1122,  1114,  9582,  1623,   102],
                      [  101,  5226,  1122,  9649,  1199,  2610,  1236,   102,     0]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
===PT-TF-SPLIT===
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[  101,   153,  7719, 21490,  1122,  1114,  9582,  1623,   102],
       [  101,  5226,  1122,  9649,  1199,  2610,  1236,   102,     0]],
      dtype=int32)>, 
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>, 
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=int32)>}
```

## Audio

Las entradas de audio se preprocesan de forma diferente a las entradas textuales, pero el objetivo final es el mismo: crear secuencias numéricas que el modelo pueda entender. Un [extractor de características](main_classes/feature_extractor) (o feature extractor en inglés) está diseñado para extraer características de datos provenientes de imágenes o audio sin procesar y convertirlos en tensores. Antes de empezar, instala 🤗 Datasets para cargar un dataset de audio para experimentar:

```bash
pip install datasets
```

Carga la tarea de detección de palabras clave del benchmark [SUPERB](https://huggingface.co/datasets/superb) (consulta el [tutorial 🤗 Dataset](https://huggingface.co/docs/datasets/load_hub.html) para que obtengas más detalles sobre cómo cargar un dataset):

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("superb", "ks")
```

Accede al primer elemento de la columna `audio` para echar un vistazo a la entrada. Al llamar a la columna `audio` se cargará y volverá a muestrear automáticamente el archivo de audio:

```py
>>> dataset["train"][0]["audio"]
{'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00592041,
        -0.00405884, -0.00253296], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/05734a36d88019a09725c20cc024e1c4e7982e37d7d55c0c1ca1742ea1cdd47f/_background_noise_/doing_the_dishes.wav',
 'sampling_rate': 16000}
```

Esto devuelve tres elementos:

* `array` es la señal de voz cargada - y potencialmente remuestreada - como un array 1D.
* `path` apunta a la ubicación del archivo de audio.
* `sampling_rate` se refiere a cuántos puntos de datos de la señal de voz se miden por segundo.

### Resample

Para este tutorial, se utilizará el modelo [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base). Como puedes ver en la model card, el modelo Wav2Vec2 está pre-entrenado en audio de voz muestreado a 16kHz. Es importante que la tasa de muestreo de tus datos de audio coincida con la tasa de muestreo del dataset utilizado para pre-entrenar el modelo. Si la tasa de muestreo de tus datos no es la misma, deberás volver a muestrear tus datos de audio. 

Por ejemplo, carga el dataset [LJ Speech](https://huggingface.co/datasets/lj_speech) que tiene una tasa de muestreo de 22050kHz. Para utilizar el modelo Wav2Vec2 con este dataset, reduce la tasa de muestreo a 16kHz:

```py
>>> lj_speech = load_dataset("lj_speech", split="train")
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}
```

1. Usa el método 🤗 Datasets' [`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.cast_column) para reducir la tasa de muestreo a 16kHz:

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

2. Carga el archivo de audio:

```py
>>> lj_speech[0]["audio"]
{'array': array([-0.00064146, -0.00074657, -0.00068768, ...,  0.00068341,
         0.00014045,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 16000}
```

Como puedes ver, el `sampling_rate` se ha reducido a 16kHz. Ahora que sabes cómo funciona el resampling, volvamos a nuestro ejemplo anterior con el dataset SUPERB.

### Extractor de características

El siguiente paso es cargar un extractor de características para normalizar y aplicar el pad a la entrada. Cuando se aplica padding a los datos textuales, se añade un "0" para las secuencias más cortas. La misma idea se aplica a los datos de audio y el extractor de características de audio añadirá un "0" - interpretado como silencio - al "array".

Carga el extractor de características con [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

Pasa el `array` de audio al extractor de características. También te recomendamos añadir el argumento `sampling_rate` en el extractor de características para poder depurar mejor los errores silenciosos que puedan producirse.

```py
>>> audio_input = [dataset["train"][0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 0.00045439,  0.00045439,  0.00045439, ..., -0.1578519 , -0.10807519, -0.06727459], dtype=float32)]}
```

### Pad y truncamiento

Al igual que el tokenizador, puedes aplicar padding o truncamiento para manejar secuencias variables en un batch. Fíjate en la longitud de la secuencia de estas dos muestras de audio:

```py
>>> dataset["train"][0]["audio"]["array"].shape
(1522930,)

>>> dataset["train"][1]["audio"]["array"].shape
(988891,)
```

Como puedes ver, el `sampling_rate` se ha reducido a 16kHz. 

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=1000000,
...         truncation=True,
...     )
...     return inputs
```

Aplica la función a los primeros ejemplos del dataset:

```py
>>> processed_dataset = preprocess_function(dataset["train"][:5])
```

Ahora echa un vistazo a las longitudes de las muestras procesadas:

```py
>>> processed_dataset["input_values"][0].shape
(1000000,)

>>> processed_dataset["input_values"][1].shape
(1000000,)
```

Las longitudes de las dos primeras muestras coinciden ahora con la longitud máxima especificada.

## Visión

También se utiliza un extractor de características para procesar imágenes para tareas de visión por computadora. Una vez más, el objetivo es convertir la imagen en bruto en un batch de tensores como entrada.

Vamos a cargar el dataset [food101](https://huggingface.co/datasets/food101) para este tutorial. Usa el parámetro 🤗 Datasets `split` para cargar solo una pequeña muestra de la división de entrenamiento ya que el dataset es bastante grande:

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

A continuación, observa la imagen con la función 🤗 Datasets [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=image#datasets.Image):

```py
>>> dataset[0]["image"]
```

![vision-preprocess-tutorial.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png)

### Extractor de características

Carga el extractor de características con [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
```

### Aumento de Datos

Para las tareas de visión por computadora es común añadir algún tipo de aumento de datos (o data augmentation) a las imágenes como parte del preprocesamiento. Puedes añadir el método de aumento de datos con cualquier librería que quieras, pero en este tutorial utilizarás el módulo [`transforms`](https://pytorch.org/vision/stable/transforms.html) de torchvision.

1. Normaliza la imagen y utiliza [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) para encadenar algunas transformaciones - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) y [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) - juntas:

```py
>>> from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor

>>> normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
>>> _transforms = Compose(
...     [RandomResizedCrop(feature_extractor.size), ColorJitter(brightness=0.5, hue=0.5), ToTensor(), normalize]
... )
```

2. El modelo acepta [`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values) como entrada. Este valor es generado por el extractor de características. Crea una función que genere `pixel_values` a partir de las transformaciones:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(image.convert("RGB")) for image in examples["image"]]
...     return examples
```

3. A continuación, utiliza 🤗 Datasets [`set_transform`](https://huggingface.co/docs/datasets/process.html#format-transform) para aplicar las transformaciones sobre la marcha:

```py
>>> dataset.set_transform(transforms)
```

4. Ahora, cuando accedes a la imagen, observarás que el extractor de características ha añadido a la entrada del modelo `pixel_values`:

```py
>>> dataset[0]["image"]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=384x512 at 0x7F1A7B0630D0>,
 'label': 6,
 'pixel_values': tensor([[[ 0.0353,  0.0745,  0.1216,  ..., -0.9922, -0.9922, -0.9922],
          [-0.0196,  0.0667,  0.1294,  ..., -0.9765, -0.9843, -0.9922],
          [ 0.0196,  0.0824,  0.1137,  ..., -0.9765, -0.9686, -0.8667],
          ...,
          [ 0.0275,  0.0745,  0.0510,  ..., -0.1137, -0.1216, -0.0824],
          [ 0.0667,  0.0824,  0.0667,  ..., -0.0588, -0.0745, -0.0980],
          [ 0.0353,  0.0353,  0.0431,  ..., -0.0039, -0.0039, -0.0588]],
 
         [[ 0.2078,  0.2471,  0.2863,  ..., -0.9451, -0.9373, -0.9451],
          [ 0.1608,  0.2471,  0.3098,  ..., -0.9373, -0.9451, -0.9373],
          [ 0.2078,  0.2706,  0.3020,  ..., -0.9608, -0.9373, -0.8275],
          ...,
          [-0.0353,  0.0118, -0.0039,  ..., -0.2392, -0.2471, -0.2078],
          [ 0.0196,  0.0353,  0.0196,  ..., -0.1843, -0.2000, -0.2235],
          [-0.0118, -0.0039, -0.0039,  ..., -0.0980, -0.0980, -0.1529]],
 
         [[ 0.3961,  0.4431,  0.4980,  ..., -0.9216, -0.9137, -0.9216],
          [ 0.3569,  0.4510,  0.5216,  ..., -0.9059, -0.9137, -0.9137],
          [ 0.4118,  0.4745,  0.5216,  ..., -0.9137, -0.8902, -0.7804],
          ...,
          [-0.2314, -0.1922, -0.2078,  ..., -0.4196, -0.4275, -0.3882],
          [-0.1843, -0.1686, -0.2000,  ..., -0.3647, -0.3804, -0.4039],
          [-0.1922, -0.1922, -0.1922,  ..., -0.2941, -0.2863, -0.3412]]])}
```

Este es el aspecto de la imagen después de preprocesarla. Como era de esperar por las transformaciones aplicadas, la imagen ha sido recortada aleatoriamente y sus propiedades de color son diferentes.

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

![preprocessed_image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png)

## Multimodal

Para las tareas multimodales utilizarás una combinación de todo lo que has aprendido hasta ahora y aplicarás tus habilidades a una tarea de reconocimiento automático de voz (ASR). Esto significa que necesitarás un:

* Extractor de características para preprocesar los datos de audio.
* Un tokenizador para procesar el texto.

Volvamos al dataset [LJ Speech](https://huggingface.co/datasets/lj_speech):

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

Suponiendo que te interesan principalmente las columnas `audio` y `texto`, elimina las demás columnas:

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

Ahora echa un vistazo a las columnas `audio` y `texto`:

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

Recuerda la sección anterior sobre el procesamiento de datos de audio, siempre debes [volver a muestrear](preprocessing#audio) la tasa de muestreo de tus datos de audio para que coincida con la tasa de muestreo del dataset utilizado para preentrenar un modelo:

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

### Processor

Un processor combina un extractor de características y un tokenizador. Cargue un procesador con [`AutoProcessor.from_pretrained]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. Crea una función para procesar los datos de audio en `input_values`, y tokeniza el texto en `labels`. Estas son las entradas del modelo:

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. Aplica la función `prepare_dataset` a una muestra:

```py
>>> prepare_dataset(lj_speech[0])
```

Observa que el método processor ha añadido `input_values` y `labels`. La tasa de muestreo también se ha reducido correctamente a 16kHz.

Genial, ahora deberías ser capaz de preprocesar datos para cualquier modalidad e incluso combinar diferentes modalidades. En el siguiente tutorial, aprenderás aplicar fine tuning a un modelo en tus datos recién preprocesados.

## Todo lo que siempre quisiste saber sobre el padding y el truncamiento

Hemos visto los comandos que funcionarán para la mayoría de los casos (hacer pad a tu batch teniendo en cuenta la longitud de la frase máxima y 
truncar a la longitud máxima que el modelo puede aceptar). Sin embargo, la API admite más estrategias si las necesitas. Los 
tres argumentos que necesitas conocer para ello son `padding`, `truncation` y `max_length`.

- `padding` controla el aplicarme padding al texto. Puede ser un booleano o una cadena que debe ser:

  - `True` o `'longest'` para aplicar el pad hasta la secuencia más larga del batch (no apliques el padding si sólo le proporcionas 
  una sola secuencia).
  - `'max_length'` para aplicar el pad hasta la longitud especificada por el argumento `max_length` o la longitud máxima aceptada 
  por el modelo si no le proporcionas `longitud_máxima` (`longitud_máxima=None`). Si sólo le proporcionas una única secuencia 
  se le aplicará el padding.
  `False` o `'do_not_pad'` para no aplicar pad a las secuencias. Como hemos visto antes, este es el comportamiento por 
  defecto.

- `truncation` controla el truncamiento. Puede ser un booleano o una string que debe ser:

  - `True` o `'longest_first'` truncan hasta la longitud máxima especificada por el argumento `max_length` o 
  la longitud máxima aceptada por el modelo si no le proporcionas `max_length` (`max_length=None`). Esto 
  truncará token por token, eliminando un token de la secuencia más larga del par hasta alcanzar la longitud 
  adecuada.
  - `'only_second'` trunca hasta la longitud máxima especificada por el argumento `max_length` o la 
  longitud máxima aceptada por el modelo si no le proporcionas `max_length` (`max_length=None`). Esto sólo truncará 
  la segunda frase de un par si le proporcionas un par de secuencias (o un batch de pares de secuencias).
  - `'only_first'` trunca hasta la longitud máxima especificada por el argumento `max_length` o la longitud máxima 
  aceptada por el modelo si no se proporciona `max_length` (`max_length=None`). Esto sólo truncará 
  la primera frase de un par si se proporciona un par de secuencias (o un lote de pares de secuencias).
  - `False` o `'do_not_truncate'` para no truncar las secuencias. Como hemos visto antes, este es el comportamiento 
  por defecto.

- `max_length` para controlar la longitud del padding/truncamiento. Puede ser un número entero o `None`, en cuyo caso 
será por defecto la longitud máxima que el modelo puede aceptar. Si el modelo no tiene una longitud máxima de entrada específica, el 
padding/truncamiento a `longitud_máxima` se desactiva.

A continuación te mostramos en una tabla que resume la forma recomendada de configurar el padding y el truncamiento. Si utilizas un par de secuencias de entrada en 
algunos de los siguientes ejemplos, puedes sustituir `truncation=True` por una `STRATEGY` seleccionada en 
`['only_first', 'only_second', 'longest_first']`, es decir, `truncation='only_second'` o `truncation= 'longest_first'` para controlar cómo se truncan ambas secuencias del par como se ha detallado anteriormente.

| Truncation                           | Padding                           | Instrucciones                                                                               |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------|
| no truncation                        | no padding                        | `tokenizer(batch_sentences)`                                                           |
|                                      | padding secuencia max del batch   | `tokenizer(batch_sentences, padding=True)` or                                          |
|                                      |                                   | `tokenizer(batch_sentences, padding='longest')`                                        |
|                                      | padding long max de input model   | `tokenizer(batch_sentences, padding='max_length')`                                     |
|                                      | padding a una long especifica     | `tokenizer(batch_sentences, padding='max_length', max_length=42)`                      |
| truncation long max del input model  | no padding                        | `tokenizer(batch_sentences, truncation=True)` or                                       |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY)`                                      |
|                                      | padding secuencia max del batch   | `tokenizer(batch_sentences, padding=True, truncation=True)` or                         |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY)`                        |
|                                      | padding long max de input model   | `tokenizer(batch_sentences, padding='max_length', truncation=True)` or                 |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY)`                |
|                                      | padding a una long especifica     | Not possible                                                                                |
| truncation a una long especifica      | no padding                        | `tokenizer(batch_sentences, truncation=True, max_length=42)` or                        |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY, max_length=42)`                       |
|                                      | padding secuencia max del batch   | `tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` or          |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY, max_length=42)`         |
|                                      | padding long max de input model   | Not possible                                                                                |
|                                      | padding a una long especifica     | `tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` or  |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY, max_length=42)` |








