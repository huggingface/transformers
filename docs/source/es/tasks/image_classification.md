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

# Clasificación de imágenes

<Youtube id="tjAIM7BOYhw"/>

La clasificación de imágenes asigna una etiqueta o clase a una imagen. A diferencia de la clasificación de texto o audio, las entradas son los valores de los píxeles que representan una imagen. La clasificación de imágenes tiene muchos usos, como la detección de daños tras una catástrofe, el control de la salud de los cultivos o la búsqueda de signos de enfermedad en imágenes médicas.

Esta guía te mostrará como hacer fine-tune al [ViT](https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/vit) en el dataset [Food-101](https://huggingface.co/datasets/food101) para clasificar un alimento en una imagen.

<Tip>

Consulta la [página de la tarea](https://huggingface.co/tasks/audio-classification) de clasificación de imágenes para obtener más información sobre sus modelos, datasets y métricas asociadas.

</Tip>

## Carga el dataset Food-101

Carga solo las primeras 5000 imágenes del dataset Food-101 de la biblioteca 🤗 de Datasets ya que es bastante grande:

```py
>>> from datasets import load_dataset

>>> food = load_dataset("food101", split="train[:5000]")
```

Divide el dataset en un train y un test set:

```py
>>> food = food.train_test_split(test_size=0.2)
```

A continuación, observa un ejemplo:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

El campo `image` contiene una imagen PIL, y cada `label` es un número entero que representa una clase. Crea un diccionario que asigne un nombre de label a un entero y viceversa. El mapeo ayudará al modelo a recuperar el nombre de label a partir del número de la misma:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

Ahora puedes convertir el número de label en un nombre de label para obtener más información:

```py
>>> id2label[str(79)]
'prime_rib'
```

Cada clase de alimento - o label - corresponde a un número; `79` indica una costilla de primera en el ejemplo anterior.

## Preprocesa

Carga el image processor de ViT para procesar la imagen en un tensor:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
```

Aplica varias transformaciones de imagen al dataset para hacer el modelo más robusto contra el overfitting. En este caso se utilizará el módulo [`transforms`](https://pytorch.org/vision/stable/transforms.html) de torchvision. Recorta una parte aleatoria de la imagen, cambia su tamaño y normalízala con la media y la desviación estándar de la imagen:

```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> _transforms = Compose([RandomResizedCrop(image_processor.size["height"]), ToTensor(), normalize])
```

Crea una función de preprocesamiento que aplique las transformaciones y devuelva los `pixel_values` - los inputs al modelo - de la imagen:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

Utiliza el método [`with_transform`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?#datasets.Dataset.with_transform) de 🤗 Dataset para aplicar las transformaciones sobre todo el dataset. Las transformaciones se aplican sobre la marcha cuando se carga un elemento del dataset:

```py
>>> food = food.with_transform(transforms)
```

Utiliza [`DefaultDataCollator`] para crear un batch de ejemplos. A diferencia de otros data collators en 🤗 Transformers, el DefaultDataCollator no aplica un preprocesamiento adicional como el padding.

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## Entrena
Carga ViT con [`AutoModelForImageClassification`]. Especifica el número de labels, y pasa al modelo el mapping entre el número de label y la clase de label:

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     "google/vit-base-patch16-224-in21k",
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

<Tip>

Si no estás familiarizado con el fine-tuning de un modelo con el [`Trainer`], echa un vistazo al tutorial básico [aquí](../training#finetune-with-trainer)!

</Tip>

Al llegar a este punto, solo quedan tres pasos:

1. Define tus hiperparámetros de entrenamiento en [`TrainingArguments`]. Es importante que no elimines las columnas que no se utilicen, ya que esto hará que desaparezca la columna `image`. Sin la columna `image` no puedes crear `pixel_values`. Establece `remove_unused_columns=False` para evitar este comportamiento.
2. Pasa los training arguments al [`Trainer`] junto con el modelo, los datasets, tokenizer y data collator.
3. Llama [`~Trainer.train`] para hacer fine-tune de tu modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="./results",
...     per_device_train_batch_size=16,
...     evaluation_strategy="steps",
...     num_train_epochs=4,
...     fp16=True,
...     save_steps=100,
...     eval_steps=100,
...     logging_steps=10,
...     learning_rate=2e-4,
...     save_total_limit=2,
...     remove_unused_columns=False,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"],
...     eval_dataset=food["test"],
...     tokenizer=image_processor,
... )

>>> trainer.train()
```

<Tip>

Para ver un ejemplo más a profundidad de cómo hacer fine-tune a un modelo para clasificación de imágenes, echa un vistazo al correspondiente [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).

</Tip>
