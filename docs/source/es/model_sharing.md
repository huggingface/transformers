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

# Compartir un modelo

Los últimos dos tutoriales mostraron cómo puedes realizar fine-tunning a un modelo con PyTorch, Keras y 🤗 Accelerate para configuraciones distribuidas. ¡El siguiente paso es compartir tu modelo con la comunidad! En Hugging Face creemos en compartir abiertamente a todos el conocimiento y los recursos para democratizar la inteligencia artificial. En este sentido, te animamos a considerar compartir tu modelo con la comunidad, de esta forma ayudas a otros ahorrando tiempo y recursos.

En este tutorial aprenderás dos métodos para compartir un modelo trained o fine-tuned en el [Model Hub](https://huggingface.co/models):

- Mediante Código, enviando (push) tus archivos al Hub.
- Con la interfaz Web, con Drag-and-drop de tus archivos al Hub.

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

Para compartir un modelo con la comunidad necesitas una cuenta en [huggingface.co](https://huggingface.co/join). También puedes unirte a una organización existente o crear una nueva.

</Tip>

## Características de los repositorios

Cada repositorio en el Model Hub se comporta como cualquier otro repositorio en GitHub. Nuestros repositorios ofrecen versioning, commit history, y la habilidad para visualizar diferencias.

El versioning desarrollado dentro del Model Hub es basado en git y [git-lfs](https://git-lfs.github.com/). En otras palabras, puedes tratar un modelo como un repositorio, brindando un mejor control de acceso y escalabilidad. Version control permite *revisions*, un método para apuntar a una versión específica de un modelo utilizando un commit hash, tag o branch.

Como resultado, puedes cargar una versión específica del modelo con el parámetro `revision`:

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # tag name, or branch name, or commit hash
... )
```

Los archivos son editados fácilmente dentro de un repositorio. Incluso puedes observar el commit history y las diferencias:

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## Configuración inicial

Antes de compartir un modelo al Hub necesitarás tus credenciales de Hugging Face. Si tienes acceso a una terminal ejecuta el siguiente comando en el entorno virtual donde 🤗 Transformers esté instalado. Esto guardará tu token de acceso dentro de tu carpeta cache de Hugging Face (~/.cache/ by default):

```bash
huggingface-cli login
```

Si usas un notebook como Jupyter o Colaboratory, asegúrate de tener instalada la biblioteca [`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library). Esta biblioteca te permitirá interactuar por código con el Hub.

```bash
pip install huggingface_hub
```

Luego usa `notebook_login` para iniciar sesión al Hub, y sigue el link [aquí](https://huggingface.co/settings/token) para generar un token con el que iniciaremos sesión:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Convertir un modelo para todos los Frameworks

Para asegurarnos que tu modelo pueda ser usado por alguien que esté trabajando con un framework diferente, te recomendamos convertir y subir tu modelo con checkpoints de pytorch y tensorflow. Aunque los usuarios aún son capaces de cargar su modelo desde un framework diferente, si se omite este paso será más lento debido a que 🤗 Transformers necesitará convertir el checkpoint sobre-la-marcha.

Convertir un checkpoint para otro framework es fácil. Asegúrate tener Pytorch y TensorFlow instalado (Véase [aquí](installation) para instrucciones de instalación), y luego encuentra el modelo específico para tu tarea en el otro Framework. 

Por ejemplo, supongamos que has entrenado DistilBert para clasificación de secuencias en PyTorch y quieres convertirlo a su equivalente en TensorFlow. Cargas el equivalente en TensorFlow de tu modelo para tu tarea y especificas `from_pt=True` así 🤗 Transformers convertirá el Pytorch checkpoint a un TensorFlow Checkpoint:

```py
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
```

Luego guardas tu nuevo modelo TensorFlow con su nuevo checkpoint:

```py
>>> tf_model.save_pretrained("path/to/awesome-name-you-picked")
```

De manera similar, especificas `from_tf=True` para convertir un checkpoint de TensorFlow a Pytorch:

```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```

Si algún modelo está disponible en Flax, también puedes convertir un checkpoint de Pytorch a Flax:

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/to/awesome-name-you-picked", from_pt=True
... )
```

## Compartir un modelo con `Trainer`

<Youtube id="Z1-XMy-GNLQ"/>

Compartir un modelo al Hub es tan simple como añadir un parámetro extra o un callback. Si recuerdas del tutorial de [fine-tuning tutorial](training), la clase [`TrainingArguments`] es donde especificas los Hiperparámetros y opciones de entrenamiento adicionales. Una de estas opciones incluye la habilidad de compartir un modelo directamente al Hub. Para ello configuras `push_to_hub=True` dentro de [`TrainingArguments`]:

```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

A continuación, como usualmente, pasa tus argumentos de entrenamiento a [`Trainer`]:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Luego que realizas fine-tune a tu modelo, llamas [`~transformers.Trainer.push_to_hub`] en [`Trainer`] para enviar el modelo al Hub!🤗 Transformers incluso añadirá automáticamente los Hiperparámetros de entrenamiento, resultados de entrenamiento y versiones del Framework a tu model card!

```py
>>> trainer.push_to_hub()
```

## Compartir un modelo con `PushToHubCallback`

Los usuarios de TensorFlow pueden activar la misma funcionalidad con [`PushToHubCallback`]. En la funcion [`PushToHubCallback`], agrega:

- Un directorio de salida para tu modelo.
- Un tokenizador.
- El `hub_model_id`, el cual es tu usuario Hub y el nombre del modelo.

```py
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
... )
```

Agregamos el callback a [`fit`](https://keras.io/api/models/model_training_apis/), y 🤗 Transformers enviará el modelo entrenado al Hub:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```

## Usando la función `push_to_hub`

Puedes llamar la función `push_to_hub` directamente en tu modelo para subirlo al Hub.

Especifica el nombre del modelo en `push_to_hub`:

```py
>>> pt_model.push_to_hub("my-awesome-model")
```

Esto creará un repositorio bajo tu usuario con el nombre del modelo `my-awesome-model`. Ahora los usuarios pueden cargar tu modelo con la función `from_pretrained`:

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

Si perteneces a una organización y quieres compartir tu modelo bajo el nombre de la organización, añade el parámetro `organization`:

```py
>>> pt_model.push_to_hub("my-awesome-model", organization="my-awesome-org")
```

La función `push_to_hub` también puede ser usada para añadir archivos al repositorio del modelo. Por ejemplo, añade un tokenizador al repositorio:

```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

O quizás te gustaría añadir la versión de TensorFlow de tu modelo fine-tuned en Pytorch:

```py
>>> tf_model.push_to_hub("my-awesome-model")
```

Ahora, cuando navegues a tu perfil en Hugging Face, deberías observar el repositorio de tu modelo creado recientemente. Si das click en el tab **Files** observarás todos los archivos que has subido al repositorio.

Para más detalles sobre cómo crear y subir archivos al repositorio, consulta la [documentación del Hub](https://huggingface.co/docs/hub/how-to-upstream).

## Compartir con la interfaz web

Los usuarios que prefieran un enfoque no-code tienen la opción de cargar su modelo a través de la interfaz gráfica del Hub. Visita la página [huggingface.co/new](https://huggingface.co/new) para crear un nuevo repositorio:

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

Desde aquí, añade información acerca del modelo:

- Selecciona el **owner** (la persona propietaria) del repositorio. Puedes ser tú o cualquier organización a la que pertenezcas.
- Escoge un nombre para tu modelo. También será el nombre del repositorio.
- Elige si tu modelo es público o privado.
- Especifica la licencia que usará tu modelo.

Ahora puedes hacer click en el tab **Files** y luego en el botón **Add file** para subir un nuevo archivo a tu repositorio. Luego arrastra y suelta un archivo a subir y le añades un mensaje al commit.

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## Añadiendo una tarjeta de modelo

Para asegurarnos que los usuarios entiendan las capacidades de tu modelo, sus limitaciones, posibles sesgos y consideraciones éticas, por favor añade una tarjeta (como una tarjeta de presentación) al repositorio del modelo. La tarjeta de modelo es definida en el archivo `README.md`. Puedes agregar una de la siguiente manera:

* Elaborando y subiendo manualmente el archivo`README.md`.
* Dando click en el botón **Edit model card** dentro del repositorio.

Toma un momento para ver la [tarjeta de modelo](https://huggingface.co/distilbert/distilbert-base-uncased) de DistilBert para que tengas un buen ejemplo del tipo de información que debería incluir. Consulta [la documentación](https://huggingface.co/docs/hub/models-cards) para más detalles acerca de otras opciones que puedes controlar dentro del archivo `README.md` como la huella de carbono del modelo o ejemplos de widgets. Consulta la documentación [aquí](https://huggingface.co/docs/hub/models-cards).
