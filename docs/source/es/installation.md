<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Instalación

En esta guía puedes encontrar información para instalar 🤗 Transformers para cualquier biblioteca de Machine Learning con la que estés trabajando. Además, encontrarás información sobre cómo establecer el caché y cómo configurar 🤗 Transformers para correrlo de manera offline (opcional).

🤗 Transformers ha sido probada en Python 3.10+ y PyTorch 2.4+. Para instalar la biblioteca de deep learning con la que desees trabajar, sigue las instrucciones correspondientes listadas a continuación:

* [PyTorch](https://pytorch.org/get-started/locally/)

## Instalación con pip

Es necesario instalar 🤗 Transformers en un [entorno virtual](https://docs.python.org/3/library/venv.html). Si necesitas más información sobre entornos virtuales de Python, consulta esta [guía](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
). Un entorno virtual facilita el manejo de proyectos y evita problemas de compatibilidad entre dependencias.

Comienza por crear un entorno virtual en el directorio de tu proyecto:

```bash
python -m venv .env
```

Activa el entorno virtual:

```bash
source .env/bin/activate
```

Ahora puedes instalar 🤗 Transformers con el siguiente comando:

```bash
pip install transformers
```

Solo para CPU, puedes instalar 🤗 Transformers y una biblioteca de deep learning con un comando de una sola línea.

Por ejemplo, instala 🤗 Transformers y Pytorch:

```bash
pip install transformers[torch]
```

Por último, revisa si 🤗 Transformers ha sido instalada exitosamente con el siguiente comando que descarga un modelo pre-entrenado:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```
Después imprime la etiqueta y el puntaje:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## Instalación desde la fuente

Instala 🤗 Transformers desde la fuente con el siguiente comando:

```bash
pip install git+https://github.com/huggingface/transformers
```

El comando de arriba instala la versión `master` más actual en vez de la última versión estable. La versión `master` es útil para obtener los últimos avances de  🤗 Transformers. Por ejemplo, se puede dar el caso de que un error fue corregido después de la última versión estable pero aún no se ha liberado un nuevo lanzamiento. Sin embargo, existe la posibilidad de que la versión `master` no sea estable. El equipo trata de mantener la versión `master` operacional y la mayoría de los errores son resueltos en unas cuantas horas o un día. Si encuentras algún problema, por favor abre un [Issue](https://github.com/huggingface/transformers/issues) para que pueda ser corregido más rápido.

Verifica si 🤗 Transformers está instalada apropiadamente con el siguiente comando:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## Instalación editable

Necesitarás una instalación editable si deseas:
* Usar la versión `master` del código fuente.
* Contribuir a 🤗 Transformers y necesitas probar cambios en el código.

Clona el repositorio e instala 🤗 Transformers con los siguientes comandos:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Éstos comandos van a ligar el directorio desde donde clonamos el repositorio al path de las bibliotecas de Python. Python ahora buscará dentro de la carpeta que clonaste además de los paths normales de la biblioteca. Por ejemplo, si los paquetes de Python se encuentran instalados en `~/anaconda3/envs/main/lib/python3.10/site-packages/`, Python también buscará en el directorio desde donde clonamos el repositorio `~/transformers/`.

<Tip warning={true}>

Debes mantener el directorio `transformers` si deseas seguir usando la biblioteca.

</Tip>

Puedes actualizar tu copia local a la última versión de 🤗 Transformers con el siguiente comando:

```bash
cd ~/transformers/
git pull
```

El entorno de Python que creaste para la instalación de 🤗 Transformers encontrará la versión `master` en la siguiente ejecución.

## Instalación con conda

Puedes instalar 🤗 Transformers desde el canal de conda `conda-forge` con el siguiente comando:

```bash
conda install conda-forge::transformers
```

## Configuración de Caché

Los modelos preentrenados se descargan y almacenan en caché localmente en: `~/.cache/huggingface/transformers/`. Este es el directorio predeterminado proporcionado por la variable de entorno de shell `HF_HUB_CACHE`. En Windows, el directorio predeterminado es dado por `C:\Users\username\.cache\huggingface\transformers`. Puedes cambiar las variables de entorno de shell que se muestran a continuación, en orden de prioridad, para especificar un directorio de caché diferente:

1. Variable de entorno del shell (por defecto): `HF_HUB_CACHE`.
2. Variable de entorno del shell:`HF_HOME` + `transformers/`.
3. Variable de entorno del shell: `XDG_CACHE_HOME` + `/huggingface/transformers`.


## Modo Offline

🤗 Transformers puede ejecutarse en un entorno con firewall o fuera de línea (offline) usando solo archivos locales. Configura la variable de entorno `HF_HUB_OFFLINE=1` para habilitar este comportamiento.

<Tip>

Puedes añadir [🤗 Datasets](https://huggingface.co/docs/datasets/) al flujo de entrenamiento offline declarando la variable de entorno  `HF_DATASETS_OFFLINE=1`.

</Tip>

### Obtener modelos y tokenizers para uso offline

Otra opción para usar 🤗 Transformers offline es descargando previamente los archivos y después apuntar al path local donde se encuentren. Hay tres maneras de hacer esto:

* Descarga un archivo mediante la interfaz de usuario del [Model Hub](https://huggingface.co/models) haciendo click en el ícono ↓.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)


* Utiliza el flujo de [`PreTrainedModel.from_pretrained`] y [`PreTrainedModel.save_pretrained`]:
    1. Descarga previamente los archivos con [`PreTrainedModel.from_pretrained`]:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```


    2. Guarda los archivos en un directorio específico con [`PreTrainedModel.save_pretrained`]:

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. Cuando te encuentres offline, recarga los archivos con [`PreTrainedModel.from_pretrained`] desde el directorio especificado:

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* Descarga de manera programática los archivos con la biblioteca [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub):

    1. Instala la biblioteca [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub) en tu entorno virtual:

    ```bash
    python -m pip install huggingface_hub
    ```

    2. Utiliza la función [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) para descargar un archivo a un path específico. Por ejemplo, el siguiente comando descarga el archivo `config.json` del modelo [T0](https://huggingface.co/bigscience/T0_3B) al path deseado:

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

Una vez que el archivo se descargue y se almacene en caché localmente, especifica tu ruta local para cargarlo y usarlo:

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

Para más detalles sobre cómo descargar archivos almacenados en el Hub consulta la sección [How to download files from the Hub](https://huggingface.co/docs/hub/how-to-downstream).

</Tip>
