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

# Compartir modelos personalizados

La biblioteca 🤗 Transformers está diseñada para ser fácilmente ampliable. Cada modelo está completamente codificado
sin abstracción en una subcarpeta determinada del repositorio, por lo que puedes copiar fácilmente un archivo del modelo
y ajustarlo según tus necesidades.

Si estás escribiendo un modelo completamente nuevo, podría ser más fácil comenzar desde cero. En este tutorial, te mostraremos
cómo escribir un modelo personalizado y su configuración para que pueda usarse dentro de Transformers, y cómo puedes compartirlo
con la comunidad (con el código en el que se basa) para que cualquiera pueda usarlo, incluso si no está presente en la biblioteca
🤗 Transformers.

Ilustraremos todo esto con un modelo ResNet, envolviendo la clase ResNet de la [biblioteca timm](https://github.com/rwightman/pytorch-image-models) en un [`PreTrainedModel`].

## Escribir una configuración personalizada

Antes de adentrarnos en el modelo, primero escribamos su configuración. La configuración de un modelo es un objeto que
contendrá toda la información necesaria para construir el modelo. Como veremos en la siguiente sección, el modelo solo puede
tomar un `config` para ser inicializado, por lo que realmente necesitamos que ese objeto esté lo más completo posible.

En nuestro ejemplo, tomaremos un par de argumentos de la clase ResNet que tal vez queramos modificar. Las diferentes
configuraciones nos darán los diferentes tipos de ResNet que son posibles. Luego simplemente almacenamos esos argumentos
después de verificar la validez de algunos de ellos.

```python
from transformers import PreTrainedConfig
from typing import List


class ResnetConfig(PreTrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: list[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

Las tres cosas importantes que debes recordar al escribir tu propia configuración son las siguientes:
- tienes que heredar de `PreTrainedConfig`,
- el `__init__` de tu `PreTrainedConfig` debe aceptar cualquier `kwargs`,
- esos `kwargs` deben pasarse a la superclase `__init__`.

La herencia es para asegurarte de obtener toda la funcionalidad de la biblioteca 🤗 Transformers, mientras que las otras dos
restricciones provienen del hecho de que una `PreTrainedConfig` tiene más campos que los que estás configurando. Al recargar una
`config` con el método `from_pretrained`, esos campos deben ser aceptados por tu `config` y luego enviados a la superclase.

Definir un `model_type` para tu configuración (en este caso `model_type="resnet"`) no es obligatorio, a menos que quieras
registrar tu modelo con las clases automáticas (ver la última sección).

Una vez hecho esto, puedes crear y guardar fácilmente tu configuración como lo harías con cualquier otra configuración de un
modelo de la biblioteca. Así es como podemos crear una configuración resnet50d y guardarla:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

Esto guardará un archivo llamado `config.json` dentro de la carpeta `custom-resnet`. Luego puedes volver a cargar tu configuración
con el método `from_pretrained`:

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

También puedes usar cualquier otro método de la clase [`PreTrainedConfig`], como [`~PreTrainedConfig.push_to_hub`], para cargar
directamente tu configuración en el Hub.

## Escribir un modelo personalizado

Ahora que tenemos nuestra configuración de ResNet, podemos seguir escribiendo el modelo. En realidad escribiremos dos: una que
extrae las características ocultas de un grupo de imágenes (como [`BertModel`]) y una que es adecuada para clasificación de
imagenes (como [`BertForSequenceClassification`]).

Como mencionamos antes, solo escribiremos un envoltura (_wrapper_) libre del modelo para simplificar este ejemplo. Lo único que debemos
hacer antes de escribir esta clase es un mapeo entre los tipos de bloques y las clases de bloques reales. Luego se define el
modelo desde la configuración pasando todo a la clase `ResNet`:

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

Para el modelo que clasificará las imágenes, solo cambiamos el método de avance (es decir, el método `forward`):

```py
import torch


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

En ambos casos, observa cómo heredamos de `PreTrainedModel` y llamamos a la inicialización de la superclase con `config`
(un poco como cuando escribes `torch.nn.Module`). La línea que establece `config_class` no es obligatoria, a menos
que quieras registrar tu modelo con las clases automáticas (consulta la última sección).

<Tip>

Si tu modelo es muy similar a un modelo dentro de la biblioteca, puedes reutilizar la misma configuración de ese modelo.

</Tip>

Puedes hacer que tu modelo devuelva lo que quieras, pero devolver un diccionario como lo hicimos para
`ResnetModelForImageClassification`, con el `loss` incluido cuando se pasan las etiquetas, hará que tu modelo se pueda
usar directamente dentro de la clase [`Trainer`]. Usar otro formato de salida está bien, siempre y cuando estés planeando usar
tu propio bucle de entrenamiento u otra biblioteca para el entrenamiento.

Ahora que tenemos nuestra clase, vamos a crear un modelo:

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

Nuevamente, puedes usar cualquiera de los métodos de [`PreTrainedModel`], como [`~PreTrainedModel.save_pretrained`] o
[`~PreTrainedModel.push_to_hub`]. Usaremos el segundo en la siguiente sección y veremos cómo pasar los pesos del modelo
con el código de nuestro modelo. Pero primero, carguemos algunos pesos previamente entrenados dentro de nuestro modelo.

En tu caso de uso, probablemente estarás entrenando tu modelo personalizado con tus propios datos. Para ir rápido en este
tutorial, usaremos la versión preentrenada de resnet50d. Dado que nuestro modelo es solo un envoltorio alrededor del resnet50d
original, será fácil transferir esos pesos:

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

Ahora veamos cómo asegurarnos de que cuando hacemos [`~PreTrainedModel.save_pretrained`] o [`~PreTrainedModel.push_to_hub`],
se guarda el código del modelo.

## Enviar el código al _Hub_

<Tip warning={true}>

Esta _API_ es experimental y puede tener algunos cambios leves en las próximas versiones.

</Tip>

Primero, asegúrate de que tu modelo esté completamente definido en un archivo `.py`. Puedes basarte en importaciones
relativas a otros archivos, siempre que todos los archivos estén en el mismo directorio (aún no admitimos submódulos
para esta característica). Para nuestro ejemplo, definiremos un archivo `modeling_resnet.py` y un archivo
`configuration_resnet.py` en una carpeta del directorio de trabajo actual llamado `resnet_model`. El archivo de configuración
contiene el código de `ResnetConfig` y el archivo del modelo contiene el código de `ResnetModel` y
`ResnetModelForImageClassification`.

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

El `__init__.py`  puede estar vacío, solo está ahí para que Python detecte que `resnet_model` se puede usar como un módulo.

<Tip warning={true}>

Si copias archivos del modelo desde la biblioteca, deberás reemplazar todas las importaciones relativas en la parte superior
del archivo para importarlos desde el paquete `transformers`.

</Tip>

Ten en cuenta que puedes reutilizar (o subclasificar) una configuración o modelo existente.

Para compartir tu modelo con la comunidad, sigue estos pasos: primero importa el modelo y la configuración de ResNet desde
los archivos recién creados:

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

Luego, debes decirle a la biblioteca que deseas copiar el código de esos objetos cuando usas el método `save_pretrained`
y registrarlos correctamente con una determinada clase automática (especialmente para modelos), simplemente ejecuta:

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

Ten en cuenta que no es necesario especificar una clase automática para la configuración (solo hay una clase automática
para ellos, [`AutoConfig`]), pero es diferente para los modelos. Tu modelo personalizado podría ser adecuado para muchas
tareas diferentes, por lo que debes especificar cuál de las clases automáticas es la correcta para tu modelo.

A continuación, vamos a crear la configuración y los modelos como lo hicimos antes:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

Ahora, para enviar el modelo al Hub, asegúrate de haber iniciado sesión. Ejecuta en tu terminal:

```bash
hf auth login
```

o desde un _notebook_:

```py
from huggingface_hub import notebook_login

notebook_login()
```

Luego puedes ingresar a tu propio espacio (o una organización de la que seas miembro) de esta manera:

```py
resnet50d.push_to_hub("custom-resnet50d")
```

Además de los pesos del modelo y la configuración en formato json, esto también copió los archivos `.py` del modelo y la
configuración en la carpeta `custom-resnet50d` y subió el resultado al Hub. Puedes verificar el resultado en este
[repositorio de modelos](https://huggingface.co/sgugger/custom-resnet50d).

Consulta el tutorial sobre cómo [compartir modelos](model_sharing) para obtener más información sobre el método para subir modelos al Hub.

## Usar un modelo con código personalizado

Puedes usar cualquier configuración, modelo o _tokenizador_ con archivos de código personalizado en tu repositorio con las
clases automáticas y el método `from_pretrained`. Todos los archivos y códigos cargados en el Hub se analizan en busca de
malware (consulta la documentación de [seguridad del Hub](https://huggingface.co/docs/hub/security#malware-scanning) para
obtener más información), pero aún debes revisar el código del modelo y el autor para evitar la ejecución de código malicioso
en tu computadora. Configura `trust_remote_code=True` para usar un modelo con código personalizado:

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

También se recomienda encarecidamente pasar un _hash_ de confirmación como una "revisión" para asegurarte de que el autor
de los modelos no actualizó el código con algunas líneas nuevas maliciosas (a menos que confíes plenamente en los autores
de los modelos).

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

Ten en cuenta que al navegar por el historial de confirmaciones del repositorio del modelo en Hub, hay un botón para copiar
fácilmente el hash de confirmación de cualquier _commit_.

## Registrar un model con código personalizado a las clases automáticas

Si estás escribiendo una biblioteca que amplía 🤗 Transformers, es posible que quieras ampliar las clases automáticas para
incluir tu propio modelo. Esto es diferente de enviar el código al Hub en el sentido de que los usuarios necesitarán importar
tu biblioteca para obtener los modelos personalizados (al contrario de descargar automáticamente el código del modelo desde Hub).

Siempre que tu configuración tenga un atributo `model_type` que sea diferente de los tipos de modelos existentes, y que tus
clases modelo tengan los atributos `config_class` correctos, puedes agregarlos a las clases automáticas de la siguiente manera:

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

Ten en cuenta que el primer argumento utilizado al registrar tu configuración personalizada en [`AutoConfig`] debe coincidir
con el `model_type` de tu configuración personalizada, y el primer argumento utilizado al registrar tus modelos personalizados
en cualquier clase del modelo automático debe coincidir con el `config_class ` de esos modelos.
