<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Exportar a TorchScript

<Tip>
Este es el comienzo de nuestros experimentos con TorchScript y todavía estamos explorando sus capacidades con modelos de variables de entrada. Es un tema de interés para nosotros y profundizaremos en nuestro análisis en las próximas versiones, con más ejemplos de código, una implementación más flexible y comparativas de rendimiento comparando códigos basados en Python con TorchScript compilado.  

</Tip>

De acuerdo con la documentación de TorchScript: 

> TorchScript es una manera de crear modelos serializables y optimizables a partir del código PyTorch.

Hay dos módulos de PyTorch, [JIT y TRACE](https://hf.co/docs/accelerate), que permiten a los desarrolladores exportar sus modelos para ser reusados en otros programas, como los programas de C++ orientados a la eficiencia.

Nosotros proveemos una interface que te permite exportar los modelos Transformes a TorchScript para que puedan ser reusados en un entorno diferente al de los programas Python basados en PyTorch. Aquí explicamos como exportar y usar nuestros modelos utilizando TorchScript.

Exportar un modelo requiere de dos cosas:

- La instanciación del modelo con la bandera TorchScript.
- Un paso hacia adelante con entradas ficticias.

Estas necesidades implican varias cosas de las que los desarrolladores deben tener cuidado, como se detalla a continuación.

## Bandera TorchScript y pesos atados.

La bandera `torchscript` es necesaria porque la mayoría de los modelos de lenguaje de Transformers tienen pesos atados entre su `capa de incrustación` (`Embedding`) y su `capa de decodificación` (`Decoding`). TorchScript no te permite exportar modelos que tienen pesos atados, por lo que es necesario desatar y clonar los pesos de antemano.

Los modelos instanciados con la bandera `torchscript` tienen su `capa de incrustación` (`Embedding`) y su `capa de decodificación` (`Decoding`) separadas, lo que significa que no deben ser entrenados más adelante. Entrenar desincronizaría las dos capas, lo que llevaría a resultados inesperados.

Esto no es así para los modelos que no tienen una cabeza de modelo de lenguaje, ya que esos modelos no tienen pesos atados. Estos modelos pueden ser exportados de manera segura sin la bandera `torchscript`.

## Entradas ficticias y longitudes estándar

Las entradas ficticias se utilizan para un paso del modelo hacia adelante. Mientras los valores de las entradas se propagan a través de las capas, PyTorch realiza un seguimiento de las diferentes operaciones ejecutadas en cada tensor. Estas operaciones registradas se utilizan luego para crear *la traza* del modelo.
La traza se crea en relación con las dimensiones de las entradas. Por lo tanto, está limitada por las dimensiones de la entrada ficticia y no funcionará para ninguna otra longitud de secuencia o tamaño de lote. Cuando se intenta con un tamaño diferente, se genera el siguiente error:

```
`El tamaño expandido del tensor (3) debe coincidir con el tamaño existente (7) en la dimensión no singleton 2`.
```

Recomendamos trazar el modelo con un tamaño de entrada ficticio al menos tan grande como la entrada más grande con la que se alimentará al modelo durante la inferencia. El relleno puede ayudar a completar los valores faltantes. Sin embargo, dado que el modelo se traza con un tamaño de entrada más grande, las dimensiones de la matriz también serán grandes, lo que resultará en más cálculos.

Ten cuidado con el número total de operaciones realizadas en cada entrada y sigue de cerca el rendimiento al exportar modelos con longitudes de secuencia variables.

## Usando TorchScript en Python

Esta sección demuestra cómo guardar y cargar modelos, así como cómo usar la traza para la inferencia.

Guardando un modelo

Para exportar un `BertModel` con TorchScript, instancia `BertModel` a partir de la clase `BertConfig` y luego guárdalo en disco bajo el nombre de archivo `traced_bert.pt`:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with *from_pretrained* you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```
### Cargando un modelo

Ahora puedes cargar el `BertModel` guardado anteriormente, `traced_bert.pt`, desde el disco y usarlo en la entrada ficticia (`dummy_input`) previamente inicializada:

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

## Usando un modelo trazado para inferencia

Utiliza el modelo trazado para inferencia utilizando su método `_call_` dunder:

```python
traced_model(tokens_tensor, segments_tensors)
```
## Despliega modelos TorchScript de Hugging Face en AWS con el Neuron SDK

AWS introdujo la familia de instancias Amazon EC2 Inf1 para inferencia de aprendizaje automático de alto rendimiento y bajo costo en la nube. Las instancias Inf1 están alimentadas por el chip AWS Inferentia, un acelerador de hardware personalizado que se especializa en cargas de trabajo de inferencia de aprendizaje profundo. [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) es el SDK para Inferentia que admite el trazado y la optimización de modelos de transformers para implementación en Inf1. El SDK Neuron proporciona:

1. Una API fácil de usar con un solo cambio de línea de código para trazar y optimizar un modelo TorchScript para inferencia en la nube.

2. Optimizaciones de rendimiento listas para usar [para mejorar el rendimiento y el costo](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/>).

3. Soporte para modelos de transformers de Hugging Face construidos tanto con [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html) como con [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html).

### Implicaciones

Los modelos transformers basados en la arquitectura [BERT (Bidirectional Encoder Representations from Transformers)](https://huggingface.co/docs/transformers/main/model_doc/bert), o sus variantes como [distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) y [roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta), funcionan mejor en Inf1 para tareas no generativas como la respuesta a preguntas extractivas, la clasificación de secuencias y la clasificación de tokens. Sin embargo, las tareas de generación de texto aún pueden adaptarse para ejecutarse en Inf1 según este [tutorial de AWS Neuron MarianMT](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html). Se puede encontrar más información sobre los modelos que se pueden convertir fácilmente para usar en Inferentia en la sección de [Model Architecture Fit](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia) de la documentación de Neuron.

### Dependencias

El uso de AWS Neuron para convertir modelos requiere un [entorno de Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide) que viene preconfigurado en [la AMI de AWS Deep Learning](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html).

### Convertir un modelo para AWS Neuron

Convierte un modelo para AWS NEURON utilizando el mismo código de [Uso de TorchScript en Python](torchscript#using-torchscript-in-python) para trazar un `BertModel`. Importa la extensión del framework `torch.neuron` para acceder a los componentes del Neuron SDK a través de una API de Python:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```
Solo necesitas la linea sigueda:

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [token_tensor, segments_tensors])
```

Esto permite que el Neuron SDK trace el modelo y lo optimice para las instancias Inf1.

Para obtener más información sobre las características, herramientas, tutoriales de ejemplo y últimas actualizaciones del AWS Neuron SDK, consulta [la documentación de AWS NeuronSDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html).