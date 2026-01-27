<!--Copyright 2020 de The HuggingFace Team. Todos los derechos reservados

Con licencia bajo la Licencia Apache, Versión 2.0 (la "Licencia"); No puedes usar este archivo excepto de conformidad con la Licencia.
Puedes obtener una copia de la Licencia en

http://www.apache.org/licenses/LICENSE-2.0

Al menos que sea requrido por la ley aplicable o acordado por escrito, el software distribuido bajo la Licencia es distribuido sobre una BASE "AS IS", SIN GARANTIAS O CONDICIONES DE
NINGÚN TIPO. Ver la Licencia para el idioma específico que rige los permisos y limitaciones bajo la Licencia.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Filosofía

🤗 Transformers es una biblioteca construida para:

- Los investigadores y educadores de NLP que busquen usar/estudiar/extender modelos transformers a gran escala 
- Profesionales que quieren optimizar esos modelos y/o ponerlos en producción 
- Ingenieros que solo quieren descargar un modelo preentrenado y usarlo para resolver una tarea NLP dada. 

La biblioteca fue diseñada con dos fuertes objetivos en mente:

- Que sea tan fácil y rápida de utilizar como sea posible:

  - Hemos limitado enormemente el número de abstracciones que el usuario tiene que aprender. De hecho, no hay casi abstracciones,
    solo tres clases estándar necesarias para usar cada modelo: [configuration](main_classes/configuration),
    [models](main_classes/model) y [tokenizer](main_classes/tokenizer).
  - Todas estas clases pueden ser inicializadas de forma simple y unificada a partir de ejemplos pre-entrenados mediante el uso de un método
    `from_pretrained()` común de solicitud que se encargará de descargar (si es necesario), almacenar y cargar la solicitud de clase relacionada y datos asociados
    (configurations' hyper-parameters, tokenizers' vocabulary, and models' weights) a partir de un control pre-entrenado proporcionado en
    [Hugging Face Hub](https://huggingface.co/models) o de tu propio control guardado.
  - Por encima de esas tres clases estándar, la biblioteca proporciona dos APIs: [`pipeline`] para usar rápidamente un modelo (junto a su configuracion y tokenizer asociados)
    sobre una tarea dada, y [`Trainer`]/`Keras.fit` para entrenar u optimizar de forma rápida un modelo dado.
  - Como consecuencia, esta biblioteca NO es una caja de herramientas modular de bloques individuales para redes neuronales. Si quieres extender/construir sobre la biblioteca,
    usa simplemente los módulos regulares de Python/PyTorch/TensorFlow/Keras y emplea las clases estándar de la biblioteca como punto de partida para reutilizar funcionalidades
    tales como abrir/guardar modelo.
    
- Proporciona modelos modernos con rendimientos lo más parecido posible a los modelos originales:

  - Proporcionamos al menos un ejemplo para cada arquitectura que reproduce un resultado proporcionado por los autores de dicha arquitectura.
  - El código normalmente es parecido al código base original, lo cual significa que algún código Pytorch puede no ser tan 
    *pytorchic* como podría ser por haber sido convertido a código TensorFlow, y viceversa. 

Unos cuantos objetivos adicionales:

- Exponer las características internas de los modelos de la forma más coherente posible:

  - Damos acceso, mediante una sola API, a todos los estados ocultos y pesos de atención.
  - Tokenizer y el modelo de API base están estandarizados para cambiar fácilmente entre modelos.

- Incorporar una selección subjetiva de herramientas de gran potencial para la optimización/investigación de estos modelos:

  - Una forma sencilla/coherente de añadir nuevos tokens al vocabulario e incrustraciones (embeddings, en inglés) para optimización.
  - Formas sencillas de camuflar y reducir "transformer heads".

- Cambiar fácilmente entre PyTorch y TensorFlow 2.0, permitiendo el entrenamiento usando un marco y la inferencia usando otro.

## Conceptos principales 

La biblioteca está construida alrededor de tres tipos de clases para cada modelo:

- **Model classes** como [`BertModel`], que consisten en más de 30 modelos PyTorch ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)) o modelos Keras ([tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)) que funcionan con pesos pre-entrenados proporcionados en la
  biblioteca.
- **Configuration classes** como [`BertConfig`], que almacena todos los parámetros necesarios para construir un modelo. 
  No siempre tienes que generarla tu. En particular, si estas usando un modelo pre-entrenado sin ninguna modificación,
  la creación del modelo se encargará automáticamente de generar la configuración (que es parte del modelo).
- **Tokenizer classes** como [`BertTokenizer`], que almacena el vocabulario para cada modelo y proporciona métodos para
  codificar/decodificar strings en una lista de índices de "token embeddings" para ser empleados en un modelo.          

Todas estas clases pueden ser generadas a partir de ejemplos pre-entrenados, y guardados localmente usando dos métodos:

- `from_pretrained()` permite generar un modelo/configuración/tokenizer a partir de una versión pre-entrenada proporcionada ya sea por
  la propia biblioteca (los modelos compatibles se pueden encontrar en [Model Hub](https://huggingface.co/models)) o
  guardados localmente (o en un servidor) por el usuario. 
- `save_pretrained()` permite guardar un modelo/configuración/tokenizer localmente, de forma que puede ser empleado de nuevo usando
  `from_pretrained()`.
