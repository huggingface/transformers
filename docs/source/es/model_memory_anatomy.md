<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Anatomía del entrenamiento de los modelos

Para entender las técnicas de optimización del rendimiento que se pueden aplicar para mejorar la eficiencia en la velocidad del entrenamiento de los modelos y la utilización de la memoria, es útil familiarizarse con cómo se utiliza la GPU durante el entrenamiento y cómo varía la intensidad de cálculo según la operación realizada.

Empecemos explorando un ejemplo enfocado en la utilización de la GPU y la ejecución del entrenamiento de un modelo. Para la demostración, necesitaremos instalar algunas bibliotecas:

```bash
pip install transformers datasets accelerate nvidia-ml-py3
```

La biblioteca `nvidia-ml-py3` nos permite monitorear la utilización de memoria de los modelos desde Python. Es posible que estés familiarizado con el comando `nvidia-smi` en la terminal, esta biblioteca nos permite acceder a la misma información en Python directamente.

Luego, creamos algunos datos ficticios: IDs de tokens aleatorios entre 100 y 30000 y etiquetas binarias para un clasificador. En total, obtenemos 512 secuencias cada una con longitud 512 y las almacenamos en un [`~datasets.Dataset`] con formato PyTorch.


```py
>>> import numpy as np
>>> from datasets import Dataset


>>> seq_len, dataset_size = 512, 512
>>> dummy_data = {
...     "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
...     "labels": np.random.randint(0, 1, (dataset_size)),
... }
>>> ds = Dataset.from_dict(dummy_data)
>>> ds.set_format("pt")
```

Para imprimir estadísticas resumidas para la utilización de la GPU y la ejecución del entrenamiento con [`Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer), definimos dos funciones auxiliares:

```py
>>> from pynvml import *


>>> def print_gpu_utilization():
...     nvmlInit()
...     handle = nvmlDeviceGetHandleByIndex(0)
...     info = nvmlDeviceGetMemoryInfo(handle)
...     print(f"GPU memory occupied: {info.used//1024**2} MB.")


>>> def print_summary(result):
...     print(f"Time: {result.metrics['train_runtime']:.2f}")
...     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
...     print_gpu_utilization()
```

Comencemos comprobando que la memoria GPU este libre:

```py
>>> print_gpu_utilization()
GPU memory occupied: 0 MB.
```

Parece estar bien: la memoria de la GPU no está ocupada como esperaríamos antes de cargar cualquier modelo. Si no es el caso en tu máquina, asegúrate de detener todos los procesos que estén utilizando la memoria de la GPU. Sin embargo, no toda la memoria libre de la GPU puede ser utilizada por el usuario. Cuando se carga un modelo en la GPU, también se cargan los kernels, lo que puede ocupar 1-2GB de memoria. Para ver cuánta memoria será ocupada por defecto, cargemos un tensor diminuto en la GPU, lo que también desencadena la carga de los kernels.

```py
>>> import torch


>>> torch.ones((1, 1)).to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 1343 MB.
```

Vemos que los kernels solos ocupan 1,3GB de memoria de la GPU. Ahora, veamos cuánto espacio ocupa el modelo.

## Cargar el Modelo

Primero, cargamos el modelo `google-bert/bert-large-uncased`. Los pesos del modelo son cargados directamente en la GPU para que podamos verificar cuánto espacio ocupan solo los pesos.

```py
>>> from transformers import AutoModelForSequenceClassification


>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-large-uncased").to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 2631 MB.
```

Podemos ver que los pesos del modelo solos ocupan 1,3 GB de memoria de la GPU. El número exacto depende de la GPU específica que estés utilizando. Ten en cuenta que en GPUs más modernas, un modelo puede ocupar más espacio ya que los pesos se cargan de manera optimizada lo cual acelera el uso del modelo. Ahora también podemos verificar rápidamente si obtenemos el mismo resultado que con la CLI de `nvidia-smi`:

```bash
nvidia-smi
```

```bash
Tue Jan 11 08:58:05 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   37C    P0    39W / 300W |   2631MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      3721      C   ...nvs/codeparrot/bin/python     2629MiB |
+-----------------------------------------------------------------------------+
```

Obtenemos el mismo número que antes y también puedes ver que estamos utilizando una GPU V100 con 16GB de memoria. Ahora podemos empezar a entrenar el modelo y ver cómo cambia el consumo de memoria de la GPU. Primero, configuramos algunos argumentos de entrenamiento estándar:

```py
default_args = {
    "output_dir": "tmp",
    "eval_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
```

<Tip>

Si planeas ejecutar varias pruebas, reinicie el kernel de Python entre cada prueba para borrar correctamente la memoria.

</Tip>

## Utilización de la memoria en el entrenamiento

Vamos a utilizar el [`Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer) y entrenar el modelo sin utilizar ninguna técnica de optimización del rendimiento de la GPU y un tamaño de lote de 4:

```py
>>> from transformers import TrainingArguments, Trainer, logging

>>> logging.set_verbosity_error()


>>> training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
>>> trainer = Trainer(model=model, args=training_args, train_dataset=ds)
>>> result = trainer.train()
>>> print_summary(result)
```

```
Time: 57.82
Samples/second: 8.86
GPU memory occupied: 14949 MB.
```

Vemos que incluso un tamaño de lote relativamente pequeño casi llena toda la memoria de nuestra GPU. Sin embargo, un tamaño de lote más grande a menudo puede resultar en una convergencia del modelo más rápida o un mejor rendimiento final. Así que idealmente queremos ajustar el tamaño del lote a las necesidades del modelo y no a las limitaciones de la GPU. Lo interesante es que utilizamos mucha más memoria que el tamaño del modelo. 
Para entender un poco mejor por qué es el caso, echemos un vistazo a las operaciones y necesidades de memoria de un modelo.

## Anatomía de las Operaciones del Modelo

La arquitectura de los transformers incluye 3 grupos principales de operaciones agrupadas a continuación por intensidad de cálculo.

1. **Contracciones de Tensores**

    Las capas lineales y componentes de la Atención Multi-Head realizan **multiplicaciones matriciales por lotes**. Estas operaciones son la parte más intensiva en cálculo del entrenamiento de los transformers.

2. **Normalizaciones Estadísticas**

    Softmax y normalización de capas son menos intensivas en cálculo que las contracciones de tensores, e implican una o más **operaciones de reducción**, cuyo resultado se aplica luego mediante un mapa.

3. **Operadores por Elemento**

    Estos son los operadores restantes: **sesgos, dropout, activaciones y conexiones residuales**. Estas son las operaciones menos intensivas en cálculo.

Este conocimiento puede ser útil al analizar cuellos de botella de rendimiento.

Este resumen se deriva de [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)


## Anatomía de la Memoria del Modelo

Hemos visto que al entrenar un modelo se utiliza mucha más memoria que solo poner el modelo en la GPU. Esto se debe a que hay muchos componentes durante el entrenamiento que utilizan memoria de la GPU. Los componentes en memoria de la GPU son los siguientes:

1. pesos del modelo
2. estados del optimizador
3. gradientes
4. activaciones hacia adelante guardadas para el cálculo del gradiente
5. buffers temporales
6. memoria específica de funcionalidad

Un modelo típico entrenado en precisión mixta con AdamW requiere 18 bytes por parámetro del modelo más memoria de activación. Para la inferencia no hay estados del optimizador ni gradientes, por lo que podemos restarlos. Y así terminamos con 6 bytes por parámetro del modelo para la inferencia en precisión mixta, más la memoria de activación.

Veámoslo a detalle:

**Pesos del Modelo:**

- 4 bytes por número de parámetros para entrenamiento en fp32
- 6 bytes por número de parámetros para entrenamiento en precisión mixta (mantiene un modelo en fp32 y uno en fp16 en memoria)

**Estados del Optimizador:**

- 8 bytes por número de parámetros para un AdamW normal (mantiene 2 estados)
- 2 bytes por número de parámetros para optimizadores de 8 bits como [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- 4 bytes por número de parámetros para optimizadores como SGD con momentum (mantiene solo 1 estado)

**Gradientes**

- 4 bytes por número de parámetros para entrenamiento en fp32 o precisión mixta (los gradientes siempre se mantienen en fp32)

**Activaciones hacia Adelante**

- El tamaño depende de muchos factores, los principales siendo la longitud de la secuencia, el tamaño oculto y el tamaño de lote.

Hay entradas y salidas que se pasan y se devuelven por las funciones hacia adelante y hacia atrás, y las activaciones hacia adelante (*forward activations*) guardadas para el cálculo del gradiente.

**Memoria Temporal**

Además, hay todas clases de variables temporales que se liberan una vez que se completa el cálculo, pero en el momento podrían requerir memoria adicional y podrían provocar un error de memoria insuficiente. Por lo tanto, al codificar es crucial pensar estratégicamente sobre tales variables temporales y a veces liberarlas explícitamente tan pronto como ya no se necesitan.

**Memoria Específica de Funcionalidad**

Entonces, su software podría tener necesidades especiales de memoria. Por ejemplo, al generar texto mediante la búsqueda por haz, el software necesita mantener múltiples copias de las entradas y salidas.

**Velocidad de Ejecución `forward` vs `backward`**

Para convoluciones y capas lineales, hay 2x flops en la ejecución hacia atrás (`backward`) en comparación con la ejecución hacia adelante (`forward`), lo que generalmente se traduce en ~2x más lento (a veces más, porque los tamaños en la ejecución hacia atrás tienden a ser más complejos). Las activaciones suelen ser limitadas por ancho de banda, y es típico que una activación tenga que leer más datos en la ejecución hacia atrás que en la ejecución hacia adelante (por ejemplo, la activación hacia adelante lee una vez, escribe una vez, la activación hacia atrás lee dos veces, gradOutput y salida de la ejecución hacia adelante, y escribe una vez, gradInput).

Como puedes ver, hay potencialmente unos pocos lugares donde podríamos ahorrar memoria de la GPU o acelerar operaciones. Ahora que entiendes qué afecta la utilización de la GPU y la velocidad de cálculo, consulta la página de documentación [Métodos y herramientas para entrenamiento eficiente en una sola GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one) para aprender sobre técnicas de optimización del rendimiento.
