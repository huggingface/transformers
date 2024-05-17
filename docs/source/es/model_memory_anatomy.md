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

Para imprimir estadísticas resumidas para la utilización de la GPU y la ejecución del entrenamiento con [`Trainer`], definimos dos funciones auxiliares:

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

Parece estar bien: la memoria de la GPU no está ocupada como esperaríamos antes de cargar cualquier modelo. Si no es el caso en tu máquina, asegúrate de detener todos los procesos que estén utilizando la memoria de la GPU. Sin embargo, no toda la memoria libre de la GPU puede ser utilizada por el usuario. Cuando se carga un modelo en la GPU, también se cargan los kernels, lo que puede ocupar 1-2GB de memoria. Para ver cuánta es, cargemos un tensor diminuto en la GPU, lo que también desencadena la carga de los kernels.

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

## Memory utilization at vanilla training

Let's use the [`Trainer`] and train the model without using any GPU performance optimization techniques and a batch size of 4:

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

We see that already a relatively small batch size almost fills up our GPU's entire memory. However, a larger batch size 
can often result in faster model convergence or better end performance. So ideally we want to tune the batch size to our
model's needs and not to the GPU limitations. What's interesting is that we use much more memory than the size of the model. 
To understand a bit better why this is the case let's have a look at a model's operations and memory needs.

## Anatomy of Model's Operations

Transformers architecture includes 3 main groups of operations grouped below by compute-intensity.

1. **Tensor Contractions**

    Linear layers and components of Multi-Head Attention all do batched **matrix-matrix multiplications**. These operations are the most compute-intensive part of training a transformer.

2. **Statistical Normalizations**

    Softmax and layer normalization are less compute-intensive than tensor contractions, and involve one or more **reduction operations**, the result of which is then applied via a map.

3. **Element-wise Operators**

    These are the remaining operators: **biases, dropout, activations, and residual connections**. These are the least compute-intensive operations.

This knowledge can be helpful to know when analyzing performance bottlenecks.

This summary is derived from [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)


## Anatomy of Model's Memory

We've seen that training the model uses much more memory than just putting the model on the GPU. This is because there 
are many components during training that use GPU memory. The components on GPU memory are the following:

1. model weights
2. optimizer states
3. gradients
4. forward activations saved for gradient computation
5. temporary buffers
6. functionality-specific memory

A typical model trained in mixed precision with AdamW requires 18 bytes per model parameter plus activation memory. For 
inference there are no optimizer states and gradients, so we can subtract those. And thus we end up with 6 bytes per 
model parameter for mixed precision inference, plus activation memory.

Let's look at the details.

**Model Weights:**

- 4 bytes * number of parameters for fp32 training
- 6 bytes * number of parameters for mixed precision training (maintains a model in fp32 and one in fp16 in memory)

**Optimizer States:**

- 8 bytes * number of parameters for normal AdamW (maintains 2 states)
- 2 bytes * number of parameters for 8-bit AdamW optimizers like [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- 4 bytes * number of parameters for optimizers like SGD with momentum (maintains only 1 state)

**Gradients**

- 4 bytes * number of parameters for either fp32 or mixed precision training (gradients are always kept in fp32)

**Forward Activations**

- size depends on many factors, the key ones being sequence length, hidden size and batch size.

There are the input and output that are being passed and returned by the forward and the backward functions and the 
forward activations saved for gradient computation.

**Temporary Memory**

Additionally, there are all kinds of temporary variables which get released once the calculation is done, but in the 
moment these could require additional memory and could push to OOM. Therefore, when coding it's crucial to think 
strategically about such temporary variables and sometimes to explicitly free those as soon as they are no longer needed.

**Functionality-specific memory**

Then, your software could have special memory needs. For example, when generating text using beam search, the software 
needs to maintain multiple copies of inputs and outputs.

**`forward` vs `backward` Execution Speed**

For convolutions and linear layers there are 2x flops in the backward compared to the forward, which generally translates 
into ~2x slower (sometimes more, because sizes in the backward tend to be more awkward). Activations are usually 
bandwidth-limited, and it’s typical for an activation to have to read more data in the backward than in the forward 
(e.g. activation forward reads once, writes once, activation backward reads twice, gradOutput and output of the forward, 
and writes once, gradInput).

As you can see, there are potentially a few places where we could save GPU memory or speed up operations. 
Now that you understand what affects GPU utilization and computation speed, refer to 
the [Methods and tools for efficient training on a single GPU](perf_train_gpu_one) documentation page to learn about 
performance optimization techniques. 
