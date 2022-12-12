<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Debugging

## Debug de problemas de Network multi-GPU

Cuando entrenas o infieres con `DistributedDataParallel` y varias GPUs, si encuentras problemas de intercomunicación entre procesos y/o nodos, puedes usar el siguiente script para diagnosticar problemas de red.
 
```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

Por ejemplo, para probar cómo interactúan 2 GPUs, haz lo siguiente:

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
Si ambos procesos pueden hablar entre sí y asignar la memoria de la GPU, cada uno imprimirá un status OK.

Para más GPUs o nodos, ajusta los argumentos en el script.

Encontrarás muchos más detalles dentro del script de diagnóstico e incluso una receta de cómo ejecutarlo en un entorno SLURM.

Un nivel adicional de debug es agregar la variable de entorno `NCCL_DEBUG=INFO` de la siguiente manera:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

Esto mostrará mucha información de debug relacionada con NCCL, que luego puedes buscar online si encuentras que reporta algún problema. O si no estás seguro de cómo interpretar el output, puedes compartir el archivo de log en un Issue.


## Detección de Underflow y Overflow

<Tip>

Esta función está disponible actualmente sólo para PyTorch.

</Tip>

<Tip>

Para el entrenamiento multi-GPU, requiere DDP (`torch.distributed.launch`).

</Tip>

<Tip>

Esta función puede utilizarse con cualquier modelo basado en `nn.Module`.

</Tip>

Si empiezas a obtener `loss=NaN` o el modelo muestra algún otro comportamiento anormal debido a `inf` o `nan` en
activations o weights hay que descubrir dónde se produce el primer underflow o overflow y qué lo ha provocado. Por suerte
puedes lograrlo fácilmente activando un módulo especial que hará la detección automáticamente.

Si estás usando [`Trainer`], solo necesitas añadir:

```bash
--debug underflow_overflow
```

a los argumentos normales de la línea de comandos, o pasar `debug="underflow_overflow"` al crear el objeto [`TrainingArguments`].

Si estás usando tu propio bucle de entrenamiento u otro Trainer puedes lograr lo mismo con:

```python
from .debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

[`~debug_utils.DebugUnderflowOverflow`] inserta hooks en el modelo que inmediatamente después de cada forward
testeará las variables de input y output y también los weights del módulo correspondiente. Tan pronto como se detecte `inf` o
`nan` se detecta en al menos un elemento de las activations o weights, el programa afirmará e imprimirá un informe
como este (esto fue capturado con `google/mt5-small` bajo fp16 mixed precision):

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

El output del ejemplo se ha recortado en el centro por razones de brevedad.

La segunda columna muestra el valor del elemento más grande en términos absolutos, por lo que si observas con detenimiento los últimos fotogramas,
los inputs y outputs estaban en el rango de `1e4`. Así que cuando este entrenamiento se hizo con fp16 mixed precision, 
el último paso sufrió overflow (ya que bajo `fp16` el mayor número antes de `inf` es `64e3`). Para evitar overflows en
`fp16` las activations deben permanecer muy por debajo de `1e4`, porque `1e4 * 1e4 = 1e8` por lo que cualquier matrix multiplication con
grandes activations va a llevar a una condición de overflow numérico.

Al principio del output puedes descubrir en qué número de batch se produjo el problema (aquí `Detected inf/nan during batch_number=0` significa que el problema se produjo en el primer batch).

Cada frame del informe comienza declarando la entrada completamente calificada para el módulo correspondiente que este frame está reportando.
Si nos fijamos sólo en este frame:

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

Aquí, `encoder.block.2.layer.1.layer_norm` indica que era una layer norm para la primera capa, del segundo
block del encoder. Y la call específica del `forward` es `T5LayerNorm`.

Veamos los últimos frames de ese informe:

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

El último frame informa para la función `Dropout.forward` con la primera entrada para el único input y la segunda para el
único output. Puedes ver que fue llamada desde un atributo `dropout` dentro de la clase `DenseReluDense`. Podemos ver
que ocurrió durante la primera capa, del segundo block, durante el primer batch. Por último, el mayor absoluto
elementos de input fue `6.27e+04` y el mismo para el output fue `inf`.

Puedes ver aquí, que `T5DenseGatedGeluDense.forward` resultó en output activations, cuyo valor máximo absoluto fue
alrededor de 62.7K, que está muy cerca del límite máximo de fp16 de 64K. En el siguiente frame tenemos `Dropout`, el cual renormaliza
los weights, después de poner a cero algunos de los elementos, lo que empuja el valor máximo absoluto a más de 64K, y obtenemos un
overflow (`inf`).

Como puedes ver son los frames anteriores los que tenemos que mirar cuando los números empiezan a ser muy grandes para números fp16.

Combinemos el informe con el código de `models/t5/modeling_t5.py`:

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

Ahora es fácil ver la call `dropout`, y también todas las calls anteriores.

Dado que la detección se produce en un forward hook, estos informes se imprimen inmediatamente después de que cada `forward`
responda.

Volviendo al informe completo, para actuar sobre él y arreglar el problema, tenemos que subir unos cuantos frames donde los números
empezaron a subir y probablemente cambiar al modo `fp32` aquí, para que los números no sufran overflow cuando se multipliquen
o al sumarlos. Por supuesto, puede haber otras soluciones. Por ejemplo, podríamos desactivar `amp` temporalmente si está
activado, después de mover el original `forward` dentro de un helper wrapper, así:

```python
def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


import torch


def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

Como el detector automático sólo informa de los inputs y outputs de los frames completos, una vez que sepas dónde buscar, puedes
analizar también las etapas intermedias de una función específica de `forward`. En este caso, puede utilizar la función
función de ayuda `detect_overflow` para inyectar el detector donde quieras, por ejemplo:

```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

Puedes ver que hemos añadido 2 de estos y ahora se trackea si `inf` o `nan` para `forwarded_states` fue detectado
en algún punto intermedio.

De hecho, el detector ya informa de esto porque cada una de las llamadas en el ejemplo anterior es un `nn.Module`, pero
digamos que si tuvieras algunos cálculos directos locales, así es como lo harías.

Además, si estás instanciando el debugger en tu propio código, puedes ajustar el número de frames impresos de
su valor por defecto, por ejemplo:

```python
from .debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### Rastreo de valores mínimos y máximos absolutos de batches específicos

La misma clase de debugging se puede utilizar para el rastreo por batches con la función de detección de underflow/overflow desactivada.

Digamos que quieres ver los valores mínimos y máximos absolutos de todos los ingredientes de cada call `forward` de un determinado
batch, y sólo hacerlo para los batches 1 y 3. Entonces instancias esta clase como:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

Y ahora los batches 1 y 3 completos serán rastreados usando el mismo formato que el detector de underflow/overflow.

Los batches son 0-index.

Esto es muy útil si sabes que el programa empieza a comportarse mal después de un determinado número de batch, para que puedas avanzar rápidamente
hasta esa área. Aquí hay un ejemplo de output recortado para tal configuración:

```
                  *** Starting batch number=1 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     not a tensor output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     not a tensor output

                  *** Starting batch number=3 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

Aquí obtendrás un gran número de frames mostrados - tantos como forward calls haya en tu modelo, por lo que puede o no ser lo que quieras, pero a veces puede ser más fácil de usar para debug que un debugger normal.
Por ejemplo, si un problema comienza a ocurrir en el batch 150. Entonces puedes mostrar las trazas de los batches 149 y 150 y comparar dónde
los números empezaron a divergir.

También puedes especificar el número de batch después del cual se debe detener el entrenamiento, con:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
