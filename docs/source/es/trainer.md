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

# El Trainer

El [`Trainer`] es un bucle completo de entrenamiento y evaluación para modelos de PyTorch implementado en la biblioteca Transformers. Solo necesitas pasarle las piezas necesarias para el entrenamiento (modelo, tokenizador, conjunto de datos, función de evaluación, hiperparámetros de entrenamiento, etc.), y la clase [`Trainer`] se encarga del resto. Esto facilita comenzar a entrenar más rápido sin tener que escribir manualmente tu propio bucle de entrenamiento. Pero al mismo tiempo, [`Trainer`] es muy personalizable y ofrece una gran cantidad de opciones de entrenamiento para que puedas adaptarlo a tus necesidades exactas de entrenamiento.

<Tip>

Además de la clase [`Trainer`], Transformers también proporciona una clase [`Seq2SeqTrainer`] para tareas de secuencia a secuencia como traducción o resumen. También está la clase [~trl.SFTTrainer] de la biblioteca [TRL](https://hf.co/docs/trl) que envuelve la clase [`Trainer`] y está optimizada para entrenar modelos de lenguaje como Llama-2 y Mistral con técnicas autoregresivas. [`~trl.SFTTrainer`] también admite funciones como el empaquetado de secuencias, LoRA, cuantización y DeepSpeed para escalar eficientemente a cualquier tamaño de modelo.

<br>

Siéntete libre de consultar [la referencia de API](./main_classes/trainer) para estas otras clases tipo [`Trainer`] para aprender más sobre cuándo usar cada una. En general, [`Trainer`] es la opción más versátil y es apropiada para una amplia gama de tareas. [`Seq2SeqTrainer`] está diseñado para tareas de secuencia a secuencia y [`~trl.SFTTrainer`] está diseñado para entrenar modelos de lenguaje.

</Tip>

Antes de comenzar, asegúrate de tener instalado [Accelerate](https://hf.co/docs/accelerate), una biblioteca para habilitar y ejecutar el entrenamiento de PyTorch en entornos distribuidos.

```bash
pip install accelerate

# upgrade
pip install accelerate --upgrade
```
Esta guía proporciona una visión general de la clase [`Trainer`].

## Uso básico

[`Trainer`] incluye todo el código que encontrarías en un bucle de entrenamiento básico:
1.  Realiza un paso de entrenamiento para calcular la pérdida
2.  Calcula los gradientes con el método [~accelerate.Accelerator.backward]
3.  Actualiza los pesos basados en los gradientes
4.  Repite este proceso hasta alcanzar un número predeterminado de épocas

La clase [`Trainer`] abstrae todo este código para que no tengas que preocuparte por escribir manualmente un bucle de entrenamiento cada vez o si estás empezando con PyTorch y el entrenamiento. Solo necesitas proporcionar los componentes esenciales requeridos para el entrenamiento, como un modelo y un conjunto de datos, y la clase [`Trainer`] maneja todo lo demás.

Si deseas especificar opciones de entrenamiento o hiperparámetros, puedes encontrarlos en la clase [`TrainingArguments`]. Por ejemplo, vamos a definir dónde guardar el modelo en output_dir y subir el modelo al Hub después del entrenamiento con `push_to_hub=True`.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
```

Pase `training_args` al [`Trainer`] con un modelo, un conjunto de datos o algo para preprocesar el conjunto de datos (dependiendo en el tipo de datos pueda ser un tokenizer, extractor de caracteristicas o procesor del imagen), un recopilador de datos y una función para calcular las métricas que desea rastrear durante el entrenamiento.

Finalmente, llame [`~Trainer.train`] para empezar entrenamiento!

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Los puntos de control

La clase [`Trainer`] guarda los puntos de control del modelo en el directorio especificado en el parámetro `output_dir` de [`TrainingArguments`]. Encontrarás los puntos de control guardados en una subcarpeta checkpoint-000 donde los números al final corresponden al paso de entrenamiento. Guardar puntos de control es útil para reanudar el entrenamiento más tarde.

```py
# resume from latest checkpoint
trainer.train(resume_from_checkpoint=True)

# resume from specific checkpoint saved in output directory
trainer.train(resume_from_checkpoint="your-model/checkpoint-1000")
```

Puedes guardar tus puntos de control (por defecto, el estado del optimizador no se guarda) en el Hub configurando `push_to_hub=True` en [`TrainingArguments`] para confirmar y enviarlos. Otras opciones para decidir cómo se guardan tus puntos de control están configuradas en el parámetro [`hub_strategy`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_strategy):

* hub_strategy="checkpoint" envía el último punto de control a una subcarpeta llamada "last-checkpoint" desde la cual puedes reanudar el entrenamiento.

* hub_strategy="all_checkpoints" envía todos los puntos de control al directorio definido en `output_dir` (verás un punto de control por carpeta en tu repositorio de modelos).

Cuando reanudas el entrenamiento desde un punto de control, el [`Trainer`] intenta mantener los estados de los generadores de números aleatorios (RNG) de Python, NumPy y PyTorch iguales a como estaban cuando se guardó el punto de control. Pero debido a que PyTorch tiene varias configuraciones predeterminadas no determinísticas, no se garantiza que los estados de RNG sean los mismos. Si deseas habilitar la plena determinismo, echa un vistazo a la guía ["Controlling sources of randomness"](https://pytorch.org/docs/stable/notes/randomness#controlling-sources-of-randomness) para aprender qué puedes habilitar para hacer que tu entrenamiento sea completamente determinista. Sin embargo, ten en cuenta que al hacer ciertas configuraciones deterministas, el entrenamiento puede ser más lento.

## Personaliza el Trainer

Si bien la clase [`Trainer`] está diseñada para ser accesible y fácil de usar, también ofrece mucha capacidad de personalización para usuarios más aventureros. Muchos de los métodos del [`Trainer`] pueden ser subclasificados y sobrescritos para admitir la funcionalidad que deseas, sin tener que reescribir todo el bucle de entrenamiento desde cero para adaptarlo. Estos métodos incluyen:

* [~Trainer.get_train_dataloader] crea un entrenamiento de DataLoader
* [~Trainer.get_eval_dataloader] crea una evaluación DataLoader
* [~Trainer.get_test_dataloader] crea una prueba de DataLoader
* [~Trainer.log] anota la información de los objetos varios que observa el entrenamiento
* [~Trainer.create_optimizer_and_scheduler] crea un optimizador y la tasa programada de aprendizaje si no lo pasaron en __init__; estos pueden ser personalizados independientes con [~Trainer.create_optimizer] y [~Trainer.create_scheduler] respectivamente
* [~Trainer.compute_loss] computa la pérdida en lote con las aportes del entrenamiento
* [~Trainer.training_step] realiza el paso del entrenamiento
* [~Trainer.prediction_step] realiza la predicción y paso de prueba
* [~Trainer.evaluate] evalua el modelo y da las metricas evaluativas
* [~Trainer.predict]  hace las predicciones (con las metricas si hay etiquetas disponibles) en lote de prueba

Por ejemplo, si deseas personalizar el método [`~Trainer.compute_loss`] para usar una pérdida ponderada en su lugar, puedes hacerlo de la siguiente manera:

```py
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```
### Callbacks

Otra opción para personalizar el [`Trainer`] es utilizar [callbacks](callbacks). Los callbacks *no cambian nada* en el bucle de entrenamiento. Inspeccionan el estado del bucle de entrenamiento y luego ejecutan alguna acción (detención anticipada, registro de resultados, etc.) según el estado. En otras palabras, un callback no puede usarse para implementar algo como una función de pérdida personalizada y necesitarás subclasificar y sobrescribir el método [`~Trainer.compute_loss`] para eso.

Por ejemplo, si deseas agregar un callback de detención anticipada al bucle de entrenamiento después de 10 pasos.

```py
from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}

```
Luego, pásalo al parámetro `callback` del [`Trainer`]:

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callback=[EarlyStoppingCallback()],
)
```

## Logging

<Tip>

Comprueba el API referencia [logging](./main_classes/logging) para mas información sobre los niveles differentes de logging.

</Tip>

El [`Trainer`] está configurado a `logging.INFO` de forma predeterminada el cual informa errores, advertencias y otra información basica. Un [`Trainer`] réplica - en entornos distributos - está configurado a `logging.WARNING` el cual solamente informa errores y advertencias. Puedes cambiar el nivel de logging con los parametros [`log_level`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level) y [`log_level_replica`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level_replica) en [`TrainingArguments`].

Para configurar el nivel de registro para cada nodo, usa el parámetro [`log_on_each_node`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.log_on_each_node) para determinar si deseas utilizar el nivel de registro en cada nodo o solo en el nodo principal.

<Tip>

[`Trainer`] establece el nivel de registro por separado para cada nodo en el método [`Trainer.init`], por lo que es posible que desees considerar establecer esto antes si estás utilizando otras funcionalidades de Transformers antes de crear el objeto [`Trainer`].

</Tip>

Por ejemplo, para establecer que tu código principal y los módulos utilicen el mismo nivel de registro según cada nodo:

```py
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```
<hfoptions id="logging">
<hfoption id="single node">

Usa diferentes combinaciones de `log_level` y `log_level_replica` para configurar qué se registra en cada uno de los nodos.

```bash
my_app.py ... --log_level warning --log_level_replica error
```

</hfoption>
<hfoption id="multi-node">

Agrega el parámetro `log_on_each_node 0` para entornos multi-nodo.

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0

# set to only report errors
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

</hfoption>
</hfoptions>

## NEFTune

[NEFTune](https://hf.co/papers/2310.05914) es una técnica que puede mejorar el rendimiento al agregar ruido a los vectores de incrustación durante el entrenamiento. Para habilitarlo en [`Trainer`], establece el parámetro `neftune_noise_alpha` en [`TrainingArguments`] para controlar cuánto ruido se agrega.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(..., neftune_noise_alpha=0.1)
trainer = Trainer(..., args=training_args)
```

NEFTune se desactiva después del entrenamiento para restaurar la capa de incrustación original y evitar cualquier comportamiento inesperado.

## Accelerate y Trainer

La clase [`Trainer`] está impulsada por [Accelerate](https://hf.co/docs/accelerate), una biblioteca para entrenar fácilmente modelos de PyTorch en entornos distribuidos con soporte para integraciones como [FullyShardedDataParallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) y [DeepSpeed](https://www.deepspeed.ai/).

<Tip>

Aprende más sobre las estrategias de fragmentación FSDP, descarga de CPU y más con el [`Trainer`] en la guía [Paralela de Datos Completamente Fragmentados](fsdp).

</Tip>

Para usar Accelerate con [`Trainer`], ejecuta el comando [`accelerate.config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) para configurar el entrenamiento para tu entorno de entrenamiento. Este comando crea un `config_file.yaml` que se utilizará cuando inicies tu script de entrenamiento. Por ejemplo, algunas configuraciones de ejemplo que puedes configurar son:

<hfoptions id="config">
<hfoption id="DistributedDataParallel">

```yml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0 #change rank as per the node
main_process_ip: 192.168.20.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
</hfoption>
<hfoption id="FSDP">

```yml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
</hfoption>
<hfoption id="DeepSpeed">

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/user/configs/ds_zero3_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
</hfoption>
<hfoption id="DeepSpeed with Accelerate plugin">

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 0.7
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

```

</hfoption>
</hfoptions>

El comando [`accelerate_launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) es la forma recomendada de lanzar tu script de entrenamiento en un sistema distribuido con Accelerate y [`Trainer`] con los parámetros especificados en `config_file.yaml`. Este archivo se guarda en la carpeta de caché de Accelerate y se carga automáticamente cuando ejecutas `accelerate_launch`.

Por ejemplo, para ejecutar el script de entrenamiento [`run_glue.py`](https://github.com/huggingface/transformers/blob/f4db565b695582891e43a5e042e5d318e28f20b8/examples/pytorch/text-classification/run_glue.py#L4) con la configuración de FSDP:

```bash
accelerate launch \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

También puedes especificar los parámetros del archivo config_file.yaml directamente en la línea de comandos:

```bash
accelerate launch --num_processes=2 \
    --use_fsdp \
    --mixed_precision=bf16 \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
    --fsdp_transformer_layer_cls_to_wrap="BertLayer" \
    --fsdp_sharding_strategy=1 \
    --fsdp_state_dict_type=FULL_STATE_DICT \
    ./examples/pytorch/text-classification/run_glue.py
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

Consulta el tutorial [Lanzamiento de tus scripts con Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch) para obtener más información sobre `accelerate_launch` y las configuraciones personalizadas.
