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

# Convertir checkpoints de Tensorflow

Te proporcionamos una interfaz de línea de comando (`CLI`, por sus siglas en inglés) para convertir puntos de control (_checkpoints_) originales de Bert/GPT/GPT-2/Transformer-XL/XLNet/XLM en modelos que se puedan cargar utilizando los métodos `from_pretrained` de la biblioteca.

<Tip>

Desde 2.3.0, el script para convertir es parte de la CLI de transformers (**transformers-cli**) disponible en cualquier instalación de transformers >= 2.3.0.

La siguiente documentación refleja el formato para el comando **transformers-cli convert**.

</Tip>

## BERT

Puedes convertir cualquier checkpoint de TensorFlow para BERT (en particular, [los modelos pre-entrenados y publicados por Google](https://github.com/google-research/bert#pre-trained-models)) en un archivo de PyTorch mediante el script [convert_bert_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py).

Esta CLI toma como entrada un checkpoint de TensorFlow (tres archivos que comienzan con `bert_model.ckpt`) y el archivo de configuración asociado (`bert_config.json`), y crea un modelo PyTorch para esta configuración, carga los pesos del checkpoint de TensorFlow en el modelo de PyTorch y guarda el modelo resultante en un archivo estándar de PyTorch que se puede importar usando `from_pretrained()` (ve el ejemplo en [Tour rápido](quicktour), [run_glue.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_glue.py)).

Solo necesitas ejecutar este script **una vez** para convertir un modelo a PyTorch. Después, puedes ignorar el checkpoint de TensorFlow (los tres archivos que comienzan con `bert_model.ckpt`), pero asegúrate de conservar el archivo de configuración (`bert_config.json`) y el archivo de vocabulario (`vocab.txt`) ya que estos también son necesarios para el modelo en PyTorch.

Para ejecutar este script deberás tener instalado TensorFlow y PyTorch (`pip install tensorflow`). El resto del repositorio solo requiere PyTorch.

Aquí hay un ejemplo del proceso para convertir un modelo `BERT-Base Uncased` pre-entrenado:

```bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
```

Puedes descargar los modelos pre-entrenados de Google para la conversión [aquí](https://github.com/google-research/bert#pre-trained-models).

## ALBERT

Convierte los checkpoints del modelo ALBERT de TensorFlow a PyTorch usando el script [convert_albert_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/albert/convert_albert_original_tf_checkpoint_to_pytorch.py).

La CLI toma como entrada un checkpoint de TensorFlow (tres archivos que comienzan con `model.ckpt-best`) y el archivo de configuración adjunto (`albert_config.json`), luego crea y guarda un modelo de PyTorch. Para ejecutar esta conversión deberás tener instalados TensorFlow y PyTorch.

Aquí hay un ejemplo del proceso para convertir un modelo `ALBERT Base` pre-entrenado:

```bash
export ALBERT_BASE_DIR=/path/to/albert/albert_base

transformers-cli convert --model_type albert \
  --tf_checkpoint $ALBERT_BASE_DIR/model.ckpt-best \
  --config $ALBERT_BASE_DIR/albert_config.json \
  --pytorch_dump_output $ALBERT_BASE_DIR/pytorch_model.bin
```

Puedes descargar los modelos pre-entrenados de Google para la conversión [aquí](https://github.com/google-research/albert#pre-trained-models).

## OpenAI GPT

Este es un ejemplo del proceso para convertir un modelo OpenAI GPT pre-entrenado, asumiendo que tu checkpoint de NumPy se guarda con el mismo formato que el modelo pre-entrenado de OpenAI (más información [aquí](https://github.com/openai/finetune-transformer-lm)):

```bash
export OPENAI_GPT_CHECKPOINT_FOLDER_PATH=/path/to/openai/pretrained/numpy/weights

transformers-cli convert --model_type gpt \
  --tf_checkpoint $OPENAI_GPT_CHECKPOINT_FOLDER_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--config OPENAI_GPT_CONFIG] \
  [--finetuning_task_name OPENAI_GPT_FINETUNED_TASK] \
```

## OpenAI GPT-2

Aquí hay un ejemplo del proceso para convertir un modelo OpenAI GPT-2 pre-entrenado (más información [aquí](https://github.com/openai/gpt-2)):

```bash
export OPENAI_GPT2_CHECKPOINT_PATH=/path/to/openai-community/gpt2/pretrained/weights

transformers-cli convert --model_type openai-community/gpt2 \
  --tf_checkpoint $OPENAI_GPT2_CHECKPOINT_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--config OPENAI_GPT2_CONFIG] \
  [--finetuning_task_name OPENAI_GPT2_FINETUNED_TASK]
```

## XLNet

Aquí hay un ejemplo del proceso para convertir un modelo XLNet pre-entrenado:

```bash
export TRANSFO_XL_CHECKPOINT_PATH=/path/to/xlnet/checkpoint
export TRANSFO_XL_CONFIG_PATH=/path/to/xlnet/config

transformers-cli convert --model_type xlnet \
  --tf_checkpoint $TRANSFO_XL_CHECKPOINT_PATH \
  --config $TRANSFO_XL_CONFIG_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--finetuning_task_name XLNET_FINETUNED_TASK] \
```

## XLM

Aquí hay un ejemplo del proceso para convertir un modelo XLM pre-entrenado:

```bash
export XLM_CHECKPOINT_PATH=/path/to/xlm/checkpoint

transformers-cli convert --model_type xlm \
  --tf_checkpoint $XLM_CHECKPOINT_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT
 [--config XML_CONFIG] \
 [--finetuning_task_name XML_FINETUNED_TASK]
```

## T5

Aquí hay un ejemplo del proceso para convertir un modelo T5 pre-entrenado:

```bash
export T5=/path/to/t5/uncased_L-12_H-768_A-12

transformers-cli convert --model_type t5 \
  --tf_checkpoint $T5/t5_model.ckpt \
  --config $T5/t5_config.json \
  --pytorch_dump_output $T5/pytorch_model.bin
```
