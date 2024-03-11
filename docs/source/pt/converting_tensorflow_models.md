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

# Convertendo checkpoints do TensorFlow para Pytorch

Uma interface de linha de comando é fornecida para converter os checkpoints originais Bert/GPT/GPT-2/Transformer-XL/XLNet/XLM em modelos
que podem ser carregados usando os métodos `from_pretrained` da biblioteca.

<Tip>

A partir da versão 2.3.0 o script de conversão agora faz parte do transformers CLI (**transformers-cli**) disponível em qualquer instalação
transformers >= 2.3.0.

A documentação abaixo reflete o formato do comando **transformers-cli convert**.

</Tip>

## BERT

Você pode converter qualquer checkpoint do BERT em TensorFlow (em particular [os modelos pré-treinados lançados pelo Google](https://github.com/google-research/bert#pre-trained-models)) em um arquivo PyTorch usando um 
[convert_bert_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py) script.

Esta Interface de Linha de Comando (CLI) recebe como entrada um checkpoint do TensorFlow (três arquivos começando com `bert_model.ckpt`) e o
arquivo de configuração (`bert_config.json`), e então cria um modelo PyTorch para esta configuração, carrega os pesos 
do checkpoint do TensorFlow no modelo PyTorch e salva o modelo resultante em um arquivo PyTorch que pode
ser importado usando `from_pretrained()` (veja o exemplo em [quicktour](quicktour) , [run_glue.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_glue.py) ).

Você só precisa executar este script de conversão **uma vez** para obter um modelo PyTorch. Você pode então desconsiderar o checkpoint em
 TensorFlow (os três arquivos começando com `bert_model.ckpt`), mas certifique-se de manter o arquivo de configuração (\
`bert_config.json`) e o arquivo de vocabulário (`vocab.txt`), pois eles também são necessários para o modelo PyTorch.

Para executar este script de conversão específico, você precisará ter o TensorFlow e o PyTorch instalados (`pip install tensorflow`). O resto do repositório requer apenas o PyTorch.

Aqui está um exemplo do processo de conversão para um modelo `BERT-Base Uncased` pré-treinado:

```bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
```

Você pode baixar os modelos pré-treinados do Google para a conversão [aqui](https://github.com/google-research/bert#pre-trained-models).

## ALBERT

Converta os checkpoints do modelo ALBERT em TensorFlow para PyTorch usando o
[convert_albert_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/albert/convert_albert_original_tf_checkpoint_to_pytorch.py) script.

A Interface de Linha de Comando (CLI) recebe como entrada um checkpoint do TensorFlow (três arquivos começando com `model.ckpt-best`) e o
arquivo de configuração (`albert_config.json`), então cria e salva um modelo PyTorch. Para executar esta conversão, você
precisa ter o TensorFlow e o PyTorch instalados.

Aqui está um exemplo do processo de conversão para o modelo `ALBERT Base` pré-treinado:

```bash
export ALBERT_BASE_DIR=/path/to/albert/albert_base

transformers-cli convert --model_type albert \
  --tf_checkpoint $ALBERT_BASE_DIR/model.ckpt-best \
  --config $ALBERT_BASE_DIR/albert_config.json \
  --pytorch_dump_output $ALBERT_BASE_DIR/pytorch_model.bin
```

Você pode baixar os modelos pré-treinados do Google para a conversão [aqui](https://github.com/google-research/albert#pre-trained-models).

## OpenAI GPT

Aqui está um exemplo do processo de conversão para um modelo OpenAI GPT pré-treinado, supondo que seu checkpoint NumPy
foi salvo com o mesmo formato do modelo pré-treinado OpenAI (veja [aqui](https://github.com/openai/finetune-transformer-lm)\
)

```bash
export OPENAI_GPT_CHECKPOINT_FOLDER_PATH=/path/to/openai/pretrained/numpy/weights

transformers-cli convert --model_type gpt \
  --tf_checkpoint $OPENAI_GPT_CHECKPOINT_FOLDER_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--config OPENAI_GPT_CONFIG] \
  [--finetuning_task_name OPENAI_GPT_FINETUNED_TASK] \
```

## OpenAI GPT-2

Aqui está um exemplo do processo de conversão para um modelo OpenAI GPT-2 pré-treinado (consulte [aqui](https://github.com/openai/gpt-2))

```bash
export OPENAI_GPT2_CHECKPOINT_PATH=/path/to/openai-community/gpt2/pretrained/weights

transformers-cli convert --model_type openai-community/gpt2 \
  --tf_checkpoint $OPENAI_GPT2_CHECKPOINT_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--config OPENAI_GPT2_CONFIG] \
  [--finetuning_task_name OPENAI_GPT2_FINETUNED_TASK]
```

## XLNet

Aqui está um exemplo do processo de conversão para um modelo XLNet pré-treinado:

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

Aqui está um exemplo do processo de conversão para um modelo XLM pré-treinado:

```bash
export XLM_CHECKPOINT_PATH=/path/to/xlm/checkpoint

transformers-cli convert --model_type xlm \
  --tf_checkpoint $XLM_CHECKPOINT_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT
 [--config XML_CONFIG] \
 [--finetuning_task_name XML_FINETUNED_TASK]
```

## T5

Aqui está um exemplo do processo de conversão para um modelo T5 pré-treinado:

```bash
export T5=/path/to/t5/uncased_L-12_H-768_A-12

transformers-cli convert --model_type t5 \
  --tf_checkpoint $T5/t5_model.ckpt \
  --config $T5/t5_config.json \
  --pytorch_dump_output $T5/pytorch_model.bin
```
