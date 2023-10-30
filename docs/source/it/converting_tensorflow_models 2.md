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

# Convertire checkpoint di Tensorflow

È disponibile un'interfaccia a linea di comando per convertire gli originali checkpoint di Bert/GPT/GPT-2/Transformer-XL/XLNet/XLM 
in modelli che possono essere caricati utilizzando i metodi `from_pretrained` della libreria.

<Tip>

A partire dalla versione 2.3.0 lo script di conversione è parte di transformers CLI (**transformers-cli**), disponibile in ogni installazione 
di transformers >=2.3.0.

La seguente documentazione riflette il formato dei comandi di **transformers-cli convert**.

</Tip>

## BERT

Puoi convertire qualunque checkpoint Tensorflow di BERT (in particolare 
[i modeli pre-allenati rilasciati da Google](https://github.com/google-research/bert#pre-trained-models)) 
in un file di salvataggio Pytorch utilizzando lo script 
[convert_bert_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py).

Questo CLI prende come input un checkpoint di Tensorflow (tre files che iniziano con `bert_model.ckpt`) ed il relativo 
file di configurazione (`bert_config.json`), crea un modello Pytorch per questa configurazione, carica i pesi dal
checkpoint di Tensorflow nel modello di Pytorch e salva il modello che ne risulta in un file di salvataggio standard di Pytorch che 
può essere importato utilizzando `from_pretrained()` (vedi l'esempio nel
[quicktour](quicktour) , [run_glue.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_glue.py) ).

Devi soltanto lanciare questo script di conversione **una volta** per ottenere un modello Pytorch. Dopodichè, potrai tralasciare 
il checkpoint di Tensorflow (i tre files che iniziano con `bert_model.ckpt`), ma assicurati di tenere il file di configurazione 
(`bert_config.json`) ed il file di vocabolario (`vocab.txt`) in quanto queste componenti sono necessarie anche per il modello di Pytorch.

Per lanciare questo specifico script di conversione avrai bisogno di un'installazione di Tensorflow e di Pytorch
(`pip install tensorflow`). Il resto della repository richiede soltanto Pytorch.

Questo è un esempio del processo di conversione per un modello `BERT-Base Uncased` pre-allenato:

```bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
```

Puoi scaricare i modelli pre-allenati di Google per la conversione [qua](https://github.com/google-research/bert#pre-trained-models).

## ALBERT

Per il modello ALBERT, converti checkpoint di Tensoflow in Pytorch utilizzando lo script 
[convert_albert_original_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/tree/main/src/transformers/models/albert/convert_albert_original_tf_checkpoint_to_pytorch.py).

Il CLI prende come input un checkpoint di Tensorflow (tre files che iniziano con `model.ckpt-best`) e i relativi file di 
configurazione (`albert_config.json`), dopodichè crea e salva un modello Pytorch. Per lanciare questa conversione 
avrai bisogno di un'installazione di Tensorflow e di Pytorch.

Ecco un esempio del procedimento di conversione di un modello `ALBERT Base` pre-allenato:

```bash
export ALBERT_BASE_DIR=/path/to/albert/albert_base
transformers-cli convert --model_type albert \
  --tf_checkpoint $ALBERT_BASE_DIR/model.ckpt-best \
  --config $ALBERT_BASE_DIR/albert_config.json \
  --pytorch_dump_output $ALBERT_BASE_DIR/pytorch_model.bin
```

Puoi scaricare i modelli pre-allenati di Google per la conversione [qui](https://github.com/google-research/albert#pre-trained-models).

## OpenAI GPT

Ecco un esempio del processo di conversione di un modello OpenAI GPT pre-allenato, assumendo che il tuo checkpoint di NumPy
sia salvato nello stesso formato dei modelli pre-allenati OpenAI (vedi [qui](https://github.com/openai/finetune-transformer-lm)):
```bash
export OPENAI_GPT_CHECKPOINT_FOLDER_PATH=/path/to/openai/pretrained/numpy/weights
transformers-cli convert --model_type gpt \
  --tf_checkpoint $OPENAI_GPT_CHECKPOINT_FOLDER_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--config OPENAI_GPT_CONFIG] \
  [--finetuning_task_name OPENAI_GPT_FINETUNED_TASK] \
```

## OpenAI GPT-2

Ecco un esempio del processo di conversione di un modello OpenAI GPT-2 pre-allenato (vedi [qui](https://github.com/openai/gpt-2)):

```bash
export OPENAI_GPT2_CHECKPOINT_PATH=/path/to/gpt2/pretrained/weights
transformers-cli convert --model_type gpt2 \
  --tf_checkpoint $OPENAI_GPT2_CHECKPOINT_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--config OPENAI_GPT2_CONFIG] \
  [--finetuning_task_name OPENAI_GPT2_FINETUNED_TASK]
```

## Transformer-XL


Ecco un esempio del processo di conversione di un modello Transformer-XL pre-allenato 
(vedi [qui](https://github.com/kimiyoung/transformer-xl/tree/master/tf#obtain-and-evaluate-pretrained-sota-models)):

```bash
export TRANSFO_XL_CHECKPOINT_FOLDER_PATH=/path/to/transfo/xl/checkpoint
transformers-cli convert --model_type transfo_xl \
  --tf_checkpoint $TRANSFO_XL_CHECKPOINT_FOLDER_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT \
  [--config TRANSFO_XL_CONFIG] \
  [--finetuning_task_name TRANSFO_XL_FINETUNED_TASK]
```

## XLNet

Ecco un esempio del processo di conversione di un modello XLNet pre-allenato:

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

Ecco un esempio del processo di conversione di un modello XLM pre-allenato:

```bash
export XLM_CHECKPOINT_PATH=/path/to/xlm/checkpoint
transformers-cli convert --model_type xlm \
  --tf_checkpoint $XLM_CHECKPOINT_PATH \
  --pytorch_dump_output $PYTORCH_DUMP_OUTPUT
 [--config XML_CONFIG] \
 [--finetuning_task_name XML_FINETUNED_TASK]
```

## T5

Ecco un esempio del processo di conversione di un modello T5 pre-allenato:

```bash
export T5=/path/to/t5/uncased_L-12_H-768_A-12
transformers-cli convert --model_type t5 \
  --tf_checkpoint $T5/t5_model.ckpt \
  --config $T5/t5_config.json \
  --pytorch_dump_output $T5/pytorch_model.bin
```
