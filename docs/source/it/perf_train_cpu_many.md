<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Addestramento effciente su multiple CPU

Quando l'addestramento su una singola CPU è troppo lento, possiamo usare CPU multiple. Quasta guida si concentra su DDP basato su PyTorch abilitando l'addetramento distribuito su CPU in maniera efficiente.

## Intel® oneCCL Bindings per PyTorch

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (collective communications library) è una libreria per l'addestramento efficiente del deep learning in distribuito e implementa collettivi come allreduce, allgather, alltoall. Per maggiori informazioni su oneCCL, fai riferimento a [oneCCL documentation](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html) e [oneCCL specification](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html).

Il modulo `oneccl_bindings_for_pytorch` (`torch_ccl` precedentemente alla versione 1.12)  implementa PyTorch C10D ProcessGroup API e può essere caricato dinamicamente com external ProcessGroup e funziona solo su piattaforma Linux al momento.

Qui trovi informazioni più dettagliate per [oneccl_bind_pt](https://github.com/intel/torch-ccl).

### Intel® oneCCL Bindings per l'installazione PyTorch:

I file wheel sono disponibili per le seguenti versioni di Python:

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :---------: |
| 1.13.0            |            | √          | √          | √          | √           |
| 1.12.100          |            | √          | √          | √          | √           |
| 1.12.0            |            | √          | √          | √          | √           |
| 1.11.0            |            | √          | √          | √          | √           |
| 1.10.0            | √          | √          | √          | √          |             |

```bash
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```

dove `{pytorch_version}` deve essere la tua versione di PyTorch, per l'stanza 1.13.0.
Verifica altri approcci per [oneccl_bind_pt installation](https://github.com/intel/torch-ccl).
Le versioni di oneCCL e PyTorch devono combaciare.

<Tip warning={true}>

oneccl_bindings_for_pytorch 1.12.0 prebuilt wheel does not work with PyTorch 1.12.1 (it is for PyTorch 1.12.0)
PyTorch 1.12.1 should work with oneccl_bindings_for_pytorch 1.12.100

</Tip>

## Intel® MPI library

Usa questa implementazione basata su standard MPI per fornire una architettura flessibile, efficiente, scalabile su cluster per Intel®. Questo componente è parte di Intel® oneAPI HPC Toolkit.

oneccl_bindings_for_pytorch è installato insieme al set di strumenti MPI. Necessità di reperire l'ambiente prima di utilizzarlo.

per Intel® oneCCL >= 1.12.0

```bash
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

per Intel® oneCCL con versione < 1.12.0

```bash
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### Installazione IPEX:

IPEX fornisce ottimizzazioni delle prestazioni per l'addestramento della CPU sia con Float32 che con BFloat16; puoi fare riferimento a [single CPU section](./perf_train_cpu).

Il seguente "Utilizzo in Trainer" prende come esempio mpirun nella libreria Intel® MPI.

## Utilizzo in Trainer

Per abilitare l'addestramento distribuito multi CPU nel Trainer con il ccl backend, gli utenti devono aggiungere **`--ddp_backend ccl`** negli argomenti del comando.

Vediamo un esempio per il [question-answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)

Il seguente comando abilita due processi sul nodo Xeon, con un processo in esecuzione per ogni socket. Le variabili OMP_NUM_THREADS/CCL_WORKER_COUNT possono essere impostate per una prestazione ottimale.

```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex
```

Il seguente comando abilita l'addestramento per un totale di quattro processi su due Xeon (node0 e node1, prendendo node0 come processo principale), ppn (processes per node) è impostato a 2, on un processo in esecuzione per ogni socket. Le variabili OMP_NUM_THREADS/CCL_WORKER_COUNT possono essere impostate per una prestazione ottimale.

In node0, è necessario creare un file di configurazione che contenga gli indirizzi IP di ciascun nodo (per esempio hostfile) e passare il percorso del file di configurazione come parametro.

```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```

A questo punto, esegui il seguente comando nel nodo0 e **4DDP** sarà abilitato in node0 e node1 con BF16 auto mixed precision:

```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex \
 --bf16
```
