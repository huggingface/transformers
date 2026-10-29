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
