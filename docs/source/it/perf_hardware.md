<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# Hardware ottimizzato per l'addestramento

L'hardware utilizzato per eseguire l'addestramento del modello e l'inferenza può avere un grande effetto sulle prestazioni. Per un analisi approfondita delle GPUs, assicurati di dare un'occhiata all'eccellente [blog post](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/) di Tim Dettmer.

Diamo un'occhiata ad alcuni consigli pratici per la configurazione della GPU.

## GPU
Quando si addestrano modelli più grandi ci sono essenzialmente tre opzioni:
- GPUs piu' grandi
- Piu' GPUs
- Piu' CPU e piu' NVMe (scaricato da [DeepSpeed-Infinity](main_classes/deepspeed#nvme-support))

Iniziamo dal caso in cui ci sia una singola GPU.

### Potenza e Raffreddamento

Se hai acquistato una costosa GPU di fascia alta, assicurati di darle la potenza corretta e un raffreddamento sufficiente.

**Potenza**:

Alcune schede GPU consumer di fascia alta hanno 2 e talvolta 3 prese di alimentazione PCI-E a 8 pin. Assicurati di avere tanti cavi PCI-E a 8 pin indipendenti da 12 V collegati alla scheda quante sono le prese. Non utilizzare le 2 fessure a un'estremità dello stesso cavo (noto anche come cavo a spirale). Cioè se hai 2 prese sulla GPU, vuoi 2 cavi PCI-E a 8 pin che vanno dall'alimentatore alla scheda e non uno che abbia 2 connettori PCI-E a 8 pin alla fine! In caso contrario, non otterrai tutte le prestazioni ufficiali.

Ciascun cavo di alimentazione PCI-E a 8 pin deve essere collegato a una guida da 12 V sul lato dell'alimentatore e può fornire fino a 150 W di potenza.

Alcune altre schede possono utilizzare connettori PCI-E a 12 pin e questi possono fornire fino a 500-600 W di potenza.

Le schede di fascia bassa possono utilizzare connettori a 6 pin, che forniscono fino a 75 W di potenza.

Inoltre vuoi un alimentatore (PSU) di fascia alta che abbia una tensione stabile. Alcuni PSU di qualità inferiore potrebbero non fornire alla scheda la tensione stabile di cui ha bisogno per funzionare al massimo.

E ovviamente l'alimentatore deve avere abbastanza Watt inutilizzati per alimentare la scheda.

**Raffreddamento**:

Quando una GPU si surriscalda, inizierà a rallentare e non fornirà le prestazioni mssimali e potrebbe persino spegnersi se diventasse troppo calda.

È difficile dire l'esatta temperatura migliore a cui aspirare quando una GPU è molto caricata, ma probabilmente qualsiasi cosa al di sotto di +80°C va bene, ma più bassa è meglio - forse 70-75°C è un intervallo eccellente in cui trovarsi. È probabile che il rallentamento inizi a circa 84-90°C. Ma oltre alla limitazione delle prestazioni, una temperatura molto elevata prolungata è probabile che riduca la durata di una GPU.

Diamo quindi un'occhiata a uno degli aspetti più importanti quando si hanno più GPU: la connettività.

### Connettività multi-GPU

Se utilizzi più GPU, il modo in cui le schede sono interconnesse può avere un enorme impatto sul tempo totale di allenamento. Se le GPU si trovano sullo stesso nodo fisico, puoi eseguire:

```
nvidia-smi topo -m
```

e ti dirà come sono interconnesse le GPU. Su una macchina con doppia GPU e collegata a NVLink, molto probabilmente vedrai qualcosa del tipo:

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

su una macchina diversa senza NVLink potremmo vedere:

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

Il rapporto include questa legenda:

```
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

Quindi il primo rapporto `NV2` ci dice che le GPU sono interconnesse con 2 NVLinks e nel secondo report `PHB` abbiamo una tipica configurazione PCIe+Bridge a livello di consumatore.

Controlla che tipo di connettività hai sulla tua configurazione. Alcuni di questi renderanno la comunicazione tra le carte più veloce (es. NVLink), altri più lenta (es. PHB).

A seconda del tipo di soluzione di scalabilità utilizzata, la velocità di connettività potrebbe avere un impatto maggiore o minore. Se le GPU devono sincronizzarsi raramente, come in DDP, l'impatto di una connessione più lenta sarà meno significativo. Se le GPU devono scambiarsi messaggi spesso, come in ZeRO-DP, una connettività più veloce diventa estremamente importante per ottenere un addestramento più veloce.

#### NVlink

[NVLink](https://en.wikipedia.org/wiki/NVLink) è un collegamento di comunicazione a corto raggio multilinea seriale basato su cavo sviluppato da Nvidia.

Ogni nuova generazione fornisce una larghezza di banda più veloce, ad es. ecco una citazione da [Nvidia Ampere GA102 GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf):

> Third-Generation NVLink®
> GA102 GPUs utilize NVIDIA’s third-generation NVLink interface, which includes four x4 links,
> with each link providing 14.0625 GB/sec bandwidth in each direction between two GPUs. Four
> links provide 56.25 GB/sec bandwidth in each direction, and 112.5 GB/sec total bandwidth
> between two GPUs. Two RTX 3090 GPUs can be connected together for SLI using NVLink.
> (Note that 3-Way and 4-Way SLI configurations are not supported.)

Quindi più `X` si ottiene nel rapporto di `NVX` nell'output di `nvidia-smi topo -m`, meglio è. La generazione dipenderà dall'architettura della tua GPU.

Confrontiamo l'esecuzione di un training del modello di linguaggio gpt2 su un piccolo campione di wikitext

I risultati sono:


| NVlink | Time |
| -----  | ---: |
| Y      | 101s |
| N      | 131s |


Puoi vedere che NVLink completa l'addestramento circa il 23% più velocemente. Nel secondo benchmark utilizziamo `NCCL_P2P_DISABLE=1` per dire alle GPU di non utilizzare NVLink.

Ecco il codice benchmark completo e gli output:

```bash
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

Hardware: 2x TITAN RTX 24GB each + NVlink with 2 NVLinks (`NV2` in `nvidia-smi topo -m`)
Software: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`