<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Addestramento efficiente su CPU

Questa guida si concentra su come addestrare in maniera efficiente grandi modelli su CPU.

## Mixed precision con IPEX

IPEX è ottimizzato per CPU con AVX-512 o superiore, e funziona per le CPU con solo AVX2. Pertanto, si prevede che le prestazioni saranno più vantaggiose per le le CPU Intel con AVX-512 o superiori, mentre le CPU con solo AVX2 (ad esempio, le CPU AMD o le CPU Intel più vecchie) potrebbero ottenere prestazioni migliori con IPEX, ma non sono garantite. IPEX offre ottimizzazioni delle prestazioni per l'addestramento della CPU sia con Float32 che con BFloat16. L'uso di BFloat16 è l'argomento principale delle seguenti sezioni.

Il tipo di dati a bassa precisione BFloat16 è stato supportato in modo nativo su 3rd Generation Xeon® Scalable Processors (aka Cooper Lake) con AVX512 e sarà supportata dalla prossima generazione di Intel® Xeon® Scalable Processors con Intel® Advanced Matrix Extensions (Intel® AMX) instruction set con prestazioni ulteriormente migliorate. L'Auto Mixed Precision per il backende della CPU è stato abilitato da PyTorch-1.10. allo stesso tempo, il supporto di Auto Mixed Precision con BFloat16 per CPU e l'ottimizzazione degli operatori BFloat16 è stata abilitata in modo massiccio in Intel® Extension per PyTorch, and parzialmente aggiornato al branch master di PyTorch. Gli utenti possono ottenere prestazioni migliori ed users experience con IPEX Auto Mixed Precision..

Vedi informazioni più dettagliate su [Auto Mixed Precision](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html).

### Installazione di IPEX:

Il rilascio di IPEX segue quello di PyTorch, da installare via pip:

| PyTorch Version   | IPEX version   |
| :---------------: | :----------:   |
| 1.13              |  1.13.0+cpu    |
| 1.12              |  1.12.300+cpu  |
| 1.11              |  1.11.200+cpu  |
| 1.10              |  1.10.100+cpu  |

```bash
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

Vedi altri approcci per [IPEX installation](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html).

### Utilizzo nel Trainer

Per abilitare la auto mixed precision con IPEX in Trainer, l'utende dovrebbe aggiungere `use_ipex`, `bf16` e `no_cuda` negli argomenti del comando di addestramento.

Vedi un sempio di un caso d'uso [Transformers question-answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)

- Training with IPEX using BF16 auto mixed precision on CPU:

<pre> python run_qa.py \
--model_name_or_path bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex \</b>
<b>--bf16 --no_cuda</b></pre> 

### Esempi pratici

Blog: [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids)
