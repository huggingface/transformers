<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Inferenza Efficiente su CPU

Questa guida si concentra sull'inferenza di modelli di grandi dimensioni in modo efficiente sulla CPU.

## `BetterTransformer` per inferenza più rapida

Abbiamo integrato di recente `BetterTransformer` per fare inferenza più rapidamente con modelli per testi, immagini e audio. Visualizza la documentazione sull'integrazione [qui](https://huggingface.co/docs/optimum/bettertransformer/overview) per maggiori dettagli.

## PyTorch JIT-mode (TorchScript)

TorchScript è un modo di creare modelli serializzabili e ottimizzabili da codice PyTorch. Ogni programmma TorchScript può esere salvato da un processo Python  e caricato in un processo dove non ci sono dipendenze Python.
Comparandolo con l'eager mode di default, jit mode in PyTorch normalmente fornisce prestazioni migliori per l'inferenza del modello da parte di metodologie di ottimizzazione come la operator fusion.

Per una prima introduzione a TorchScript, vedi la Introduction to [PyTorch TorchScript tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules).

### IPEX Graph Optimization con JIT-mode

Intel® Extension per PyTorch fornnisce ulteriori ottimizzazioni in jit mode per i modelli della serie Transformers. Consigliamo vivamente agli utenti di usufruire dei vantaggi di Intel® Extension per PyTorch con jit mode. Alcuni operator patterns usati fequentemente dai modelli Transformers models sono già supportati in Intel® Extension per PyTorch con jit mode fusions. Questi fusion patterns come Multi-head-attention fusion, Concat Linear, Linear+Add, Linear+Gelu, Add+LayerNorm fusion and etc. sono abilitati e hanno buone performance. I benefici della fusion è fornito agli utenti in modo trasparente. In base alle analisi, il ~70% dei problemi più popolari in NLP question-answering, text-classification, and token-classification possono avere benefici sulle performance grazie ai fusion patterns sia per Float32 precision che per BFloat16 Mixed precision.

Vedi maggiori informazioni per [IPEX Graph Optimization](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html).

#### Installazione di IPEX

I rilasci di IPEX seguono PyTorch, verifica i vari approcci per [IPEX installation](https://intel.github.io/intel-extension-for-pytorch/).

### Utilizzo del JIT-mode

Per abilitare JIT-mode in Trainer per evaluation e prediction, devi aggiungere `jit_mode_eval` negli argomenti di Trainer.

<Tip warning={true}>

per PyTorch >= 1.14.0. JIT-mode potrebe giovare a qualsiasi modello di prediction e evaluaion visto che il dict input è supportato in jit.trace

per PyTorch < 1.14.0. JIT-mode potrebbe giovare ai modelli il cui ordine dei parametri corrisponde all'ordine delle tuple in ingresso in jit.trace, come i modelli per question-answering.
Nel caso in cui l'ordine dei parametri seguenti non corrisponda all'ordine delle tuple in ingresso in jit.trace, come nei modelli di text-classification, jit.trace fallirà e lo cattureremo con una eccezione al fine di renderlo un fallback. Il logging è usato per notificare gli utenti.

</Tip>

Trovi un esempo con caso d'uso in [Transformers question-answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)

- Inference using jit mode on CPU:

<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--jit_mode_eval </b></pre> 

- Inference with IPEX using jit mode on CPU:

<pre>python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
<b>--use_ipex \</b>
<b>--jit_mode_eval</b></pre> 
