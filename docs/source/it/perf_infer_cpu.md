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

## PyTorch JIT-mode (TorchScript)

TorchScript è un modo di creare modelli serializzabili e ottimizzabili da codice PyTorch. Ogni programma TorchScript può esere salvato da un processo Python  e caricato in un processo dove non ci sono dipendenze Python.
Comparandolo con l'eager mode di default, jit mode in PyTorch normalmente fornisce prestazioni migliori per l'inferenza del modello da parte di metodologie di ottimizzazione come la operator fusion.

Per una prima introduzione a TorchScript, vedi la Introduction to [PyTorch TorchScript tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#tracing-modules).

### IPEX Graph Optimization con JIT-mode

Intel® Extension per PyTorch fornnisce ulteriori ottimizzazioni in jit mode per i modelli della serie Transformers. Consigliamo vivamente agli utenti di usufruire dei vantaggi di Intel® Extension per PyTorch con jit mode. Alcuni operator patterns usati fequentemente dai modelli Transformers models sono già supportati in Intel® Extension per PyTorch con jit mode fusions. Questi fusion patterns come Multi-head-attention fusion, Concat Linear, Linear+Add, Linear+Gelu, Add+LayerNorm fusion and etc. sono abilitati e hanno buone performance. I benefici della fusion è fornito agli utenti in modo trasparente. In base alle analisi, il ~70% dei problemi più popolari in NLP question-answering, text-classification, and token-classification possono avere benefici sulle performance grazie ai fusion patterns sia per Float32 precision che per BFloat16 Mixed precision.

Vedi maggiori informazioni per [IPEX Graph Optimization](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html).

#### Installazione di IPEX

I rilasci di IPEX seguono PyTorch, verifica i vari approcci per [IPEX installation](https://intel.github.io/intel-extension-for-pytorch/).
