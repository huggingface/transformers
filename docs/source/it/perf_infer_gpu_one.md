<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Inferenza efficiente su GPU singola

Questo documento sarà presto completato con informazioni su come effetture l'inferenza su una singola GPU. Nel frattempo è possibile consultare [la guida per l'addestramento su una singola GPU](perf_train_gpu_one) e [la guida per l'inferenza su CPU](perf_infer_cpu).

## `BetterTransformer` per l'inferenza più veloce

Abbiamo recentemente integrato `BetterTransformer` per velocizzare l'inferenza su GPU per modelli di testo, immagini e audio. Per maggiori dettagli, consultare la documentazione su questa integrazione [qui](https://huggingface.co/docs/optimum/bettertransformer/overview).

## Integrazione di `bitsandbytes` per Int8 mixed-precision matrix decomposition

<Tip>

Nota che questa funzione può essere utilizzata anche nelle configurazioni multi GPU.

</Tip>

Dal paper [`LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale`](https://arxiv.org/abs/2208.07339), noi supportiamo l'integrazione di Hugging Face per tutti i modelli dell'Hub con poche righe di codice.
Il metodo `nn.Linear` riduce la dimensione di 2 per i pesi `float16` e `bfloat16` e di 4 per i pesi `float32`, con un impatto quasi nullo sulla qualità, operando sugli outlier in half-precision.

![HFxbitsandbytes.png](https://s3.amazonaws.com/moonup/production/uploads/1659861207959-62441d1d9fdefb55a0b7d12c.png)

Il metodo Int8 mixed-precision matrix decomposition funziona separando la moltiplicazione tra matrici in due flussi: (1) una matrice di flusso di outlier di caratteristiche sistematiche moltiplicata in fp16, (2) in flusso regolare di moltiplicazione di matrici int8 (99,9%). Con questo metodo, è possibile effettutare inferenza int8 per modelli molto grandi senza degrado predittivo.
Per maggiori dettagli sul metodo, consultare il [paper](https://arxiv.org/abs/2208.07339) o il nostro [blogpost sull'integrazione](https://huggingface.co/blog/hf-bitsandbytes-integration).

![MixedInt8.gif](https://s3.amazonaws.com/moonup/production/uploads/1660567469965-62441d1d9fdefb55a0b7d12c.gif)

Nota che è necessaria una GPU per eseguire modelli di tipo mixed-8bit, poiché i kernel sono stati compilati solo per le GPU. Prima di utilizzare questa funzione, assicurarsi di disporre di memoria sufficiente sulla GPU per memorizzare un quarto del modello (o la metà se i pesi del modello sono in mezza precisione).
Di seguito sono riportate alcune note per aiutarvi a utilizzare questo modulo, oppure seguite le dimostrazioni su [Google colab](#colab-demos).

### Requisiti

- Se si dispone di `bitsandbytes<0.37.0`, assicurarsi di eseguire su GPU NVIDIA che supportano tensor cores a 8 bit (Turing, Ampere o architetture più recenti - ad esempio T4, RTX20s RTX30s, A40-A100). Per `bitsandbytes>=0.37.0`, tutte le GPU dovrebbero essere supportate.
- Installare la versione corretta di `bitsandbytes` eseguendo:
`pip install bitsandbytes>=0.31.5`.
- Installare `accelerate`
`pip install accelerate>=0.12.0`

### Esecuzione di modelli mixed-Int8 - configurazione per singola GPU

Dopo aver installato le librerie necessarie, per caricare il tuo modello mixed 8-bit è il seguente:

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```

Per la generazione di testo, si consiglia di:

* utilizzare il metodo `generate()` del modello invece della funzione `pipeline()`. Sebbene l'inferenza sia possibile con la funzione `pipeline()`, essa non è ottimizzata per i modelli mixed-8bit e sarà più lenta rispetto all'uso del metodo `generate()`. Inoltre, alcune strategie di campionamento, come il campionamento nucleaus, non sono supportate dalla funzione `pipeline()` per i modelli mixed-8bit.
* collocare tutti gli ingressi sullo stesso dispositivo del modello.

Ecco un semplice esempio:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

text = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```


### Esecuzione di modelli mixed-8bit - configurazione multi GPU

Usare il seguente modo caricare il modello mixed-8bit su più GPU (stesso comando della configurazione a GPU singola):
```py
model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```
Puoi controllare la RAM della GPU che si vuole allocare su ogni GPU usando `accelerate`. Utilizzare l'argomento `max_memory` come segue:

```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```
In questo esempio, la prima GPU utilizzerà 1 GB di memoria e la seconda 2 GB.

### Colab demos

Con questo metodo è possibile inferire modelli che prima non era possibile inferire su Google Colab.
Guardate la demo per l'esecuzione di T5-11b (42GB in fp32)! Utilizzo la quantizzazione a 8 bit su Google Colab:

[![Open In Colab: T5-11b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)

Oppure questa demo di BLOOM-3B:

[![Open In Colab: BLOOM-3b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing)