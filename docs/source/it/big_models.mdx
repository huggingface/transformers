<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Istanziare un big model

Quando vuoi utilizzare un modello preaddestrato (pretrained) molto grande, una sfida è minimizzare l'uso della RAM. Il workflow classico
in PyTorch è:

1. Crea il tuo modello con pesi casuali (random weights).
2. Carica i tuoi pesi preaddestrati.
3. Inserisci i pesi preaddestrati nel tuo modello casuale.

I passi 1 e 2 una versione completa del modello in memoria, in molti casi non è un problema, ma se il modello inizia a pesare diversi GigaBytes, queste due copie possono sturare la nostra RAM. Ancora peggio, se stai usando `torch.distributed` per seguire l'addestramento (training) in distribuito, ogni processo caricherà il modello preaddestrato e memorizzerà queste due copie nella RAM.

<Tip>

Nota che il modello creato casualmente è inizializzato con tensori "vuoti", che occupano spazio in memoria ma senza riempirlo (quindi i valori casuali sono quelli che si trovavano in questa porzione di memoria in un determinato momento). L'inizializzazione casuale che segue la distribuzione appropriata per il tipo di modello/parametri istanziato (come la distribuzione normale per le istanze) è eseguito solo dopo il passaggio 3 sui pesi non inizializzati, per essere più rapido possibile!

</Tip>

In questa guida, esploreremo le soluzioni che Transformers offre per affrontare questo problema. C'è da tenere in conto che questa è un'area in cui si sta attualmente sviluppando, quindi le API spiegate qui possono variare velocemente in futuro.

## Checkpoints condivisi

Dalla versione 4.18.0, i checkpoints dei modelli che occupano più di 10GB di spazio vengono automaticamente frammentati in più parti. Per quanto riguarda la possibilità di avere un unico checkpoint quando si utilizza `model.save_pretrained(save_dir)`, si hanno diversi checkpoint parziali (ognuno con dimensione < 10GB) e un  indice che mappa i nomi dei parametri ai file in cui sono memorizzati.

Puoi controllare la dimensione massima dopo la frammentazione con il parametro `max_shard_size`, nel prossimo esempio, useremo modelli di dimensioni normali con frammenti di piccoli dimensioni: prendiamo un modello BERT classico.

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

Se tu salvi usando [`~PreTrainedModel.save_pretrained`], avrai una nuova cartella con due file: il config del modello e i suoi pesi:

```py
>>> import os
>>> import tempfile

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir)
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model.bin']
```

Adesso usiamo una dimensione massima di frammentazione di 200MB:

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']
```

In aggiunta alla configurazione del modello, vediamo tre differenti file dei pesi, e un file `index.json` che è il nostro indice. Un checkpoint può essere ricaricato totalmente usando il metodo [`~PreTrainedModel.from_pretrained`]:

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

Il vantaggio principale di applicare questo metodo per modelli grandi è che durante il passo 2 del workflow illustrato in precedenza, ogni frammento del checkpoint viene caricato dopo il precedente, limitando l'utilizzo della RAM alla dimensione del modello più la dimensione del frammento più grande.

Dietro le quinte, il file indice è utilizzato per determinare quali chiavi sono nel checkpoint, e dove i corrispondenti pesi sono memorizzati. Possiamo caricare l'indice come un qualsiasi json e ottenere un dizionario:

```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     with open(os.path.join(tmp_dir, "pytorch_model.bin.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

I metadati consistono solo nella dimensione totale del modello per ora. Abbiamo in programma di aggiungere altre informazioni in futuro:

```py
>>> index["metadata"]
{'total_size': 433245184}
```

La mappa dei pesi è la parte principale di questo indice, che mappa ogni nome dei parametri (si trova solitamente nei modelli PyTorch come `state_dict`) al file in cui è memorizzato:

```py
>>> index["weight_map"]
{'embeddings.LayerNorm.bias': 'pytorch_model-00001-of-00003.bin',
 'embeddings.LayerNorm.weight': 'pytorch_model-00001-of-00003.bin',
 ...
```

Se vuoi caricare direttamente un checkpoint frammentato in un modello senza usare [`~PreTrainedModel.from_pretrained`] (come si farebbe con `model.load_state_dict()` per un checkpoint completo) devi usare [`~modeling_utils.load_sharded_checkpoint`]:

```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     load_sharded_checkpoint(model, tmp_dir)
```

## Caricamento low memory

Frammentare i checkpoint l'utilizzo di memoria al passo 2 del workflow citato in precedenza, ma per utilizzare questo modello in un ambiente con poca memoria, consigliamo di utilizzare i nostri strumenti basati sulla libreria Accelerate.

Per ulteriori informazioni, leggere la seguente guida: [Large model loading using Accelerate](./main_classes/model#large-model-loading)