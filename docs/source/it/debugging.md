<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Debugging

## Debug dei problemi di rete multi-GPU

Quando addestri o fai inferenza con `DistributedDataParallel` e GPU multiple, se si verificano problemi di intercomunicazione tra processi e/o nodi, puoi utilizzare il seguente script per diagnosticare i problemi della rete.

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

Per esempio per testare come 2 GPU interagiscono fai:

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

Se entrambi i processi sono in grado di comunicare tra loro e di allocare la memoria della GPU, ciascuno di essi stamperà lo stato OK.

Per più GPU o nodi adatta gli argumenti nello script.

All'interno dello script di diagnostica troverai molti altri dettagli e anche una guida per eseguirlo in ambiente SLURM.

Un livello di debug superiore è aggiungere la variabile d'ambiente `NCCL_DEBUG=INFO` come di seguito:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

In questo modo si scaricano molte informazioni di debug relative a NCCL, che puoi cercare online in caso di problemi. Oppure, se non hai la sicurezza di come interpretare l'output, puoi condividere il file di log in una Issue.

## Rilevamento di Underflow e Overflow

<Tip>

Questa funzionalità al momento è disponibile solo per PyTorch.

</Tip>

<Tip>

Per addestramento multi-GPU richiede DDP (`torch.distributed.launch`).

</Tip>

<Tip>

Questa funzionalità può essere usata con modelli basati su `nn.Module`.

</Tip>

Se inizi a ottenere `loss=NaN` o il modello presenta qualche altro comportamento anomalo a causa di valori `inf` o `nan` in
attivazioni o nei pesi, è necessario scoprire dove si verifica il primo underflow o overflow e cosa lo ha determinato. Fortunatamente
è possibile farlo facilmente attivando un modulo speciale che effettuerà il rilevamento automaticamente.

Se stai usando [`Trainer`], hai bisogno di aggiungere solo:

```bash
--debug underflow_overflow
```

ai normali argomenti della riga di comando, o passa `debug="underflow_overflow"` quando viene creato l'oggetto
[`TrainingArguments`].

Se stai usando il tuo ciclo di allenamento o un altro trainer, puoi ottenere lo stesso risultato con:

```python
from .debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

[`~debug_utils.DebugUnderflowOverflow`] inserisce dei ganci nel modello che dopo ogni chiamata
testeranno le variabili di ingresso e di uscita e anche i pesi del modulo corrispondente. Non appena viene rilevato `inf` o
o `nan` in almeno un elemento delle attivazioni o dei pesi, il programma lo notifica e stampa un rapporto come il seguente (questo è stato rilevato con `google/mt5-small` sotto fp16 mixed precision):

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

L'output di esempio è stato tagliato al centro per brevità.

La seconda colonna mostra il valore dell'elemento più grande in assoluto,così se osserviamo da vicino gli ultimi istanti,
input e output sono nel range di `1e4`. Questo addestramento è stato eseguito con una mixed precision fp16 e l'ultimo passo usciva fuori (sotto `fp16` il valore più grande prima di `inf` è `64e3`). Per evitare overflows sotto `fp16` le attivazionioni devono rimanere molto al di sotto di `1e4`, perché `1e4 * 1e4 = 1e8` quindi qualsiasi moltiplicazione di matrice con grandi attivazioni porterà a una condizione di overflow numerico.

All'inizio della traccia è possibile scoprire a quale lotto si è verificato il problema (questo `Detected inf/nan during batch_number=0` significa che il problema si è verificato nel primo lotto).

Ogni frame segnalato inizia dichiarando la voce completamente qualificata per il modulo corrispondente per il quale il frame è stato segnalato. 
Se osserviamo il seguente frame:

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

Questo, `encoder.block.2.layer.1.layer_norm` indica che si tratta di un layer norm nel primo layer, del secondo blocco dell'encoder. E le chiamata specifica di `forward` è `T5LayerNorm`.

Osserviamo gli ultimi frame del report:

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

L'ultimo frame report per la funzione `Dropout.forward` con la prima voce per l'unico input e la seconda per l'unico output. Si può notare che è stato richiamato da un attibuto `dropout` dentro la classe `DenseReluDense`. Si può notare che ciò è avvenuto durante il primo strato, del 2° blocco, durante il primissimo lotto. Infine, gli elementi di input più grandi in assoluto sono stati `6.27e+04` e l'equivalente per l'output era `inf`.

Puoi vedere qui, che `T5DenseGatedGeluDense.forward` risulta in output activations, il cui valore massimo assoluto era circa 62,7K, che è molto vicino al limite massimo di 64K di fp16. Nel prossimo frame abbiamo `Dropout` che rinormalizza i pesi, dopo aver azzerato alcuni elementi, il che spinge il valore massimo assoluto a più di 64K e si verifica un overflow.(`inf`).

Come puoi notare, è nei frames precedenti che occorre esaminare quando i numeri iniziano a diventare molto grandi per i valori fp16.

Confrontiamo il report al codice `models/t5/modeling_t5.py`:

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

Ora è facile vedere la chiamata `dropout`, e tutte le chiamate precedenti.

Poiché il rilevamento avviene in un avanzamento (forward hook in eng.), i rapporti vengono creati immeditamente dopo ogni rientro da `forward` (forward returns in eng.).

Tornando al rapporto completo, per agire e risolvere il problema, dobbiamo andare qualche frame più in alto, dove i numeri hanno iniziato a salire, e probabilmente passare alla modalità `fp32`, in modo che i numeri non trabocchino quando vengono moltiplicati o sommati. Naturalmente, potrebbero esserci altre soluzioni. Per esempio, potremmo spegnere temporanemante `amp` se è abilitato, successivamente spostare `forward` in un helper wrapper, come:

```python
def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


import torch


def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

Poiché il rilevatore automatico riporta solo gli ingressi e le uscite di fotogrammi completi, una volta che si sa dove cercare, si può
analizzare anche le fasi intermedie di una specifica funzione `forward`. In alcuni casi puoi usare la funzione di supporto `detect_overflow` per indirizzare il rilevatore dove preferisci, ad esempio:

```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

Si può vedere che abbiamo aggiunto 2 di questi e ora teniamo traccia se `inf` o `nan` per `forwarded_states` è stato rilevato
da qualche parte.

In realtà, il rilevatore li riporta già, perché ciascuna delle chiamate nell'esempio precedente è un `nn.Module`, ma
diciamo che se avessimo dei calcoli diretti locali, questo è il modo in cui lo faremmo.

Inoltre, se si istanzia il debugger nel proprio codice, è possibile modificare il numero di fotogrammi stampati rispetto a
predefinito, ad esempio.:

```python
from .debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### Tracciamento della mistura assoluta del lotto specifico e del valore massimo

La stessa classe di debug può essere utilizzata per il tracciamento per-batch con la funzione di rilevamento di underflow/overflow disattivata.

Supponiamo di voler osservare i valori minimi e massimi assoluti per tutti gli ingredienti di ogni chiamata `forward` di un dato lotto.
lotto, e che lo si voglia fare solo per i lotti 1 e 3. Si istanzia questa classe come:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

Ora i batch completi 1 e 3 saranno tracciati utilizzando lo stesso formato del rilevatore di underflow/overflow.

I batches sono 0-indexed.

Questo è utile se si sa che il programma inizia a comportarsi male dopo un certo numero di batch, in modo da poter avanzare velocemente fino a quell'area.
direttamente a quell'area. Ecco un esempio di output troncato per questa configurazione:

```
                  *** Starting batch number=1 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     not a tensor output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     not a tensor output

                  *** Starting batch number=3 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

Qui verrà scaricato un numero enorme di fotogrammi, tanti quanti sono le chiamate in avanti nel modello, quindi può essere o non essere quello che volete, ma a volte può essere più utile usarlo di un classico debugger. Per esempio, se il problema inizia a verificarsi a partire dal lotto numero 150. Quindi è possibile scaricare le tracce dei lotti 149 e 150 e confrontare i punti in cui i numeri hanno iniziato a divergere.

È inoltre possibile specificare il numero di batch dopo il quale interrompere l'addestramento, con:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
