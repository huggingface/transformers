<!--- 
Copyright 2020 The HuggingFace Team. Tutti i diritti riservati. 

Concesso in licenza in base alla Licenza Apache, Versione 2.0 (la "Licenza"); 
non è possibile utilizzare questo file se non in conformità con la Licenza. 
È possibile ottenere una copia della Licenza all'indirizzo 

http://www.apache.org/licenses/LICENSE-2.0 

A meno che non sia richiesto dalla legge applicabile o concordato per iscritto, il software 
distribuito con la Licenza è distribuito su BASE "COSÌ COM'È", 
SENZA GARANZIE O CONDIZIONI DI ALCUN TIPO, espresse o implicite. 
Per la lingua specifica vedi la Licenza che regola le autorizzazioni e 
le limitazioni ai sensi della STESSA. 

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

--> 

# Migrazione da pacchetti precedenti 

## Migrazione da transformers `v3.x` a `v4.x` 

Un paio di modifiche sono state introdotte nel passaggio dalla versione 3 alla versione 4. Di seguito è riportato un riepilogo delle 
modifiche previste: 

#### 1. AutoTokenizer e pipeline ora utilizzano tokenizer veloci (rust) per impostazione predefinita. 

I tokenizer python e rust hanno all'incirca le stesse API, ma i tokenizer rust hanno un set di funzionalità più completo. 

Ciò introduce due modifiche sostanziali: 
- La gestione dei token in overflow tra i tokenizer Python e Rust è diversa. 
- I tokenizers di rust non accettano numeri interi nei metodi di codifica. 

##### Come ottenere lo stesso comportamento di v3.x in v4.x 

- Le pipeline ora contengono funzionalità aggiuntive pronte all'uso. Vedi la [pipeline di classificazione dei token con il flag `grouped_entities`](main_classes/pipelines#transformers.TokenClassificationPipeline). 
- Gli auto-tokenizer ora restituiscono tokenizer rust. Per ottenere invece i tokenizer python, l'utente deve usare il flag `use_fast` impostandolo `False`: 

Nella versione `v3.x`: 
```py 
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
``` 
per ottenere lo stesso nella versione `v4.x`: 
```py 
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False) 
``` 

#### 2. SentencePiece è stato rimosso dalle dipendenze richieste 

Il requisito sulla dipendenza SentencePiece è stato rimosso da `setup.py`. È stato fatto per avere un canale su anaconda cloud senza basarsi su `conda-forge`. Ciò significa che i tokenizer che dipendono dalla libreria SentencePiece non saranno disponibili con un'installazione standard di `transformers`. 

Ciò include le versioni **lente** di: 
- `XLNetTokenizer` 
- `AlbertTokenizer` 
- `CamembertTokenizer` 
- `MBartTokenizer` 
- `PegasusTokenizer` 
- `T5Tokenizer` 
- `ReformerTokenizer` 
- `XLMRobertaTokenizer` 

##### Come ottenere lo stesso comportamento della v3.x nella v4.x 

Per ottenere lo stesso comportamento della versione `v3.x`, devi installare anche `sentencepiece`: 

Nella versione `v3.x`: 
```bash 
pip install transformers 
``` 
per ottenere lo stesso nella versione `v4.x`: 
```bash 
pip install transformers[sentencepiece] 
``` 
o 
```bash 
pip install transformers stentencepiece 
``` 
#### 3. L'architettura delle repo è stato aggiornata in modo che ogni modello abbia la propria cartella 

Con l’aggiunta di nuovi modelli, il numero di file nella cartella `src/transformers` continua a crescere e diventa più difficile navigare e capire. Abbiamo fatto la scelta di inserire ogni modello e i file che lo accompagnano nelle proprie sottocartelle. 

Si tratta di una modifica sostanziale in quanto l'importazione di layer intermedi utilizzando direttamente il modulo di un modello deve essere eseguita tramite un percorso diverso. 

##### Come ottenere lo stesso comportamento della v3.x nella v4.x 

Per ottenere lo stesso comportamento della versione `v3.x`, devi aggiornare il percorso utilizzato per accedere ai layer. 

Nella versione `v3.x`: 
```bash 
from transformers.modeling_bert import BertLayer 
``` 
per ottenere lo stesso nella versione `v4.x`: 
```bash 
from transformers.models.bert.modeling_bert import BertLayer 
``` 

#### 4. Impostare l'argomento `return_dict` su `True` per impostazione predefinita 

L'[argomento `return_dict`](main_classes/output) abilita la restituzione di oggetti python dict-like contenenti gli output del modello, invece delle tuple standard. Questo oggetto è self-documented poiché le chiavi possono essere utilizzate per recuperare valori, comportandosi anche come una tupla e gli utenti possono recuperare oggetti per indexing o slicing. 

Questa è una modifica sostanziale poiché la tupla non può essere decompressa: `value0, value1 = outputs` non funzionerà. 

##### Come ottenere lo stesso comportamento della v3.x nella v4.x 

Per ottenere lo stesso comportamento della versione `v3.x`, specifica l'argomento `return_dict` come `False`, sia nella configurazione del modello che nel passaggio successivo. 

Nella versione `v3.x`: 
```bash 
model = BertModel.from_pretrained("bert-base-cased") 
outputs = model(**inputs) 
``` 
per ottenere lo stesso nella versione `v4.x`: 
```bash 
model = BertModel.from_pretrained("bert-base-cased") 
outputs = model(**inputs, return_dict=False) 
``` 
o 
```bash 
model = BertModel.from_pretrained("bert-base-cased", return_dict=False) 
outputs = model(**inputs) 
``` 

#### 5. Rimozione di alcuni attributi deprecati 

Gli attributi sono stati rimossi se deprecati da almeno un mese. L'elenco completo degli attributi obsoleti è disponibile in [#8604](https://github.com/huggingface/transformers/pull/8604). 

Ecco un elenco di questi attributi/metodi/argomenti e quali dovrebbero essere le loro sostituzioni: 

In diversi modelli, le etichette diventano coerenti con gli altri modelli: 
- `masked_lm_labels` diventa `labels` in `AlbertForMaskedLM` e `AlbertForPreTraining`. 
- `masked_lm_labels` diventa `labels` in `BertForMaskedLM` e `BertForPreTraining`. 
- `masked_lm_labels` diventa `labels` in `DistilBertForMaskedLM`. 
- `masked_lm_labels` diventa `labels` in `ElectraForMaskedLM`. 
- `masked_lm_labels` diventa `labels` in `LongformerForMaskedLM`. 
- `masked_lm_labels` diventa `labels` in `MobileBertForMaskedLM`. 
- `masked_lm_labels` diventa `labels` in `RobertaForMaskedLM`. 
- `lm_labels` diventa `labels` in `BartForConditionalGeneration`. 
- `lm_labels` diventa `labels` in `GPT2DoubleHeadsModel`. 
- `lm_labels` diventa `labels` in `OpenAIGPTDoubleHeadsModel`. 
- `lm_labels` diventa `labels` in `T5ForConditionalGeneration`. 

In diversi modelli, il meccanismo di memorizzazione nella cache diventa coerente con gli altri: 
- `decoder_cached_states` diventa `past_key_values` in tutti i modelli BART-like, FSMT e T5. 
- `decoder_past_key_values` diventa `past_key_values` in tutti i modelli BART-like, FSMT e T5. 
- `past` diventa `past_key_values` in tutti i modelli CTRL. 
- `past` diventa `past_key_values` in tutti i modelli GPT-2. 

Per quanto riguarda le classi tokenizer: 
- L'attributo tokenizer `max_len` diventa `model_max_length`. 
- L'attributo tokenizer `return_lengths` diventa `return_length`. 
- L'argomento di codifica del tokenizer `is_pretokenized` diventa `is_split_into_words`. 

Per quanto riguarda la classe `Trainer`: 
- L'argomento `tb_writer` di `Trainer` è stato rimosso in favore della funzione richiamabile `TensorBoardCallback(tb_writer=...)`. 
- L'argomento `prediction_loss_only` di `Trainer` è stato rimosso in favore dell'argomento di classe `args.prediction_loss_only`. 
- L'attributo `data_collator` di `Trainer` sarà richiamabile. 
- Il metodo `_log` di `Trainer` è deprecato a favore di `log`. 
- Il metodo `_training_step` di `Trainer` è deprecato a favore di `training_step`. 
- Il metodo `_prediction_loop` di `Trainer` è deprecato a favore di `prediction_loop`. 
- Il metodo `is_local_master` di `Trainer` è deprecato a favore di `is_local_process_zero`. 
- Il metodo `is_world_master` di `Trainer` è deprecato a favore di `is_world_process_zero`. 

Per quanto riguarda la classe `TFTrainer`: 
- L'argomento `prediction_loss_only` di `TFTrainer` è stato rimosso a favore dell'argomento di classe `args.prediction_loss_only`. 
- Il metodo `_log` di `Trainer` è deprecato a favore di `log`. 
- Il metodo `_prediction_loop` di `TFTrainer` è deprecato a favore di `prediction_loop`. 
- Il metodo `_setup_wandb` di `TFTrainer` è deprecato a favore di `setup_wandb`. 
- Il metodo `_run_model` di `TFTrainer` è deprecato a favore di `run_model`. 

Per quanto riguarda la classe `TrainingArguments`: 
- L'argomento `evaluate_during_training` di `TrainingArguments` è deprecato a favore di `evaluation_strategy`. 

Per quanto riguarda il modello Transfo-XL: 
- L'attributo di configurazione `tie_weight` di Transfo-XL diventa `tie_words_embeddings`. 
- Il metodo di modellazione `reset_length` di Transfo-XL diventa `reset_memory_length`. 

Per quanto riguarda le pipeline: 
- L'argomento `topk` di `FillMaskPipeline` diventa `top_k`. 



## Passaggio da pytorch-transformers a 🤗 Transformers 

Ecco un breve riepilogo di ciò a cui prestare attenzione durante il passaggio da `pytorch-transformers` a 🤗 Transformers. 

### L’ordine posizionale di alcune parole chiave di input dei modelli (`attention_mask`, `token_type_ids`...) è cambiato 

Per usare Torchscript (vedi #1010, #1204 e #1195) l'ordine specifico delle **parole chiave di input** di alcuni modelli (`attention_mask`, `token_type_ids`...) è stato modificato. 

Se inizializzavi i modelli usando parole chiave per gli argomenti, ad esempio `model(inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)`, questo non dovrebbe causare alcun cambiamento. 

Se inizializzavi i modelli con input posizionali per gli argomenti, ad esempio `model(inputs_ids, attention_mask, token_type_ids)`, potrebbe essere necessario ricontrollare l'ordine esatto degli argomenti di input. 

## Migrazione da pytorch-pretrained-bert 

Ecco un breve riepilogo di ciò a cui prestare attenzione durante la migrazione da `pytorch-pretrained-bert` a 🤗 Transformers 

### I modelli restituiscono sempre `tuple` 

La principale modifica di rilievo durante la migrazione da `pytorch-pretrained-bert` a 🤗 Transformers è che il metodo dei modelli di previsione dà sempre una `tupla` con vari elementi a seconda del modello e dei parametri di configurazione. 

Il contenuto esatto delle tuple per ciascun modello è mostrato in dettaglio nelle docstring dei modelli e nella [documentazione](https://huggingface.co/transformers/). 

In quasi tutti i casi, andrà bene prendendo il primo elemento dell'output come quello che avresti precedentemente utilizzato in `pytorch-pretrained-bert`. 

Ecco un esempio di conversione da `pytorch-pretrained-bert`
 a 🤗 Transformers per un modello di classificazione `BertForSequenceClassification`: 

```python 
# Carichiamo il nostro modello 
model = BertForSequenceClassification.from_pretrained("bert-base-uncased") 

# Se usavi questa riga in pytorch-pretrained-bert : 
loss = model(input_ids, labels=labels) 

# Ora usa questa riga in 🤗 Transformers per estrarre la perdita dalla tupla di output: 
outputs = model(input_ids, labels=labels) 
loss = outputs[0] 

# In 🤗 Transformers puoi anche avere accesso ai logit: 
loss, logits = outputs[:2] 

# Ed anche agli attention weight se configuri il modello per restituirli (e anche altri output, vedi le docstring e la documentazione) 
model = BertForSequenceClassification.from_pretrained(" bert-base-uncased", output_attentions=True) 
outputs = model(input_ids, labels=labels) 
loss, logits, attentions = outputs 
``` 

### Serializzazione 

Modifica sostanziale nel metodo `from_pretrained()`: 

1. I modelli sono ora impostati in modalità di valutazione in maniera predefinita quando usi il metodo `from_pretrained()`. Per addestrarli non dimenticare di riportarli in modalità di addestramento (`model.train()`) per attivare i moduli di dropout. 

2. Gli argomenti aggiuntivi `*inputs` e `**kwargs` forniti al metodo `from_pretrained()` venivano passati direttamente al metodo `__init__()` della classe sottostante del modello. Ora sono usati per aggiornare prima l'attributo di configurazione del modello, che può non funzionare con le classi del modello derivate costruite basandosi sui precedenti esempi di `BertForSequenceClassification`. Più precisamente, gli argomenti posizionali `*inputs` forniti a `from_pretrained()` vengono inoltrati direttamente al metodo `__init__()`  del modello mentre gli argomenti keyword `**kwargs` (i) che corrispondono agli attributi della classe di configurazione, vengono utilizzati per aggiornare tali attributi (ii) che non corrispondono ad alcun attributo della classe di configurazione, vengono inoltrati al metodo `__init__()`. 

Inoltre, sebbene non si tratti di una modifica sostanziale, i metodi di serializzazione sono stati standardizzati e probabilmente dovresti passare al nuovo metodo `save_pretrained(save_directory)` se prima usavi qualsiasi altro metodo di serializzazione. 

Ecco un esempio: 

```python 
### Carichiamo un modello e un tokenizer 
model = BertForSequenceClassification.from_pretrained("bert-base-uncased") 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 

### Facciamo fare alcune cose al nostro modello e tokenizer 
# Es: aggiungiamo nuovi token al vocabolario e agli embending del nostro modello 
tokenizer.add_tokens(["[SPECIAL_TOKEN_1]", "[SPECIAL_TOKEN_2]"]) 
model.resize_token_embeddings(len(tokenizer)) 
# Alleniamo il nostro modello
train(model) 

### Ora salviamo il nostro modello e il tokenizer in una cartella 
model.save_pretrained("./my_saved_model_directory/") 
tokenizer.save_pretrained("./my_saved_model_directory/") 

### Ricarichiamo il modello e il tokenizer 
model = BertForSequenceClassification.from_pretrained("./my_saved_model_directory/") 
tokenizer = BertTokenizer.from_pretrained("./my_saved_model_directory/") 
``` 

### Ottimizzatori: BertAdam e OpenAIAdam ora sono AdamW, lo scheduling è quello standard PyTorch 

I due ottimizzatori precedenti inclusi, `BertAdam` e `OpenAIAdam`, sono stati sostituiti da un singolo `AdamW` che presenta alcune differenze: 

- implementa solo la correzione del weights decay, 
- lo scheduling ora è esterno (vedi sotto), 
- anche il gradient clipping ora è esterno (vedi sotto). 

Il nuovo ottimizzatore `AdamW` corrisponde alle API di `Adam` di PyTorch e ti consente di utilizzare metodi PyTorch o apex per lo scheduling e il clipping.

Lo scheduling è ora standard [PyTorch learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) e non fanno più parte dell'ottimizzatore. 

Ecco un esempio di linear warmup e decay con `BertAdam` e con `AdamW`: 

```python 
# Parametri: 
lr = 1e-3 
max_grad_norm = 1.0 
num_training_steps = 1000 
num_warmup_steps = 100 
warmup_proportion = float( num_warmup_steps) / float(num_training_steps) # 0.1 

### In precedenza l'ottimizzatore BertAdam veniva istanziato in questo modo: 
optimizer = BertAdam( 
   model.parameters(), 
   lr=lr, 
   schedule="warmup_linear", 
   warmup=warmup_proportion, 
   num_training_steps=num_training_steps, 
) 
### e usato in questo modo: 
for batch in train_data: 
   loss = model(batch) 
   loss.backward() 
   optimizer.step() 

### In 🤗 Transformers, ottimizzatore e schedule sono divisi e usati in questo modo: 
optimizer = AdamW( 
   model.parameters(), lr=lr, correct_bias=False 
) # Per riprodurre il comportamento specifico di BertAdam impostare correct_bias=False 
scheduler = get_linear_schedule_with_warmup( 
   optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps 
) # PyTorch scheduler 
### e va usato così: 
for batch in train_data: 
   loss = model(batch) 
   loss.backward() 
   torch.nn.utils.clip_grad_norm_( 
   model.parameters(), max_grad_norm 
   ) # Gradient clipping non è più in AdamW (quindi puoi usare amp senza problemi) 
   optimizer.step() 
   scheduler.step()
```
