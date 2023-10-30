<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fine-tuning di un modello pre-addestrato

[[open-in-colab]]

Ci sono benefici significativi nell'usare un modello pre-addestrato. Si riducono i costi computazionali, l'impronta di carbonio e ti consente di usare modelli stato dell'arte senza doverli addestrare da zero. 🤗 Transformers consente l'accesso a migliaia di modelli pre-addestrati per un'ampia gamma di compiti. Quando usi un modello pre-addestrato, lo alleni su un dataset specifico per il tuo compito. Questo è conosciuto come fine-tuning, una tecnica di addestramento incredibilmente potente. In questa esercitazione, potrai fare il fine-tuning di un modello pre-addestrato, con un framework di deep learning a tua scelta:

* Fine-tuning di un modello pre-addestrato con 🤗 Transformers [`Trainer`].
* Fine-tuning di un modello pre-addestrato in TensorFlow con Keras.
* Fine-tuning di un modello pre-addestrato con PyTorch.

<a id='data-processing'></a>

## Preparare un dataset

<Youtube id="_BZearw7f0w"/>

Prima di poter fare il fine-tuning di un modello pre-addestrato, scarica un dataset e preparalo per l'addestramento. La precedente esercitazione ti ha mostrato come processare i dati per l'addestramento e adesso hai l'opportunità di metterti alla prova!

Inizia caricando il dataset [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full):

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

Come già sai, hai bisogno di un tokenizer per processare il testo e includere una strategia di padding e truncation per gestire sequenze di lunghezza variabile. Per processare il dataset in un unico passo, usa il metodo [`map`](https://huggingface.co/docs/datasets/process.html#map) di 🤗 Datasets che applica la funzione di preprocessing all'intero dataset:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

Se vuoi, puoi creare un sottoinsieme più piccolo del dataset per il fine-tuning così da ridurre il tempo necessario:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Addestramento

<frameworkcontent>
<pt>
<Youtube id="nvBXf7s7vTI"/>

🤗 Transformers mette a disposizione la classe [`Trainer`] ottimizzata per addestrare modelli 🤗 Transformers, rendendo semplice iniziare l'addestramento senza scrivere manualmente il tuo ciclo di addestramento. L'API [`Trainer`] supporta un'ampia gamma di opzioni e funzionalità di addestramento come logging, gradient accumulation e mixed precision.

Inizia caricando il tuo modello e specificando il numero di etichette (labels) attese. Nel dataset Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields), sai che ci sono cinque etichette:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

<Tip>

Potresti vedere un warning dato che alcuni dei pesi pre-addestrati non sono stati utilizzati e altri pesi sono stati inizializzati casualmente. Non preoccuparti, è completamente normale! L'head pre-addestrata del modello BERT viene scartata e rimpiazzata da una classification head inizializzata casualmente. Farai il fine-tuning di questa nuova head del modello sul tuo compito di classificazione, trasferendogli la conoscenza del modello pre-addestrato.

</Tip>

### Iperparametri per il training

Successivamente, crea una classe [`TrainingArguments`] contenente tutti gli iperparametri che si possono regore nonché le variabili per attivare le differenti opzioni di addestramento. Per questa esercitazione puoi iniziare con gli [iperparametri](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) di ddestramento predefiniti, ma sentiti libero di sperimentare per trovare la configurazione ottimale per te.

Specifica dove salvare i checkpoints del tuo addestramento:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### Metriche

[`Trainer`] non valuta automaticamente le performance del modello durante l'addestramento. Dovrai passare a [`Trainer`] una funzione che calcola e restituisce le metriche. La libreria 🤗 Datasets mette a disposizione una semplice funzione [`accuracy`](https://huggingface.co/metrics/accuracy) che puoi caricare con la funzione `load_metric` (guarda questa [esercitazione](https://huggingface.co/docs/datasets/metrics.html) per maggiori informazioni):

```py
>>> import numpy as np
>>> from datasets import load_metric

>>> metric = load_metric("accuracy")
```

Richiama `compute` su `metric` per calcolare l'accuratezza delle tue previsioni. Prima di passare le tue previsioni a `compute`, hai bisogno di convertirle in logits (ricorda che tutti i modelli 🤗 Transformers restituiscono logits):

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

Se preferisci monitorare le tue metriche di valutazione durante il fine-tuning, specifica il parametro `evaluation_strategy` nei tuoi training arguments per restituire le metriche di valutazione ad ogni epoca di addestramento:

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```

### Trainer

Crea un oggetto [`Trainer`] col tuo modello, training arguments, dataset di training e test, e funzione di valutazione:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Poi metti a punto il modello richiamando [`~transformers.Trainer.train`]:

```py
>>> trainer.train()
```
</pt>
<tf>
<a id='keras'></a>

<Youtube id="rnTGBy2ax1c"/>

I modelli 🤗 Transformers supportano anche l'addestramento in TensorFlow usando l'API di Keras.

### Convertire dataset nel formato per TensorFlow

Il [`DefaultDataCollator`] assembla tensori in lotti su cui il modello si addestrerà. Assicurati di specificare di restituire tensori per TensorFlow in `return_tensors`:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```

<Tip>

[`Trainer`] usa [`DataCollatorWithPadding`] in maniera predefinita in modo da non dover specificare esplicitamente un collettore di dati.

</Tip>

Successivamente, converti i datasets tokenizzati in TensorFlow datasets con il metodo [`to_tf_dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.to_tf_dataset). Specifica il tuo input in `columns` e le tue etichette in `label_cols`:

```py
>>> tf_train_dataset = small_train_dataset.to_tf_dataset(
...     columns=["attention_mask", "input_ids", "token_type_ids"],
...     label_cols=["labels"],
...     shuffle=True,
...     collate_fn=data_collator,
...     batch_size=8,
... )

>>> tf_validation_dataset = small_eval_dataset.to_tf_dataset(
...     columns=["attention_mask", "input_ids", "token_type_ids"],
...     label_cols=["labels"],
...     shuffle=False,
...     collate_fn=data_collator,
...     batch_size=8,
... )
```

### Compilazione e addestramento

Carica un modello TensorFlow col numero atteso di etichette:

```py
>>> import tensorflow as tf
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

Poi compila e fai il fine-tuning del tuo modello usando [`fit`](https://keras.io/api/models/model_training_apis/) come faresti con qualsiasi altro modello di Keras:

```py
>>> model.compile(
...     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
...     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
...     metrics=tf.metrics.SparseCategoricalAccuracy(),
... )

>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)
```
</tf>
</frameworkcontent>

<a id='pytorch_native'></a>

## Addestramento in PyTorch nativo

<frameworkcontent>
<pt>
<Youtube id="Dh9CL8fyG80"/>

[`Trainer`] si occupa del ciclo di addestramento e ti consente di mettere a punto un modello con una sola riga di codice. Per chi preferisse scrivere un proprio ciclo di addestramento personale, puoi anche fare il fine-tuning di un modello 🤗 Transformers in PyTorch nativo.

A questo punto, potresti avere bisogno di riavviare il tuo notebook o eseguire il seguente codice per liberare un po' di memoria:

```py
del model
del pytorch_model
del trainer
torch.cuda.empty_cache()
```

Successivamente, postprocessa manualmente il `tokenized_dataset` per prepararlo ad essere allenato.

1. Rimuovi la colonna `text` perché il modello non accetta testo grezzo come input:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. Rinomina la colonna `label` in `labels` perché il modello si aspetta che questo argomento si chiami `labels`:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. Imposta il formato del dataset per farti restituire tensori di PyTorch all'interno delle liste:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

Poi crea un piccolo sottocampione del dataset come visto precedentemente per velocizzare il fine-tuning:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

Crea un `DataLoader` per i tuoi datasets di train e test così puoi iterare sui lotti di dati:

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

Carica il tuo modello con il numero atteso di etichette:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```

### Ottimizzatore e learning rate scheduler

Crea un ottimizzatore e il learning rate scheduler per fare il fine-tuning del modello. Usa l'ottimizzatore [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) di PyTorch:

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

Crea il learning rate scheduler predefinito da [`Trainer`]:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

Infine specifica come `device` da usare una GPU se ne hai una. Altrimenti, l'addestramento su una CPU può richiedere diverse ore invece di un paio di minuti.

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

Ottieni l'accesso gratuito a una GPU sul cloud se non ne possiedi una usando un notebook sul web come [Colaboratory](https://colab.research.google.com/) o [SageMaker StudioLab](https://studiolab.sagemaker.aws/).

</Tip>

Ottimo, adesso possiamo addestrare! 🥳 

### Training loop

Per tenere traccia dei tuoi progressi durante l'addestramento, usa la libreria [tqdm](https://tqdm.github.io/) per aggiungere una progress bar sopra il numero dei passi di addestramento:

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### Metriche

Proprio come è necessario aggiungere una funzione di valutazione del [`Trainer`], è necessario fare lo stesso quando si scrive il proprio ciclo di addestramento. Ma invece di calcolare e riportare la metrica alla fine di ogni epoca, questa volta accumulerai tutti i batch con [`add_batch`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=add_batch#datasets.Metric.add_batch) e calcolerai la metrica alla fine.

```py
>>> metric = load_metric("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```
</pt>
</frameworkcontent>

<a id='additional-resources'></a>

## Altre risorse

Per altri esempi sul fine-tuning, fai riferimento a:

- [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) include scripts per addestrare compiti comuni di NLP in PyTorch e TensorFlow.

- [🤗 Transformers Notebooks](notebooks) contiene diversi notebooks su come mettere a punto un modello per compiti specifici in PyTorch e TensorFlow.
