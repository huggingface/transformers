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

# Condividi un modello

Gli ultimi due tutorial ti hanno mostrato come puoi fare fine-tuning di un modello con PyTorch, Keras e 🤗 Accelerate per configurazioni distribuite. Il prossimo passo è quello di condividere il tuo modello con la community! In Hugging Face, crediamo nella condivisione della conoscenza e delle risorse in modo da democratizzare l'intelligenza artificiale per chiunque. Ti incoraggiamo a considerare di condividere il tuo modello con la community per aiutare altre persone a risparmiare tempo e risorse.

In questo tutorial, imparerai due metodi per la condivisione di un modello trained o fine-tuned nel [Model Hub](https://huggingface.co/models):

- Condividi in modo programmatico i tuoi file nell'Hub.
- Trascina i tuoi file nell'Hub mediante interfaccia grafica.

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

Per condividere un modello con la community, hai bisogno di un account su [huggingface.co](https://huggingface.co/join). Puoi anche unirti ad un'organizzazione esistente o crearne una nuova.

</Tip>

## Caratteristiche dei repository

Ogni repository nel Model Hub si comporta come un tipico repository di GitHub. I nostri repository offrono il versionamento, la cronologia dei commit, e la possibilità di visualizzare le differenze.

Il versionamento all'interno del Model Hub è basato su git e [git-lfs](https://git-lfs.github.com/). In altre parole, puoi trattare un modello come un unico repository, consentendo un maggiore controllo degli accessi e maggiore scalabilità. Il controllo delle versioni consente *revisions*, un metodo per appuntare una versione specifica di un modello con un hash di commit, un tag o un branch.

Come risultato, puoi caricare una specifica versione di un modello con il parametro `revision`:

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # nome di un tag, di un branch, o commit hash
... )
```

Anche i file possono essere modificati facilmente in un repository ed è possibile visualizzare la cronologia dei commit e le differenze:

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## Configurazione

Prima di condividere un modello nell'Hub, hai bisogno delle tue credenziali di Hugging Face. Se hai accesso ad un terminale, esegui il seguente comando nell'ambiente virtuale in cui è installata la libreria 🤗 Transformers. Questo memorizzerà il tuo token di accesso nella cartella cache di Hugging Face (di default `~/.cache/`):

```bash
huggingface-cli login
```

Se stai usando un notebook come Jupyter o Colaboratory, assicurati di avere la libreria [`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library) installata. Questa libreria ti permette di interagire in maniera programmatica con l'Hub.

```bash
pip install huggingface_hub
```

Utilizza `notebook_login` per accedere all'Hub, e segui il link [qui](https://huggingface.co/settings/token) per generare un token con cui effettuare il login:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Converti un modello per tutti i framework

Per assicurarti che il tuo modello possa essere utilizzato da persone che lavorano con un framework differente, ti raccomandiamo di convertire e caricare il tuo modello sia con i checkpoint di PyTorch che con quelli di TensorFlow. Anche se è possibile caricare il modello da un framework diverso, se si salta questo passaggio, il caricamento sarà più lento perché 🤗 Transformers ha bisogno di convertire i checkpoint al momento.

Convertire un checkpoint per un altro framework è semplice. Assicurati di avere PyTorch e TensorFlow installati (vedi [qui](installation) per le istruzioni d'installazione), e poi trova il modello specifico per il tuo compito nell'altro framework.

<frameworkcontent>
<pt>
Specifica `from_tf=True` per convertire un checkpoint da TensorFlow a PyTorch:

```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained(
...     "path/verso/il-nome-magnifico-che-hai-scelto", from_tf=True
... )
>>> pt_model.save_pretrained("path/verso/il-nome-magnifico-che-hai-scelto")
```
</pt>
<tf>
Specifica `from_pt=True` per convertire un checkpoint da PyTorch a TensorFlow:

```py
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained(
...     "path/verso/il-nome-magnifico-che-hai-scelto", from_pt=True
... )
```

Poi puoi salvare il tuo nuovo modello in TensorFlow con il suo nuovo checkpoint:

```py
>>> tf_model.save_pretrained("path/verso/il-nome-magnifico-che-hai-scelto")
```
</tf>
<jax>
Se un modello è disponibile in Flax, puoi anche convertire un checkpoint da PyTorch a Flax:

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/verso/il-nome-magnifico-che-hai-scelto", from_pt=True
... )
```
</jax>
</frameworkcontent>

## Condividi un modello durante il training

<frameworkcontent>
<pt>
<Youtube id="Z1-XMy-GNLQ"/>

Condividere un modello nell'Hub è tanto semplice quanto aggiungere un parametro extra o un callback. Ricorda dal [tutorial sul fine-tuning](training), la classe [`TrainingArguments`] è dove specifichi gli iperparametri e le opzioni addizionali per l'allenamento. Una di queste opzioni di training include l'abilità di condividere direttamente un modello nell'Hub. Imposta `push_to_hub=True` in [`TrainingArguments`]:

```py
>>> training_args = TrainingArguments(output_dir="il-mio-bellissimo-modello", push_to_hub=True)
```

Passa gli argomenti per il training come di consueto al [`Trainer`]:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Dopo aver effettuato il fine-tuning del tuo modello, chiama [`~transformers.Trainer.push_to_hub`] sul [`Trainer`] per condividere il modello allenato nell'Hub. 🤗 Transformers aggiungerà in modo automatico persino gli iperparametri, i risultati del training e le versioni del framework alla scheda del tuo modello (model card, in inglese)!

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
Condividi un modello nell'Hub con [`PushToHubCallback`]. Nella funzione [`PushToHubCallback`], aggiungi:

- Una directory di output per il tuo modello.
- Un tokenizer.
- L'`hub_model_id`, che è il tuo username sull'Hub e il nome del modello.

```py
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./il_path_dove_salvare_il_tuo_modello",
...     tokenizer=tokenizer,
...     hub_model_id="il-tuo-username/il-mio-bellissimo-modello",
... )
```

Aggiungi il callback a [`fit`](https://keras.io/api/models/model_training_apis/), e 🤗 Transformers caricherà il modello allenato nell'Hub:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```
</tf>
</frameworkcontent>

## Utilizzare la funzione `push_to_hub`

Puoi anche chiamare `push_to_hub` direttamente sul tuo modello per caricarlo nell'Hub.

Specifica il nome del tuo modello in `push_to_hub`:

```py
>>> pt_model.push_to_hub("il-mio-bellissimo-modello")
```

Questo crea un repository sotto il proprio username con il nome del modello `il-mio-bellissimo-modello`. Ora chiunque può caricare il tuo modello con la funzione `from_pretrained`:

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("il-tuo-username/il-mio-bellissimo-modello")
```

Se fai parte di un'organizzazione e vuoi invece condividere un modello sotto il nome dell'organizzazione, aggiungi il parametro `organization`:

```py
>>> pt_model.push_to_hub("il-mio-bellissimo-modello", organization="la-mia-fantastica-org")
```

La funzione `push_to_hub` può essere anche utilizzata per aggiungere altri file al repository del modello. Per esempio, aggiungi un tokenizer ad un repository di un modello:

```py
>>> tokenizer.push_to_hub("il-mio-bellissimo-modello")
```

O magari potresti voler aggiungere la versione di TensorFlow del tuo modello PyTorch a cui hai fatto fine-tuning:

```py
>>> tf_model.push_to_hub("il-mio-bellissimo-modello")
```

Ora quando navighi nel tuo profilo Hugging Face, dovresti vedere il tuo repository del modello appena creato. Premendo sulla scheda **Files** vengono visualizzati tutti i file caricati nel repository.

Per maggiori dettagli su come creare e caricare file ad un repository, fai riferimento alla documentazione [qui](https://huggingface.co/docs/hub/how-to-upstream).

## Carica un modello utilizzando l'interfaccia web

Chi preferisce un approccio senza codice può caricare un modello tramite l'interfaccia web dell'hub. Visita [huggingface.co/new](https://huggingface.co/new) per creare un nuovo repository:

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

Da qui, aggiungi alcune informazioni sul tuo modello:

- Seleziona il/la **owner** del repository. Puoi essere te o qualunque organizzazione di cui fai parte.
- Scegli un nome per il tuo modello, il quale sarà anche il nome del repository.
- Scegli se il tuo modello è pubblico o privato.
- Specifica la licenza utilizzata per il tuo modello.

Ora premi sulla scheda **Files** e premi sul pulsante **Add file** per caricare un nuovo file al tuo repository. Trascina poi un file per caricarlo e aggiungere un messaggio di commit.

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## Aggiungi una scheda del modello

Per assicurarti che chiunque possa comprendere le abilità, limitazioni, i potenziali bias e le considerazioni etiche del tuo modello, per favore aggiungi una scheda del modello (model card, in inglese) al tuo repository. La scheda del modello è definita nel file `README.md`. Puoi aggiungere una scheda del modello:

* Creando manualmente e caricando un file `README.md`.
* Premendo sul pulsante **Edit model card** nel repository del tuo modello.

Dai un'occhiata alla [scheda del modello](https://huggingface.co/distilbert/distilbert-base-uncased) di DistilBert per avere un buon esempio del tipo di informazioni che una scheda di un modello deve includere. Per maggiori dettagli legati ad altre opzioni che puoi controllare nel file `README.md`, come l'impatto ambientale o widget di esempio, fai riferimento alla documentazione [qui](https://huggingface.co/docs/hub/models-cards).
