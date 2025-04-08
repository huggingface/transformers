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

# Installazione

Installa 🤗 Transformers per qualsiasi libreria di deep learning con cui stai lavorando, imposta la tua cache, e opzionalmente configura 🤗 Transformers per l'esecuzione offline.

🤗 Transformers è testato su Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, e Flax. Segui le istruzioni di installazione seguenti per la libreria di deep learning che stai utilizzando:

* [PyTorch](https://pytorch.org/get-started/locally/) istruzioni di installazione.
* [TensorFlow 2.0](https://www.tensorflow.org/install/pip) istruzioni di installazione.
* [Flax](https://flax.readthedocs.io/en/latest/) istruzioni di installazione.

## Installazione con pip

Puoi installare 🤗 Transformers in un [ambiente virtuale](https://docs.python.org/3/library/venv.html). Se non sei familiare con gli ambienti virtuali in Python, dai un'occhiata a questa [guida](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Un ambiente virtuale rende più semplice la gestione di progetti differenti, evitando problemi di compatibilità tra dipendenze.

Inizia creando un ambiente virtuale nella directory del tuo progetto:

```bash
python -m venv .env
```

Attiva l'ambiente virtuale:

```bash
source .env/bin/activate
```

Ora puoi procedere con l'installazione di 🤗 Transformers eseguendo il comando seguente:

```bash
pip install transformers
```

Per il solo supporto della CPU, puoi installare facilmente 🤗 Transformers e una libreria di deep learning in solo una riga. Ad esempio, installiamo 🤗 Transformers e PyTorch con:

```bash
pip install transformers[torch]
```

🤗 Transformers e TensorFlow 2.0:

```bash
pip install transformers[tf-cpu]
```

🤗 Transformers e Flax:

```bash
pip install transformers[flax]
```

Infine, verifica se 🤗 Transformers è stato installato in modo appropriato eseguendo il seguente comando. Questo scaricherà un modello pre-allenato:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

Dopodiché stampa l'etichetta e il punteggio:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## Installazione dalla fonte

Installa 🤗 Transformers dalla fonte con il seguente comando:

```bash
pip install git+https://github.com/huggingface/transformers
```

Questo comando installa la versione `main` più attuale invece dell'ultima versione stabile. Questo è utile per stare al passo con gli ultimi sviluppi. Ad esempio, se un bug è stato sistemato da quando è uscita l'ultima versione ufficiale ma non è stata ancora rilasciata una nuova versione. Tuttavia, questo significa che questa versione `main` può non essere sempre stabile. Ci sforziamo per mantenere la versione `main` operativa, e la maggior parte dei problemi viene risolta in poche ore o in un giorno. Se riscontri un problema, per favore apri una [Issue](https://github.com/huggingface/transformers/issues) così possiamo sistemarlo ancora più velocemente!

Controlla se 🤗 Transformers è stata installata in modo appropriato con il seguente comando:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## Installazione modificabile

Hai bisogno di un'installazione modificabile se vuoi:

* Usare la versione `main` del codice dalla fonte.
* Contribuire a 🤗 Transformers e hai bisogno di testare i cambiamenti nel codice.

Clona il repository e installa 🤗 Transformers con i seguenti comandi:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Questi comandi collegheranno la cartella in cui è stato clonato il repository e i path delle librerie Python. Python guarderà ora all'interno della cartella clonata, oltre ai normali path delle librerie. Per esempio, se i tuoi pacchetti Python sono installati tipicamente in `~/anaconda3/envs/main/lib/python3.7/site-packages/`, Python cercherà anche nella cartella clonata: `~/transformers/`.

<Tip warning={true}>

Devi tenere la cartella `transformers` se vuoi continuare ad utilizzare la libreria.

</Tip>

Ora puoi facilmente aggiornare il tuo clone all'ultima versione di 🤗 Transformers con il seguente comando:

```bash
cd ~/transformers/
git pull
```

Il tuo ambiente Python troverà la versione `main` di 🤗 Transformers alla prossima esecuzione.

## Installazione con conda

Installazione dal canale conda `conda-forge`:

```bash
conda install conda-forge::transformers
```

## Impostazione della cache

I modelli pre-allenati sono scaricati e memorizzati localmente nella cache in: `~/.cache/huggingface/transformers/`. Questa è la directory di default data dalla variabile d'ambiente della shell `TRANSFORMERS_CACHE`. Su Windows, la directory di default è data da `C:\Users\username\.cache\huggingface\transformers`. Puoi cambiare le variabili d'ambiente della shell indicate in seguito, in ordine di priorità, per specificare una directory differente per la cache:

1. Variabile d'ambiente della shell (default): `TRANSFORMERS_CACHE`.
2. Variabile d'ambiente della shell: `HF_HOME` + `transformers/`.
3. Variabile d'ambiente della shell: `XDG_CACHE_HOME` + `/huggingface/transformers`.

<Tip>

🤗 Transformers utilizzerà le variabili d'ambiente della shell `PYTORCH_TRANSFORMERS_CACHE` o `PYTORCH_PRETRAINED_BERT_CACHE` se si proviene da un'iterazione precedente di questa libreria e sono state impostate queste variabili d'ambiente, a meno che non si specifichi la variabile d'ambiente della shell `TRANSFORMERS_CACHE`.

</Tip>

## Modalità Offline

🤗 Transformers può essere eseguita in un ambiente firewalled o offline utilizzando solo file locali. Imposta la variabile d'ambiente `HF_HUB_OFFLINE=1` per abilitare questo comportamento.

<Tip>

Aggiungi [🤗 Datasets](https://huggingface.co/docs/datasets/) al tuo flusso di lavoro offline di training impostando la variabile d'ambiente `HF_DATASETS_OFFLINE=1`.

</Tip>

Ad esempio, in genere si esegue un programma su una rete normale, protetta da firewall per le istanze esterne, con il seguente comando:

```bash
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

Esegui lo stesso programma in un'istanza offline con:

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

Lo script viene ora eseguito senza bloccarsi o attendere il timeout, perché sa di dover cercare solo file locali.

### Ottenere modelli e tokenizer per l'uso offline

Un'altra opzione per utilizzare offline 🤗 Transformers è scaricare i file in anticipo, e poi puntare al loro path locale quando hai la necessità di utilizzarli offline. Ci sono tre modi per fare questo:

* Scarica un file tramite l'interfaccia utente sul [Model Hub](https://huggingface.co/models) premendo sull'icona ↓.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

* Utilizza il flusso [`PreTrainedModel.from_pretrained`] e [`PreTrainedModel.save_pretrained`]:

    1. Scarica i tuoi file in anticipo con [`PreTrainedModel.from_pretrained`]:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. Salva i tuoi file in una directory specificata con [`PreTrainedModel.save_pretrained`]:

    ```py
    >>> tokenizer.save_pretrained("./il/tuo/path/bigscience_t0")
    >>> model.save_pretrained("./il/tuo/path/bigscience_t0")
    ```

    3. Ora quando sei offline, carica i tuoi file con [`PreTrainedModel.from_pretrained`] dalla directory specificata:

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./il/tuo/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./il/tuo/path/bigscience_t0")
    ```

* Scarica in maniera programmatica i file con la libreria [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub):

    1. Installa la libreria `huggingface_hub` nel tuo ambiente virtuale:

    ```bash
    python -m pip install huggingface_hub
    ```

    2. Utilizza la funzione [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) per scaricare un file in un path specifico. Per esempio, il seguente comando scarica il file `config.json` dal modello [T0](https://huggingface.co/bigscience/T0_3B) nel path che desideri:

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./il/tuo/path/bigscience_t0")
    ```

Una volta che il tuo file è scaricato e salvato in cache localmente, specifica il suo path locale per caricarlo e utilizzarlo:

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./il/tuo/path/bigscience_t0/config.json")
```

<Tip>

Fai riferimento alla sezione [How to download files from the Hub](https://huggingface.co/docs/hub/how-to-downstream) per avere maggiori dettagli su come scaricare modelli presenti sull Hub.

</Tip>
