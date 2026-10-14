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

# Installation

Installieren Sie 🤗 Transformers für die Deep-Learning-Bibliothek, mit der Sie arbeiten, richten Sie Ihren Cache ein und konfigurieren Sie 🤗 Transformers optional für den Offline-Betrieb.

🤗 Transformers wurde unter Python 3.10+ und PyTorch 2.4+ getestet. Folgen Sie den Installationsanweisungen unten für die von Ihnen verwendete Deep-Learning-Bibliothek:

* [PyTorch](https://pytorch.org/get-started/locally/) installation instructions.

## Installation mit pip

Sie sollten 🤗 Transformers in einer [virtuellen Umgebung](https://docs.python.org/3/library/venv.html) installieren. Wenn Sie mit virtuellen Python-Umgebungen nicht vertraut sind, werfen Sie einen Blick auf diese [Anleitung](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Eine virtuelle Umgebung macht es einfacher, verschiedene Projekte zu verwalten und Kompatibilitätsprobleme zwischen Abhängigkeiten zu vermeiden.

Beginnen wir mit der Erstellung einer virtuellen Umgebung in Ihrem Projektverzeichnis:


```bash
python -m venv .env
```

Aktivieren wir die virtuelle Umgebung. Unter Linux und MacOs:

```bash
source .env/bin/activate
```
Aktivieren wir die virtuelle Umgebung unter Windows

```bash
.env/Scripts/activate
```

Jetzt können wir die 🤗 Transformers mit dem folgenden Befehl installieren:

```bash
pip install transformers
```

Bei reiner CPU-Unterstützung können wir 🤗 Transformers und eine Deep-Learning-Bibliothek bequem in einer Zeile installieren. Installieren wir zum Beispiel 🤗 Transformers und PyTorch mit:

```bash
pip install transformers[torch]
```

Überprüfen wir abschließend, ob 🤗 Transformers ordnungsgemäß installiert wurde, indem wir den folgenden Befehl ausführen. Es wird ein vortrainiertes Modell heruntergeladen:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

Dann wird die Kategorie und die Wahrscheinlichkeit ausgegeben:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## Installation aus dem Code

Installieren wir 🤗 Transformers aus dem Quellcode mit dem folgenden Befehl:

```bash
pip install git+https://github.com/huggingface/transformers
```

Dieser Befehl installiert die aktuelle `main` Version und nicht die neueste `stable` Version. Die `main`-Version ist nützlich, um mit den neuesten Entwicklungen Schritt zu halten. Zum Beispiel, wenn ein Fehler seit der letzten offiziellen Version behoben wurde, aber eine neue Version noch nicht veröffentlicht wurde. Das bedeutet jedoch, dass die "Hauptversion" nicht immer stabil ist. Wir bemühen uns, die Hauptversion einsatzbereit zu halten, und die meisten Probleme werden normalerweise innerhalb weniger Stunden oder eines Tages behoben. Wenn Sie auf ein Problem stoßen, öffnen Sie bitte ein [Issue](https://github.com/huggingface/transformers/issues), damit wir es noch schneller beheben können!

Überprüfen wir, ob 🤗 Transformers richtig installiert wurde, indem Sie den folgenden Befehl ausführen:


```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## Editierbare Installation

Sie benötigen eine bearbeitbare Installation, wenn Sie:

* die "Haupt"-Version des Quellcodes verwenden möchten.
* Zu 🤗 Transformers beitragen und Änderungen am Code testen wollen.

Klonen Sie das Repository und installieren 🤗 Transformers mit den folgenden Befehlen:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Diese Befehle verknüpfen den Ordner, in den Sie das Repository geklont haben, mit den Pfaden Ihrer Python-Bibliotheken. Python wird nun in dem Ordner suchen, in den Sie geklont haben, zusätzlich zu den normalen Bibliothekspfaden. Wenn zum Beispiel Ihre Python-Pakete normalerweise in `~/anaconda3/envs/main/lib/python3.10/site-packages/` installiert sind, wird Python auch den Ordner durchsuchen, in den Sie geklont haben: `~/transformers/`.


<Tip warning={true}>

Sie müssen den Ordner `transformers` behalten, wenn Sie die Bibliothek weiter verwenden wollen.

</Tip>

Jetzt können Sie Ihren Klon mit dem folgenden Befehl ganz einfach auf die neueste Version von 🤗 Transformers aktualisieren:


```bash
cd ~/transformers/
git pull
```

Ihre Python-Umgebung wird beim nächsten Ausführen die `main`-Version von 🤗 Transformers finden.

## Installation mit conda

Installation von dem conda Kanal `conda-forge`:

```bash
conda install conda-forge::transformers
```

## Cache Einrichtung

Vorgefertigte Modelle werden heruntergeladen und lokal zwischengespeichert unter: `~/.cache/huggingface/hub`. Dies ist das Standardverzeichnis, das durch die Shell-Umgebungsvariable "HF_HUB_CACHE" vorgegeben ist. Unter Windows wird das Standardverzeichnis durch `C:\Benutzer\Benutzername\.cache\huggingface\hub` angegeben. Sie können die unten aufgeführten Shell-Umgebungsvariablen - in der Reihenfolge ihrer Priorität - ändern, um ein anderes Cache-Verzeichnis anzugeben:

1. Shell-Umgebungsvariable (Standard): `HF_HUB_CACHE`.
2. Shell-Umgebungsvariable: `HF_HOME`.
3. Shell-Umgebungsvariable: `XDG_CACHE_HOME` + `/huggingface`.


## Offline Modus

Transformers ist in der Lage, in einer Firewall- oder Offline-Umgebung zu laufen, indem es nur lokale Dateien verwendet. Setzen Sie die Umgebungsvariable `HF_HUB_OFFLINE=1`, um dieses Verhalten zu aktivieren.

<Tip>

Fügen sie [🤗 Datasets](https://huggingface.co/docs/datasets/) zu Ihrem Offline-Trainingsworkflow hinzufügen, indem Sie die Umgebungsvariable `HF_DATASETS_OFFLINE=1` setzen.

</Tip>

### Abrufen von Modellen und Tokenizern zur Offline-Verwendung

Eine andere Möglichkeit, 🤗 Transformers offline zu verwenden, besteht darin, die Dateien im Voraus herunterzuladen und dann auf ihren lokalen Pfad zu verweisen, wenn Sie sie offline verwenden müssen. Es gibt drei Möglichkeiten, dies zu tun:

* Laden Sie eine Datei über die Benutzeroberfläche des [Model Hub](https://huggingface.co/models) herunter, indem Sie auf das ↓-Symbol klicken.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

* Verwenden Sie den [PreTrainedModel.from_pretrained] und [PreTrainedModel.save_pretrained] Workflow:

    1. Laden Sie Ihre Dateien im Voraus mit [`PreTrainedModel.from_pretrained`] herunter:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. Speichern Sie Ihre Dateien in einem bestimmten Verzeichnis mit [`PreTrainedModel.save_pretrained`]:

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. Wenn Sie nun offline sind, laden Sie Ihre Dateien mit [`PreTrainedModel.from_pretrained`] aus dem bestimmten Verzeichnis:

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* Programmatisches Herunterladen von Dateien mit der [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub) Bibliothek:

    1. Installieren Sie die "huggingface_hub"-Bibliothek in Ihrer virtuellen Umgebung:

    ```bash
    python -m pip install huggingface_hub
    ```

    2. Verwenden Sie die Funktion [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub), um eine Datei in einen bestimmten Pfad herunterzuladen. Der folgende Befehl lädt zum Beispiel die Datei "config.json" aus dem Modell [T0](https://huggingface.co/bigscience/T0_3B) in den gewünschten Pfad herunter:

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

Sobald Ihre Datei heruntergeladen und lokal zwischengespeichert ist, geben Sie den lokalen Pfad an, um sie zu laden und zu verwenden:

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

Weitere Informationen zum Herunterladen von Dateien, die auf dem Hub gespeichert sind, finden Sie im Abschnitt [Wie man Dateien vom Hub herunterlädt](https://huggingface.co/docs/hub/how-to-downstream).

</Tip>
