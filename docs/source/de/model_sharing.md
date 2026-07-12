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

# Ein Modell teilen

Die letzten beiden Tutorials haben gezeigt, wie man ein Modell mit PyTorch, Keras und 🤗 Accelerate für verteilte Setups feinabstimmen kann. Der nächste Schritt besteht darin, Ihr Modell mit der Community zu teilen! Bei Hugging Face glauben wir an den offenen Austausch von Wissen und Ressourcen, um künstliche Intelligenz für alle zu demokratisieren. Wir ermutigen Sie, Ihr Modell mit der Community zu teilen, um anderen zu helfen, Zeit und Ressourcen zu sparen.

In diesem Tutorial lernen Sie zwei Methoden kennen, wie Sie ein trainiertes oder verfeinertes Modell auf dem [Model Hub](https://huggingface.co/models) teilen können:

- Programmgesteuertes Übertragen Ihrer Dateien auf den Hub.
- Ziehen Sie Ihre Dateien per Drag-and-Drop über die Weboberfläche in den Hub.

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

Um ein Modell mit der Öffentlichkeit zu teilen, benötigen Sie ein Konto auf [huggingface.co](https://huggingface.co/join). Sie können auch einer bestehenden Organisation beitreten oder eine neue Organisation gründen.

</Tip>

## Repository-Funktionen

Jedes Repository im Model Hub verhält sich wie ein typisches GitHub-Repository. Unsere Repositorys bieten Versionierung, Commit-Historie und die Möglichkeit, Unterschiede zu visualisieren.

Die integrierte Versionierung des Model Hub basiert auf Git und [git-lfs](https://git-lfs.github.com/). Mit anderen Worten: Sie können ein Modell als ein Repository behandeln, was eine bessere Zugriffskontrolle und Skalierbarkeit ermöglicht. Die Versionskontrolle ermöglicht *Revisionen*, eine Methode zum Anheften einer bestimmten Version eines Modells mit einem Commit-Hash, Tag oder Branch.

Folglich können Sie eine bestimmte Modellversion mit dem Parameter "Revision" laden:

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="4c77982"  # tag name, or branch name, or commit hash
... )
```

Dateien lassen sich auch in einem Repository leicht bearbeiten, und Sie können die Commit-Historie sowie die Unterschiede einsehen:

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## Einrichtung

Bevor Sie ein Modell für den Hub freigeben, benötigen Sie Ihre Hugging Face-Anmeldedaten. Wenn Sie Zugang zu einem Terminal haben, führen Sie den folgenden Befehl in der virtuellen Umgebung aus, in der 🤗 Transformers installiert ist. Dadurch werden Ihre Zugangsdaten in Ihrem Hugging Face-Cache-Ordner (standardmäßig `~/.cache/`) gespeichert:

```bash
hf auth login
```

Wenn Sie ein Notebook wie Jupyter oder Colaboratory verwenden, stellen Sie sicher, dass Sie die [`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library) Bibliothek installiert haben. Diese Bibliothek ermöglicht Ihnen die programmatische Interaktion mit dem Hub.

```bash
pip install huggingface_hub
```

Verwenden Sie dann `notebook_login`, um sich beim Hub anzumelden, und folgen Sie dem Link [hier](https://huggingface.co/settings/token), um ein Token für die Anmeldung zu generieren:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Ein Modell für alle Frameworks konvertieren

Um sicherzustellen, dass Ihr Modell von jemandem verwendet werden kann, der mit einem anderen Framework arbeitet, empfehlen wir Ihnen, Ihr Modell sowohl mit PyTorch- als auch mit TensorFlow-Checkpoints zu konvertieren und hochzuladen. Während Benutzer immer noch in der Lage sind, Ihr Modell von einem anderen Framework zu laden, wenn Sie diesen Schritt überspringen, wird es langsamer sein, weil 🤗 Transformers den Checkpoint on-the-fly konvertieren müssen.

Die Konvertierung eines Checkpoints für ein anderes Framework ist einfach. Stellen Sie sicher, dass Sie PyTorch und TensorFlow installiert haben (siehe [hier](installation) für Installationsanweisungen), und finden Sie dann das spezifische Modell für Ihre Aufgabe in dem anderen Framework.

Geben Sie `from_tf=True` an, um einen Prüfpunkt von TensorFlow nach PyTorch zu konvertieren:

```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```

## Ein Modell während des Trainings hochladen

<Youtube id="Z1-XMy-GNLQ"/>

Die Weitergabe eines Modells an den Hub ist so einfach wie das Hinzufügen eines zusätzlichen Parameters oder Rückrufs. Erinnern Sie sich an das [Feinabstimmungs-Tutorial](training), in der Klasse [`TrainingArguments`] geben Sie Hyperparameter und zusätzliche Trainingsoptionen an. Eine dieser Trainingsoptionen beinhaltet die Möglichkeit, ein Modell direkt an den Hub zu pushen. Setzen Sie `push_to_hub=True` in Ihrer [`TrainingArguments`]:

```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

Übergeben Sie Ihre Trainingsargumente wie gewohnt an [`Trainer`]:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Nach der Feinabstimmung Ihres Modells rufen Sie [`~transformers.Trainer.push_to_hub`] auf [`Trainer`] auf, um das trainierte Modell an den Hub zu übertragen. Transformers fügt sogar automatisch Trainings-Hyperparameter, Trainingsergebnisse und Framework-Versionen zu Ihrer Modellkarte hinzu!

```py
>>> trainer.push_to_hub()
```

## Verwenden Sie die Funktion `push_to_hub`.

Sie können `push_to_hub` auch direkt für Ihr Modell aufrufen, um es in den Hub hochzuladen.

Geben Sie den Namen Ihres Modells in "push_to_hub" an:

```py
>>> pt_model.push_to_hub("my-awesome-model")
```

Dadurch wird ein Repository unter Ihrem Benutzernamen mit dem Modellnamen `my-awesome-model` erstellt. Benutzer können nun Ihr Modell mit der Funktion `from_pretrained` laden:

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

Wenn Sie zu einer Organisation gehören und Ihr Modell stattdessen unter dem Namen der Organisation pushen wollen, fügen Sie diesen einfach zur `repo_id` hinzu:

```py
>>> pt_model.push_to_hub("my-awesome-org/my-awesome-model")
```

Die Funktion "push_to_hub" kann auch verwendet werden, um andere Dateien zu einem Modell-Repository hinzuzufügen. Zum Beispiel kann man einen Tokenizer zu einem Modell-Repository hinzufügen:

```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

Oder vielleicht möchten Sie die TensorFlow-Version Ihres fein abgestimmten PyTorch-Modells hinzufügen:

```py
>>> tf_model.push_to_hub("my-awesome-model")
```

Wenn Sie nun zu Ihrem Hugging Face-Profil navigieren, sollten Sie Ihr neu erstelltes Modell-Repository sehen. Wenn Sie auf die Registerkarte **Dateien** klicken, werden alle Dateien angezeigt, die Sie in das Repository hochgeladen haben.

Weitere Einzelheiten zum Erstellen und Hochladen von Dateien in ein Repository finden Sie in der Hub-Dokumentation [hier](https://huggingface.co/docs/hub/how-to-upstream).

## Hochladen mit der Weboberfläche

Benutzer, die einen no-code Ansatz bevorzugen, können ein Modell über das Webinterface des Hubs hochladen. Besuchen Sie [huggingface.co/new](https://huggingface.co/new) um ein neues Repository zu erstellen:

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

Fügen Sie von hier aus einige Informationen über Ihr Modell hinzu:

- Wählen Sie den **Besitzer** des Repositorys. Dies können Sie selbst oder eine der Organisationen sein, denen Sie angehören.
- Wählen Sie einen Namen für Ihr Modell, der auch der Name des Repositorys sein wird.
- Wählen Sie, ob Ihr Modell öffentlich oder privat ist.
- Geben Sie die Lizenzverwendung für Ihr Modell an.

Klicken Sie nun auf die Registerkarte **Dateien** und klicken Sie auf die Schaltfläche **Datei hinzufügen**, um eine neue Datei in Ihr Repository hochzuladen. Ziehen Sie dann eine Datei per Drag-and-Drop hoch und fügen Sie eine Übergabemeldung hinzu.

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## Hinzufügen einer Modellkarte

Um sicherzustellen, dass die Benutzer die Fähigkeiten, Grenzen, möglichen Verzerrungen und ethischen Aspekte Ihres Modells verstehen, fügen Sie bitte eine Modellkarte zu Ihrem Repository hinzu. Die Modellkarte wird in der Datei `README.md` definiert. Sie können eine Modellkarte hinzufügen, indem Sie:

* Manuelles Erstellen und Hochladen einer "README.md"-Datei.
* Klicken Sie auf die Schaltfläche **Modellkarte bearbeiten** in Ihrem Modell-Repository.

Werfen Sie einen Blick auf die DistilBert [model card](https://huggingface.co/distilbert/distilbert-base-uncased) als gutes Beispiel für die Art von Informationen, die eine Modellkarte enthalten sollte. Weitere Details über andere Optionen, die Sie in der Datei "README.md" einstellen können, wie z.B. den Kohlenstoff-Fußabdruck eines Modells oder Beispiele für Widgets, finden Sie in der Dokumentation [hier](https://huggingface.co/docs/hub/models-cards).
