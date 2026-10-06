<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Zu 🤗 Transformers beitragen

Jeder ist willkommen, einen Beitrag zu leisten, und wir schätzen den Beitrag jedes Einzelnen. Codebeiträge sind nicht der einzige Weg, der Community zu helfen. Fragen zu beantworten, anderen zu helfen und die Dokumentation zu verbessern, sind ebenfalls äußerst wertvoll.

Es hilft uns auch, wenn Sie das Projekt weiterempfehlen! Erwähnen Sie die Bibliothek in Blogposts über die großartigen Projekte, die sie ermöglicht hat, tweeten Sie, wenn sie Ihnen geholfen hat, oder hinterlassen Sie dem Repository ein ⭐️, um Danke zu sagen.

Wie auch immer Sie sich entscheiden beizutragen, seien Sie achtsam und respektieren Sie unseren [Verhaltenskodex](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md).

**Dieser Leitfaden wurde stark durch den fantastischen [scikit-learn-Leitfaden für Beiträge](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md) inspiriert.**

## Beitragsmöglichkeiten

Es gibt mehrere Wege, wie Sie zu 🤗 Transformers beitragen können:

* Beheben Sie bestehende Probleme im vorhandenen Code.
* Erstellen Sie Issues im Zusammenhang mit Fehlern oder gewünschten neuen Funktionen.
* Implementieren Sie neue Modelle.
* Tragen Sie zu den Beispielen oder zur Dokumentation bei.

Wenn Sie nicht wissen, wo Sie anfangen sollen, gibt es eine spezielle Liste von [Good First Issues](https://github.com/huggingface/transformers/contribute). Sie bietet Ihnen eine Liste offener und anfängerfreundlicher Probleme und hilft Ihnen, einen ersten Beitrag zu Open-Source zu leisten. Idealerweise erstellen Sie eine Pull-Anfrage und verlinken sie mit dem Issue, an dem Sie arbeiten möchten. Wir versuchen, erstellte PRs bevorzugt zu behandeln, da wir so den Fortschritt leicht verfolgen können, und die Option besteht, dass jemand anderes den PR übernehmen kann, falls der Beitragende keine Zeit mehr hat.

Für etwas mehr Herausforderung, können Sie auch einen Blick auf die Liste der [Good Second Issues](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue) werfen. Generell gilt: Legen Sie los, wenn Sie sich den Anforderungen gewachsen sehen und wir helfen Ihnen dabei! 🚀

> Alle Beiträge sind für die Community gleichermaßen wertvoll. 🥰

## Bestehende Probleme beheben

Wenn Ihnen ein Problem im vorhandenen Code auffällt und Sie eine Lösung im Sinn haben, können Sie gerne einen Beitrag leisten und [eine Pull-Anfrage erstellen](#eine-pull-anfrage-erstellen)!

## Ein fehlerspezifisches Issue oder eine Feature-Anfrage erstellen

Tun Sie Ihr Bestes, diesen Richtlinien zu folgen, wenn Sie ein fehlerspezifisches Issue erstellen oder eine Feature-Anfrage einreichen. Das macht es uns leichter, Ihnen schnell und mit gutem Feedback zu antworten.

### Haben Sie einen Fehler gefunden?

Die 🤗 Transformers-Bibliothek verdankt ihre Robustheit und Zuverlässigkeit aller Nutzer, die frisch entdeckte Probleme melden.

Wir würden es wirklich schätzen, wenn Sie **sicherstellen könnten, dass der Fehler noch nicht gemeldet wurde** (verwenden Sie die Suchleiste auf GitHub unter Issues), bevor Sie ein Issue erstellen. Ihr Problem sollte sich auch auf Fehler in der Bibliothek selbst und nicht auf Ihren eigenen Code beziehen. Wenn Sie sich nicht sicher sind, ob der Fehler in Ihrem eigenen Code oder der Bibliothek liegt, fragen Sie bitte zuerst im [Forum](https://discuss.huggingface.co/) nach. Das hilft uns, schneller auf Probleme im Zusammenhang mit der Bibliothek zu reagieren, anstatt auf allgemeine Fragen.

Wenn Sie sich vergewissert haben, dass der Fehler noch nicht gemeldet wurde, geben Sie bitte die folgenden Informationen in Ihrem Issue an, damit wir es schnell beheben können:

* Ihr **Betriebssystem und Version** sowie die Versionen von **Python**, **PyTorch** und **TensorFlow**, falls zutreffend.
* Ein kurzes und unabhängiges Code-Snippet, das es uns ermöglicht, den Fehler in weniger als 30 Sekunden nachzustellen.
* Den *vollständigen* Traceback, wenn eine Ausnahme geworfen wird.
* Fügen Sie weitere hilfreiche Informationen, wie z. B. Screenshots, an.

Um das Betriebssystem und die Softwareversionen automatisch auszugeben, führen Sie den folgenden Befehl aus:

```bash
transformers env
```

Sie können denselben Befehl auch im Hauptverzeichnis des Repositorys ausführen:

```bash
python src/transformers/commands/transformers_cli.py env
```

### Möchten Sie eine neue Funktion?

Wenn Sie eine bestimmte neue Funktion in 🤗 Transformers sehen möchten, erstellen Sie bitte ein Issue und fügen Sie eine Beschreibung hinzu:

1. Was ist die *Motivation* hinter dieser Funktion? Steht sie in Zusammenhang mit einem Problem oder einer Frustration mit der Bibliothek? Ist es eine Funktion, die Sie für ein Projekt benötigen? Ist es etwas, an dem Sie gearbeitet haben und denken, dass es der Community nutzen könnte?

   Was auch immer es ist, wir würden uns freuen, davon zu hören!

1. Beschreiben Sie Ihre gewünschte Funktion so detailliert wie möglich. Je mehr Sie uns darüber erzählen können, desto besser können wir Ihnen helfen.
1. Stellen Sie einen *Code-Schnipsel* bereit, der die Funktionsweise demonstriert.
1. Falls die Funktion auf einem Paper beruht, verlinken Sie dieses bitte.

Wenn Ihr Issue gut geschrieben ist, sind wir zum Zeitpunkt seiner Erstellung bereits zu 80 % fertig.

Wir haben [Vorlagen](https://github.com/huggingface/transformers/tree/main/templates) hinzugefügt, um Ihnen den Start Ihres Issues zu erleichtern.

## Möchten Sie ein neues Modell implementieren?

Es werden ständig neue Modelle veröffentlicht. Wenn Sie ein neues Modell implementieren möchten, geben Sie bitte folgende Informationen an:

* Eine kurze Beschreibung des Modells und einen Link zum Paper.
* Link zur Implementierung, falls sie Open-Source ist.
* Link zu den Modellgewichten, falls verfügbar.

Lassen Sie es uns wissen, wenn Sie bereit sind, das Modell selbst beizutragen. Dann können wir Ihnen helfen, es zu 🤗 Transformers hinzuzufügen!

Wir haben auch einen technischen Leitfaden dazu, [wie man ein Modell zu 🤗 Transformers hinzufügt](https://huggingface.co/docs/transformers/add_new_model).

## Möchten Sie die Dokumentation erweitern?

Wir sind immer auf der Suche nach Verbesserungen, die die Dokumentation klarer und präziser machen. Bitte teilen Sie uns Verbesserungsvorschläge mit, wie z. B. Tippfehler und fehlende, unklare oder ungenaue Inhalte. Wir übernehmen gerne die Änderungen oder helfen Ihnen, einen Beitrag zu leisten, wenn Sie daran interessiert sind!

Für weitere Einzelheiten darüber, wie man die Dokumentation generiert, erstellt und schreibt, werfen Sie einen Blick auf das [README](https://github.com/huggingface/transformers/tree/main/docs) der Dokumentation.

## Eine Pull-Anfrage erstellen

Bevor Sie irgendwelchen Code schreiben, empfehlen wir Ihnen dringend, die bestehenden PRs oder Issues zu durchsuchen, um sicherzustellen, dass niemand bereits an diesem Thema arbeitet. Wenn Sie sich unsicher sind, ist es immer eine gute Idee, nach Feedback in einem neuen Issue zu fragen.

Sie benötigen grundlegende `git`-Kenntnisse, um zu 🤗 Transformers beizutragen. Obwohl `git` nicht das einfachste Werkzeug ist, hat es ein sehr gutes Handbuch. Geben Sie `git --help` in eine Shell ein und genießen Sie es! Wenn Sie Bücher bevorzugen, ist [Pro Git](https://git-scm.com/book/en/v2) eine gute Anlaufstelle.

Sie benötigen **[Python 3.9](https://github.com/huggingface/transformers/blob/main/setup.py#L426)** oder höher, um zu 🤗 Transformers beizutragen. Folgen Sie den nachstehenden Schritten, um mit dem Beitrag zu beginnen:

1. Forken Sie das [Repository](https://github.com/huggingface/transformers), indem Sie auf den **[Fork](https://github.com/huggingface/transformers/fork)**-Button auf der Seite des Repositorys klicken. Dadurch wird eine Kopie des Codes auf Ihrem GitHub-Account erstellt.

1. Klonen Sie Ihren Fork auf Ihre lokale Festplatte und fügen Sie das ursprüngliche Repository als Remote hinzu:

   ```bash
   git clone git@github.com:<your Github handle>/transformers.git
   cd transformers
   git remote add upstream https://github.com/huggingface/transformers.git
   ```

1. Erstellen Sie einen neuen Branch, um Ihre Änderungen zu speichern:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 Arbeiten Sie **nicht** auf dem `main` Branch!

1. Richten Sie eine Entwicklungsumgebung ein, indem Sie den folgenden Befehl in einer virtuellen Umgebung ausführen:

   ```bash
   pip install -e ".[dev]"
   ```

   Wenn 🤗 Transformers bereits in der virtuellen Umgebung installiert war, entfernen Sie es mit `pip uninstall transformers`, bevor Sie es im bearbeitbaren Modus mit dem `-e` Flag neu installieren.

   Abhängig von Ihrem Betriebssystem und durch die wachsende Anzahl der optionalen Abhängigkeiten von Transformers könnten Sie mit diesem Befehl einen Fehler verursachen. Wenn das der Fall ist, stellen Sie sicher, dass Sie ihr bevorzugtes Deep-Learning-Framework (PyTorch, TensorFlow und/oder Flax) installieren und anschließend den folgenden Befehl ausführen:

   ```bash
   pip install -e ".[quality]"
   ```

   Dies sollte für die meisten Anwendungsfälle ausreichend sein.

1. Entwickeln Sie die Funktionen in Ihrem Branch.

   Während Sie an Ihrem Code arbeiten, sollten Sie sicherstellen, dass die Test-Suite erfolgreich durchläuft. Führen Sie die von Ihren Änderungen betroffenen Tests wie folgt aus:

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   Weitere Informationen über Tests finden Sie in der Anleitung zum Thema [Testen](https://huggingface.co/docs/transformers/testing).

   🤗 Transformers stützt sich auf `black` und `ruff`, um seinen Quellcode konsistent zu formatieren. Nachdem Sie Änderungen vorgenommen haben, wenden Sie automatische Stilkorrekturen und Codeprüfungen, die nicht automatisiert werden können, in einem Schritt an:

   ```bash
   make style
   ```

   Dieser Task ist optimiert, nur mit Dateien zu arbeiten, die von Ihrer PR modifiziert wurden.

   Wenn Sie die Prüfungen nacheinander ausführen möchten, wendet der folgende Befehl die Stilkorrekturen an:

   ```bash
   make style
   ```

   🤗 Transformers verwendet auch `ruff` und einige benutzerdefinierte Skripte, um auf Programmierfehler zu prüfen. Qualitätskontrollen werden von der CI durchgeführt, aber Sie können die gleichen Überprüfungen auch selbst ausführen:

   ```bash
   make check-repo
   ```

   Um mehr über diese Prüfungen zu erfahren und wie man mit ihnen Probleme behebt, lesen Sie den Leitfaden zu [Überprüfungen bei einer Pull-Anfrage](https://huggingface.co/docs/transformers/pr_checks).

   Wenn Sie Dokumente im Verzeichnis `docs/source` ändern, stellen Sie sicher, dass die Dokumentation noch generiert werden kann. Diese Prüfung wird auch im CI laufen, wenn Sie eine Pull-Anfrage erstellen. Um eine lokale Prüfung durchzuführen, müssen Sie den Dukumentation-Builder installieren:

   ```bash
   pip install ".[docs]"
   ```

   Führen Sie den folgenden Befehl im Hauptverzeichnis des Repositorys aus:

   ```bash
   doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
   ```

   Dadurch wird die Dokumentation im Ordner `~/tmp/test-build` erstellt, wo Sie die erzeugten Markdown-Dateien mit Ihrem bevorzugten Editor überprüfen können. Sie können auch eine Vorschau der Dokumentation auf GitHub sehen, wenn Sie eine Pull-Anfrage öffnen.

   Wenn Sie mit Ihren Änderungen zufrieden sind, fügen Sie die geänderten Dateien mit `git add` hinzu und speichern Sie Ihre Änderungen lokal mit `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Bitte achten Sie darauf, [gute Commit-Nachrichten](https://chris.beams.io/posts/git-commit/) zu schreiben, um die von Ihnen vorgenommenen Änderungen klar zu kommunizieren!

   Um Ihre Kopie des Codes auf dem aktuellen Stand des ursprünglichen Repositorys zu halten, rebasen Sie Ihren Branch auf `upstream/branch` *bevor* Sie eine Pull-Anfrage öffnen oder falls Sie von einem Maintainer dazu aufgefordert werden:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Pushen Sie Ihre Änderungen in Ihrem Branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   Wenn Sie bereits eine Pull-Anfrage erstellt haben, müssen Sie den Push mit dem `--force` Flag erzwingen. Andernfalls, wenn die Pull-Anfrage noch nicht erstellt wurde, können Sie Ihre Änderungen normal pushen.

1. Jetzt können Sie zu Ihrem Fork des Repositorys auf GitHub gehen und auf **Pull-Anfrage** klicken, um eine Pull-Anfrage zu erstellen. Stellen Sie sicher, dass Sie alle Punkte auf unserer [Checkliste](#checkliste-für-pull-anfragen) unten abhaken. Wenn Sie fertig sind, können Sie Ihre Änderungen zur Überprüfung an die Projektverantwortlichen senden.

1. Es ist kein Problem, wenn die Maintainer Änderungen beantragen, das geschieht auch bei unseren Kernmitarbeitern! Damit jeder die Änderungen in der Pull-Anfrage sehen kann, arbeiten Sie in Ihrem lokalen Branch und pushen die Änderungen zu Ihrem Fork. Sie werden automatisch in der Pull-Anfrage erscheinen.

### Checkliste für Pull-Anfragen

☐ Der Titel der Pull-Anfrage sollte Ihren Beitrag zusammenfassen.<br>
☐ Wenn Ihre Pull-Anfrage ein bestimmtes Issue bearbeitet, erwähnen Sie bitte die zugehörige Nummer in der Beschreibung der Pull-Anfrage, sodass diese verlinkt sind (und Personen, die das Issue lesen, wissen, dass Sie daran arbeiten).<br>
☐ Um eine fortlaufende Bearbeitung anzuzeigen, versehen Sie bitte den Titel mit einem `[WIP]` Präfix. Diese sind nützlich, um doppelte Arbeit zu verhindern und sie von PRs abzuheben, die bereit zum Zusammenführen sind.<br>
☐ Stellen Sie sicher, dass existierende Tests bestanden werden.<br>
☐ Wenn Sie eine neue Funktion hinzufügen, erstellen Sie auch Tests dafür.<br>

* Wenn Sie ein neues Modell hinzufügen, stellen Sie sicher, dass Sie `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)` verwenden, um die gemeinsamen Tests auszulösen.
* Wenn Sie neue `@slow` Tests hinzufügen, stellen Sie mit `RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py` sicher, dass diese erfolgreich durchlaufen.
* Wenn Sie einen neuen Tokenizer hinzufügen, schreiben Sie Tests und stellen Sie mit `RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py` sicher, dass diese erfolgreich durchlaufen.
* CircleCI führt die langsamen Tests nicht aus, aber GitHub Actions tut dies jede Nacht!<br>

☐ Alle public Methoden müssen informative Docstrings haben (siehe [`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) als Beispiel).<br>
☐ Aufgrund des schnell wachsenden Repositorys fügen Sie bitte keine Bilder, Videos oder andere Nicht-Textdateien hinzu, die das Repository erheblich belasten würden. Verwenden Sie stattdessen ein Hub-Repository wie [`hf-internal-testing`](https://huggingface.co/hf-internal-testing), um diese Dateien zu hosten und sie per URL zu verlinken. Wir empfehlen Bilder, die zur Dokumentation gehören, im folgenden Repository abzulegen: [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images). Sie können eine PR in diesem Datasets-Repository erstellen und ein Hugging-Face-Mitglied bitten, sie zu mergen.

Um mehr über die Prüfungen zu erfahren, die bei einer Pull-Anfrage ausgelöst werden, lesen Sie unseren Leitfaden zu [Überprüfungen bei einer Pull-Anfrage](https://huggingface.co/docs/transformers/pr_checks).

### Tests

Eine umfangreiche Test-Suite ist enthalten, um das Verhalten der Bibliothek zu testen. Tests für die Bibliothek finden Sie im [tests](https://github.com/huggingface/transformers/tree/main/tests) Ordner.

Wir bevorzugen `pytest` und `pytest-xdist`, weil es schneller ist. Geben Sie einen *Pfad zu einem Unterordner oder einer Testdatei* vom Hauptverzeichnis des Repositorys aus an, um den Test auszuführen:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

Tatsächlich ist dies genau, wie unser `make test` Befehl implementiert ist!

Sie können auch eine kleinere Anzahl an Tests angeben, um nur die Funktion, an der Sie arbeiten, zu testen.

Standardmäßig werden langsame Tests übersprungen, aber Sie können die Umgebungsvariable `RUN_SLOW` auf `yes` setzen, um sie auszuführen. Dies wird den Download vieler Gigabyte an Modellen starten - stellen Sie also sicher, dass Sie sowohl genügend Festplattenspeicher als auch eine gute Internetverbindung oder die nötige Geduld haben!

<Tip warning={true}>

Vergessen Sie nicht, einen *Pfad zu einem Unterordner oder einer Testdatei* anzugeben, um den Test auszuführen. Sonst führen Sie alle Tests im `tests` Ordner aus, was sehr lange dauern wird!

</Tip>

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

Wie bei den langsamen Tests gibt es auch andere Umgebungsvariablen, die standardmäßig beim Testen nicht gesetzt sind:

* `RUN_CUSTOM_TOKENIZERS`: Aktiviert Tests für benutzerdefinierte Tokenizer.

Weitere Umgebungsvariablen und zusätzliche Informationen finden Sie in der [testing_utils.py](src/transformers/testing_utils.py).

🤗 Transformers verwendet `pytest` nur als Test-Runner. Es verwendet keine `pytest`-spezifischen Funktionen in der Test-Suite selbst.

Das bedeutet, `unittest` wird vollständig unterstützt. Folgend wird beschrieben, wie man Tests mit `unittest` ausführt:

```bash
python -m unittest discover -s tests -t . -v
```

### Stil-Leitfaden

Für Docstrings befolgt 🤗 Transformers den [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Lesen Sie unseren [Leitfaden zum Schreiben von Dokumentationen](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification) für weitere Informationen.

### Entwickeln unter Windows

Unter Windows (falls Sie nicht im [Windows-Subsystem für Linux](https://learn.microsoft.com/en-us/windows/wsl/) oder WSL arbeiten) müssen Sie git so konfigurieren, dass Windows `CRLF` in Linux `LF` Zeilenenden umgewandelt werden:

```bash
git config core.autocrlf input
```

Eine Möglichkeit, den `make`-Befehl unter Windows auszuführen, ist mit MSYS2:

1. Laden Sie [MSYS2](https://www.msys2.org/) herunter und installieren Sie es nach `C:\msys64`.
1. Öffnen Sie die Kommandozeile `C:\msys64\msys2.exe` (sie sollte vom **Start**-Menü aus verfügbar sein).
1. Führen Sie den Befehl in der Shell aus: `pacman -Syu` und installieren Sie `make` mit `pacman -S make`.
1. Fügen Sie `C:\msys64\usr\bin` an Ihrer PATH-Umgebungsvariable an.

Sie können nun `make` aus jedem Terminal heraus verwenden (PowerShell, cmd.exe usw.)! 🎉

### Ein geforktes Repository mit dem Haupt-Repository von Hugging Face synchronisieren

Beim Aktualisieren des main-Branches eines geforkten Repositories beachten Sie bitte die folgenden Schritte, um das Anpingen des Haupt-Repositorys zu vermeiden, was unnötige Verweise in abhängigen PRs vermerkt und beteiligte Entwickler benachrichtigt:

1. Wenn möglich, vermeiden Sie die Synchronisation mit dem Haupt-Repository über einen Branch und PR im geforkten Repository. Mergen Sie stattdessen direkt in den main-Branch des Forks.
1. Wenn ein PR unbedingt notwendig ist, verwenden Sie die folgenden Schritte, nachdem Sie Ihren Branch ausgecheckt haben:

   ```bash
   git checkout -b your-branch-for-syncing
   git pull --squash --no-commit upstream main
   git commit -m '<your message without GitHub references>'
   git push --set-upstream origin your-branch-for-syncing
   ```
