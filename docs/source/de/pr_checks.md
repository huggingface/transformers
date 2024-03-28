<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

# Überprüfungen bei einer Pull-Anfrage

Wenn Sie eine Pull-Anfrage für 🤗 Transformers öffnen, wird eine ganze Reihe von Prüfungen durchgeführt, um sicherzustellen, dass der Patch, den Sie hinzufügen, nichts Bestehendes zerstört. Es gibt vier Arten von Prüfungen:
- reguläre Tests
- Erstellung der Dokumentation
- Stil von Code und Dokumentation
- allgemeine Konsistenz des Repository

In diesem Dokument werden wir versuchen zu erklären, worum es sich bei diesen verschiedenen Prüfungen handelt und wie Sie sie lokal debuggen können, wenn eine der Prüfungen in Ihrer PR fehlschlägt.

Beachten Sie, dass Sie im Idealfall eine Dev-Installation benötigen:

```bash
pip install transformers[dev]
```

oder für eine bearbeitbare Installation:

```bash
pip install -e .[dev]
```

innerhalb des Transformers Repo. Da die Anzahl der optionalen Abhängigkeiten von Transformers stark zugenommen hat, ist es möglich, dass Sie nicht alle davon bekommen können. Wenn die Dev-Installation fehlschlägt, stellen Sie sicher, dass Sie das Deep Learning-Framework, mit dem Sie arbeiten, installieren (PyTorch, TensorFlow und/oder Flax).

```bash
pip install transformers[quality]
```

oder für eine bearbeitbare Installation:

```bash
pip install -e .[quality]
```


## Tests

Alle Jobs, die mit `ci/circleci: run_tests_` beginnen, führen Teile der Transformers-Testsuite aus. Jeder dieser Jobs konzentriert sich auf einen Teil der Bibliothek in einer bestimmten Umgebung: `ci/circleci: run_tests_pipelines_tf` zum Beispiel führt den Pipelines-Test in einer Umgebung aus, in der nur TensorFlow installiert ist.

Beachten Sie, dass nur ein Teil der Testsuite jedes Mal ausgeführt wird, um zu vermeiden, dass Tests ausgeführt werden, wenn es keine wirkliche Änderung in den Modulen gibt, die sie testen: ein Dienstprogramm wird ausgeführt, um die Unterschiede in der Bibliothek zwischen vor und nach dem PR zu ermitteln (was GitHub Ihnen auf der Registerkarte "Files changes" anzeigt) und die Tests auszuwählen, die von diesem Unterschied betroffen sind. Dieses Dienstprogramm kann lokal mit ausgeführt werden:

```bash
python utils/tests_fetcher.py
```

aus dem Stammverzeichnis des Transformers-Repositoriums. Es wird:

1. Überprüfen Sie für jede Datei im Diff, ob die Änderungen im Code oder nur in Kommentaren oder Docstrings enthalten sind. Nur die Dateien mit echten Codeänderungen werden beibehalten.
2. Erstellen Sie eine interne Map, die für jede Datei des Quellcodes der Bibliothek alle Dateien angibt, auf die sie rekursiv Einfluss nimmt. Von Modul A wird gesagt, dass es sich auf Modul B auswirkt, wenn Modul B Modul A importiert. Für die rekursive Auswirkung benötigen wir eine Kette von Modulen, die von Modul A zu Modul B führt und in der jedes Modul das vorherige importiert.
3. Wenden Sie diese Zuordnung auf die in Schritt 1 gesammelten Dateien an. So erhalten wir die Liste der Modelldateien, die von der PR betroffen sind.
4. Ordnen Sie jede dieser Dateien der/den entsprechenden Testdatei(en) zu und erhalten Sie die Liste der auszuführenden Tests.

Wenn Sie das Skript lokal ausführen, sollten Sie die Ergebnisse von Schritt 1, 3 und 4 ausgegeben bekommen und somit wissen, welche Tests ausgeführt werden. Das Skript erstellt außerdem eine Datei namens `test_list.txt`, die die Liste der auszuführenden Tests enthält, die Sie mit dem folgenden Befehl lokal ausführen können:

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

Für den Fall, dass Ihnen etwas entgangen ist, wird die komplette Testreihe ebenfalls täglich ausgeführt.

## Dokumentation erstellen

Der Job `build_pr_documentation` erstellt und generiert eine Vorschau der Dokumentation, um sicherzustellen, dass alles in Ordnung ist, wenn Ihr PR zusammengeführt wird. Ein Bot fügt einen Link zur Vorschau der Dokumentation zu Ihrem PR hinzu. Alle Änderungen, die Sie an dem PR vornehmen, werden automatisch in der Vorschau aktualisiert. Wenn die Dokumentation nicht erstellt werden kann, klicken Sie auf **Details** neben dem fehlgeschlagenen Auftrag, um zu sehen, wo der Fehler liegt. Oft ist der Fehler so einfach wie eine fehlende Datei im `toctree`.

Wenn Sie daran interessiert sind, die Dokumentation lokal zu erstellen oder in der Vorschau anzusehen, werfen Sie einen Blick in die [`README.md`](https://github.com/huggingface/transformers/tree/main/docs) im Ordner docs.

## Code und Dokumentationsstil

Die Formatierung des Codes erfolgt für alle Quelldateien, die Beispiele und die Tests mit `black` und `ruff`. Wir haben auch ein benutzerdefiniertes Tool, das sich um die Formatierung von docstrings und `rst`-Dateien kümmert (`utils/style_doc.py`), sowie um die Reihenfolge der Lazy-Importe, die in den Transformers `__init__.py`-Dateien durchgeführt werden (`utils/custom_init_isort.py`). All dies können Sie starten, indem Sie Folgendes ausführen

```bash
make style
```

Das CI prüft, ob diese innerhalb der Prüfung `ci/circleci: check_code_quality` angewendet wurden. Es führt auch `ruff` aus, das einen grundlegenden Blick auf Ihren Code wirft und sich beschwert, wenn es eine undefinierte Variable findet oder eine, die nicht verwendet wird. Um diese Prüfung lokal auszuführen, verwenden Sie

```bash
make quality
```

Dies kann sehr viel Zeit in Anspruch nehmen. Um dasselbe nur für die Dateien zu tun, die Sie im aktuellen Zweig geändert haben, führen Sie

```bash
make fixup
```

Dieser letzte Befehl führt auch alle zusätzlichen Prüfungen für die Konsistenz des Repositorys durch. Schauen wir uns diese an.

## Repository-Konsistenz

Dies fasst alle Tests zusammen, die sicherstellen, dass Ihr PR das Repository in einem guten Zustand verlässt. Sie können diese Prüfung lokal durchführen, indem Sie Folgendes ausführen:

```bash
make repo-consistency
```

Dies überprüft, ob:

- Alle zum Init hinzugefügten Objekte sind dokumentiert (ausgeführt von `utils/check_repo.py`)
- Alle `__init__.py`-Dateien haben in ihren beiden Abschnitten den gleichen Inhalt (ausgeführt von `utils/check_inits.py`)
- Der gesamte Code, der als Kopie eines anderen Moduls identifiziert wurde, stimmt mit dem Original überein (ausgeführt von `utils/check_copies.py`)
- Alle Konfigurationsklassen haben mindestens einen gültigen Prüfpunkt, der in ihren Dokumentationen erwähnt wird (ausgeführt von `utils/check_config_docstrings.py`)
- Alle Konfigurationsklassen enthalten nur Attribute, die in den entsprechenden Modellierungsdateien verwendet werden (ausgeführt von `utils/check_config_attributes.py`)
- Die Übersetzungen der READMEs und der Index des Dokuments haben die gleiche Modellliste wie die Haupt-README (durchgeführt von `utils/check_copies.py`)
- Die automatisch generierten Tabellen in der Dokumentation sind auf dem neuesten Stand (ausgeführt von `utils/check_table.py`)
- Die Bibliothek verfügt über alle Objekte, auch wenn nicht alle optionalen Abhängigkeiten installiert sind (ausgeführt von `utils/check_dummies.py`)

Sollte diese Prüfung fehlschlagen, müssen die ersten beiden Punkte manuell korrigiert werden, die letzten vier können automatisch für Sie korrigiert werden, indem Sie den Befehl

```bash
make fix-copies
```

Zusätzliche Prüfungen betreffen PRs, die neue Modelle hinzufügen, vor allem, dass:

- Alle hinzugefügten Modelle befinden sich in einer Auto-Zuordnung (durchgeführt von `utils/check_repo.py`)
<!-- TODO Sylvain, add a check that makes sure the common tests are implemented.-->
- Alle Modelle werden ordnungsgemäß getestet (ausgeführt von `utils/check_repo.py`)

<!-- TODO Sylvain, add the following
- All models are added to the main README, inside the main doc
- All checkpoints used actually exist on the Hub

-->

### Kopien prüfen

Da die Transformers-Bibliothek in Bezug auf den Modellcode sehr eigenwillig ist und jedes Modell vollständig in einer einzigen Datei implementiert sein sollte, ohne sich auf andere Modelle zu stützen, haben wir einen Mechanismus hinzugefügt, der überprüft, ob eine Kopie des Codes einer Ebene eines bestimmten Modells mit dem Original übereinstimmt. Auf diese Weise können wir bei einer Fehlerbehebung alle anderen betroffenen Modelle sehen und entscheiden, ob wir die Änderung weitergeben oder die Kopie zerstören.

<Tip>

Wenn eine Datei eine vollständige Kopie einer anderen Datei ist, sollten Sie sie in der Konstante `FULL_COPIES` von `utils/check_copies.py` registrieren.

</Tip>

Dieser Mechanismus stützt sich auf Kommentare der Form `# Kopiert von xxx`. Das `xxx` sollte den gesamten Pfad zu der Klasse der Funktion enthalten, die darunter kopiert wird. Zum Beispiel ist `RobertaSelfOutput` eine direkte Kopie der Klasse `BertSelfOutput`. Sie können also [hier](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L289) sehen, dass sie einen Kommentar hat:

```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
```

Beachten Sie, dass Sie dies nicht auf eine ganze Klasse anwenden, sondern auf die entsprechenden Methoden, von denen kopiert wird. Zum Beispiel [hier](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L598) können Sie sehen, wie `RobertaPreTrainedModel._init_weights` von der gleichen Methode in `BertPreTrainedModel` mit dem Kommentar kopiert wird:

```py
# Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
```

Manchmal ist die Kopie bis auf die Namen genau gleich: zum Beispiel verwenden wir in `RobertaAttention` `RobertaSelfAttention` anstelle von `BertSelfAttention`, aber ansonsten ist der Code genau derselbe. Aus diesem Grund unterstützt `#Copied from` einfache String-Ersetzungen mit der folgenden Syntax: `Kopiert von xxx mit foo->bar`. Das bedeutet, dass der Code kopiert wird, wobei alle Instanzen von "foo" durch "bar" ersetzt werden. Sie können sehen, wie es [hier](https://github.com/huggingface/transformers/blob/2bd7a27a671fd1d98059124024f580f8f5c0f3b5/src/transformers/models/roberta/modeling_roberta.py#L304C1-L304C86) in `RobertaAttention` mit dem Kommentar verwendet wird:

```py
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
```

Beachten Sie, dass um den Pfeil herum keine Leerzeichen stehen sollten (es sei denn, das Leerzeichen ist Teil des zu ersetzenden Musters, natürlich).

Sie können mehrere Muster durch ein Komma getrennt hinzufügen. Zum Beispiel ist hier `CamemberForMaskedLM` eine direkte Kopie von `RobertaForMaskedLM` mit zwei Ersetzungen: `Roberta` zu `Camembert` und `ROBERTA` zu `CAMEMBERT`. Sie können [hier](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/camembert/modeling_camembert.py#L929) sehen, wie dies mit dem Kommentar gemacht wird:

```py
# Copied from transformers.models.roberta.modeling_roberta.RobertaForMaskedLM with Roberta->Camembert, ROBERTA->CAMEMBERT
```

Wenn die Reihenfolge eine Rolle spielt (weil eine der Ersetzungen mit einer vorherigen in Konflikt geraten könnte), werden die Ersetzungen von links nach rechts ausgeführt.

<Tip>

Wenn die Ersetzungen die Formatierung ändern (wenn Sie z.B. einen kurzen Namen durch einen sehr langen Namen ersetzen), wird die Kopie nach Anwendung des automatischen Formats überprüft.

</Tip>

Eine andere Möglichkeit, wenn es sich bei den Mustern nur um verschiedene Umschreibungen derselben Ersetzung handelt (mit einer groß- und einer kleingeschriebenen Variante), besteht darin, die Option `all-casing` hinzuzufügen. [Hier](https://github.com/huggingface/transformers/blob/15082a9dc6950ecae63a0d3e5060b2fc7f15050a/src/transformers/models/mobilebert/modeling_mobilebert.py#L1237) ist ein Beispiel in `MobileBertForSequenceClassification` mit dem Kommentar:

```py
# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

In diesem Fall wird der Code von `BertForSequenceClassification` kopiert, indem er ersetzt wird:
- `Bert` durch `MobileBert` (zum Beispiel bei der Verwendung von `MobileBertModel` in der Init)
- `bert` durch `mobilebert` (zum Beispiel bei der Definition von `self.mobilebert`)
- `BERT` durch `MOBILEBERT` (in der Konstante `MOBILEBERT_INPUTS_DOCSTRING`)
