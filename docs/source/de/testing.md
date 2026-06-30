<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Testen


Werfen wir einen Blick darauf, wie 🤗 Transformers-Modelle getestet werden und wie Sie neue Tests schreiben und die vorhandenen verbessern können.

Es gibt 2 Testsuiten im Repository:

1. `tests` -- Tests für die allgemeine API
2. `examples` -- Tests hauptsächlich für verschiedene Anwendungen, die nicht Teil der API sind

## Wie Transformatoren getestet werden

1. Sobald ein PR eingereicht wurde, wird er mit 9 CircleCi Jobs getestet. Jeder neue Commit zu diesem PR wird erneut getestet. Diese Aufträge
   sind in dieser [Konfigurationsdatei](https://github.com/huggingface/transformers/tree/main/.circleci/config.yml) definiert, so dass Sie bei Bedarf die gleiche Umgebung auf Ihrem Rechner reproduzieren können.
   Umgebung auf Ihrem Rechner reproduzieren können.

   Diese CI-Jobs führen keine `@slow`-Tests durch.

2. Es gibt 3 Jobs, die von [github actions](https://github.com/huggingface/transformers/actions) ausgeführt werden:

   - [torch hub integration](https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml): prüft, ob die torch hub
     Integration funktioniert.

   - [self-hosted (push)](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml): führt schnelle Tests auf der GPU nur bei Commits auf
     `main`. Es wird nur ausgeführt, wenn ein Commit auf `main` den Code in einem der folgenden Ordner aktualisiert hat: `src`,
     `tests`, `.github` (um zu verhindern, dass er auf hinzugefügten Modellkarten, Notebooks usw. läuft)

   - [self-hosted runner](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml): führt normale und langsame Tests auf GPU in
     `tests` und `examples`:

```bash
RUN_SLOW=1 pytest tests/
RUN_SLOW=1 pytest examples/
```

   Die Ergebnisse können Sie [hier](https://github.com/huggingface/transformers/actions) sehen.



## Tests ausführen





### Auswahl der auszuführenden Tests

In diesem Dokument wird ausführlich erläutert, wie Tests ausgeführt werden können. Wenn Sie nach der Lektüre noch mehr Details benötigen
finden Sie diese [hier](https://docs.pytest.org/en/latest/usage.html).

Hier sind einige der nützlichsten Möglichkeiten, Tests auszuführen.

Alle ausführen:

```console
pytest
```

oder:

```bash
make test
```

Beachten Sie, dass Letzteres wie folgt definiert ist:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

was pytest anweist:

- so viele Testprozesse laufen zu lassen, wie es CPU-Kerne gibt (was zu viele sein könnten, wenn Sie nicht über eine Menge RAM verfügen!)
- sicherzustellen, dass alle Tests aus derselben Datei von demselben Testprozess ausgeführt werden
- Erfassen Sie keine Ausgaben
- im ausführlichen Modus laufen lassen



### Abrufen der Liste aller Tests

Alle Tests der Testsuite:

```bash
pytest --collect-only -q
```

Alle Tests einer bestimmten Testdatei:

```bash
pytest tests/test_optimization.py --collect-only -q
```

### Führen Sie ein bestimmtes Testmodul aus

Um ein einzelnes Testmodul auszuführen:

```bash
pytest tests/utils/test_logging.py
```

### Spezifische Tests ausführen

Da unittest in den meisten Tests verwendet wird, müssen Sie, um bestimmte Untertests auszuführen, den Namen der unittest
Klasse, die diese Tests enthält. Er könnte zum Beispiel lauten:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

Hier:

- `tests/test_optimization.py` - die Datei mit den Tests
- `OptimizationTest` - der Name der Klasse
- `test_adam_w` - der Name der spezifischen Testfunktion

Wenn die Datei mehrere Klassen enthält, können Sie auswählen, dass nur die Tests einer bestimmten Klasse ausgeführt werden sollen. Zum Beispiel:

```bash
pytest tests/test_optimization.py::OptimizationTest
```

führt alle Tests innerhalb dieser Klasse aus.

Wie bereits erwähnt, können Sie sehen, welche Tests in der Klasse `OptimizationTest` enthalten sind, indem Sie sie ausführen:

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

Sie können Tests mit Hilfe von Schlüsselwortausdrücken ausführen.

Um nur Tests auszuführen, deren Name `adam` enthält:

```bash
pytest -k adam tests/test_optimization.py
```

Die logischen `und` und `oder` können verwendet werden, um anzugeben, ob alle Schlüsselwörter übereinstimmen sollen oder nur eines. `nicht` kann verwendet werden, um
negieren.

Um alle Tests auszuführen, außer denen, deren Name `adam` enthält:

```bash
pytest -k "not adam" tests/test_optimization.py
```

Und Sie können die beiden Muster in einem kombinieren:

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

Um zum Beispiel sowohl `test_adafactor` als auch `test_adam_w` auszuführen, können Sie verwenden:

```bash
pytest -k "test_adam_w or test_adam_w" tests/test_optimization.py
```

Beachten Sie, dass wir hier `oder` verwenden, da wir wollen, dass eines der Schlüsselwörter übereinstimmt, um beide einzuschließen.

Wenn Sie nur Tests einschließen möchten, die beide Muster enthalten, müssen Sie `und` verwenden:

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### Führen Sie `accelerate` Tests durch

Manchmal müssen Sie `accelerate` Tests für Ihre Modelle ausführen. Dazu fügen Sie einfach `-m accelerate_tests` zu Ihrem Befehl hinzu, wenn Sie diese Tests bei einem `OPT`-Lauf ausführen möchten:
```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py
```


### Dokumentationstests ausführen

Um zu testen, ob die Dokumentationsbeispiele korrekt sind, sollten Sie überprüfen, ob die `doctests` erfolgreich sind.
Lassen Sie uns als Beispiel den docstring von [WhisperModel.forward](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1017-L1035) verwenden:

```python
r"""
Returns:

Example:
    ```python
    >>> import torch
    >>> from transformers import WhisperModel, WhisperFeatureExtractor
    >>> from datasets import load_dataset

    >>> model = WhisperModel.from_pretrained("openai/whisper-base")
    >>> feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features
    >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 2, 512]
    ```"""

```

Führen Sie einfach die folgende Zeile aus, um automatisch jedes docstring-Beispiel in der gewünschten Datei zu testen:
```bash
pytest --doctest-modules <path_to_file_or_dir>
```
Wenn die Datei eine Markdown-Erweiterung hat, sollten Sie das Argument `--doctest-glob="*.md"` hinzufügen.

### Nur geänderte Tests ausführen

Mit [pytest-picked](https://github.com/anapaulagomes/pytest-picked) können Sie die Tests ausführen, die sich auf die unstaged Dateien oder den aktuellen Zweig (gemäß Git) beziehen. Auf diese Weise können Sie schnell testen, ob Ihre Änderungen nichts kaputt gemacht haben.
nichts kaputt gemacht haben, da die Tests für Dateien, die Sie nicht verändert haben, nicht ausgeführt werden.

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

Alle Tests werden von Dateien und Ordnern ausgeführt, die geändert, aber noch nicht übergeben wurden.

### Fehlgeschlagene Tests bei Änderung der Quelle automatisch wiederholen

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) bietet eine sehr nützliche Funktion zur Erkennung aller fehlgeschlagenen
Tests zu erkennen und dann darauf zu warten, dass Sie Dateien ändern, um die fehlgeschlagenen Tests so lange zu wiederholen, bis sie erfolgreich sind, während Sie die
sie reparieren. So müssen Sie pytest nicht erneut starten, nachdem Sie die Korrektur vorgenommen haben. Dies wird so lange wiederholt, bis alle Tests bestanden sind.
Danach wird erneut ein vollständiger Durchlauf durchgeführt.

```bash
pip install pytest-xdist
```

So rufen Sie den Modus auf: `pytest -f` oder `pytest --looponfail`

Datei-Änderungen werden erkannt, indem die Wurzelverzeichnisse von `looponfailroots` und alle ihre Inhalte (rekursiv) untersucht werden.
Wenn die Vorgabe für diesen Wert für Sie nicht funktioniert, können Sie ihn in Ihrem Projekt ändern, indem Sie eine Konfigurations
Option in der Datei `setup.cfg` ändern:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

oder die Dateien `pytest.ini`/`tox.ini``:

```ini
[pytest]
looponfailroots = transformers tests
```

Dies würde dazu führen, dass nur nach Dateiänderungen in den jeweiligen Verzeichnissen gesucht wird, die relativ zum Verzeichnis der ini-Datei angegeben sind.
Verzeichnis.

[pytest-watch](https://github.com/joeyespo/pytest-watch) ist eine alternative Implementierung dieser Funktionalität.


### Überspringen eines Testmoduls

Wenn Sie alle Testmodule ausführen möchten, mit Ausnahme einiger weniger, können Sie diese ausschließen, indem Sie eine explizite Liste der auszuführenden Tests angeben. Für
Beispiel: Um alle Tests außer `test_modeling_*.py` auszuführen:

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### Status leeren

CI-Builds und wenn Isolation wichtig ist (gegen Geschwindigkeit), sollte der Cache geleert werden:

```bash
pytest --cache-clear tests
```

### Tests parallel ausführen

Wie bereits erwähnt, führt `make test` über das Plugin `pytest-xdist` Tests parallel aus (Argument `-n X`, z.B. `-n 2`
um 2 Jobs parallel laufen zu lassen).

Mit der Option `--dist=` von `pytest-xdist` können Sie steuern, wie die Tests gruppiert werden. Mit `--dist=loadfile` werden die
Tests, die sich in einer Datei befinden, in denselben Prozess.

Da die Reihenfolge der ausgeführten Tests unterschiedlich und nicht vorhersehbar ist, kann die Ausführung der Testsuite mit `pytest-xdist`
zu Fehlern führt (was bedeutet, dass wir einige unentdeckte gekoppelte Tests haben), verwenden Sie [pytest-replay](https://github.com/ESSS/pytest-replay), um die Tests in der gleichen Reihenfolge abzuspielen, was dabei helfen sollte
diese fehlgeschlagene Sequenz auf ein Minimum zu reduzieren.

### Testreihenfolge und Wiederholung

Es ist gut, die Tests mehrmals zu wiederholen, nacheinander, zufällig oder in Gruppen, um mögliche
Abhängigkeiten und zustandsbezogene Fehler zu erkennen (Abriss). Und die einfache, mehrfache Wiederholung ist einfach gut, um
einige Probleme zu erkennen, die durch die Zufälligkeit von DL aufgedeckt werden.


#### Wiederholungstests

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

Und führen Sie dann jeden Test mehrmals durch (standardmäßig 50):

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>

Dieses Plugin funktioniert nicht mit dem `-n` Flag von `pytest-xdist`.

</Tip>

<Tip>

Es gibt noch ein anderes Plugin `pytest-repeat`, aber es funktioniert nicht mit `unittest`.

</Tip>

#### Run tests in a random order

```bash
pip install pytest-random-order
```

Wichtig: Das Vorhandensein von `pytest-random-order` sorgt für eine automatische Zufallsanordnung der Tests, es sind keine Konfigurationsänderungen oder
Befehlszeilenoptionen sind nicht erforderlich.

Wie bereits erläutert, ermöglicht dies die Erkennung von gekoppelten Tests - bei denen der Zustand eines Tests den Zustand eines anderen beeinflusst. Wenn
`pytest-random-order` installiert ist, gibt es den Zufallswert aus, der für diese Sitzung verwendet wurde, z.B:

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

Wenn eine bestimmte Sequenz fehlschlägt, können Sie sie reproduzieren, indem Sie genau diesen Seed hinzufügen, z.B:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

Es wird nur dann die exakte Reihenfolge reproduzieren, wenn Sie genau dieselbe Liste von Tests (oder gar keine Liste) verwenden. Sobald Sie beginnen, die Liste
die Liste manuell einzugrenzen, können Sie sich nicht mehr auf den Seed verlassen, sondern müssen die Tests manuell in der genauen Reihenfolge auflisten
auflisten und pytest anweisen, sie nicht zu randomisieren, indem Sie `--random-order-bucket=none` verwenden, z.B.:

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

So deaktivieren Sie das Shuffling für alle Tests:

```bash
pytest --random-order-bucket=none
```

Standardmäßig ist `--random-order-bucket=module` impliziert, wodurch die Dateien auf den Modulebenen gemischt werden. Es kann auch
auf den Ebenen `class`, `package`, `global` und `none` mischen. Die vollständigen Details entnehmen Sie bitte der
[Dokumentation](https://github.com/jbasko/pytest-random-order).

Eine weitere Alternative zur Randomisierung ist: [`pytest-random`](https://github.com/pytest-dev/pytest-randomly). Dieses
Modul hat eine sehr ähnliche Funktionalität/Schnittstelle, aber es hat nicht die Eimermodi, die in
`pytest-random-order` zur Verfügung. Es hat das gleiche Problem, dass es sich nach der Installation aufdrängt.

### Variationen von Aussehen und Bedienung

#### pytest-zucker

[pytest-sugar](https://github.com/Frozenball/pytest-sugar) ist ein Plugin, das das Erscheinungsbild verbessert, eine
Fortschrittsbalken hinzufügt und Tests, die fehlschlagen, sowie die Bestätigung sofort anzeigt. Es wird bei der Installation automatisch aktiviert.

```bash
pip install pytest-sugar
```

Um Tests ohne sie durchzuführen, führen Sie aus:

```bash
pytest -p no:sugar
```

oder deinstallieren Sie es.



#### Melden Sie den Namen jedes Subtests und seinen Fortschritt

Für einen einzelnen oder eine Gruppe von Tests über `pytest` (nach `pip install pytest-pspec`):

```bash
pytest --pspec tests/test_optimization.py
```

#### Zeigt fehlgeschlagene Tests sofort an

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) zeigt Fehlschläge und Fehler sofort an, anstatt
bis zum Ende der Testsitzung zu warten.

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### Zu GPU oder nicht zu GPU

Bei einem GPU-aktivierten Setup fügen Sie zum Testen im reinen CPU-Modus `CUDA_VISIBLE_DEVICES=""` hinzu:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

oder wenn Sie mehrere Grafikprozessoren haben, können Sie angeben, welcher von `pytest` verwendet werden soll. Wenn Sie zum Beispiel nur den
zweiten Grafikkarte zu verwenden, wenn Sie die Grafikkarten `0` und `1` haben, können Sie folgendes ausführen:

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

Dies ist praktisch, wenn Sie verschiedene Aufgaben auf verschiedenen GPUs ausführen möchten.

Einige Tests müssen nur auf der CPU ausgeführt werden, andere entweder auf der CPU, der GPU oder der TPU und wieder andere auf mehreren GPUs. Die folgenden skip
Dekorateure werden verwendet, um die Anforderungen von Tests in Bezug auf CPU/GPU/TPU festzulegen:

- `require_torch` - dieser Test wird nur unter Torch ausgeführt
- `require_torch_gpu` - wie `require_torch` plus erfordert mindestens 1 GPU
- `require_torch_multi_gpu` - wie `require_torch` und zusätzlich mindestens 2 GPUs erforderlich
- `require_torch_non_multi_gpu` - wie `require_torch` plus benötigt 0 oder 1 GPUs
- `require_torch_up_to_2_gpus` - wie `require_torch` plus erfordert 0 oder 1 oder 2 GPUs
- `require_torch_xla` - wie `require_torch` plus erfordert mindestens 1 TPU

Lassen Sie uns die GPU-Anforderungen in der folgenden Tabelle darstellen:


| n gpus | decorator                      |
|--------|--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


Hier ist zum Beispiel ein Test, der nur ausgeführt werden muss, wenn 2 oder mehr GPUs verfügbar sind und pytorch installiert ist:

```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

Diese Dekors können gestapelt werden. Wenn zum Beispiel ein Test langsam ist und mindestens eine GPU unter pytorch benötigt, können Sie
wie Sie ihn einrichten können:

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

Einige Dekoratoren wie `@parametrized` schreiben Testnamen um, daher müssen `@require_*`-Sprungdekoratoren als letztes aufgeführt werden.
zuletzt aufgeführt werden, damit sie korrekt funktionieren. Hier ist ein Beispiel für die korrekte Verwendung:

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

Dieses Problem mit der Reihenfolge gibt es bei `@pytest.mark.parametrize` nicht, Sie können es an den Anfang oder an den Schluss setzen und es wird trotzdem funktionieren.
funktionieren. Aber es funktioniert nur bei Nicht-Unittests.

Innerhalb von Tests:

- Wie viele GPUs sind verfügbar:

```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count()  # works with torch and tf
```

### Testen mit einem bestimmten PyTorch-Backend oder Gerät

Um die Testsuite auf einem bestimmten Torch-Gerät auszuführen, fügen Sie `TRANSFORMERS_TEST_DEVICE="$Gerät"` hinzu, wobei `$Gerät` das Ziel-Backend ist. Zum Beispiel, um nur auf der CPU zu testen:
```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/utils/test_logging.py
```

Diese Variable ist nützlich, um benutzerdefinierte oder weniger verbreitete PyTorch-Backends wie `mps` zu testen. Sie kann auch verwendet werden, um den gleichen Effekt wie `CUDA_VISIBLE_DEVICES` zu erzielen, indem Sie bestimmte GPUs anvisieren oder im reinen CPU-Modus testen.

Bestimmte Geräte erfordern einen zusätzlichen Import, nachdem Sie `torch` zum ersten Mal importiert haben. Dies kann über die Umgebungsvariable `TRANSFORMERS_TEST_BACKEND` festgelegt werden:
```bash
TRANSFORMERS_TEST_BACKEND="torch_npu" pytest tests/utils/test_logging.py
```


### Verteiltes Training

`pytest` kann nicht direkt mit verteiltem Training umgehen. Wenn dies versucht wird, tun die Unterprozesse nicht das Richtige
und denken am Ende, sie seien `pytest` und beginnen, die Testsuite in Schleifen auszuführen. Es funktioniert jedoch, wenn man
einen normalen Prozess erzeugt, der dann mehrere Worker erzeugt und die IO-Pipes verwaltet.

Hier sind einige Tests, die dies verwenden:

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/distributed/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)

Um direkt mit der Ausführung zu beginnen, suchen Sie in diesen Tests nach dem Aufruf `execute_subprocess_async`.

Sie benötigen mindestens 2 GPUs, um diese Tests in Aktion zu sehen:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

### Erfassung von Ausgaben

Während der Testausführung werden alle Ausgaben, die an `stdout` und `stderr` gesendet werden, aufgezeichnet. Wenn ein Test oder eine Setup-Methode fehlschlägt, wird die
wird die entsprechende aufgezeichnete Ausgabe in der Regel zusammen mit dem Fehler-Traceback angezeigt.

Um die Aufzeichnung von Ausgaben zu deaktivieren und `stdout` und `stderr` normal zu erhalten, verwenden Sie `-s` oder `--capture=no`:

```bash
pytest -s tests/utils/test_logging.py
```

So senden Sie Testergebnisse an die JUnit-Formatausgabe:

```bash
py.test tests --junitxml=result.xml
```

### Farbsteuerung

Keine Farbe zu haben (z.B. gelb auf weißem Hintergrund ist nicht lesbar):

```bash
pytest --color=no tests/utils/test_logging.py
```

### Testbericht an den Online-Dienst pastebin senden

Erstellen Sie eine URL für jeden Testfehler:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

Dadurch werden Informationen über den Testlauf an einen entfernten Paste-Dienst übermittelt und eine URL für jeden Fehlschlag bereitgestellt. Sie können die
Tests wie gewohnt auswählen oder z.B. -x hinzufügen, wenn Sie nur einen bestimmten Fehler senden möchten.

Erstellen einer URL für ein ganzes Testsitzungsprotokoll:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## Tests schreiben

🤗 Die Tests von Transformers basieren auf `unittest`, werden aber von `pytest` ausgeführt, so dass die meiste Zeit Funktionen aus beiden Systemen
verwendet werden können.

Sie können [hier](https://docs.pytest.org/en/stable/unittest.html) nachlesen, welche Funktionen unterstützt werden, aber das Wichtigste ist
Wichtig ist, dass die meisten `pytest`-Fixtures nicht funktionieren. Auch die Parametrisierung nicht, aber wir verwenden das Modul
`parametrisiert`, das auf ähnliche Weise funktioniert.


### Parametrisierung

Oft besteht die Notwendigkeit, denselben Test mehrmals auszuführen, aber mit unterschiedlichen Argumenten. Das könnte innerhalb des Tests geschehen
des Tests gemacht werden, aber dann gibt es keine Möglichkeit, den Test mit nur einem Satz von Argumenten auszuführen.

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert_equal(math.floor(input), expected)
```

Nun wird dieser Test standardmäßig 3 Mal ausgeführt, wobei jedes Mal die letzten 3 Argumente von `test_floor` den entsprechenden Argumenten in der Parameterliste zugeordnet werden.
die entsprechenden Argumente in der Parameterliste.

Sie können auch nur die Parameter `negativ` und `ganzzahlig` mit ausführen:

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

oder alle Untertests außer `negativ`, mit:

```bash
pytest -k "not negative" tests/test_mytest.py
```

Neben der Verwendung des gerade erwähnten Filters `-k` können Sie auch den genauen Namen jedes Untertests herausfinden und jeden
oder alle unter Verwendung ihrer genauen Namen ausführen.

```bash
pytest test_this1.py --collect-only -q
```

und es wird aufgelistet:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

Jetzt können Sie also nur 2 spezifische Untertests durchführen:

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

Das Modul [parametrisiert](https://pypi.org/project/parameterized/), das sich bereits in den Entwickler-Abhängigkeiten befindet
von `transformers` befindet, funktioniert sowohl für `unittests` als auch für `pytest` Tests.

Wenn es sich bei dem Test jedoch nicht um einen `Unittest` handelt, können Sie `pytest.mark.parametrize` verwenden (oder Sie können sehen, dass es in
einigen bestehenden Tests verwendet wird, meist unter `Beispiele`).

Hier ist das gleiche Beispiel, diesmal unter Verwendung der `parametrize`-Markierung von `pytest`:

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert_equal(math.floor(input), expected)
```

Genau wie bei `parametrisiert` können Sie mit `pytest.mark.parametrize` genau steuern, welche Subtests ausgeführt werden
ausgeführt werden, wenn der Filter `-k` nicht ausreicht. Allerdings erzeugt diese Parametrisierungsfunktion einen etwas anderen Satz von
Namen für die Untertests. Sie sehen folgendermaßen aus:

```bash
pytest test_this2.py --collect-only -q
```

und es wird aufgelistet:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

Jetzt können Sie also nur den spezifischen Test durchführen:

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

wie im vorherigen Beispiel.



### Dateien und Verzeichnisse

In Tests müssen wir oft wissen, wo sich Dinge relativ zur aktuellen Testdatei befinden, und das ist nicht trivial, da der Test
von mehreren Verzeichnissen aus aufgerufen werden kann oder sich in Unterverzeichnissen mit unterschiedlicher Tiefe befinden kann. Eine Hilfsklasse
`transformers.test_utils.TestCasePlus` löst dieses Problem, indem sie alle grundlegenden Pfade sortiert und einfache
Zugriffsmöglichkeiten auf sie bietet:

- `pathlib`-Objekte (alle vollständig aufgelöst):

  - `test_file_path` - der aktuelle Testdateipfad, d.h. `__file__`
  - `test_file_dir` - das Verzeichnis, das die aktuelle Testdatei enthält
  - `tests_dir` - das Verzeichnis der `tests` Testreihe
  - `examples_dir` - das Verzeichnis der `examples` Test-Suite
  - `repo_root_dir` - das Verzeichnis des Repositorys
  - `src_dir` - das Verzeichnis von `src` (d.h. wo sich das Unterverzeichnis `transformers` befindet)

- stringifizierte Pfade - wie oben, aber diese geben Pfade als Strings zurück, anstatt als `pathlib`-Objekte:

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

Um diese zu verwenden, müssen Sie lediglich sicherstellen, dass der Test in einer Unterklasse von
`transformers.test_utils.TestCasePlus` befindet. Zum Beispiel:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

Wenn Sie Pfade nicht über `pathlib` manipulieren müssen oder nur einen Pfad als String benötigen, können Sie jederzeit
`str()` auf das `pathlib`-Objekt anwenden oder die Accessoren mit der Endung `_str` verwenden. Zum Beispiel:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

### Temporäre Dateien und Verzeichnisse

Die Verwendung eindeutiger temporärer Dateien und Verzeichnisse ist für die parallele Durchführung von Tests unerlässlich, damit sich die Tests nicht gegenseitig überschreiben.
Daten gegenseitig überschreiben. Außerdem möchten wir, dass die temporären Dateien und Verzeichnisse am Ende jedes Tests, der sie erstellt hat, gelöscht werden.
erstellt hat. Daher ist die Verwendung von Paketen wie `tempfile`, die diese Anforderungen erfüllen, unerlässlich.

Beim Debuggen von Tests müssen Sie jedoch sehen können, was in der temporären Datei oder dem temporären Verzeichnis gespeichert wird und Sie möchten
Sie müssen den genauen Pfad kennen und dürfen ihn nicht bei jedem neuen Testdurchlauf zufällig ändern.

Für solche Zwecke ist die Hilfsklasse `transformers.test_utils.TestCasePlus` am besten geeignet. Sie ist eine Unterklasse von
Unittest.TestCase`, so dass wir in den Testmodulen einfach von ihr erben können.

Hier ist ein Beispiel für die Verwendung dieser Klasse:

```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

Dieser Code erstellt ein eindeutiges temporäres Verzeichnis und setzt `tmp_dir` auf dessen Speicherort.

- Erstellen Sie ein eindeutiges temporäres Verzeichnis:

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

tmp_dir" enthält den Pfad zu dem erstellten temporären Verzeichnis. Es wird am Ende des Tests automatisch entfernt.
Tests entfernt.

- Erstellen Sie ein temporäres Verzeichnis meiner Wahl, stellen Sie sicher, dass es leer ist, bevor der Test beginnt, und leeren Sie es nach dem Test nicht.

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

Dies ist nützlich für die Fehlersuche, wenn Sie ein bestimmtes Verzeichnis überwachen und sicherstellen möchten, dass die vorherigen Tests keine Daten darin hinterlassen haben.
keine Daten dort hinterlassen haben.

- Sie können das Standardverhalten außer Kraft setzen, indem Sie die Argumente `before` und `after` direkt überschreiben, was zu einem der folgenden Verhaltensweisen führt
  folgenden Verhaltensweisen:

  - `before=True`: das temporäre Verzeichnis wird immer zu Beginn des Tests gelöscht.
  - `before=False`: wenn das temporäre Verzeichnis bereits existiert, bleiben alle vorhandenen Dateien dort erhalten.
  - `after=True`: das temporäre Verzeichnis wird immer am Ende des Tests gelöscht.
  - `after=False`: das temporäre Verzeichnis wird am Ende des Tests immer beibehalten.

<Tip>

Um das Äquivalent von `rm -r` sicher ausführen zu können, sind nur Unterverzeichnisse des Projektarchivs checkout erlaubt, wenn
ein explizites `tmp_dir` verwendet wird, so dass nicht versehentlich ein `/tmp` oder ein ähnlich wichtiger Teil des Dateisystems vernichtet wird.
d.h. geben Sie bitte immer Pfade an, die mit `./` beginnen.

</Tip>

<Tip>

Jeder Test kann mehrere temporäre Verzeichnisse registrieren, die alle automatisch entfernt werden, sofern nicht anders gewünscht.
anders.

</Tip>

### Temporäre Überschreibung von sys.path

Wenn Sie `sys.path` vorübergehend überschreiben müssen, um z.B. von einem anderen Test zu importieren, können Sie den
Kontextmanager `ExtendSysPath` verwenden. Beispiel:


```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### Überspringen von Tests

Dies ist nützlich, wenn ein Fehler gefunden und ein neuer Test geschrieben wird, der Fehler aber noch nicht behoben ist. Damit wir ihn
in das Haupt-Repository zu übertragen, müssen wir sicherstellen, dass er bei `make test` übersprungen wird.

Methoden:

- Ein **Skip** bedeutet, dass Sie erwarten, dass Ihr Test nur dann erfolgreich ist, wenn einige Bedingungen erfüllt sind, andernfalls sollte pytest den Test überspringen.
  die Ausführung des Tests ganz überspringen. Übliche Beispiele sind das Überspringen von Tests, die nur unter Windows laufen, auf Nicht-Windows-Plattformen oder das Überspringen von
  Tests, die von einer externen Ressource abhängen, die im Moment nicht verfügbar ist (z.B. eine Datenbank).

- Ein **xfail** bedeutet, dass Sie erwarten, dass ein Test aus irgendeinem Grund fehlschlägt. Ein gängiges Beispiel ist ein Test für eine Funktion, die noch nicht
  noch nicht implementiert oder ein noch nicht behobener Fehler. Wenn ein Test trotz eines erwarteten Fehlschlags bestanden wird (markiert mit
  pytest.mark.xfail), ist dies ein xpass und wird in der Testzusammenfassung gemeldet.

Einer der wichtigsten Unterschiede zwischen den beiden ist, dass `skip` den Test nicht ausführt, während `xfail` dies tut. Wenn also der
Code, der fehlerhaft ist, einen schlechten Zustand verursacht, der sich auf andere Tests auswirkt, sollten Sie also nicht `xfail` verwenden.

#### Implementierung

- Hier sehen Sie, wie Sie einen ganzen Test bedingungslos überspringen können:

```python no-style
@unittest.skip(reason="this bug needs to be fixed")
def test_feature_x():
```

oder mit pytest:

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

oder mit dem `xfail` Weg:

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

- Hier erfahren Sie, wie Sie einen Test aufgrund einer internen Prüfung innerhalb des Tests auslassen können:

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

oder das ganze Modul:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

oder mit dem `xfail` Weg:

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- Hier erfahren Sie, wie Sie alle Tests in einem Modul überspringen können, wenn ein Import fehlt:

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- Einen Test aufgrund einer Bedingung überspringen:

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

oder:

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

oder überspringen Sie das ganze Modul:

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

Weitere Details, Beispiele und Möglichkeiten finden Sie [hier](https://docs.pytest.org/en/latest/skipping.html).

### Langsame Tests

Die Bibliothek der Tests wächst ständig, und einige der Tests brauchen Minuten, um ausgeführt zu werden, daher können wir es uns nicht leisten, eine Stunde zu warten, bis die
eine Stunde auf die Fertigstellung der Testsuite auf CI zu warten. Daher sollten langsame Tests, mit einigen Ausnahmen für wichtige Tests, wie im folgenden Beispiel
wie im folgenden Beispiel markiert werden:

```python no-style
from transformers.testing_utils import slow
@slow
def test_integration_foo():
```

Sobald ein Test als `@slow` markiert ist, setzen Sie die Umgebungsvariable `RUN_SLOW=1`, um solche Tests auszuführen, z.B:

```bash
RUN_SLOW=1 pytest tests
```

Einige Dekoratoren wie `@parameterized` schreiben Testnamen um, daher müssen `@slow` und die übrigen Skip-Dekoratoren
`@require_*` müssen als letztes aufgeführt werden, damit sie korrekt funktionieren. Hier ist ein Beispiel für die korrekte Verwendung:

```python no-style
@parameterized.expand(...)
@slow
def test_integration_foo():
```

Wie zu Beginn dieses Dokuments erläutert, werden langsame Tests nach einem Zeitplan ausgeführt und nicht in PRs CI
Prüfungen. Es ist also möglich, dass einige Probleme bei der Einreichung eines PRs übersehen werden und zusammengeführt werden. Solche Probleme werden
werden beim nächsten geplanten CI-Job abgefangen. Das bedeutet aber auch, dass es wichtig ist, die langsamen Tests auf Ihrem
Rechner auszuführen, bevor Sie den PR einreichen.

Hier ist ein grober Entscheidungsmechanismus für die Auswahl der Tests, die als langsam markiert werden sollen:

Wenn der Test auf eine der internen Komponenten der Bibliothek ausgerichtet ist (z.B. Modellierungsdateien, Tokenisierungsdateien,
Pipelines), dann sollten wir diesen Test in der nicht langsamen Testsuite ausführen. Wenn er sich auf einen anderen Aspekt der Bibliothek bezieht,
wie z.B. die Dokumentation oder die Beispiele, dann sollten wir diese Tests in der langsamen Testsuite durchführen. Und dann, zur Verfeinerung
Ansatz zu verfeinern, sollten wir Ausnahmen einführen:

- Alle Tests, die einen umfangreichen Satz von Gewichten oder einen Datensatz mit einer Größe von mehr als ~50MB herunterladen müssen (z.B. Modell- oder
  Tokenizer-Integrationstests, Pipeline-Integrationstests) sollten auf langsam gesetzt werden. Wenn Sie ein neues Modell hinzufügen, sollten Sie
  sollten Sie eine kleine Version des Modells (mit zufälligen Gewichtungen) für Integrationstests erstellen und in den Hub hochladen. Dies wird
  wird in den folgenden Abschnitten erläutert.
- Alle Tests, die ein Training durchführen müssen, das nicht speziell auf Schnelligkeit optimiert ist, sollten auf langsam gesetzt werden.
- Wir können Ausnahmen einführen, wenn einige dieser Tests, die nicht langsam sein sollten, unerträglich langsam sind, und sie auf
  `@slow`. Auto-Modellierungstests, die große Dateien auf der Festplatte speichern und laden, sind ein gutes Beispiel für Tests, die als
  als `@slow` markiert sind.
- Wenn ein Test in weniger als 1 Sekunde auf CI abgeschlossen wird (einschließlich eventueller Downloads), sollte es sich trotzdem um einen normalen Test handeln.

Insgesamt müssen alle nicht langsamen Tests die verschiedenen Interna abdecken und dabei schnell bleiben. Zum Beispiel,
kann eine signifikante Abdeckung erreicht werden, indem Sie mit speziell erstellten kleinen Modellen mit zufälligen Gewichten testen. Solche Modelle
haben eine sehr geringe Anzahl von Schichten (z.B. 2), Vokabeln (z.B. 1000), usw. Dann können die `@slow`-Tests große
langsame Modelle verwenden, um qualitative Tests durchzuführen. Um die Verwendung dieser Modelle zu sehen, suchen Sie einfach nach *winzigen* Modellen mit:

```bash
grep tiny tests examples
```

Hier ist ein Beispiel für ein [Skript](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py), das das winzige Modell erstellt hat
[stas/tiny-wmt19-en-de](https://huggingface.co/stas/tiny-wmt19-en-de). Sie können es ganz einfach an Ihre eigene
Architektur Ihres Modells anpassen.

Es ist leicht, die Laufzeit falsch zu messen, wenn zum Beispiel ein großes Modell heruntergeladen wird, aber wenn
Sie es lokal testen, würden die heruntergeladenen Dateien zwischengespeichert und somit die Download-Zeit nicht gemessen werden. Prüfen Sie daher den
Ausführungsgeschwindigkeitsbericht in den CI-Protokollen (die Ausgabe von `pytest --durations=0 tests`).

Dieser Bericht ist auch nützlich, um langsame Ausreißer zu finden, die nicht als solche gekennzeichnet sind oder die neu geschrieben werden müssen, um schnell zu sein.
Wenn Sie bemerken, dass die Testsuite beim CI langsam wird, zeigt die oberste Liste dieses Berichts die langsamsten
Tests.


### Testen der stdout/stderr-Ausgabe

Um Funktionen zu testen, die in `stdout` und/oder `stderr` schreiben, kann der Test auf diese Ströme zugreifen, indem er die
[capsys system](https://docs.pytest.org/en/latest/capture.html) von `pytest` zugreifen. So wird dies bewerkstelligt:

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # consume the captured output streams
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # test:
    assert msg in out
    assert msg in err
```

Und natürlich wird `stderr` in den meisten Fällen als Teil einer Ausnahme auftreten, so dass try/except in einem solchen Fall verwendet werden muss
Fall verwendet werden:

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "Not a good value"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} is in the exception:\n{error}"
```

Ein anderer Ansatz zur Erfassung von stdout ist `contextlib.redirect_stdout`:

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    # test:
    assert msg in out
```

Ein wichtiges potenzielles Problem beim Erfassen von stdout ist, dass es `r` Zeichen enthalten kann, die bei normalem `print`
alles zurücksetzen, was bisher gedruckt wurde. Mit `pytest` gibt es kein Problem, aber mit `pytest -s` werden diese
werden diese Zeichen in den Puffer aufgenommen. Um den Test mit und ohne `-s` laufen zu lassen, müssen Sie also eine zusätzliche Bereinigung
zusätzliche Bereinigung der erfassten Ausgabe vornehmen, indem Sie `re.sub(r'~.*\r', '', buf, 0, re.M)` verwenden.

Aber dann haben wir einen Hilfskontextmanager-Wrapper, der sich automatisch um alles kümmert, unabhängig davon, ob er
einige "*.*.*.*" enthält oder nicht:

```python
from transformers.testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

Hier ist ein vollständiges Testbeispiel:

```python
from transformers.testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

Wenn Sie `stderr` aufzeichnen möchten, verwenden Sie stattdessen die Klasse `CaptureStderr`:

```python
from transformers.testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

Wenn Sie beide Streams auf einmal erfassen müssen, verwenden Sie die übergeordnete Klasse `CaptureStd`:

```python
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

Um das Debuggen von Testproblemen zu erleichtern, geben diese Kontextmanager standardmäßig die aufgezeichneten Streams beim Verlassen
aus dem Kontext wieder.


### Erfassen von Logger-Streams

Wenn Sie die Ausgabe eines Loggers validieren müssen, können Sie `CaptureLogger` verwenden:

```python
from transformers import logging
from transformers.testing_utils import CaptureLogger

msg = "Testing 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

### Testen mit Umgebungsvariablen

Wenn Sie die Auswirkungen von Umgebungsvariablen für einen bestimmten Test testen möchten, können Sie einen Hilfsdekorator verwenden
`transformers.testing_utils.mockenv`

```python
from transformers.testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

Manchmal muss ein externes Programm aufgerufen werden, was die Einstellung von `PYTHONPATH` in `os.environ` erfordert, um mehrere lokale Pfade einzuschließen.
mehrere lokale Pfade. Eine Hilfsklasse `transformers.test_utils.TestCasePlus` hilft Ihnen dabei:

```python
from transformers.testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # now call the external program, passing `env` to it
```

Je nachdem, ob die Testdatei in der Testsuite `tests` oder in `examples` war, wird sie korrekt eingerichtet
`env[PYTHONPATH]` eines dieser beiden Verzeichnisse und auch das `src` Verzeichnis, um sicherzustellen, dass der Test gegen das aktuelle
um sicherzustellen, dass der Test mit dem aktuellen Projektarchiv durchgeführt wird, und schließlich mit dem, was in `env[PYTHONPATH]` bereits eingestellt war, bevor der Test aufgerufen wurde.
wenn überhaupt.

Diese Hilfsmethode erstellt eine Kopie des Objekts `os.environ`, so dass das Original intakt bleibt.


### Reproduzierbare Ergebnisse erhalten

In manchen Situationen möchten Sie vielleicht die Zufälligkeit Ihrer Tests beseitigen. Um identische, reproduzierbare Ergebnisse zu erhalten, müssen Sie
müssen Sie den Seed festlegen:

```python
seed = 42

# python RNG
import random

random.seed(seed)

# pytorch RNGs
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np

np.random.seed(seed)
```

### Tests debuggen

Um einen Debugger an der Stelle zu starten, an der die Warnung auftritt, gehen Sie wie folgt vor:

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```

## Arbeiten mit Github-Aktionen-Workflows

Um einen CI-Job für einen Self-Push-Workflow auszulösen, müssen Sie:

1. Erstellen Sie einen neuen Zweig auf `transformers` Ursprung (keine Gabelung!).
2. Der Name der Verzweigung muss entweder mit `ci_` oder `ci-` beginnen (`main` löst ihn auch aus, aber wir können keine PRs auf
   `main`). Es wird auch nur für bestimmte Pfade ausgelöst - Sie können die aktuelle Definition finden, falls sie
   falls sie sich seit der Erstellung dieses Dokuments geändert hat [hier](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-push.yml) unter *push:*
3. Erstellen Sie einen PR von diesem Zweig.
4. Dann können Sie sehen, wie der Job erscheint [hier](https://github.com/huggingface/transformers/actions/workflows/self-push.yml). Er wird möglicherweise nicht sofort ausgeführt, wenn es
   ein Backlog vorhanden ist.




## Testen experimenteller CI-Funktionen

Das Testen von CI-Funktionen kann potenziell problematisch sein, da es die normale CI-Funktion beeinträchtigen kann. Wenn also eine
neue CI-Funktion hinzugefügt werden soll, sollte dies wie folgt geschehen.

1. Erstellen Sie einen neuen Auftrag, der die zu testende Funktion testet.
2. Der neue Job muss immer erfolgreich sein, so dass er uns ein grünes ✓ gibt (Details unten).
3. Lassen Sie ihn einige Tage lang laufen, um zu sehen, dass eine Vielzahl verschiedener PR-Typen darauf laufen (Benutzer-Gabelzweige,
   nicht geforkte Zweige, Zweige, die von github.com UI direct file edit stammen, verschiedene erzwungene Pushes, etc. - es gibt
   es gibt so viele), während Sie die Protokolle des experimentellen Jobs überwachen (nicht den gesamten Job grün, da er absichtlich immer
   grün)
4. Wenn klar ist, dass alles in Ordnung ist, fügen Sie die neuen Änderungen in die bestehenden Jobs ein.

Auf diese Weise wird der normale Arbeitsablauf nicht durch Experimente mit der CI-Funktionalität selbst beeinträchtigt.

Wie können wir nun dafür sorgen, dass der Auftrag immer erfolgreich ist, während die neue CI-Funktion entwickelt wird?

Einige CIs, wie TravisCI, unterstützen ignore-step-failure und melden den gesamten Job als erfolgreich, aber CircleCI und
Github Actions unterstützen dies zum jetzigen Zeitpunkt nicht.

Sie können also die folgende Abhilfe verwenden:

1. Setzen Sie `set +euo pipefail` am Anfang des Ausführungsbefehls, um die meisten potenziellen Fehler im Bash-Skript zu unterdrücken.
2. Der letzte Befehl muss ein Erfolg sein: `echo "done"` oder einfach `true` reicht aus.

Hier ist ein Beispiel:

```yaml
- run:
    name: run CI experiment
    command: |
        set +euo pipefail
        echo "setting run-all-despite-any-errors-mode"
        this_command_will_fail
        echo "but bash continues to run"
        # emulate another failure
        false
        # but the last command must be a success
        echo "during experiment do not remove: reporting success to CI, even if there were failures"
```

Für einfache Befehle können Sie auch Folgendes tun:

```bash
cmd_that_may_fail || true
```

Wenn Sie mit den Ergebnissen zufrieden sind, integrieren Sie den experimentellen Schritt oder Job natürlich in den Rest der normalen Jobs,
Entfernen Sie dabei `set +euo pipefail` oder andere Dinge, die Sie eventuell hinzugefügt haben, um sicherzustellen, dass der experimentelle Auftrag nicht
den normalen CI-Betrieb nicht beeinträchtigt.

Dieser ganze Prozess wäre viel einfacher gewesen, wenn wir nur etwas wie `allow-failure` für den
experimentellen Schritt festlegen könnten und ihn scheitern lassen würden, ohne den Gesamtstatus der PRs zu beeinträchtigen. Aber wie bereits erwähnt, haben CircleCI und
Github Actions dies im Moment nicht unterstützen.

Sie können in diesen CI-spezifischen Threads für diese Funktion stimmen und sehen, wo sie steht:

- [Github Actions:](https://github.com/actions/toolkit/issues/399)
- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344)
