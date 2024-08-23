<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Wie erstellt man eine benutzerdefinierte Pipeline?

In dieser Anleitung sehen wir uns an, wie Sie eine benutzerdefinierte Pipeline erstellen und sie auf dem [Hub](hf.co/models) freigeben oder sie der
🤗 Transformers-Bibliothek hinzufügen.

Zuallererst müssen Sie entscheiden, welche Roheingaben die Pipeline verarbeiten kann. Es kann sich um Strings, rohe Bytes,
Dictionaries oder was auch immer die wahrscheinlichste gewünschte Eingabe ist. Versuchen Sie, diese Eingaben so rein wie möglich in Python zu halten
denn das macht die Kompatibilität einfacher (auch mit anderen Sprachen über JSON). Dies werden die Eingaben der
Pipeline (`Vorverarbeitung`).

Definieren Sie dann die `Outputs`. Dieselbe Richtlinie wie für die Eingänge. Je einfacher, desto besser. Dies werden die Ausgaben der
Methode `Postprocess`.

Beginnen Sie damit, die Basisklasse `Pipeline` mit den 4 Methoden zu erben, die für die Implementierung von `preprocess` benötigt werden,
Weiterleiten", "Nachbearbeitung" und "Parameter säubern".


```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

Die Struktur dieser Aufteilung soll eine relativ nahtlose Unterstützung für CPU/GPU ermöglichen und gleichzeitig die Durchführung von
Vor-/Nachbearbeitung auf der CPU in verschiedenen Threads

Preprocess" nimmt die ursprünglich definierten Eingaben und wandelt sie in etwas um, das in das Modell eingespeist werden kann. Es kann
mehr Informationen enthalten und ist normalerweise ein `Dict`.

`_forward` ist das Implementierungsdetail und ist nicht dafür gedacht, direkt aufgerufen zu werden. Weiterleiten" ist die bevorzugte
aufgerufene Methode, da sie Sicherheitsvorkehrungen enthält, die sicherstellen, dass alles auf dem erwarteten Gerät funktioniert. Wenn etwas
mit einem realen Modell verknüpft ist, gehört es in die Methode `_forward`, alles andere gehört in die Methoden preprocess/postprocess.

Die Methode `Postprocess` nimmt die Ausgabe von `_forward` und verwandelt sie in die endgültige Ausgabe, die zuvor festgelegt wurde.
zuvor entschieden wurde.

Die Methode `_sanitize_parameters` ermöglicht es dem Benutzer, beliebige Parameter zu übergeben, wann immer er möchte, sei es bei der Initialisierung
Zeit `pipeline(...., maybe_arg=4)` oder zur Aufrufzeit `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`.

Die Rückgabe von `_sanitize_parameters` sind die 3 Dicts von kwargs, die direkt an `preprocess` übergeben werden,
`_forward` und `postprocess` übergeben werden. Füllen Sie nichts aus, wenn der Aufrufer keinen zusätzlichen Parameter angegeben hat. Das
erlaubt es, die Standardargumente in der Funktionsdefinition beizubehalten, was immer "natürlicher" ist.

Ein klassisches Beispiel wäre das Argument `top_k` in der Nachbearbeitung bei Klassifizierungsaufgaben.

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

In order to achieve that, we'll update our `postprocess` method with a default parameter to `5`. and edit
`_sanitize_parameters` to allow this new parameter.


```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

Versuchen Sie, die Eingaben/Ausgaben sehr einfach und idealerweise JSON-serialisierbar zu halten, da dies die Verwendung der Pipeline sehr einfach macht
ohne dass die Benutzer neue Arten von Objekten verstehen müssen. Es ist auch relativ üblich, viele verschiedene Arten von Argumenten zu unterstützen
von Argumenten zu unterstützen (Audiodateien, die Dateinamen, URLs oder reine Bytes sein können).



## Hinzufügen zur Liste der unterstützten Aufgaben

Um Ihre `neue Aufgabe` in die Liste der unterstützten Aufgaben aufzunehmen, müssen Sie sie zur `PIPELINE_REGISTRY` hinzufügen:

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

Wenn Sie möchten, können Sie ein Standardmodell angeben. In diesem Fall sollte es mit einer bestimmten Revision (die der Name einer Verzweigung oder ein Commit-Hash sein kann, hier haben wir `"abcdef"` genommen) sowie mit dem Typ versehen sein:

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## Teilen Sie Ihre Pipeline auf dem Hub

Um Ihre benutzerdefinierte Pipeline auf dem Hub freizugeben, müssen Sie lediglich den benutzerdefinierten Code Ihrer `Pipeline`-Unterklasse in einer
Python-Datei speichern. Nehmen wir zum Beispiel an, Sie möchten eine benutzerdefinierte Pipeline für die Klassifizierung von Satzpaaren wie folgt verwenden:

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

Die Implementierung ist Framework-unabhängig und funktioniert für PyTorch- und TensorFlow-Modelle. Wenn wir dies in einer Datei
einer Datei namens `pair_classification.py` gespeichert haben, können wir sie importieren und wie folgt registrieren:

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

Sobald dies geschehen ist, können wir es mit einem vortrainierten Modell verwenden. Zum Beispiel wurde `sgugger/finetuned-bert-mrpc` auf den
auf den MRPC-Datensatz abgestimmt, der Satzpaare als Paraphrasen oder nicht klassifiziert.

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

Dann können wir sie auf dem Hub mit der Methode `save_pretrained` in einem `Repository` freigeben:

```py
from huggingface_hub import Repository

repo = Repository("test-dynamic-pipeline", clone_from="{your_username}/test-dynamic-pipeline")
classifier.save_pretrained("test-dynamic-pipeline")
repo.push_to_hub()
```

Dadurch wird die Datei, in der Sie `PairClassificationPipeline` definiert haben, in den Ordner `"test-dynamic-pipeline"` kopiert,
und speichert das Modell und den Tokenizer der Pipeline, bevor Sie alles in das Repository verschieben
`{Ihr_Benutzername}/test-dynamic-pipeline`. Danach kann jeder die Pipeline verwenden, solange er die Option
`trust_remote_code=True` angeben:

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## Hinzufügen der Pipeline zu 🤗 Transformers

Wenn Sie Ihre Pipeline zu 🤗 Transformers beitragen möchten, müssen Sie ein neues Modul im Untermodul `pipelines` hinzufügen
mit dem Code Ihrer Pipeline hinzufügen. Fügen Sie es dann der Liste der in `pipelines/__init__.py` definierten Aufgaben hinzu.

Dann müssen Sie noch Tests hinzufügen. Erstellen Sie eine neue Datei `tests/test_pipelines_MY_PIPELINE.py` mit Beispielen für die anderen Tests.

Die Funktion `run_pipeline_test` ist sehr allgemein gehalten und läuft auf kleinen Zufallsmodellen auf jeder möglichen
Architektur, wie durch `model_mapping` und `tf_model_mapping` definiert.

Dies ist sehr wichtig, um die zukünftige Kompatibilität zu testen, d.h. wenn jemand ein neues Modell für
`XXXForQuestionAnswering` hinzufügt, wird der Pipeline-Test versuchen, mit diesem Modell zu arbeiten. Da die Modelle zufällig sind, ist es
ist es unmöglich, die tatsächlichen Werte zu überprüfen. Deshalb gibt es eine Hilfsfunktion `ANY`, die einfach versucht, die
Ausgabe der Pipeline TYPE.

Außerdem *müssen* Sie 2 (idealerweise 4) Tests implementieren.

- test_small_model_pt` : Definieren Sie 1 kleines Modell für diese Pipeline (es spielt keine Rolle, ob die Ergebnisse keinen Sinn ergeben)
  und testen Sie die Ausgaben der Pipeline. Die Ergebnisse sollten die gleichen sein wie bei `test_small_model_tf`.
- test_small_model_tf : Definieren Sie 1 kleines Modell für diese Pipeline (es spielt keine Rolle, ob die Ergebnisse keinen Sinn ergeben)
  und testen Sie die Ausgaben der Pipeline. Die Ergebnisse sollten die gleichen sein wie bei `test_small_model_pt`.
- test_large_model_pt` (`optional`): Testet die Pipeline an einer echten Pipeline, bei der die Ergebnisse
  Sinn machen. Diese Tests sind langsam und sollten als solche gekennzeichnet werden. Hier geht es darum, die Pipeline zu präsentieren und sicherzustellen
  sicherzustellen, dass es in zukünftigen Versionen keine Abweichungen gibt.
- test_large_model_tf` (`optional`): Testet die Pipeline an einer echten Pipeline, bei der die Ergebnisse
  Sinn machen. Diese Tests sind langsam und sollten als solche gekennzeichnet werden. Hier geht es darum, die Pipeline zu präsentieren und sicherzustellen
  sicherzustellen, dass es in zukünftigen Versionen keine Abweichungen gibt.
