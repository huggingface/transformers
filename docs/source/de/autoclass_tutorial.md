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

# Vortrainierte Instanzen mit einer AutoClass laden

Bei so vielen verschiedenen Transformator-Architekturen kann es eine Herausforderung sein, eine für Ihren Checkpoint zu erstellen. Als Teil der 🤗 Transformers Kernphilosophie, die Bibliothek leicht, einfach und flexibel nutzbar zu machen, leitet eine `AutoClass` automatisch die richtige Architektur aus einem gegebenen Checkpoint ab und lädt sie. Mit der Methode `from_pretrained()` kann man schnell ein vortrainiertes Modell für eine beliebige Architektur laden, so dass man keine Zeit und Ressourcen aufwenden muss, um ein Modell von Grund auf zu trainieren. Die Erstellung dieser Art von Checkpoint-agnostischem Code bedeutet, dass Ihr Code, wenn er für einen Checkpoint funktioniert, auch mit einem anderen Checkpoint funktionieren wird - solange er für eine ähnliche Aufgabe trainiert wurde - selbst wenn die Architektur unterschiedlich ist.

<Tip>

Denken Sie daran, dass sich die Architektur auf das Skelett des Modells bezieht und die Checkpoints die Gewichte für eine bestimmte Architektur sind. Zum Beispiel ist [BERT](https://huggingface.co/bert-base-uncased) eine Architektur, während `bert-base-uncased` ein Checkpoint ist. Modell ist ein allgemeiner Begriff, der entweder Architektur oder Prüfpunkt bedeuten kann.

</Tip>

In dieser Anleitung lernen Sie, wie man:

* Einen vortrainierten Tokenizer lädt.
* Einen vortrainierten Merkmalsextraktor lädt.
* Einen vortrainierten Prozessor lädt.
* Ein vortrainiertes Modell lädt.

## AutoTokenizer

Nahezu jede NLP-Aufgabe beginnt mit einem Tokenizer. Ein Tokenizer wandelt Ihre Eingabe in ein Format um, das vom Modell verarbeitet werden kann.

Laden Sie einen Tokenizer mit [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

Dann tokenisieren Sie Ihre Eingabe wie unten gezeigt:

```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoFeatureExtractor

Für Audio- und Bildverarbeitungsaufgaben verarbeitet ein Merkmalsextraktor das Audiosignal oder Bild in das richtige Eingabeformat.

Laden Sie einen Merkmalsextraktor mit [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

Multimodale Aufgaben erfordern einen Prozessor, der zwei Arten von Vorverarbeitungswerkzeugen kombiniert. Das Modell [LayoutLMV2](model_doc/layoutlmv2) beispielsweise benötigt einen Feature-Extraktor für Bilder und einen Tokenizer für Text; ein Prozessor kombiniert beide.

Laden Sie einen Prozessor mit [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

<frameworkcontent>
<pt>
Mit den `AutoModelFor`-Klassen können Sie schließlich ein vortrainiertes Modell für eine bestimmte Aufgabe laden (siehe [hier](model_doc/auto) für eine vollständige Liste der verfügbaren Aufgaben). Laden Sie zum Beispiel ein Modell für die Sequenzklassifikation mit [`AutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

Sie können denselben Prüfpunkt problemlos wiederverwenden, um eine Architektur für eine andere Aufgabe zu laden:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

<Tip warning={true}>

Für PyTorch-Modelle verwendet die Methode `from_pretrained()` `torch.load()`, die intern `pickle` verwendet und als unsicher bekannt ist. Generell sollte man niemals ein Modell laden, das aus einer nicht vertrauenswürdigen Quelle stammen könnte, oder das manipuliert worden sein könnte. Dieses Sicherheitsrisiko wird für öffentliche Modelle, die auf dem Hugging Face Hub gehostet werden, teilweise gemildert, da diese bei jeder Übertragung [auf Malware](https://huggingface.co/docs/hub/security-malware) gescannt werden. Siehe die [Hub-Dokumentation](https://huggingface.co/docs/hub/security) für Best Practices wie [signierte Commit-Verifizierung](https://huggingface.co/docs/hub/security-gpg#signing-commits-with-gpg) mit GPG.

TensorFlow- und Flax-Checkpoints sind nicht betroffen und können in PyTorch-Architekturen mit den Kwargs `from_tf` und `from_flax` für die Methode `from_pretrained` geladen werden, um dieses Problem zu umgehen.

</Tip>

Im Allgemeinen empfehlen wir die Verwendung der Klasse "AutoTokenizer" und der Klasse "AutoModelFor", um trainierte Instanzen von Modellen zu laden. Dadurch wird sichergestellt, dass Sie jedes Mal die richtige Architektur laden. Im nächsten [Tutorial] (Vorverarbeitung) erfahren Sie, wie Sie Ihren neu geladenen Tokenizer, Feature Extractor und Prozessor verwenden, um einen Datensatz für die Feinabstimmung vorzuverarbeiten.
</pt>
<tf>
Mit den Klassen `TFAutoModelFor` schließlich können Sie ein vortrainiertes Modell für eine bestimmte Aufgabe laden (siehe [hier](model_doc/auto) für eine vollständige Liste der verfügbaren Aufgaben). Laden Sie zum Beispiel ein Modell für die Sequenzklassifikation mit [`TFAutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

Sie können denselben Prüfpunkt problemlos wiederverwenden, um eine Architektur für eine andere Aufgabe zu laden:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

Im Allgemeinen empfehlen wir, die Klasse "AutoTokenizer" und die Klasse "TFAutoModelFor" zu verwenden, um vortrainierte Instanzen von Modellen zu laden. Dadurch wird sichergestellt, dass Sie jedes Mal die richtige Architektur laden. Im nächsten [Tutorial] (Vorverarbeitung) erfahren Sie, wie Sie Ihren neu geladenen Tokenizer, Feature Extractor und Prozessor verwenden, um einen Datensatz für die Feinabstimmung vorzuverarbeiten.
</tf>
</frameworkcontent>
