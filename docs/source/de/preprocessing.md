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

# Vorverarbeiten

[[open-in-colab]]

Bevor Sie Ihre Daten in einem Modell verwenden können, müssen die Daten in ein für das Modell akzeptables Format gebracht werden. Ein Modell versteht keine Rohtexte, Bilder oder Audiodaten. Diese Eingaben müssen in Zahlen umgewandelt und zu Tensoren zusammengesetzt werden. In dieser Anleitung werden Sie:

* Textdaten mit einem Tokenizer vorverarbeiten.
* Bild- oder Audiodaten mit einem Feature Extractor vorverarbeiten.
* Daten für eine multimodale Aufgabe mit einem Prozessor vorverarbeiten.

## NLP

<Youtube id="Yffk5aydLzg"/>

Das wichtigste Werkzeug zur Verarbeitung von Textdaten ist ein [Tokenizer](main_classes/tokenizer). Ein Tokenizer zerlegt Text zunächst nach einer Reihe von Regeln in *Token*. Die Token werden in Zahlen umgewandelt, die zum Aufbau von Tensoren als Eingabe für ein Modell verwendet werden. Alle zusätzlichen Eingaben, die ein Modell benötigt, werden ebenfalls vom Tokenizer hinzugefügt.

<Tip>

Wenn Sie ein vortrainiertes Modell verwenden möchten, ist es wichtig, den zugehörigen vortrainierten Tokenizer zu verwenden. Dadurch wird sichergestellt, dass der Text auf die gleiche Weise aufgeteilt wird wie das Pretraining-Korpus und die gleichen entsprechenden Token-zu-Index (in der Regel als *vocab* bezeichnet) während des Pretrainings verwendet werden.

</Tip>

Laden Sie einen vortrainierten Tokenizer mit der Klasse [AutoTokenizer], um schnell loszulegen. Damit wird das *vocab* heruntergeladen, das verwendet wird, wenn ein Modell vortrainiert wird.

### Tokenize

Laden Sie einen vortrainierten Tokenizer mit [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

Dann übergeben Sie Ihren Satz an den Tokenizer:

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Der Tokenizer gibt ein Wörterbuch mit drei wichtigen Elementen zurück:

* [input_ids](glossary#input-ids) sind die Indizes, die den einzelnen Token im Satz entsprechen.
* [attention_mask](glossary#attention-mask) gibt an, ob ein Token beachtet werden soll oder nicht.
* [token_type_ids](glossary#token-type-ids) gibt an, zu welcher Sequenz ein Token gehört, wenn es mehr als eine Sequenz gibt.

Sie können die `input_ids` dekodieren, um die ursprüngliche Eingabe zurückzugeben:

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

Wie Sie sehen können, hat der Tokenisierer zwei spezielle Token - `CLS` und `SEP` (Klassifikator und Separator) - zum Satz hinzugefügt. Nicht alle Modelle benötigen
spezielle Token, aber wenn dies der Fall ist, fügt der Tokenisierer sie automatisch für Sie hinzu.

Wenn Sie mehrere Sätze verarbeiten wollen, übergeben Sie die Sätze als Liste an den Tokenizer:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### Pad

Dies bringt uns zu einem wichtigen Thema. Wenn Sie einen Haufen von Sätzen verarbeiten, sind diese nicht immer gleich lang. Das ist ein Problem, weil Tensoren, die Eingabe für das Modell, eine einheitliche Form haben müssen. Padding ist eine Strategie, die sicherstellt, dass Tensoren rechteckig sind, indem ein spezielles *Padding-Token* zu Sätzen mit weniger Token hinzugefügt wird.

Setzen Sie den Parameter "padding" auf "true", um die kürzeren Sequenzen im Stapel so aufzufüllen, dass sie der längsten Sequenz entsprechen:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

Beachten Sie, dass der Tokenizer den ersten und den dritten Satz mit einer "0" aufgefüllt hat, weil sie kürzer sind!

### Kürzung

Auf der anderen Seite des Spektrums kann es vorkommen, dass eine Sequenz zu lang für ein Modell ist. In diesem Fall müssen Sie die Sequenz auf eine kürzere Länge kürzen.

Setzen Sie den Parameter "truncation" auf "true", um eine Sequenz auf die vom Modell akzeptierte Höchstlänge zu kürzen:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

### Tensoren erstellen

Schließlich möchten Sie, dass der Tokenizer die tatsächlichen Tensoren zurückgibt, die dem Modell zugeführt werden.

Setzen Sie den Parameter `return_tensors` entweder auf `pt` für PyTorch, oder `tf` für TensorFlow:

<frameworkcontent>
<pt>

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt>
<tf>
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>, 
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>, 
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
```
</tf>
</frameworkcontent>

## Audio

Audioeingaben werden anders vorverarbeitet als Texteingaben, aber das Endziel bleibt dasselbe: numerische Sequenzen zu erstellen, die das Modell verstehen kann. Ein [feature extractor](main_classes/feature_extractor) dient dem ausdrücklichen Zweck, Merkmale aus Rohbild- oder Audiodaten zu extrahieren und in Tensoren zu konvertieren. Bevor Sie beginnen, installieren Sie 🤗 Datasets, um einen Audio-Datensatz zu laden, mit dem Sie experimentieren können:

```bash
pip install datasets
```

Laden Sie den [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) Datensatz (weitere Informationen zum Laden eines Datensatzes finden Sie im 🤗 [Datasets tutorial](https://huggingface.co/docs/datasets/load_hub)):

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

Greifen Sie auf das erste Element der `audio`-Spalte zu, um einen Blick auf die Eingabe zu werfen. Durch den Aufruf der Spalte "audio" wird die Audiodatei automatisch geladen und neu gesampelt:

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

Dies gibt drei Elemente zurück:

* "array" ist das Sprachsignal, das als 1D-Array geladen - und möglicherweise neu gesampelt - wurde.
* Pfad" zeigt auf den Speicherort der Audiodatei.
* `sampling_rate` bezieht sich darauf, wie viele Datenpunkte im Sprachsignal pro Sekunde gemessen werden.

### Resample

Für dieses Tutorial werden Sie das Modell [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) verwenden. Wie Sie aus der Modellkarte ersehen können, ist das Wav2Vec2-Modell auf 16kHz abgetastetes Sprachaudio vortrainiert. Es ist wichtig, dass die Abtastrate Ihrer Audiodaten mit der Abtastrate des Datensatzes übereinstimmt, der für das Pre-Training des Modells verwendet wurde. Wenn die Abtastrate Ihrer Daten nicht dieselbe ist, müssen Sie Ihre Audiodaten neu abtasten. 

Der Datensatz [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) hat zum Beispiel eine Abtastrate von 8000 kHz. Um das Wav2Vec2-Modell mit diesem Datensatz verwenden zu können, müssen Sie die Abtastrate auf 16 kHz erhöhen:

```py
>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

1. Verwenden Sie die Methode [`~datasets.Dataset.cast_column`] von 🤗 Datasets, um die Abtastrate auf 16kHz zu erhöhen:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. Laden Sie die Audiodatei:

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

Wie Sie sehen können, ist die Abtastrate jetzt 16kHz!

### Merkmalsextraktor

Der nächste Schritt ist das Laden eines Merkmalsextraktors, um die Eingabe zu normalisieren und aufzufüllen. Beim Auffüllen von Textdaten wird für kürzere Sequenzen ein `0` hinzugefügt. Die gleiche Idee gilt für Audiodaten, und der Audio-Feature-Extraktor fügt eine `0` - interpretiert als Stille - zu `array` hinzu.

Laden Sie den Merkmalsextraktor mit [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

Übergeben Sie das Audio-"Array" an den Feature-Extraktor. Wir empfehlen auch, das Argument `sampling_rate` im Feature Extractor hinzuzufügen, um eventuell auftretende stille Fehler besser zu beheben.

```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

### Auffüllen und Kürzen

Genau wie beim Tokenizer können Sie variable Sequenzen in einem Stapel durch Auffüllen oder Abschneiden behandeln. Werfen Sie einen Blick auf die Sequenzlänge dieser beiden Audiobeispiele:

```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

Wie Sie sehen können, hat das erste Beispiel eine längere Sequenz als das zweite Beispiel. Lassen Sie uns eine Funktion erstellen, die den Datensatz vorverarbeitet. Geben Sie eine maximale Länge der Probe an, und der Feature-Extraktor wird die Sequenzen entweder auffüllen oder abschneiden, damit sie dieser Länge entsprechen:

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

Wenden Sie die Funktion auf die ersten paar Beispiele im Datensatz an:

```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

Schauen Sie sich nun noch einmal die verarbeiteten Beispiel-Längen an:

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

Die Länge der ersten beiden Beispiele entspricht nun der von Ihnen angegebenen Maximallänge.

## Bildverarbeitung

Ein Merkmalsextraktor wird auch verwendet, um Bilder für Bildverarbeitungsaufgaben zu verarbeiten. Auch hier besteht das Ziel darin, das Rohbild in eine Reihe von Tensoren als Eingabe zu konvertieren.

Laden wir den [food101](https://huggingface.co/datasets/food101) Datensatz für dieses Tutorial. Verwenden Sie den Parameter 🤗 Datasets `split`, um nur eine kleine Stichprobe aus dem Trainingssplit zu laden, da der Datensatz recht groß ist:

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

Als Nächstes sehen Sie sich das Bild mit dem Merkmal 🤗 Datensätze [Bild](https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=image#datasets.Image) an:

```py
>>> dataset[0]["image"]
```

![vision-preprocess-tutorial.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png)

### Merkmalsextraktor

Laden Sie den Merkmalsextraktor mit [`AutoImageProcessor.from_pretrained`]:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

### Datenerweiterung

Bei Bildverarbeitungsaufgaben ist es üblich, den Bildern als Teil der Vorverarbeitung eine Art von Datenerweiterung hinzuzufügen. Sie können Erweiterungen mit jeder beliebigen Bibliothek hinzufügen, aber in diesem Tutorial werden Sie das Modul [`transforms`](https://pytorch.org/vision/stable/transforms.html) von torchvision verwenden.

1. Normalisieren Sie das Bild und verwenden Sie [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html), um einige Transformationen - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) und [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) - miteinander zu verknüpfen:

```py
>>> from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> _transforms = Compose(
...     [RandomResizedCrop(image_processor.size["height"]), ColorJitter(brightness=0.5, hue=0.5), ToTensor(), normalize]
... )
```

2. Das Modell akzeptiert [`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values) als Eingabe. Dieser Wert wird vom Merkmalsextraktor erzeugt. Erstellen Sie eine Funktion, die `pixel_values` aus den Transformationen erzeugt:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(image.convert("RGB")) for image in examples["image"]]
...     return examples
```

3. Dann verwenden Sie 🤗 Datasets [`set_transform`](https://huggingface.co/docs/datasets/process#format-transform), um die Transformationen im laufenden Betrieb anzuwenden:

```py
>>> dataset.set_transform(transforms)
```

4. Wenn Sie nun auf das Bild zugreifen, werden Sie feststellen, dass der Feature Extractor die Modelleingabe "pixel_values" hinzugefügt hat:

```py
>>> dataset[0]["image"]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=384x512 at 0x7F1A7B0630D0>,
 'label': 6,
 'pixel_values': tensor([[[ 0.0353,  0.0745,  0.1216,  ..., -0.9922, -0.9922, -0.9922],
          [-0.0196,  0.0667,  0.1294,  ..., -0.9765, -0.9843, -0.9922],
          [ 0.0196,  0.0824,  0.1137,  ..., -0.9765, -0.9686, -0.8667],
          ...,
          [ 0.0275,  0.0745,  0.0510,  ..., -0.1137, -0.1216, -0.0824],
          [ 0.0667,  0.0824,  0.0667,  ..., -0.0588, -0.0745, -0.0980],
          [ 0.0353,  0.0353,  0.0431,  ..., -0.0039, -0.0039, -0.0588]],
 
         [[ 0.2078,  0.2471,  0.2863,  ..., -0.9451, -0.9373, -0.9451],
          [ 0.1608,  0.2471,  0.3098,  ..., -0.9373, -0.9451, -0.9373],
          [ 0.2078,  0.2706,  0.3020,  ..., -0.9608, -0.9373, -0.8275],
          ...,
          [-0.0353,  0.0118, -0.0039,  ..., -0.2392, -0.2471, -0.2078],
          [ 0.0196,  0.0353,  0.0196,  ..., -0.1843, -0.2000, -0.2235],
          [-0.0118, -0.0039, -0.0039,  ..., -0.0980, -0.0980, -0.1529]],
 
         [[ 0.3961,  0.4431,  0.4980,  ..., -0.9216, -0.9137, -0.9216],
          [ 0.3569,  0.4510,  0.5216,  ..., -0.9059, -0.9137, -0.9137],
          [ 0.4118,  0.4745,  0.5216,  ..., -0.9137, -0.8902, -0.7804],
          ...,
          [-0.2314, -0.1922, -0.2078,  ..., -0.4196, -0.4275, -0.3882],
          [-0.1843, -0.1686, -0.2000,  ..., -0.3647, -0.3804, -0.4039],
          [-0.1922, -0.1922, -0.1922,  ..., -0.2941, -0.2863, -0.3412]]])}
```

Hier sehen Sie, wie das Bild nach der Vorverarbeitung aussieht. Wie von den angewandten Transformationen zu erwarten, wurde das Bild willkürlich beschnitten und seine Farbeigenschaften sind anders.

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

![preprocessed_image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png)

## Multimodal

Für multimodale Aufgaben werden Sie eine Kombination aus allem, was Sie bisher gelernt haben, verwenden und Ihre Fähigkeiten auf eine Aufgabe der automatischen Spracherkennung (ASR) anwenden. Dies bedeutet, dass Sie einen:

* Feature Extractor zur Vorverarbeitung der Audiodaten.
* Tokenizer, um den Text zu verarbeiten.

Kehren wir zum [LJ Speech](https://huggingface.co/datasets/lj_speech) Datensatz zurück:

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

Da Sie hauptsächlich an den Spalten "Audio" und "Text" interessiert sind, entfernen Sie die anderen Spalten:

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

Schauen Sie sich nun die Spalten "Audio" und "Text" an:

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

Erinnern Sie sich an den früheren Abschnitt über die Verarbeitung von Audiodaten: Sie sollten immer die Abtastrate Ihrer Audiodaten [resample](preprocessing#audio), damit sie mit der Abtastrate des Datensatzes übereinstimmt, der für das Vortraining eines Modells verwendet wird:

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

### Prozessor

Ein Processor kombiniert einen Feature-Extraktor und einen Tokenizer. Laden Sie einen Processor mit [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. Erstellen Sie eine Funktion, die die Audiodaten zu `input_values` verarbeitet und den Text zu `labels` tokenisiert. Dies sind Ihre Eingaben für das Modell:

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. Wenden Sie die Funktion "prepare_dataset" auf ein Beispiel an:

```py
>>> prepare_dataset(lj_speech[0])
```

Beachten Sie, dass der Processor `input_values` und `labels` hinzugefügt hat. Auch die Abtastrate wurde korrekt auf 16kHz heruntergerechnet.

Toll, Sie sollten jetzt in der Lage sein, Daten für jede Modalität vorzuverarbeiten und sogar verschiedene Modalitäten zu kombinieren! Im nächsten Kurs lernen Sie, wie Sie ein Modell mit Ihren neu aufbereiteten Daten feinabstimmen können.
