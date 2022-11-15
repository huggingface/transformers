<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pipelines für Inferenzen

Die [`pipeline`] macht es einfach, jedes beliebige Modell aus dem [Hub](https://huggingface.co/models) für die Inferenz auf jede Sprache, Computer Vision, Sprache und multimodale Aufgaben zu verwenden. Selbst wenn Sie keine Erfahrung mit einer bestimmten Modalität haben oder nicht mit dem zugrundeliegenden Code hinter den Modellen vertraut sind, können Sie sie mit der [`pipeline`] für Inferenzen verwenden! In diesem Beispiel lernen Sie, wie:

* Eine [`pipeline`] für Inferenz zu verwenden.
* Einen bestimmten Tokenizer oder ein bestimmtes Modell zu verwenden.
* Eine [`pipeline`] für Audio-, Vision- und multimodale Aufgaben zu verwenden.

<Tip>

Eine vollständige Liste der unterstützten Aufgaben und verfügbaren Parameter finden Sie in der [`pipeline`]-Dokumentation.

</Tip>

## Verwendung von Pipelines

Obwohl jede Aufgabe eine zugehörige [`pipeline`] hat, ist es einfacher, die allgemeine [`pipeline`]-Abstraktion zu verwenden, die alle aufgabenspezifischen Pipelines enthält. Die [`pipeline`] lädt automatisch ein Standardmodell und eine Vorverarbeitungsklasse, die für Ihre Aufgabe inferenzfähig ist.

1. Beginnen Sie mit der Erstellung einer [`pipeline`] und geben Sie eine Inferenzaufgabe an:

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="text-generation")
```

2. Übergeben Sie Ihren Eingabetext an die [`pipeline`]:

```py
>>> generator(
...     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
... )  # doctest: +SKIP
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Iron-priests at the door to the east, and thirteen for the Lord Kings at the end of the mountain'}]
```

Wenn Sie mehr als eine Eingabe haben, übergeben Sie die Eingabe als Liste:

```py
>>> generator(
...     [
...         "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
...         "Nine for Mortal Men, doomed to die, One for the Dark Lord on his dark throne",
...     ]
... )  # doctest: +SKIP
```

Alle zusätzlichen Parameter für Ihre Aufgabe können auch in die [`pipeline`] aufgenommen werden. Die Aufgabe `Text-Generierung` hat eine [`~generation.GenerationMixin.generate`]-Methode mit mehreren Parametern zur Steuerung der Ausgabe. Wenn Sie zum Beispiel mehr als eine Ausgabe erzeugen wollen, setzen Sie den Parameter `num_return_sequences`:

```py
>>> generator(
...     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
...     num_return_sequences=2,
... )  # doctest: +SKIP
```

### Wählen Sie ein Modell und einen Tokenizer

Die [`pipeline`] akzeptiert jedes Modell aus dem [Hub] (https://huggingface.co/models). Auf dem Hub gibt es Tags, mit denen Sie nach einem Modell filtern können, das Sie für Ihre Aufgabe verwenden möchten. Sobald Sie ein passendes Modell ausgewählt haben, laden Sie es mit der entsprechenden `AutoModelFor` und [`AutoTokenizer`] Klasse. Laden Sie zum Beispiel die Klasse [`AutoModelForCausalLM`] für eine kausale Sprachmodellierungsaufgabe:

```py
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

Erstellen Sie eine [`pipeline`] für Ihre Aufgabe, und geben Sie das Modell und den Tokenizer an, die Sie geladen haben:

```py
>>> from transformers import pipeline

>>> generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
```

Übergeben Sie Ihren Eingabetext an die [`pipeline`] , um einen Text zu erzeugen:

```py
>>> generator(
...     "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
... )  # doctest: +SKIP
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Dragon-lords (for them to rule in a world ruled by their rulers, and all who live within the realm'}]
```

## Audio-Pipeline

Die [`pipeline`] unterstützt auch Audioaufgaben wie Audioklassifizierung und automatische Spracherkennung.

Lassen Sie uns zum Beispiel die Emotion in diesem Audioclip klassifizieren:

```py
>>> from datasets import load_dataset
>>> import torch

>>> torch.manual_seed(42)  # doctest: +IGNORE_RESULT
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> audio_file = ds[0]["audio"]["path"]
```

Finden Sie ein [Audioklassifikation](https://huggingface.co/models?pipeline_tag=audio-classification) Modell auf dem Model Hub für Emotionserkennung und laden Sie es in die [`pipeline`]:

```py
>>> from transformers import pipeline

>>> audio_classifier = pipeline(
...     task="audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

Übergeben Sie die Audiodatei an die [`pipeline`]:

```py
>>> preds = audio_classifier(audio_file)
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.1315, 'label': 'calm'}, {'score': 0.1307, 'label': 'neutral'}, {'score': 0.1274, 'label': 'sad'}, {'score': 0.1261, 'label': 'fearful'}, {'score': 0.1242, 'label': 'happy'}]
```

## Bildverarbeitungs-Pipeline

Die Verwendung einer [`pipeline`] für Bildverarbeitungsaufgaben ist praktisch identisch.

Geben Sie Ihre Aufgabe an und übergeben Sie Ihr Bild an den Klassifikator. Das Bild kann ein Link oder ein lokaler Pfad zu dem Bild sein. Zum Beispiel: Welche Katzenart ist unten abgebildet?

![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

```py
>>> from transformers import pipeline

>>> vision_classifier = pipeline(task="image-classification")
>>> preds = vision_classifier(
...     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4335, 'label': 'lynx, catamount'}, {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}, {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}, {'score': 0.0239, 'label': 'Egyptian cat'}, {'score': 0.0229, 'label': 'tiger cat'}]
```

## Multimodale Pipeline

Die [`pipeline`] unterstützt mehr als eine Modalität. Eine Aufgabe zur Beantwortung visueller Fragen (VQA) kombiniert zum Beispiel Text und Bild. Verwenden Sie einen beliebigen Bildlink und eine Frage, die Sie zu dem Bild stellen möchten. Das Bild kann eine URL oder ein lokaler Pfad zu dem Bild sein.

Wenn Sie zum Beispiel das gleiche Bild wie in der obigen Vision-Pipeline verwenden:

```py
>>> image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
>>> question = "Where is the cat?"
```

Erstellen Sie eine Pipeline für "vqa" und übergeben Sie ihr das Bild und die Frage:

```py
>>> from transformers import pipeline

>>> vqa = pipeline(task="vqa")
>>> preds = vqa(image=image, question=question)
>>> preds = [{"score": round(pred["score"], 4), "answer": pred["answer"]} for pred in preds]
>>> preds
[{'score': 0.9112, 'answer': 'snow'}, {'score': 0.8796, 'answer': 'in snow'}, {'score': 0.6717, 'answer': 'outside'}, {'score': 0.0291, 'answer': 'on ground'}, {'score': 0.027, 'answer': 'ground'}]
```
