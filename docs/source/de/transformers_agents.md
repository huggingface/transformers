<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Transformers Agents

<Tip warning={true}>

Transformers Agents ist eine experimentelle API, die jederzeit geändert werden kann. Die von den Agenten zurückgegebenen Ergebnisse
zurückgegeben werden, können variieren, da sich die APIs oder die zugrunde liegenden Modelle ändern können.

</Tip>

Transformers Version v4.29.0, die auf dem Konzept von *Tools* und *Agenten* aufbaut. Sie können damit spielen in
[dieses Colab](https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj).

Kurz gesagt, es bietet eine API für natürliche Sprache auf der Grundlage von Transformers: Wir definieren eine Reihe von kuratierten Tools und entwerfen einen 
Agenten, um natürliche Sprache zu interpretieren und diese Werkzeuge zu verwenden. Es ist von vornherein erweiterbar; wir haben einige relevante Tools kuratiert, 
aber wir werden Ihnen zeigen, wie das System einfach erweitert werden kann, um jedes von der Community entwickelte Tool zu verwenden.

Beginnen wir mit einigen Beispielen dafür, was mit dieser neuen API erreicht werden kann. Sie ist besonders leistungsfähig, wenn es um 
Sie ist besonders leistungsstark, wenn es um multimodale Aufgaben geht. Lassen Sie uns also eine Runde drehen, um Bilder zu erzeugen und Text vorzulesen.

```py
agent.run("Caption the following image", image=image)
```

| **Input**                                                                                                                   | **Output**                        |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png" width=200> | A beaver is swimming in the water |

---

```py
agent.run("Read the following text out loud", text=text)
```
| **Input**                                                                                                               | **Output**                                   |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| A beaver is swimming in the water | <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tts_example.wav" type="audio/wav"> your browser does not support the audio element. </audio>

---

```py
agent.run(
    "In the following `document`, where will the TRRF Scientific Advisory Council Meeting take place?",
    document=document,
)
```
| **Input**                                                                                                                   | **Output**     |
|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| <img src="https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/0/image/image.jpg" width=200> | ballroom foyer |

## Schnellstart

Bevor Sie `agent.run` verwenden können, müssen Sie einen Agenten instanziieren, der ein großes Sprachmodell (LLM) ist. 
Wir bieten Unterstützung für openAI-Modelle sowie für OpenSource-Alternativen von BigCode und OpenAssistant. Die openAI
Modelle sind leistungsfähiger (erfordern aber einen openAI-API-Schlüssel, können also nicht kostenlos verwendet werden); Hugging Face
bietet kostenlosen Zugang zu Endpunkten für BigCode- und OpenAssistant-Modelle.

To start with, please install the `agents` extras in order to install all default dependencies.
```bash
pip install transformers[agents]
```

Um openAI-Modelle zu verwenden, instanziieren Sie einen [`OpenAiAgent`], nachdem Sie die `openai`-Abhängigkeit installiert haben:

```bash
pip install openai
```


```py
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key="<your_api_key>")
```

Um BigCode oder OpenAssistant zu verwenden, melden Sie sich zunächst an, um Zugriff auf die Inference API zu erhalten:

```py
from huggingface_hub import login

login("<YOUR_TOKEN>")
```

Dann instanziieren Sie den Agenten

```py
from transformers import HfAgent

# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
```

Dies geschieht mit der Inferenz-API, die Hugging Face derzeit kostenlos zur Verfügung stellt. Wenn Sie Ihren eigenen Inferenz
Endpunkt für dieses Modell (oder einen anderen) haben, können Sie die obige URL durch Ihren URL-Endpunkt ersetzen.

<Tip>

StarCoder und OpenAssistant sind kostenlos und leisten bei einfachen Aufgaben bewundernswert gute Arbeit. Allerdings halten die Kontrollpunkte
nicht, wenn es um komplexere Aufforderungen geht. Wenn Sie mit einem solchen Problem konfrontiert sind, empfehlen wir Ihnen, das OpenAI
Modell auszuprobieren, das zwar leider nicht quelloffen ist, aber zur Zeit eine bessere Leistung erbringt.

</Tip>

Sie sind jetzt startklar! Lassen Sie uns in die beiden APIs eintauchen, die Ihnen jetzt zur Verfügung stehen.

### Einzelne Ausführung (run)

Die Methode der einmaligen Ausführung ist die Verwendung der [`~Agent.run`] Methode des Agenten:

```py
agent.run("Draw me a picture of rivers and lakes.")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200>

Es wählt automatisch das (oder die) Werkzeug(e) aus, das (die) für die von Ihnen gewünschte Aufgabe geeignet ist (sind) und führt es (sie) entsprechend aus. Es
kann eine oder mehrere Aufgaben in der gleichen Anweisung ausführen (je komplexer Ihre Anweisung ist, desto wahrscheinlicher ist ein
der Agent scheitern).

```py
agent.run("Draw me a picture of the sea then transform the picture to add an island")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sea_and_island.png" width=200>

<br/>


Jede [`~Agent.run`] Operation ist unabhängig, so dass Sie sie mehrmals hintereinander mit unterschiedlichen Aufgaben ausführen können.

Beachten Sie, dass Ihr `Agent` nur ein großsprachiges Modell ist, so dass kleine Variationen in Ihrer Eingabeaufforderung völlig unterschiedliche Ergebnisse liefern können.
unterschiedliche Ergebnisse liefern. Es ist wichtig, dass Sie die Aufgabe, die Sie ausführen möchten, so genau wie möglich erklären. Wir gehen noch weiter ins Detail
wie man gute Prompts schreibt [hier](custom_tools#writing-good-user-inputs).

Wenn Sie einen Status über Ausführungszeiten hinweg beibehalten oder dem Agenten Nicht-Text-Objekte übergeben möchten, können Sie dies tun, indem Sie
Variablen, die der Agent verwenden soll. Sie könnten zum Beispiel das erste Bild von Flüssen und Seen erzeugen, 
und das Modell bitten, dieses Bild zu aktualisieren und eine Insel hinzuzufügen, indem Sie Folgendes tun:

```python
picture = agent.run("Generate a picture of rivers and lakes.")
updated_picture = agent.run("Transform the image in `picture` to add an island to it.", picture=picture)
```

<Tip>

Dies kann hilfreich sein, wenn das Modell Ihre Anfrage nicht verstehen kann und die Werkzeuge verwechselt. Ein Beispiel wäre:

```py
agent.run("Draw me the picture of a capybara swimming in the sea")
```

Hier könnte das Modell auf zwei Arten interpretieren:
- Die Funktion `Text-zu-Bild` erzeugt ein Wasserschwein, das im Meer schwimmt.
- Oder Sie lassen das `Text-zu-Bild` ein Wasserschwein erzeugen und verwenden dann das Werkzeug `Bildtransformation`, um es im Meer schwimmen zu lassen.

Falls Sie das erste Szenario erzwingen möchten, können Sie dies tun, indem Sie die Eingabeaufforderung als Argument übergeben:

```py
agent.run("Draw me a picture of the `prompt`", prompt="a capybara swimming in the sea")
```

</Tip>


### Chat-basierte Ausführung (Chat)

Der Agent verfügt auch über einen Chat-basierten Ansatz, der die Methode [`~Agent.chat`] verwendet:

```py
agent.chat("Generate a picture of rivers and lakes")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

```py
agent.chat("Transform the picture so that there is a rock in there")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_and_beaver.png" width=200>

<br/>

Dies ist ein interessanter Ansatz, wenn Sie den Zustand über Anweisungen hinweg beibehalten möchten. Er ist besser für Experimente geeignet, 
eignet sich aber eher für einzelne Anweisungen als für komplexe Anweisungen (die die [`~Agent.run`]
Methode besser verarbeiten kann).

Diese Methode kann auch Argumente entgegennehmen, wenn Sie Nicht-Text-Typen oder bestimmte Aufforderungen übergeben möchten.

### ⚠️ Fernausführung

Zu Demonstrationszwecken und damit es mit allen Setups verwendet werden kann, haben wir Remote-Executors für mehrere 
der Standard-Tools erstellt, auf die der Agent in dieser Version Zugriff hat. Diese werden erstellt mit 
[inference endpoints](https://huggingface.co/inference-endpoints).

Wir haben diese vorerst deaktiviert, aber um zu sehen, wie Sie selbst Remote Executors Tools einrichten können,
empfehlen wir die Lektüre des [custom tool guide](./custom_tools).

### Was passiert hier? Was sind Tools und was sind Agenten?

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/diagram.png">

#### Agenten

Der "Agent" ist hier ein großes Sprachmodell, das wir auffordern, Zugang zu einem bestimmten Satz von Tools zu erhalten.

LLMs sind ziemlich gut darin, kleine Codeproben zu erzeugen. Diese API macht sich das zunutze, indem sie das 
LLM ein kleines Codebeispiel gibt, das eine Aufgabe mit einer Reihe von Werkzeugen ausführt. Diese Aufforderung wird dann ergänzt durch die 
Aufgabe, die Sie Ihrem Agenten geben, und die Beschreibung der Werkzeuge, die Sie ihm geben. Auf diese Weise erhält er Zugriff auf die Dokumentation der 
Tools, insbesondere die erwarteten Eingaben und Ausgaben, und kann den entsprechenden Code generieren.

#### Tools

Tools sind sehr einfach: Sie bestehen aus einer einzigen Funktion mit einem Namen und einer Beschreibung. Wir verwenden dann die Beschreibungen dieser Tools 
um den Agenten aufzufordern. Anhand der Eingabeaufforderung zeigen wir dem Agenten, wie er die Tools nutzen kann, um das zu tun, was in der 
in der Abfrage angefordert wurde.

Dies geschieht mit brandneuen Tools und nicht mit Pipelines, denn der Agent schreibt besseren Code mit sehr atomaren Tools. 
Pipelines sind stärker refaktorisiert und fassen oft mehrere Aufgaben in einer einzigen zusammen. Tools sind dafür gedacht, sich auf
eine einzige, sehr einfache Aufgabe konzentrieren.

#### Code-Ausführung?!

Dieser Code wird dann mit unserem kleinen Python-Interpreter auf den mit Ihren Tools übergebenen Eingaben ausgeführt. 
Wir hören Sie schon schreien "Willkürliche Codeausführung!", aber lassen Sie uns erklären, warum das nicht der Fall ist.

Die einzigen Funktionen, die aufgerufen werden können, sind die von Ihnen zur Verfügung gestellten Tools und die Druckfunktion, so dass Sie bereits eingeschränkt sind 
eingeschränkt, was ausgeführt werden kann. Sie sollten sicher sein, wenn es sich auf die Werkzeuge für das Umarmungsgesicht beschränkt. 

Dann lassen wir keine Attributsuche oder Importe zu (die ohnehin nicht benötigt werden, um die 
Inputs/Outputs an eine kleine Gruppe von Funktionen), so dass alle offensichtlichen Angriffe (und Sie müssten den LLM 
dazu auffordern, sie auszugeben) kein Problem darstellen sollten. Wenn Sie auf Nummer sicher gehen wollen, können Sie die 
run()-Methode mit dem zusätzlichen Argument return_code=True ausführen. In diesem Fall gibt der Agent nur den auszuführenden Code 
zur Ausführung zurück und Sie können entscheiden, ob Sie ihn ausführen möchten oder nicht.

Die Ausführung bricht bei jeder Zeile ab, in der versucht wird, eine illegale Operation auszuführen, oder wenn ein regulärer Python-Fehler 
mit dem vom Agenten generierten Code.

### Ein kuratierter Satz von Tools

Wir haben eine Reihe von Tools identifiziert, die solche Agenten unterstützen können. Hier ist eine aktualisierte Liste der Tools, die wir integriert haben 
in `transformers` integriert haben:

- **Beantwortung von Fragen zu Dokumenten**: Beantworten Sie anhand eines Dokuments (z.B. PDF) im Bildformat eine Frage zu diesem Dokument ([Donut](./model_doc/donut))
- Beantworten von Textfragen**: Geben Sie einen langen Text und eine Frage an, beantworten Sie die Frage im Text ([Flan-T5](./model_doc/flan-t5))
- **Unbedingte Bildunterschriften**: Beschriften Sie das Bild! ([BLIP](./model_doc/blip))
- **Bildfragebeantwortung**: Beantworten Sie bei einem Bild eine Frage zu diesem Bild ([VILT](./model_doc/vilt))
- **Bildsegmentierung**: Geben Sie ein Bild und einen Prompt an und geben Sie die Segmentierungsmaske dieses Prompts aus ([CLIPSeg](./model_doc/clipseg))
- **Sprache in Text**: Geben Sie eine Audioaufnahme einer sprechenden Person an und transkribieren Sie die Sprache in Text ([Whisper](./model_doc/whisper))
- **Text in Sprache**: wandelt Text in Sprache um ([SpeechT5](./model_doc/speecht5))
- **Zero-Shot-Textklassifizierung**: Ermitteln Sie anhand eines Textes und einer Liste von Bezeichnungen, welcher Bezeichnung der Text am ehesten entspricht ([BART](./model_doc/bart))
- **Textzusammenfassung**: fassen Sie einen langen Text in einem oder wenigen Sätzen zusammen ([BART](./model_doc/bart))
- **Übersetzung**: Übersetzen des Textes in eine bestimmte Sprache ([NLLB](./model_doc/nllb))

Diese Tools sind in Transformatoren integriert und können auch manuell verwendet werden, zum Beispiel:

```py
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

### Benutzerdefinierte Tools

Wir haben zwar eine Reihe von Tools identifiziert, sind aber der festen Überzeugung, dass der Hauptwert dieser Implementierung darin besteht 
die Möglichkeit, benutzerdefinierte Tools schnell zu erstellen und weiterzugeben.

Indem Sie den Code eines Tools in einen Hugging Face Space oder ein Modell-Repository stellen, können Sie das Tool 
direkt mit dem Agenten nutzen. Wir haben ein paar neue Funktionen hinzugefügt 
**transformers-agnostic** Tools zur [`huggingface-tools` Organisation](https://huggingface.co/huggingface-tools) hinzugefügt:

- **Text-Downloader**: zum Herunterladen eines Textes von einer Web-URL
- **Text zu Bild**: erzeugt ein Bild nach einer Eingabeaufforderung und nutzt dabei stabile Diffusion
- **Bildtransformation**: verändert ein Bild anhand eines Ausgangsbildes und einer Eingabeaufforderung, unter Ausnutzung der stabilen pix2pix-Diffusion
- **Text zu Video**: Erzeugen eines kleinen Videos nach einer Eingabeaufforderung, unter Verwendung von damo-vilab

Das Text-zu-Bild-Tool, das wir von Anfang an verwendet haben, ist ein Remote-Tool, das sich in 
[*huggingface-tools/text-to-image*](https://huggingface.co/spaces/huggingface-tools/text-to-image)! Wir werden
weiterhin solche Tools für diese und andere Organisationen veröffentlichen, um diese Implementierung weiter zu verbessern.

Die Agenten haben standardmäßig Zugriff auf die Tools, die sich auf [*huggingface-tools*](https://huggingface.co/huggingface-tools) befinden.
Wie Sie Ihre eigenen Tools schreiben und freigeben können und wie Sie jedes benutzerdefinierte Tool, das sich auf dem Hub befindet, nutzen können, erklären wir in [folgender Anleitung](custom_tools).

### Code-Erzeugung

Bisher haben wir gezeigt, wie Sie die Agenten nutzen können, um Aktionen für Sie durchzuführen. Der Agent generiert jedoch nur Code
den wir dann mit einem sehr eingeschränkten Python-Interpreter ausführen. Falls Sie den generierten Code in einer anderen Umgebung verwenden möchten 
einer anderen Umgebung verwenden möchten, können Sie den Agenten auffordern, den Code zusammen mit einer Tooldefinition und genauen Importen zurückzugeben.

Zum Beispiel die folgende Anweisung
```python
agent.run("Draw me a picture of rivers and lakes", return_code=True)
```

gibt den folgenden Code zurück

```python
from transformers import load_tool

image_generator = load_tool("huggingface-tools/text-to-image")

image = image_generator(prompt="rivers and lakes")
```

die Sie dann selbst ändern und ausführen können.
