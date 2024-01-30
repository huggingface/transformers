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


# Generation with LLMs

[[open-in-colab]]

LLMs (Large Language Models) sind die Schlüsselkomponente bei der Texterstellung. Kurz gesagt, bestehen sie aus großen, vortrainierten Transformationsmodellen, die darauf trainiert sind, das nächste Wort (oder genauer gesagt Token) aus einem Eingabetext vorherzusagen. Da sie jeweils ein Token vorhersagen, müssen Sie etwas Aufwändigeres tun, um neue Sätze zu generieren, als nur das Modell aufzurufen - Sie müssen eine autoregressive Generierung durchführen.

Die autoregressive Generierung ist ein Verfahren zur Inferenzzeit, bei dem ein Modell mit seinen eigenen generierten Ausgaben iterativ aufgerufen wird, wenn einige anfängliche Eingaben vorliegen. In 🤗 Transformers wird dies von der Methode [`~generation.GenerationMixin.generate`] übernommen, die allen Modellen mit generativen Fähigkeiten zur Verfügung steht.

Dieses Tutorial zeigt Ihnen, wie Sie:

* Text mit einem LLM generieren
* Vermeiden Sie häufige Fallstricke
* Nächste Schritte, damit Sie das Beste aus Ihrem LLM herausholen können

Bevor Sie beginnen, stellen Sie sicher, dass Sie alle erforderlichen Bibliotheken installiert haben:

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```


## Text generieren

Ein Sprachmodell, das für [causal language modeling](tasks/language_modeling) trainiert wurde, nimmt eine Folge von Text-Token als Eingabe und gibt die Wahrscheinlichkeitsverteilung für das nächste Token zurück.

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
    <figcaption>"Forward pass of an LLM"</figcaption>
</figure>

Ein wichtiger Aspekt der autoregressiven Generierung mit LLMs ist die Auswahl des nächsten Tokens aus dieser Wahrscheinlichkeitsverteilung. In diesem Schritt ist alles möglich, solange Sie am Ende ein Token für die nächste Iteration haben. Das heißt, es kann so einfach sein wie die Auswahl des wahrscheinlichsten Tokens aus der Wahrscheinlichkeitsverteilung oder so komplex wie die Anwendung von einem Dutzend Transformationen vor der Stichprobenziehung aus der resultierenden Verteilung.

<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
    <figcaption>"Die autoregressive Generierung wählt iterativ das nächste Token aus einer Wahrscheinlichkeitsverteilung aus, um Text zu erzeugen"</figcaption>
</figure>

Der oben dargestellte Prozess wird iterativ wiederholt, bis eine bestimmte Abbruchbedingung erreicht ist. Im Idealfall wird die Abbruchbedingung vom Modell vorgegeben, das lernen sollte, wann es ein Ende-der-Sequenz-Token (EOS) ausgeben muss. Ist dies nicht der Fall, stoppt die Generierung, wenn eine vordefinierte Maximallänge erreicht ist.

Damit sich Ihr Modell so verhält, wie Sie es für Ihre Aufgabe erwarten, müssen Sie den Schritt der Token-Auswahl und die Abbruchbedingung richtig einstellen. Aus diesem Grund haben wir zu jedem Modell eine [`~generation.GenerationConfig`]-Datei, die eine gute generative Standardparametrisierung enthält und zusammen mit Ihrem Modell geladen wird.

Lassen Sie uns über Code sprechen!

<Tip>

Wenn Sie an der grundlegenden Verwendung von LLMs interessiert sind, ist unsere High-Level-Schnittstelle [`Pipeline`](pipeline_tutorial) ein guter Ausgangspunkt. LLMs erfordern jedoch oft fortgeschrittene Funktionen wie Quantisierung und Feinsteuerung des Token-Auswahlschritts, was am besten über [`~generation.GenerationMixin.generate`] erfolgt. Die autoregressive Generierung mit LLMs ist ebenfalls ressourcenintensiv und sollte für einen angemessenen Durchsatz auf einer GPU ausgeführt werden.

</Tip>

<!-- TODO: update example to llama 2 (or a newer popular baseline) when it becomes ungated -->
Zunächst müssen Sie das Modell laden.

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

Sie werden zwei Flags in dem Aufruf `from_pretrained` bemerken:

 - `device_map` stellt sicher, dass das Modell auf Ihre GPU(s) übertragen wird
 - `load_in_4bit` wendet [dynamische 4-Bit-Quantisierung](main_classes/quantization) an, um die Ressourcenanforderungen massiv zu reduzieren

Es gibt noch andere Möglichkeiten, ein Modell zu initialisieren, aber dies ist eine gute Grundlage, um mit einem LLM zu beginnen.

Als nächstes müssen Sie Ihre Texteingabe mit einem [tokenizer](tokenizer_summary) vorverarbeiten.

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

Die Variable `model_inputs` enthält die tokenisierte Texteingabe sowie die Aufmerksamkeitsmaske. Obwohl [`~generation.GenerationMixin.generate`] sein Bestes tut, um die Aufmerksamkeitsmaske abzuleiten, wenn sie nicht übergeben wird, empfehlen wir, sie für optimale Ergebnisse wann immer möglich zu übergeben.

Rufen Sie schließlich die Methode [~generation.GenerationMixin.generate] auf, um die generierten Token zurückzugeben, die vor dem Drucken in Text umgewandelt werden sollten.

```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, black, white, and brown'
```

Und das war's! Mit ein paar Zeilen Code können Sie sich die Macht eines LLM zunutze machen.


## Häufige Fallstricke

Es gibt viele [Generierungsstrategien](generation_strategies), und manchmal sind die Standardwerte für Ihren Anwendungsfall vielleicht nicht geeignet. Wenn Ihre Ausgaben nicht mit dem übereinstimmen, was Sie erwarten, haben wir eine Liste der häufigsten Fallstricke erstellt und wie Sie diese vermeiden können.

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

### Generierte Ausgabe ist zu kurz/lang

Wenn in der Datei [~generation.GenerationConfig`] nichts angegeben ist, gibt `generate` standardmäßig bis zu 20 Token zurück. Wir empfehlen dringend, `max_new_tokens` in Ihrem `generate`-Aufruf manuell zu setzen, um die maximale Anzahl neuer Token zu kontrollieren, die zurückgegeben werden können. Beachten Sie, dass LLMs (genauer gesagt, [decoder-only models](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)) auch die Eingabeaufforderung als Teil der Ausgabe zurückgeben.


```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### Falscher Generierungsmodus

Standardmäßig und sofern nicht in der Datei [~generation.GenerationConfig`] angegeben, wählt `generate` bei jeder Iteration das wahrscheinlichste Token aus (gierige Dekodierung). Je nach Aufgabe kann dies unerwünscht sein; kreative Aufgaben wie Chatbots oder das Schreiben eines Aufsatzes profitieren vom Sampling. Andererseits profitieren Aufgaben, bei denen es auf die Eingabe ankommt, wie z.B. Audiotranskription oder Übersetzung, von der gierigen Dekodierung. Aktivieren Sie das Sampling mit `do_sample=True`. Mehr zu diesem Thema erfahren Sie in diesem [Blogbeitrag] (https://huggingface.co/blog/how-to-generate).

```py
>>> # Set seed or reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(0)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.\nI just need to be. I am always.\nEvery time'
```

### Falsche Auffüllseite

LLMs sind [decoder-only](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt)-Architekturen, d.h. sie iterieren weiter über Ihre Eingabeaufforderung. Wenn Ihre Eingaben nicht die gleiche Länge haben, müssen sie aufgefüllt werden. Da LLMs nicht darauf trainiert sind, mit aufgefüllten Token fortzufahren, muss Ihre Eingabe links aufgefüllt werden. Vergessen Sie auch nicht, die Aufmerksamkeitsmaske an generate zu übergeben!

```py
>>> # The tokenizer initialized above has right-padding active by default: the 1st sequence,
>>> # which is shorter, has padding on the right side. Generation fails.
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
''

>>> # With left-padding, it works as expected!
>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", padding_side="left")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

<!-- TODO: when the prompting guide is ready, mention the importance of setting the right prompt in this section -->

## Weitere Ressourcen

Während der Prozess der autoregressiven Generierung relativ einfach ist, kann die optimale Nutzung Ihres LLM ein schwieriges Unterfangen sein, da es viele bewegliche Teile gibt. Für Ihre nächsten Schritte, die Ihnen helfen, tiefer in die LLM-Nutzung und das Verständnis einzutauchen:

<!-- TODO: mit neuen Anleitungen vervollständigen -->
### Fortgeschrittene Nutzung generieren

1. [Leitfaden](generation_strategies) zur Steuerung verschiedener Generierungsmethoden, zur Einrichtung der Generierungskonfigurationsdatei und zum Streaming der Ausgabe;
2. API-Referenz zu [`~generation.GenerationConfig`], [`~generation.GenerationMixin.generate`] und [generate-bezogene Klassen](internal/generation_utils).

### LLM-Ranglisten

1. [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), das sich auf die Qualität der Open-Source-Modelle konzentriert;
2. [Open LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard), das sich auf den LLM-Durchsatz konzentriert.

### Latenz und Durchsatz

1. [Leitfaden](main_classes/quantization) zur dynamischen Quantisierung, der Ihnen zeigt, wie Sie Ihren Speicherbedarf drastisch reduzieren können.

### Verwandte Bibliotheken

1. [text-generation-inference](https://github.com/huggingface/text-generation-inference), ein produktionsreifer Server für LLMs;
2. [`optimum`](https://github.com/huggingface/optimum), eine Erweiterung von 🤗 Transformers, die für bestimmte Hardware-Geräte optimiert.
