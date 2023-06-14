<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Carica istanze pre-allenate con AutoClass

Con così tante architetture Transformer differenti, può essere sfidante crearne una per il tuo checkpoint. Come parte della filosofia centrale di 🤗 Transformers per rendere la libreria facile, semplice e flessibile da utilizzare, una `AutoClass` inferisce e carica automaticamente l'architettura corretta da un dato checkpoint. Il metodo `from_pretrained` ti permette di caricare velocemente un modello pre-allenato per qualsiasi architettura, così non devi utilizzare tempo e risorse per allenare un modello da zero. Produrre questo codice agnostico ai checkpoint significa che se il tuo codice funziona per un checkpoint, funzionerà anche per un altro checkpoint, purché sia stato allenato per un compito simile, anche se l'architettura è differente.

<Tip>

Ricorda, con architettura ci si riferisce allo scheletro del modello e con checkpoint ai pesi di una determinata architettura. Per esempio, [BERT](https://huggingface.co/bert-base-uncased) è un'architettura, mentre `bert-base-uncased` è un checkpoint. Modello è un termine generale che può significare sia architettura che checkpoint.

</Tip>

In questo tutorial, imparerai a:

* Caricare un tokenizer pre-allenato.
* Caricare un estrattore di caratteristiche (feature extractor, in inglese) pre-allenato.
* Caricare un processore pre-allenato.
* Caricare un modello pre-allenato.

## AutoTokenizer

Quasi tutti i compiti di NLP iniziano con un tokenizer. Un tokenizer converte il tuo input in un formato che possa essere elaborato dal modello.

Carica un tokenizer con [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
```

Poi tokenizza il tuo input come mostrato in seguito:

```py
>>> sequenza = "In un buco nel terreno viveva uno Hobbit."
>>> print(tokenizer(sequenza))
{'input_ids': [0, 360, 51, 373, 587, 1718, 54644, 22597, 330, 3269, 2291, 22155, 18, 5, 2],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoFeatureExtractor

Per compiti inerenti a audio e video, un feature extractor processa il segnale audio o l'immagine nel formato di input corretto.

Carica un feature extractor con [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

Compiti multimodali richiedono un processore che combini i due tipi di strumenti di elaborazione. Per esempio, il modello [LayoutLMV2](model_doc/layoutlmv2) richiede un feature extractor per gestire le immagine e un tokenizer per gestire il testo; un processore li combina entrambi.

Carica un processore con [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

<frameworkcontent>
<pt>
Infine, le classi `AutoModelFor` ti permettono di caricare un modello pre-allenato per un determinato compito (guarda [qui](model_doc/auto) per una lista completa di compiti presenti). Per esempio, carica un modello per la classificazione di sequenze con [`AutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

Semplicemente utilizza lo stesso checkpoint per caricare un'architettura per un task differente:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

Generalmente, raccomandiamo di utilizzare la classe `AutoTokenizer` e la classe `AutoModelFor` per caricare istanze pre-allenate dei modelli. Questo ti assicurerà di aver caricato la corretta architettura ogni volta. Nel prossimo [tutorial](preprocessing), imparerai come utilizzare il tokenizer, il feature extractor e il processore per elaborare un dataset per il fine-tuning.

</pt>
<tf>
Infine, le classi `TFAutoModelFor` ti permettono di caricare un modello pre-allenato per un determinato compito (guarda [qui](model_doc/auto) per una lista completa di compiti presenti). Per esempio, carica un modello per la classificazione di sequenze con [`TFAutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```

Semplicemente utilizza lo stesso checkpoint per caricare un'architettura per un task differente:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
```

Generalmente, raccomandiamo di utilizzare la classe `AutoTokenizer` e la classe `TFAutoModelFor` per caricare istanze pre-allenate dei modelli. Questo ti assicurerà di aver caricato la corretta architettura ogni volta. Nel prossimo [tutorial](preprocessing), imparerai come utilizzare il tokenizer, il feature extractor e il processore per elaborare un dataset per il fine-tuning.
</tf>
</frameworkcontent>
