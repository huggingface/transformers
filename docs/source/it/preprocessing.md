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

# Preprocess

[[open-in-colab]]

Prima di poter usare i dati in un modello, bisogna processarli in un formato accettabile per quest'ultimo. Un modello non comprende il testo grezzo, le immagini o l'audio. Bisogna convertire questi input in numeri e assemblarli all'interno di tensori. In questa esercitazione, tu potrai:

* Preprocessare dati testuali con un tokenizer.
* Preprocessare immagini o dati audio con un estrattore di caratteristiche.
* Preprocessare dati per attività multimodali mediante un processore.

## NLP

<Youtube id="Yffk5aydLzg"/>

Lo strumento principale per processare dati testuali è un [tokenizer](main_classes/tokenizer). Un tokenizer inizia separando il testo in *tokens* secondo una serie di regole. I tokens sono convertiti in numeri, questi vengono utilizzati per costruire i tensori di input del modello. Anche altri input addizionali se richiesti dal modello vengono aggiunti dal tokenizer.

<Tip>

Se stai pensando si utilizzare un modello preaddestrato, è importante utilizzare il tokenizer preaddestrato associato. Questo assicura che il testo sia separato allo stesso modo che nel corpus usato per l'addestramento, e venga usata la stessa mappatura tokens-to-index (solitamente indicato come il *vocabolario*) come nel preaddestramento.

</Tip>

Iniziamo subito caricando un tokenizer preaddestrato con la classe [`AutoTokenizer`]. Questo scarica il *vocabolario* usato quando il modello è stato preaddestrato.

### Tokenize

Carica un tokenizer preaddestrato con [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

Poi inserisci le tue frasi nel tokenizer:

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Il tokenizer restituisce un dizionario contenente tre oggetti importanti:

* [input_ids](glossary#input-ids) sono gli indici che corrispondono ad ogni token nella frase.
* [attention_mask](glossary#attention-mask) indicata se un token deve essere elaborato o no.
* [token_type_ids](glossary#token-type-ids) identifica a quale sequenza appartiene un token se è presente più di una sequenza.

Si possono decodificare gli `input_ids` per farsi restituire l'input originale:

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

Come si può vedere, il tokenizer aggiunge due token speciali - `CLS` e `SEP` (classificatore e separatore) - alla frase. Non tutti i modelli hanno bisogno dei token speciali, ma se servono, il tokenizer li aggiungerà automaticamente.

Se ci sono più frasi che vuoi processare, passale come una lista al tokenizer:

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

Questo è un argomento importante. Quando processi un insieme di frasi potrebbero non avere tutte la stessa lunghezza. Questo è un problema perchè i tensori, in input del modello, devono avere dimensioni uniformi. Il padding è una strategia per assicurarsi che i tensori siano rettangolari aggiungendo uno speciale *padding token* alle frasi più corte.

Imposta il parametro `padding` a `True` per imbottire le frasi più corte nel gruppo in modo che combacino con la massima lunghezza presente:

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

Nota che il tokenizer aggiunge alle sequenze degli `0` perchè sono troppo corte!

### Truncation

L'altra faccia della medaglia è che avolte le sequenze possono essere troppo lunghe per essere gestite dal modello. In questo caso, avrai bisogno di troncare la sequenza per avere una lunghezza minore.

Imposta il parametro `truncation` a `True` per troncare una sequenza alla massima lunghezza accettata dal modello:

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

### Costruire i tensori

Infine, vuoi che il tokenizer restituisca i tensori prodotti dal modello.

Imposta il parametro `return_tensors` su `pt` per PyTorch, o `tf` per TensorFlow:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[  101,   153,  7719, 21490,  1122,  1114,  9582,  1623,   102],
                      [  101,  5226,  1122,  9649,  1199,  2610,  1236,   102,     0]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
===PT-TF-SPLIT===
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[  101,   153,  7719, 21490,  1122,  1114,  9582,  1623,   102],
       [  101,  5226,  1122,  9649,  1199,  2610,  1236,   102,     0]],
      dtype=int32)>, 
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>, 
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=int32)>}
```

## Audio

Gli input audio sono processati in modo differente rispetto al testo, ma l'obiettivo rimane lo stesso: creare sequenze numeriche che il modello può capire. Un [estrattore di caratteristiche](main_classes/feature_extractor) è progettato con lo scopo preciso di estrarre caratteristiche da immagini o dati audio grezzi e convertirli in tensori. Prima di iniziare, installa 🤗 Datasets per caricare un dataset audio e sperimentare:

```bash
pip install datasets
```

Carica il dataset [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) (vedi il 🤗 [Datasets tutorial](https://huggingface.co/docs/datasets/load_hub) per avere maggiori dettagli su come caricare un dataset):

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

Accedi al primo elemento della colonna `audio` per dare uno sguardo all'input. Richiamando la colonna `audio` sarà caricato automaticamente e ricampionato il file audio:

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

Questo restituisce tre oggetti:

* `array` è il segnale vocale caricato - e potenzialmente ricampionato - come vettore 1D.
* `path` il percorso del file audio.
* `sampling_rate` si riferisce al numero di campioni del segnale vocale misurati al secondo.

### Ricampionamento

Per questo tutorial, puoi usare il modello [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base). Come puoi vedere dalla model card, il modello Wav2Vec2 è preaddestrato su un campionamento vocale a 16kHz.È importante che la frequenza di campionamento dei tuoi dati audio combaci con la frequenza di campionamento del dataset usato per preaddestrare il modello. Se la frequenza di campionamento dei tuoi dati non è uguale dovrai ricampionare i tuoi dati audio.

Per esempio, il dataset [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) ha una frequenza di campionamento di 8000kHz. Utilizzando il modello Wav2Vec2 su questo dataset, alzala a 16kHz:

```py
>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

1. Usa il metodo di 🤗 Datasets' [`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.cast_column) per alzare la frequenza di campionamento a 16kHz:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. Carica il file audio:

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

Come puoi notare, la `sampling_rate` adesso è 16kHz!

### Feature extractor

Il prossimo passo è caricare un estrattore di caratteristiche per normalizzare e fare padding sull'input. Quando applichiamo il padding sui dati testuali, uno `0` è aggiunto alle sequenze più brevi. La stessa idea si applica ai dati audio, l'estrattore di caratteristiche per gli audio aggiungerà uno `0` - interpretato come silenzio - agli `array`.

Carica l'estrattore delle caratteristiche con [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

Inserisci l' `array` audio nell'estrattore delle caratteristiche. Noi raccomandiamo sempre di aggiungere il parametro `sampling_rate` nell'estrattore delle caratteristiche per correggere meglio qualche errore, dovuto ai silenzi, che potrebbe verificarsi.

```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

### Pad e truncate

Come per il tokenizer, puoi applicare le operazioni padding o truncation per manipolare sequenze di variabili a lotti. Dai uno sguaro alla lunghezza delle sequenze di questi due campioni audio:

```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

Come puoi vedere, il primo campione ha una sequenza più lunga del secondo. Crea una funzione che preprocesserà il dataset. Specifica una lunghezza massima del campione, e l'estrattore di features si occuperà di riempire o troncare la sequenza per coincidervi:

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

Applica la funzione ai primi esempi nel dataset:

```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

Adesso guarda la lunghezza dei campioni elaborati:

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

La lunghezza dei campioni adesso coincide con la massima lunghezza impostata nelle funzione.

## Vision

Un estrattore di caratteristiche si può usare anche per processare immagini e per compiti di visione. Ancora una volta, l'obiettivo è convertire l'immagine grezza in un lotto di tensori come input.

Carica il dataset [food101](https://huggingface.co/datasets/food101) per questa esercitazione. Usa il parametro `split` di 🤗 Datasets  per caricare solo un piccolo campione dal dataset di addestramento poichè il set di dati è molto grande:

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

Secondo passo, dai uno sguardo alle immagini usando la caratteristica [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=image#datasets.Image) di 🤗 Datasets:

```py
>>> dataset[0]["image"]
```

![vision-preprocess-tutorial.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png)

### Feature extractor

Carica l'estrattore di caratteristiche [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
```

### Data augmentation

Per le attività di visione, è usuale aggiungere alcuni tipi di data augmentation alle immagini come parte del preprocessing. Puoi aggiungere augmentations con qualsiasi libreria che preferisci, ma in questa esercitazione, userai il modulo [`transforms`](https://pytorch.org/vision/stable/transforms.html) di torchvision.

1. Normalizza l'immagine e usa [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) per concatenare alcune trasformazioni - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) e [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) - insieme:

```py
>>> from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor

>>> normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
>>> _transforms = Compose(
...     [RandomResizedCrop(feature_extractor.size), ColorJitter(brightness=0.5, hue=0.5), ToTensor(), normalize]
... )
```

2. Il modello accetta [`pixel_values`](model_doc/visionencoderdecoder#transformers.VisionEncoderDecoderModel.forward.pixel_values) come input. Questo valore è generato dall'estrattore di caratteristiche. Crea una funzione che genera `pixel_values` dai transforms:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(image.convert("RGB")) for image in examples["image"]]
...     return examples
```

3. Poi utilizza 🤗 Datasets [`set_transform`](https://huggingface.co/docs/datasets/process#format-transform)per applicare al volo la trasformazione:

```py
>>> dataset.set_transform(transforms)
```

4. Adesso quando accedi all'immagine, puoi notare che l'estrattore di caratteristiche ha aggiunto `pixel_values` allo schema di input:

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

Di seguito come si vede l'immagine dopo la fase di preprocessing. Come ci si aspetterebbe dalle trasformazioni applicate, l'immagine è stata ritagliata in modo casuale e le proprietà del colore sono diverse.

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

![preprocessed_image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png)

## Multimodal

Per attività multimodali userai una combinazione di tutto quello che hai imparato poco fa e applicherai le tue competenze alla comprensione automatica del parlato (Automatic Speech Recognition -  ASR). Questo significa che avrai bisogno di:

* Un estrattore delle caratteristiche per processare i dati audio.
* Il Tokenizer per processare i testi.

Ritorna sul datasere [LJ Speech](https://huggingface.co/datasets/lj_speech):

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

Visto che sei interessato solo alle colonne `audio` e `text`, elimina tutte le altre:

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

Adesso guarda le colonne `audio` e `text`:

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

Ricorda dalla sezione precedente sull'elaborazione dei dati audio, tu dovresti sempre [ricampionare](preprocessing#audio) la frequenza di campionamento dei tuoi dati audio per farla coincidere con quella del dataset usato dal modello preaddestrato:

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

### Processor

Un processor combina un estrattore di caratteristiche e un tokenizer. Carica un processor con [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. Crea una funzione che processi i dati audio in `input_values`, e tokenizza il testo in `labels`. Questi sono i tuoi input per il modello:

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. Applica la funzione `prepare_dataset` ad un campione:

```py
>>> prepare_dataset(lj_speech[0])
```

Nota che il processor ha aggiunto `input_values` e `labels`. La frequenza di campionamento è stata corretta riducendola a 16kHz.

Fantastico, ora dovresti essere in grado di preelaborare i dati per qualsiasi modalità e persino di combinare modalità diverse! Nella prossima esercitazione, impareremo a mettere a punto un modello sui dati appena pre-elaborati.