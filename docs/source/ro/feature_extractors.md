<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Feature extractors

Feature extractors preprocesează datele audio în formatul corect pentru un model dat. Preia semnalul audio brut și îl convertește într-un tensor care poate fi pasat unui model. Forma tensorului depinde de model, dar feature extractor-ul va preprocesa corect datele audio pentru modelul pe care îl folosești. Feature extractors includ și metode pentru padding, trunchiere și resampling.

Apelează [`~AutoFeatureExtractor.from_pretrained`] ca să încarci un feature extractor și configurația sa de preprocesare de pe Hub-ul Hugging Face sau dintr-un director local. Configurația feature extractor-ului și a preprocesatorului este salvată într-un fișier [preprocessor_config.json](https://hf.co/openai/whisper-tiny/blob/main/preprocessor_config.json).

Pasează semnalul audio, stocat de obicei în `array`, feature extractor-ului și setează parametrul `sampling_rate` la rata de eșantionare a modelului audio pre-antrenat. Este important ca rata de eșantionare a datelor audio să se potrivească cu rata de eșantionare a datelor pe care a fost antrenat modelul audio pre-antrenat.

```py
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
processed_sample = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=16000)
processed_sample
{'input_values': [array([ 9.4472744e-05,  3.0777880e-03, -2.8888427e-03, ...,
       -2.8888427e-03,  9.4472744e-05,  9.4472744e-05], dtype=float32)]}
```

Feature extractor-ul returnează un input, `input_values`, gata să fie consumat de model.

Acest ghid te ghidează prin clasele de feature extractor și cum să preprocesezi date audio.

## Clasele de feature extractor

Feature extractors din Transformers moștenesc din clasa de bază [`SequenceFeatureExtractor`] care subclasează [`FeatureExtractionMixin`].

- [`SequenceFeatureExtractor`] furnizează o metodă de [`~SequenceFeatureExtractor.pad`] pentru secvențe la o anumită lungime ca să eviți lungimile inegale ale secvențelor.
- [`FeatureExtractionMixin`] furnizează [`~FeatureExtractionMixin.from_pretrained`] și [`~FeatureExtractionMixin.save_pretrained`] ca să încarci și să salvezi un feature extractor.

Există două moduri de a încărca un feature extractor: [`AutoFeatureExtractor`] și o clasă de feature extractor specifică modelului.

<hfoptions id="feature-extractor-classes">
<hfoption id="AutoFeatureExtractor">

API-ul [AutoClass] încarcă automat feature extractor-ul corect pentru un model dat.

Folosește [`~AutoFeatureExtractor.from_pretrained`] ca să încarci un feature extractor.

```py
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
<hfoption id="model-specific feature extractor">

Fiecare model audio preantrenat are un feature extractor specific asociat pentru procesarea corectă a datelor audio. Când încarci un feature extractor, acesta preia configurația feature extractor-ului (dimensiunea feature-ului, lungimea chunk-ului etc.) din [preprocessor_config.json](https://hf.co/openai/whisper-tiny/blob/main/preprocessor_config.json).

Un feature extractor poate fi încărcat direct din clasa sa specifică modelului.

```py
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
</hfoptions>

## Preprocesare

Un feature extractor așteaptă un tensor PyTorch de o anumită formă ca input. Forma exactă a input-ului poate varia în funcție de modelul audio specific pe care îl folosești.

De exemplu, [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) se așteaptă ca `input_features` să fie un tensor de forma `(batch_size, feature_size, sequence_length)`, dar [Wav2Vec2](https://hf.co/docs/transformers/model_doc/wav2vec2) se așteaptă ca `input_values` să fie un tensor de forma `(batch_size, sequence_length)`.

Feature extractor-ul generează forma corectă de input pentru orice model audio folosești.

Un feature extractor setează și rata de eșantionare (numărul de valori ale semnalului audio preluate pe secundă) ale fișierelor audio. Rata de eșantionare a datelor tale audio trebuie să se potrivească cu rata de eșantionare a dataset-ului pe care a fost antrenat un model preantrenat. Această valoare este dată de obicei în card-ul modelului.

Încarcă un dataset și un feature extractor cu [`~FeatureExtractionMixin.from_pretrained`].

```py
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

Uită-te la primul exemplu din dataset și accesează coloana `audio` care conține `array`, semnalul audio brut.

```py
dataset[0]["audio"]["array"]
array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
        0.        ,  0.        ])
```

Feature extractor-ul preprocesează `array` în formatul de input așteptat pentru un model audio dat. Folosește parametrul `sampling_rate` ca să setezi rata de eșantionare corespunzătoare.

```py
processed_dataset = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=16000)
processed_dataset
{'input_values': [array([ 9.4472744e-05,  3.0777880e-03, -2.8888427e-03, ...,
       -2.8888427e-03,  9.4472744e-05,  9.4472744e-05], dtype=float32)]}
```

### Padding

Lungimile diferite ale secvențelor audio sunt o problemă pentru că Transformers se așteaptă ca toate secvențele să aibă aceeași lungime ca să fie grupate în batch-uri. Secvențele de lungimi inegale nu pot fi grupate în batch-uri.

```py
dataset[0]["audio"]["array"].shape
(86699,)

dataset[1]["audio"]["array"].shape
(53248,)
```

Padding-ul adaugă un *token de padding* special ca să se asigure că toate secvențele au aceeași lungime. Feature extractor-ul adaugă un `0` — interpretat ca tăcere — la `array` ca să facă padding. Setează `padding=True` ca să faci padding secvențelor la lungimea celei mai lungi secvențe din batch.

```py
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])
processed_dataset["input_values"][0].shape
(86699,)

processed_dataset["input_values"][1].shape
(86699,)
```

### Trunchiere

Modelele pot procesa secvențe doar până la o anumită lungime înainte să se blocheze.

Truncherea este o strategie de eliminare a token-urilor în exces dintr-o secvență ca să se asigure că nu depășește lungimea maximă. Setează `truncation=True` ca să trunchiezi o secvență la lungimea din parametrul `max_length`.

```py
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        max_length=50000,
        truncation=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])
processed_dataset["input_values"][0].shape
(50000,)

processed_dataset["input_values"][1].shape
(50000,)
```

### Resampling

Biblioteca [Datasets](https://hf.co/docs/datasets/index) poate și să resample datele audio ca să se potrivească cu rata de eșantionare așteptată de un model audio. Metoda asta resamplează datele audio din mers când sunt încărcate, ceea ce poate fi mai rapid decât resamplingul întregului dataset în loc.

Dataset-ul audio cu care lucrezi are o rată de eșantionare de 8kHz, iar modelul preantrenat se așteaptă la 16kHz.

```py
dataset[0]["audio"]
{'path': '/root/.cache/huggingface/datasets/downloads/extracted/f507fdca7f475d961f5bb7093bcc9d544f16f8cab8608e772a2ed4fbeb4d6f50/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ]),
 'sampling_rate': 8000}
```

Apelează [`~datasets.Dataset.cast_column`] pe coloana `audio` ca să mărești rata de eșantionare la 16kHz.

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

Când încarci sample-ul din dataset, acum este resampled la 16kHz.

```py
dataset[0]["audio"]
{'path': '/root/.cache/huggingface/datasets/downloads/extracted/f507fdca7f475d961f5bb7093bcc9d544f16f8cab8608e772a2ed4fbeb4d6f50/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'array': array([ 1.70562416e-05,  2.18727451e-04,  2.28099874e-04, ...,
         3.43842403e-05, -5.96364771e-06, -1.76846661e-05]),
 'sampling_rate': 16000}
```
