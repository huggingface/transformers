<!---
Copyright 2024 The HuggingFace Team. Alle rettigheder forbeholdes.

Licenseret under Apache License, Version 2.0 (the "License");
du må ikke bruge denne fil undtagen i overensstemmelse med Licensen.
Du kan få en kopi af Licensen på

    http://www.apache.org/licenses/LICENSE-2.0

Medmindre det kræves af gældende lov eller skriftligt aftalt,
software distribueret under Licensen distribueres på en "AS IS" BASIS,
UDEN GARANTIER ELLER BETINGELSER AF NOGEN ART, hverken udtrykt eller underforstået.
Se Licensen for de specifikke regler om rettigheder og begrænsninger under Licensen.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_fr.md">Français</a> |
        <b>Deutsch</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_vi.md">Tiếng Việt</a> |
    </p>
</h4>

<h3 align="center">
    <p>State-of-the-art maskinlæring for JAX, PyTorch og TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers tilbyder tusindvis af fortrænede modeller til at udføre opgaver i forskellige modaliteter såsom tekst, billede og lyd.

Disse modeller kan anvendes til:

* 📝 Tekst - til opgaver såsom tekstklassifikation, informationsekstraktion, spørgsmålssvar, automatisk tekstopsummering, maskinoversættelse og tekstgenerering på mere end 100 sprog.
* 🖼️ Billeder - til opgaver såsom billedklassifikation, objektgenkendelse og segmentering.
* 🗣️ Lyd - til opgaver såsom talegenkendelse og lydklassifikation.

Transformer-modeller kan også udføre opgaver for **flere modaliteter i kombination**, f.eks. tabelbaseret spørgsmålssvar, optisk tegngenkendelse, informationsekstraktion fra scannede dokumenter, videoklassifikation og visuel spørgsmålssvar.

🤗 Transformers tilbyder API'er til hurtigt at downloade disse fortrænede modeller og bruge dem på en given tekst, finjustere dem på dine egne datasæt og derefter dele dem med fællesskabet i vores [Model Hub](https://huggingface.co/models). Samtidig er hvert Python-modul, der definerer en arkitektur, fuldstændig selvstændigt og kan modificeres for at muliggøre hurtige forskningsforsøg.

🤗 Transformers understøtter sømløs integration af tre af de mest populære dybe læringsbiblioteker: [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) og [TensorFlow](https://www.tensorflow.org/). Træn din model i et framework og indlæs det til inferens med et andet uden problemer.

## Online-demos

Du kan teste de fleste af vores modeller direkte på deres sider i [Model Hub](https://huggingface.co/models). Vi tilbyder også [privat model-hosting, versionering, & en inferens-API](https://huggingface.co/pricing) for offentlige og private modeller.

Her er nogle eksempler:

Inden for naturlig sprogbehandling:

- [Maskeret ordudfyldning med BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Navngivningsgenkendelse med Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Tekstgenerering med GPT-2](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [Natural Language Inference med RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Automatisk tekstopsummering med BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+

of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct.)

Inden for computer vision:

- [Billedklassifikation med Vision Transformer](https://huggingface.co/google/vit-base-patch16-224)
- [Objektgenkendelse med DETR](https://huggingface.co/facebook/detr-resnet-50)

Inden for tale:

- [Talegenkendelse med Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)

Inden for kombinerede modaliteter:

- [Billedspørgsmålssvar med Flava](https://huggingface.co/docs/transformers/main/en/model_doc/flava)

## Quick tour

Hugging Face biblioteket har to hovedmoduler: [*transformers*](https://github.com/huggingface/transformers/tree/main/src/transformers) og [*datasets*](https://github.com/huggingface/datasets).

### *transformers* biblioteket

Det modulære bibliotek, der tilbyder alle muligheder for at indlæse og tilpasse modeller samt deres tokenizere.

#### Installation

Den nemmeste måde at installere det på er ved at bruge [pip](https://pip.pypa.io/en/stable/):

```bash
pip install transformers
```

For installation via conda, følg venligst [denne vejledning](https://huggingface.co/docs/transformers/installation#conda).

#### Brug af pip

Hent en transformer model fra Hugging Face Model Hub, og brug den i tre linjer af Python:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Jeg elsker Hugging Face!"))
```

For mere information om installation og brug, se vores [dokumentation](https://huggingface.co/docs/transformers/installation).

### *datasets* biblioteket

Dette modul tilbyder sømløs download, pre-procesning og håndtering af datasæt til maskinlæringsmodeller.

#### Installation

Installer *datasets* biblioteket med pip:

```bash
pip install datasets
```

#### Brug af pip

Hent et datasæt og forbered det til brug med en model:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset["train"][0])
```

For yderligere detaljer om installation og brug af *datasets*, besøg vores [dokumentation](https://huggingface.co/docs/datasets/installation).

Vi inviterer dig til at bidrage til udviklingen af Hugging Face biblioteket. For mere information om hvordan man bidrager, besøg vores [Contributing](https://huggingface.co/docs/transformers/main/en/community/contributing) side.
