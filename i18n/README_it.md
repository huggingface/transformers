<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Português</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <b>Italiano</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/ReADME_id.md">Bahasa Indonesia</a> |
    </p>
</h4>

<h3 align="center">
    <p>Modelli preaddestrati all'avanguardia per l'inferenza e l'addestramento</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers funge da framework di definizione dei modelli per modelli di machine learning all'avanguardia nei
modelli di testo, visione artificiale, audio, video e multimodali, sia per l'inferenza che per l'addestramento.

Centralizza la definizione del modello in modo che tale definizione sia concordata all'interno dell'ecosistema.
`transformers` è il perno tra i framework: se una definizione di modello è supportata, sarà compatibile con la
maggior parte dei framework di addestramento (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), motori
di inferenza (vLLM, SGLang, TGI, ...) e librerie di modellazione adiacenti (llama.cpp, mlx, ...) che sfruttano
la definizione del modello da `transformers`.

Ci impegniamo a sostenere nuovi modelli all'avanguardia e a democratizzarne l'utilizzo rendendo la loro definizione
semplice, personalizzabile ed efficiente.

Ci sono oltre 1 milione di Transformers [model checkpoint](https://huggingface.co/models?library=transformers&sort=trending) su [Hugging Face Hub](https://huggingface.com/models) che puoi utilizzare.

Esplora oggi stesso l'[Hub](https://huggingface.com/) per trovare un modello e utilizzare Transformers per aiutarti a iniziare subito.

## Installazione

Transformers funziona con Python 3.9+ e [PyTorch](https://pytorch.org/get-started/locally/) 2.1+.

Crea e attiva un ambiente virtuale con [venv](https://docs.python.org/3/library/venv.html) o [uv](https://docs.astral.sh/uv/), un pacchetto Python veloce basato su Rust e un gestore di progetti.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

Installa Transformers nel tuo ambiente virtuale.

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

Installa Transformers dal sorgente se desideri le ultime modifiche nella libreria o sei interessato a contribuire. Tuttavia, la versione *più recente* potrebbe non essere stabile. Non esitare ad aprire una [issue](https://github.com/huggingface/transformers/issues) se riscontri un errore.

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Quickstart

Inizia subito a utilizzare Transformers con l'API [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial). Pipeline è una classe di inferenza di alto livello che supporta attività di testo, audio, visione e multimodali. Gestisce la pre-elaborazione dell'input e restituisce l'output appropriato.

Istanziare una pipeline e specificare il modello da utilizzare per la generazione di testo. Il modello viene scaricato e memorizzato nella cache in modo da poterlo riutilizzare facilmente. Infine, passare del testo per attivare il modello.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("il segreto per preparare una torta davvero buona è ")
[{'generated_text': 'il segreto per preparare una torta davvero buona è 1) usare gli ingredienti giusti e 2) seguire alla lettera la ricetta. la ricetta della torta è la seguente: 1 tazza di zucchero, 1 tazza di farina, 1 tazza di latte, 1 tazza di burro, 1 tazza di uova, 1 tazza di gocce di cioccolato. se vuoi preparare 2 torte, quanto zucchero ti serve? Per preparare 2 torte, avrete bisogno di 2 tazze di zucchero.'}]
```

Per chattare con un modello, lo schema di utilizzo è lo stesso. L'unica differenza è che è necessario creare una cronologia delle chat (l'input per `Pipeline`) tra l'utente e il sistema.

> [!TIP]
> È anche possibile chattare con un modello direttamente dalla riga di comando.
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "Sei un robot sfacciato e spiritoso, proprio come lo immaginava Hollywood nel 1986."},
    {"role": "user", "content": "Ehi, mi puoi suggerire qualcosa di divertente da fare a New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Espandi gli esempi riportati di seguito per vedere come funziona `Pipeline` per diverse modalità e attività.

<details>
<summary>Riconoscimento vocale automatico</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' Ho un sogno: che un giorno questa nazione si solleverà e vivrà il vero significato del suo credo.'}
```

</details>

<details>
<summary>Classificazione delle immagini</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'label': 'macaw', 'score': 0.997848391532898},
 {'label': 'cacatua dal ciuffo giallo, Kakatoe galerita, Cacatua galerita',
  'score': 0.0016551691805943847},
 {'label': 'lorichetto', 'score': 0.00018523589824326336},
 {'label': 'Pappagallo grigio africano, Psittacus erithacus',
  'score': 7.85409429227002e-05},
 {'label': 'quaglia', 'score': 5.502637941390276e-05}]
```

</details>

<details>
<summary>Risposta a domande visive</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="Cosa c'è nell'immagine?",
)
[{'answer': 'statua della libertà'}]
```

</details>

## Perché dovrei usare Transformers?

1. Modelli all'avanguardia facili da usare:
    - Prestazioni elevate nella comprensione e generazione del linguaggio naturale, nella visione artificiale, nell'audio, nel video e nelle attività multimodali.
    - Bassa barriera di ingresso per ricercatori, ingegneri e sviluppatori.
    - Poche astrazioni rivolte all'utente con solo tre classi da imparare.
    - Un'API unificata per l'utilizzo di tutti i nostri modelli preaddestrati.

1. Riduzione dei costi di calcolo e dell'impronta di carbonio:
    - Condivisione dei modelli addestrati invece di addestrarli da zero.
    - Riduzione dei tempi di calcolo e dei costi di produzione.
    - Decine di architetture di modelli con oltre 1 milione di checkpoint preaddestrati in tutte le modalità.

1. Scegli il framework giusto per ogni fase del ciclo di vita di un modello:
    - Addestra modelli all'avanguardia con sole 3 righe di codice.
    - Sposta un singolo modello tra i framework PyTorch/JAX/TF2.0 a tuo piacimento.
    - Scegli il framework giusto per l'addestramento, la valutazione e la produzione.

1. Personalizza facilmente un modello o un esempio in base alle tue esigenze:
    - Forniamo esempi per ogni architettura per riprodurre i risultati pubblicati dagli autori originali.
    - Gli interni del modello sono esposti nel modo più coerente possibile.
    - I file del modello possono essere utilizzati indipendentemente dalla libreria per esperimenti rapidi.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## Perché non dovrei usare Transformers?

- Questa libreria non è un toolbox modulare di blocchi costitutivi per reti neurali. Il codice nei file dei modelli non è stato rifattorizzato con ulteriori astrazioni di proposito, in modo che i ricercatori possano iterare rapidamente su ciascuno dei modelli senza dover approfondire ulteriori astrazioni/file.
- L'API di addestramento è ottimizzata per funzionare con i modelli PyTorch forniti da Transformers. Per i loop generici di machine learning, è necessario utilizzare un'altra libreria come [Accelerate](https://huggingface.co/docs/accelerate).
- Gli [script di esempio](https://github.com/huggingface/transformers/tree/main/examples) sono solo *esempi*. Potrebbero non funzionare immediatamente nel vostro caso specifico e potrebbe essere necessario adattare il codice affinché funzioni.

## 100 progetti che usano Transformers

Transformers è più di un semplice toolkit per l'utilizzo di modelli preaddestrati, è una comunità di progetti costruita attorno ad esso e all'
Hugging Face Hub. Vogliamo che Transformers consenta a sviluppatori, ricercatori, studenti, professori, ingegneri e chiunque altro
di realizzare i propri progetti dei sogni.

Per celebrare le 100.000 stelle di Transformers, abbiamo voluto puntare i riflettori sulla
comunità con la pagina [awesome-transformers](./awesome-transformers.md), che elenca 100
incredibili progetti realizzati con Transformers.

Se possiedi o utilizzi un progetto che ritieni debba essere inserito nell'elenco, apri una PR per aggiungerlo!

## Modelli di esempio

È possibile testare la maggior parte dei nostri modelli direttamente sulle loro [pagine dei modelli Hub](https://huggingface.co/models).

Espandi ciascuna modalità qui sotto per vedere alcuni modelli di esempio per vari casi d'uso.

<details>
<summary>Audio</summary>

- Classificazione audio con [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
- Riconoscimento vocale automatico con [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
- Individuazione delle keyword con [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- Generazione da discorso a discorso con [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
- Testo in audio con [MusicGen](https://huggingface.co/facebook/musicgen-large)
- Sintesi vocale con [Bark](https://huggingface.co/suno/bark)

</details>

<details>
<summary>Visione artificiale</summary>

- Generazione automatica di maschere con [SAM](https://huggingface.co/facebook/sam-vit-base)
- Stima della profondità con [DepthPro](https://huggingface.co/apple/DepthPro-hf)
- Classificazione delle immagini con [DINO v2](https://huggingface.co/facebook/dinov2-base)
- Rilevamento dei punti chiave con [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)
- Corrispondenza dei punti chiave con [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
- Rilevamento degli oggetti con [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
- Stima della posa con [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
- Segmentazione universale con [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
- Classificazione dei video con [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>Multimodale</summary>

- Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
- Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)
- Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
- Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>NLP</summary>

- Completamento parole mascherate con [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- Riconoscimento delle entità denominate con [Gemma](https://huggingface.co/google/gemma-2-2b)
- Risposte alle domande con [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- Sintesi con [BART](https://huggingface.co/facebook/bart-large-cnn)
- Traduzione con [T5](https://huggingface.co/google-t5/t5-base)
- Generazione di testo con [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Classificazione del testo con [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

## Citazione

Ora abbiamo un [paper](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) che puoi citare per la libreria 🤗 Transformers:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
