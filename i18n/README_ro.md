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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fa.md">فارسی</a> |
         <b>Română</b> |
    </p>
</h4>

<h3 align="center">
    <p>Modele pre-antrenate de ultimă generație pentru inferență și antrenare</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers funcționează ca framework-ul de definire a modelelor pentru tehnologii de ultimă generație în machine learning aplicate pe text, computer vision, audio, video și modele multimodale, atât pentru inferență, cât și pentru antrenare.

Acesta centralizează definirea modelelor astfel încât această definiție să fie agreată la nivelul întregului ecosistem. `transformers` este pivotul dintre framework-uri: dacă definirea unui model este suportată, acesta va fi compatibil cu majoritatea framework-urilor de antrenare (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), a motoarelor de inferență (vLLM, SGLang, TGI, ...),
și a bibliotecilor de modelare adiacente (llama.cpp, mlx, ...) care utilizează definirea modelului din `transformers`.

Ne angajăm să ajutăm suportarea noilor modele de ultimă generație și să le democratizăm utilizarea prin oferirea unei definiri a modelului simplă, personalizabilă și eficientă.

Avem peste 1M de [checkpoint-uri de model](https://huggingface.co/models?library=transformers&sort=trending) Transformers pe [Hub-ul Hugging Face](https://huggingface.co/models) pe care le poți utiliza.

Explorează [Hub-ul](https://huggingface.co/) chiar azi pentru a găsi un model și folosește Transformers pentru a începe imediat.

## Instalarea

Transformers este compatibil cu Python 3.10+ și [PyTorch](https://pytorch.org/get-started/locally/) 2.4+.

Creează și activează un virtual environment folosind [venv](https://docs.python.org/3/library/venv.html) sau [uv](https://docs.astral.sh/uv/), un Python package manager și project manager rapid, scris în Rust.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

Instalează Transformers în virtual environment-ul tău.

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

Instalează Transformers din codul sursă dacă vrei cele mai noi schimbări din bibliotecă sau ești interesat în a contribui. Totuși, s-ar putea ca *cea mai recentă* versiune să nu fie stabilă. Deschide un [issue](https://github.com/huggingface/transformers/issues) dacă întâmpini o eroare.

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

## Pornire rapidă

Începe să utilizezi Transformers imediat folosind API-ul [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial). `Pipeline-ul` este o clasă de inferență high-level ce suportă text, audio, vision și task-uri multimodale. Se ocupă de preprocesarea input-ului și returnează output-ul corespunzător.

Inițializează un pipeline și specifică modelul pentru generarea de text. Modelul este descărcat și salvat în cache pentru o reutilizare ușoară. În final, scrie un prompt pentru model.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

Pentru a conversa cu un model, utilizarea este aceeași. Singura diferență este că va trebui să construiești un istoric al conversației (input-ul pentru `Pipeline`) dintre tine și sistem.

> [!TIP]
> Poți conversa cu un model și din linia de comandă, atât timp cât [`transformers serve` rulează].
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Vezi exemplele de mai jos pentru a vedea cum funcționează `Pipeline` pentru different modalități și task-uri.

<details>
<summary>Recunoaștere vocală automată</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>Clasificare de imagini</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'label': 'macaw', 'score': 0.997848391532898},
 {'label': 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
  'score': 0.0016551691805943847},
 {'label': 'lorikeet', 'score': 0.00018523589824326336},
 {'label': 'African grey, African gray, Psittacus erithacus',
  'score': 7.85409429227002e-05},
 {'label': 'quail', 'score': 5.502637941390276e-05}]
```

</details>

<details>
<summary>Răspundere vizuală la întrebări</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</details>

## De ce să folosesc Transformers?

1. Modele de ultimă generație ușor de utilizat:
    - Performanță înaltă la generarea și procesarea de limbaj natural, computer vision, audio, video și task-uri multimodale.
    - Barieră scăzută de intrare pentru cercetători, ingineri și developeri.
    - Puține niveluri de abstractizare pentru utilizator, având doar trei clase de învățat.
    - Un API unificat pentru utilizarea tuturor modelelor noastre pre-antrenate.

1. Costuri de calcul mai mici, amprentă de carbon mai mică:
    - Utilizează modelele antrenate în loc să le antrenezi de la zero.
    - Redu timpul de calcul și costurile de producție.
    - Sute de arhitecturi de modele cu peste 1M de checkpoint-uri pre-antrenate pentru toate modalitățile de date.

1. Alege framework-ul potrivit pentru fiecare etapă din ciclul de viață al unui model:
    - Antrenează modele de ultimă generație în doar 3 linii de cod.
    - Mută un singur model între framework-urile PyTorch / JAX / TF2.0 după bunul plac.
    - Alege framework-ul potrivit pentru antrenare, evaluare și producție.

1. Personalizează cu ușurință un model sau un exemplu în funcție de nevoile tale:
    - Oferim exemple pentru fiecare arhitectură pentru a reproduce rezultatele publicate de autorii originali.
    - Mecanismele interne ale modelelor sunt expuse într-un mod cât mai consecvent posibil.
    - Fișierele modelului pot fi utilizate independent de librărie pentru experimente rapide.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## Când nu ar trebui să utilizez Transformers?

- Această bibliotecă nu este un toolbox de block-uri pentru construirea rețelelor neuronale. Codul din fișierele modelelor nu este refactorizat cu mai multă abstractizare pentru ca cercetătorii să poată utiliza rapid fiecare dintre modele fără să aibă de-a face cu fișiere/abstractizări adiționale.
- API-ul de antrenare este optimizat pentru utilizarea cu modele PyTorch oferite de Transformers. Pentru loop-uri generice de machine learning, utilizează o bibliotecă precum [Accelerate](https://huggingface.co/docs/accelerate).
- [Script-urile de exemplu](https://github.com/huggingface/transformers/tree/main/examples) sunt doar *exemple*. S-ar putea ca acestea să nu funcționeze în toate cazurile și va trebui să adaptezi codul pentru ca acestea să funcționeze.

## 100 de proiecte folosind Transformers

Transformers este mai mult decât un toolkit pentru utilizarea modelelor pre-antrenate, este o comunitate de proiecte construite în jurul acestuia și a Hub-ului Hugging Face. Vrem ca Transformers să ajute developerii, cercetătorii, studenții, profesorii, inginerii și pe toți ceilalți oameni să-și construiască propriile proiecte.

Pentru a sărbători atingerea a 100,000 de stars pentru proiectul Transformers, am vrut să aducem comunitatea în centrul atenției prin pagina [awesome-transformers](./awesome-transformers.md) care este o listă de 100 de proiecte incredibile construite utilizând Transformers.

Dacă deții sau utilizezi un proiect și crezi că ar trebui să facă parte din listă, deschide un PR pentru a-l adăuga!

## Modele de exemplu

Poți testa majoritatea modelelor noastre direct pe [paginile lor de pe Hub](https://huggingface.co/models).

Vezi mai jos modele de exemplu pentru diverse cazuri de utilizare.

<details>
<summary>Audio</summary>

- Clasificare audio cu [CLAP](https://huggingface.co/laion/clap-htsat-fused)
- Recunoaștere vocală automată cu [Parakeet](https://huggingface.co/nvidia/parakeet-ctc-1.1b#transcribing-using-transformers-%F0%9F%A4%97), [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) și [Moonshine-Streaming](https://huggingface.co/UsefulSensors/moonshine-streaming-medium)
- Detectare de cuvinte-cheie cu [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- Generare speech-to-speech cu [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
- Text-to-audio cu [MusicGen](https://huggingface.co/facebook/musicgen-large)
- Text-to-speech cu [CSM](https://huggingface.co/sesame/csm-1b)

</details>

<details>
<summary>Computer vision</summary>

- Generare automată de măști cu [SAM](https://huggingface.co/facebook/sam-vit-base)
- Estimare de depth cu [DepthPro](https://huggingface.co/apple/DepthPro-hf)
- Clasificare de imagini cu [DINO v2](https://huggingface.co/facebook/dinov2-base)
- Detectare de keypoint-uri cu [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)
- Potrivire de keypoint-uri cu [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
- Detectare de obiecte cu [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
- Estimare de postură corporală cu [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
- Segmentare universală cu [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
- Clasificare video cu [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>Multimodale</summary>

- Audio-to-text sau text-to-text cu [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [Audio Flamingo](https://huggingface.co/nvidia/audio-flamingo-3-hf)
- Răspunsuri la întrebări din documente cu [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- Imagine-to-text sau text-to-text cu [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Descrierea imaginilor cu [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- Înțelegerea documentelor pe bază de OCR cu [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- Răspunsuri la întrebări pe bază de tabele cu [TAPAS](https://huggingface.co/google/tapas-base)
- Generare și înțelegere multimodală unificată cu [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
- Vision-to-text cu [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- Răspunsuri la întrebări vizuale cu [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- Segmentare vizuală pe bază de expresii de referință cu [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>NLP</summary>

- Completarea cuvintelor mascate cu [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- Recunoașterea entităților numite cu [Gemma](https://huggingface.co/google/gemma-2-2b)
- Răspunsuri la întrebări cu [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- Sumarizare cu [BART](https://huggingface.co/facebook/bart-large-cnn)
- Traducere cu [T5](https://huggingface.co/google-t5/t5-base)
- Generare de text cu [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Clasificare de text cu [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

## Citare

Avem un [articol](https://aclanthology.org/2020.emnlp-demos.6/) pe care îl poți cita pentru biblioteca 🤗 Transformers:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-demos.6/",
    pages = "38--45"
}
```
