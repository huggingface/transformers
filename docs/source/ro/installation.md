<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Instalarea

Transformers depinde de [PyTorch](https://pytorch.org/get-started/locally/). A fost testat cu Python 3.10+ și PyTorch 2.4+.

## Virtual environment

[uv](https://docs.astral.sh/uv/) este un Python Package manager și project manager foarte rapid scris în Rust și necesită un [virtual environment](https://docs.astral.sh/uv/pip/environments/) pentru a gestiona diferite proiecte, evitând conflictele dintre dependencies.

Poate fi utilizat drept înlocuitor pentru [pip](https://pip.pypa.io/en/stable/), dar, dacă preferi să utilizezi pip, omite `uv` din comenzile de mai jos.

> [!TIP]
> Uită-te la [ghidul de instalare](https://docs.astral.sh/uv/guides/install-python/) uv pentru a instala uv.

Creează un virtual environment în care să instalezi Transformers.

```bash
uv venv .env
source .env/bin/activate
```

## Python

Instalează Transformers cu următoarea comandă.

[uv](https://docs.astral.sh/uv/) este un Python Package manager și project manager rapid scris în Rust.

```bash
uv pip install transformers
```

Pentru GPU Acceleration descarcă driverele CUDA pentru [PyTorch](https://pytorch.org/get-started/locally).

Rulează comanda de mai jos pentru a vedea dacă sistemul tău detectează un GPU NVIDIA.

```bash
nvidia-smi
```

Pentru a instala o versiune doar de CPU a Transformers, rulează comanda de mai jos.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers
```

Testează dacă instalarea s-a realizat cu succes cu următoarea comandă. Ar trebui să returneze un label și un score pentru textul următor.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Instalarea din codul sursă

Instalând de la sursă se instalează *cea mai recentă* versiune în loc de versiunea *stabilă*. Aceasta asigură cele mai noi schimbări din Transformers și folosește la experimentarea cu cele mai noi funcții sau repararea unui bug care nu a fost încă lansat în versiunea stabilă.

Dezavantajul este că cea mai recentă versiune ar putea să nu fie stabilă. Dacă întâmpini vreo problemă, te rugăm să deschizi un [GitHub Issue](https://github.com/huggingface/transformers/issues) ca să o putem repara cât de repede posibil.

Instalează din codul sursă cu următoarea comandă.

```bash
uv pip install git+https://github.com/huggingface/transformers
```

Testează dacă instalarea s-a realizat cu succes cu următoarea comandă. Ar trebui să returneze un label și un score pentru textul următor.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Instalare editabilă

O [instalare editabilă](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) este utilă dacă faci development local cu Transformers. Aceasta conectează copia ta locală a Transformers cu [repository-ul](https://github.com/huggingface/transformers) Transformers în loc să copieze fișierele. Acestea sunt adăugate la path-ul de import Python.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .
```

> [!WARNING]
> Trebuie să păstrezi folder-ul local Transformers ca să continui utilizarea.

Actualizează-ți copia locală de Transformers cu cele mai noi schimbări din repository-ul principal cu comanda de mai jos.

```bash
cd ~/transformers/
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/#) este un package manager independent de un limbaj de programare. Instalează Transformers de la canalul [conda-forge](https://anaconda.org/conda-forge/transformers) în noul tău virtual environment.

```bash
conda install conda-forge::transformers
```

## Configurare

După instalare, poți seta locația cache pentru Transformers sau poți configura biblioteca pentru utilizare offline.

### Folder-ul cache

Când încarci un model folosind [`~PreTrainedModel.from_pretrained`], modelul este descărcat de pe Hub și salvat local în cache.

De fiecare dată când încarci un model se verifică dacă modelul salvat în cache este la cea mai nouă versiune. Dacă da, modelul salvat în cache este încărcat. Dacă nu, modelul mai nou este descărcat și salvat în cache.

Folder-ul implicit dat de variabila de mediu `HF_HUB_CACHE` este `~/.cache/huggingface/hub`. Pe Windows, folder-ul implicit este `C:\Users\username\.cache\huggingface\hub`.

Salvează un model în cache într-un alt folder schimbând path-ul în următoarele variabile de mediu (sortate după prioritate).

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) (implicit)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (doar dacă `HF_HOME` nu este setat)

### Modul Offline

Folosirea Transformers într-un mediu offline sau cu firewall necesită fișierele descărcate și salvate în cache. Descarcă un repository de modele folosind metoda [`~huggingface_hub.snapshot_download`] .

> [!TIP]
> Urmează ghidul [Descarcă fișiere de pe Hub](https://hf.co/docs/huggingface_hub/guides/download) pentru mai multe opțiuni de a descărca fișiere de pe Hub. Poți descărca fișiere din revizii specifice, utilizând linia de comandă și poți filtra ce fișiere să descarci din repository.

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

Setează variabila de mediu `HF_HUB_OFFLINE=1` pentru a preveni HTTP calls către Hub la descărcarea unui model.

```bash
HF_HUB_OFFLINE=1 \
python examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext ...
```

O altă opțiune pentru încărcarea exclusivă a fișierelor salvate în cache este setarea `local_files_only=True` în [`~PreTrainedModel.from_pretrained`].

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```
