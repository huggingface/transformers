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

# Distribuirea modelelor

[Hub-ul](https://hf.co/models) Hugging Face este o platformă pentru distribuirea, descoperirea și utilizarea modelelor de toate tipurile și dimensiunile. Îți recomandăm să distribui modelul tău pe Hub pentru a avansa machine learning-ul open-source pentru toți!

Acest ghid îți va arăta cum să distribui un model pe Hub direct din Transformers.

## Configurare

Pentru a partaja un model pe Hub, ai nevoie de un [cont](https://hf.co/join) Hugging Face. Creează un [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) (stocat în [cache](./installation#folder-ul-cache) în mod implicit) și autentifică-te în contul tău din linia de comandă sau notebook.

<hfoptions id="share">
<hfoption id="huggingface-CLI">

```bash
hf auth login
```

</hfoption>
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
</hfoptions>

## Funcțiile repository-ului

<Youtube id="XvSGPZFEjDY"/>

Fiecare repository de model include versionare, istoricul commit-urilor și vizualizarea diff-urilor.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png"/>
</div>

Versionarea se bazează pe [Git](https://git-scm.com/) și [Git Large File Storage (LFS)](https://git-lfs.github.com/) și permite revizuiri, o modalitate de a specifica o versiune a modelului cu un hash de commit, tag sau branch.

De exemplu, folosește parametrul `revision` în [`~PreTrainedModel.from_pretrained`] pentru a încărca o versiune specifică a modelului dintr-un hash de commit.

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="4c77982"
)
```

Repository-urile de modele suportă și [gating](https://hf.co/docs/hub/models-gated) pentru a controla cine poate accesa un model. Gating-ul este comun pentru a permite unui grup selectat de utilizatori să previzualizeze un model de cercetare înainte de a fi făcut public.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/gated-model.png"/>
</div>

Un repository de model include și un [widget](https://hf.co/docs/hub/models-widgets) de inferență pentru ca utilizatorii să interacționeze direct cu un model pe Hub.

Consultă documentația Hub [Models](https://hf.co/docs/hub/models) pentru mai multe informații.

## Încărcarea unui model pe Hub

Există mai multe modalități de a încărca un model pe Hub în funcție de preferința ta de workflow. Poți publica un model cu [`Trainer`], apela [`~PreTrainedModel.push_to_hub`] direct pe un model sau folosi interfața web a Hub-ului.

<Youtube id="Z1-XMy-GNLQ"/>

### Trainer

[`Trainer`] poate publica un model direct pe Hub după antrenare. Setează `push_to_hub=True` în [`TrainingArguments`] și pasează-l la [`Trainer`]. Odată ce antrenarea este completă, apelează [`~transformers.Trainer.push_to_hub`] pentru a încărca modelul.

[`~transformers.Trainer.push_to_hub`] adaugă automat informații utile precum hyperparameters de antrenare și rezultate la model card.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.push_to_hub()
```

### PushToHubMixin

[`~utils.PushToHubMixin`] oferă funcționalitate pentru publicarea unui model sau tokenizer pe Hub.

Apelează [`~utils.PushToHubMixin.push_to_hub`] direct pe un model pentru a-l încărca pe Hub. Creează un repository sub namespace-ul tău cu numele modelului specificat în [`~utils.PushToHubMixin.push_to_hub`].

```py
model.push_to_hub("my-awesome-model")
```

Alte obiecte precum un tokenizer sunt publicate pe Hub în același mod.

```py
tokenizer.push_to_hub("my-awesome-model")
```

Profilul tău Hugging Face ar trebui să afișeze acum repository-ul de model nou creat. Navighează la tab-ul **Files** pentru a vedea toate fișierele încărcate.

Consultă ghidul [Upload files to the Hub](https://hf.co/docs/hub/how-to-upstream) pentru mai multe informații despre publicarea fișierelor pe Hub.

### Interfața web a Hub-ului

Interfața web a Hub-ului este o abordare fără cod pentru încărcarea unui model.

1. Creează un nou repository selectând [**New Model**](https://huggingface.co/new).

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png"/>
</div>

Adaugă câteva informații despre modelul tău:

- Selectează **owner**-ul repository-ului. Acesta poate fi tu însuți sau oricare dintre organizațiile din care faci parte.
- Alege un nume pentru modelul tău, care va fi și numele repository-ului.
- Alege dacă modelul tău este public sau privat.
- Setează utilizarea licenței.

2. Click pe **Create model** pentru a crea repository-ul de model.

3. Selectează tab-ul **Files** și click pe butonul **Add file** pentru a trage și plasa un fișier în repository-ul tău. Adaugă un mesaj de commit și click pe **Commit changes to main** pentru a face commit fișierului.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png"/>
</div>

## Model card

[Model card-urile](https://hf.co/docs/hub/model-cards#model-cards) informează utilizatorii despre performanța, limitările, posibilele biasuri și considerațiile etice ale unui model. Îți recomandăm cu căldură să adaugi un model card la repository-ul tău!

Un model card este un fișier `README.md` din repository-ul tău. Adaugă acest fișier prin:

- crearea și încărcarea manuală a unui fișier `README.md`
- click pe butonul **Edit model card** din repository

Aruncă o privire la [model card-ul](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) Llama 3.1 pentru un exemplu de ce să incluzi într-un model card.

Află mai multe despre metadata model card-ului (emisii de carbon, licență, link la articol, etc.) în ghidul [Model Cards](https://hf.co/docs/hub/model-cards#model-cards).
