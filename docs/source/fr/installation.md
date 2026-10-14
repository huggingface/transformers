<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# Installation

Installez 🤗 Transformers pour n'importe quelle librairie d'apprentissage profond avec laquelle vous avez l'habitude de travaillez, configurez votre cache et configurez 🤗 Transformers pour un usage hors ligne (facultatif).

🤗 Transformers est testé avec Python 3.10+ et PyTorch 2.4+.
Consulter les instructions d'installation ci-dessous pour la librairie d'apprentissage profond que vous utilisez:

  * Instructions d'installation pour [PyTorch](https://pytorch.org/get-started/locally/).

## Installation avec pip

Vous devriez installer 🤗 Transformers dans un [environnement virtuel](https://docs.python.org/3/library/venv.html).
Si vous n'êtes pas à l'aise avec les environnements virtuels, consultez ce [guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
Utiliser un environnement virtuel permet de facilement gérer différents projets et d'éviter des erreurs de compatibilité entre les différentes dépendances.

Commencez par créer un environnement virtuel dans l'espace de travail de votre projet :

```bash
python -m venv .env
```

Activez l'environnement virtuel. Sur Linux ou MacOs :

```bash
source .env/bin/activate
```

Activez l'environnement virtuel sur Windows :

```bash
.env/Scripts/activate
```

Maintenant, 🤗 Transformers peut être installé avec la commande suivante :

```bash
pip install transformers
```

Pour une utilisation avec CPU seulement, 🤗 Transformers et la librairie d'apprentissage profond de votre choix peuvent être installés en une seule ligne.
Par exemple, installez 🤗 Transformers et PyTorch avec la commande suivante :

```bash
pip install 'transformers[torch]'
```

Vérifiez que 🤗 Transformers a bien été installé avec la commande suivante. La commande va télécharger un modèle pré-entraîné :

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

Le label et score sont ensuite affichés :

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## Installation depuis le code source

Installez 🤗 Transformers depuis le code source avec la commande suivante :

```bash
pip install git+https://github.com/huggingface/transformers
```

Cette commande installe la version depuis la branche `main` au lieu de la dernière version stable. La version de la branche `main` est utile pour avoir les derniers développements. Par exemple, si un bug a été résolu depuis la dernière version stable mais n'a pas encore été publié officiellement. Cependant, cela veut aussi dire que la version de la branche `main` n'est pas toujours stable. Nous nous efforçons de maintenir la version de la branche `main` opérationnelle, et la plupart des problèmes sont généralement résolus en l'espace de quelques heures ou d'un jour. Si vous recontrez un problème, n'hésitez pas à créer une [Issue](https://github.com/huggingface/transformers/issues) pour que l'on puisse trouver une solution au plus vite !

Vérifiez que 🤗 Transformers a bien été installé avec la commande suivante :

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## Installation modifiable

Vous aurez besoin d'une installation modifiable si vous le souhaitez :

  * Utiliser la version de la branche `main` du code source.
  * Contribuer à 🤗 Transformers et vouler tester vos modifications du code source.

Clonez le projet et installez 🤗 Transformers avec les commandes suivantes :

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Ces commandes créent des liens entre le dossier où le projet a été cloné et les chemins de vos librairies Python. Python regardera maintenant dans le dossier que vous avez cloné en plus des dossiers où sont installées vos autres librairies. Par exemple, si vos librairies Python sont installées dans `~/anaconda3/envs/main/lib/python3.10/site-packages/`, Python cherchera aussi dans le dossier où vous avez cloné : `~/transformers/`.

<Tip warning={true}>

Vous devez garder le dossier `transformers` si vous voulez continuer d'utiliser la librairie.

</Tip>

Maintenant, vous pouvez facilement mettre à jour votre clone avec la dernière version de 🤗 Transformers en utilisant la commande suivante :

```bash
cd ~/transformers/
git pull
```

Votre environnement Python utilisera la version de la branche `main` lors de la prochaine exécution.

## Installation avec conda

Installation via le canal `conda-forge` de conda :

```bash
conda install conda-forge::transformers
```

## Configuration du cache

Les modèles pré-entraînés sont téléchargés et mis en cache localement dans le dossier suivant : `~/.cache/huggingface/hub`. C'est le dossier par défaut donné par la variable d'environnement `HF_HUB_CACHE`. Sur Windows, le dossier par défaut est `C:\Users\nom_utilisateur\.cache\huggingface\hub`. Vous pouvez modifier les variables d'environnement indiquées ci-dessous - par ordre de priorité - pour spécifier un dossier de cache différent :

1. Variable d'environnement (par défaut) : `HF_HUB_CACHE`.
2. Variable d'environnement : `HF_HOME`.
3. Variable d'environnement : `XDG_CACHE_HOME` + `/huggingface`.

## Mode hors ligne

🤗 Transformers peut fonctionner dans un environnement cloisonné ou hors ligne en n'utilisant que des fichiers locaux. Définissez la variable d'environnement `HF_HUB_OFFLINE=1` pour activer ce mode.

<Tip>

Ajoutez [🤗 Datasets](https://huggingface.co/docs/datasets/) à votre processus d'entraînement hors ligne en définissant la variable d'environnement `HF_DATASETS_OFFLINE=1`.

</Tip>

Vous pouvez aussi éviter de télécharger un modèle à chaque appel de la fonction [`~PreTrainedModel.from_pretrained`] en utilisant le paramètre [local_files_only]. Seuls les fichiers locaux sont chargés lorsque ce paramètre est activé (c.-à-d. `local_files_only=True`) :

```py
from transformers import T5Model

model = T5Model.from_pretrained("./path/to/local/directory", local_files_only=True)
```

### Récupérer des modèles et des tokenizers pour une utilisation hors ligne

Une autre option pour utiliser 🤗 Transformers hors ligne est de télécharger les fichiers à l'avance, puis d'utiliser les chemins locaux lorsque vous en avez besoin en mode hors ligne. Il existe trois façons de faire cela :

  * Téléchargez un fichier via l'interface utilisateur sur le [Model Hub](https://huggingface.co/models) en cliquant sur l'icône ↓.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

  * Utilisez les fonctions [`PreTrainedModel.from_pretrained`] et [`PreTrainedModel.save_pretrained`] :

    1. Téléchargez vos fichiers à l'avance avec [`PreTrainedModel.from_pretrained`]:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. Sauvegardez les fichiers dans un dossier de votre choix avec [`PreTrainedModel.save_pretrained`]:

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. Maintenant, lorsque vous êtes hors ligne, rechargez vos fichiers avec [`PreTrainedModel.from_pretrained`] depuis le dossier où vous les avez sauvegardés :

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

  * Téléchargez des fichiers de manière automatique avec la librairie [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub) :

    1. Installez la librairie `huggingface_hub`  dans votre environnement virtuel :

    ```bash
    python -m pip install huggingface_hub
    ```

    2. Utilisez la fonction [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) pour télécharger un fichier vers un chemin de votre choix.  Par exemple, la commande suivante télécharge le fichier `config.json` du modèle [T0](https://huggingface.co/bigscience/T0_3B) vers le chemin de votre choix :

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

Une fois que votre fichier est téléchargé et caché localement, spécifiez son chemin local pour le charger et l'utiliser :

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

Consultez la section [How to download files from the Hub (Comment télécharger des fichiers depuis le Hub)](https://huggingface.co/docs/hub/how-to-downstream) pour plus de détails sur le téléchargement de fichiers stockés sur le Hub.

</Tip>
