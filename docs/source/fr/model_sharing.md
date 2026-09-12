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

# Partage

Le [Hub](https://hf.co/models) Hugging Face est une plateforme pour partager, découvrir et utiliser des modèles de tous types et de toutes tailles. Nous vous recommandons vivement de partager votre modèle sur le Hub pour faire avancer le machine learning open-source pour tout le monde !

Ce guide vous montre comment partager un modèle sur le Hub depuis Transformers.

## Configuration

Pour partager un modèle sur le Hub, il vous faut un [compte](https://hf.co/join) Hugging Face. Créez un [token d'accès](https://hf.co/docs/hub/security-tokens#user-access-tokens) (stocké dans le [cache](./installation#configuration-du-cache) par défaut) et connectez-vous à votre compte depuis la ligne de commande ou un notebook.

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

## Fonctionnalités des dépôts

<Youtube id="XvSGPZFEjDY"/>

Chaque dépôt de modèle propose le versioning, l'historique des commits et la visualisation des diffs.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png"/>
</div>

Le versioning repose sur [Git](https://git-scm.com/) et [Git Large File Storage (LFS)](https://git-lfs.github.com/), et il permet les révisions : vous pouvez désigner une version précise d'un modèle par un hash de commit, un tag ou une branche.

Par exemple, utilisez le paramètre `revision` dans [`~PreTrainedModel.from_pretrained`] pour charger une version précise d'un modèle à partir d'un hash de commit.

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="4c77982"
)
```

Les dépôts de modèles prennent aussi en charge le [gating](https://hf.co/docs/hub/models-gated) pour contrôler qui peut accéder à un modèle. Le gating est couramment utilisé pour permettre à un groupe restreint d'utilisateurs de découvrir un modèle de recherche avant sa publication.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/gated-model.png"/>
</div>

Un dépôt de modèle inclut également un [widget](https://hf.co/docs/hub/models-widgets) d'inférence qui permet d'interagir directement avec un modèle sur le Hub.

Consultez la documentation [Models](https://hf.co/docs/hub/models) du Hub pour plus d'informations.

## Uploader un modèle

Il existe plusieurs façons d'uploader un modèle sur le Hub, selon votre manière de travailler. Vous pouvez pusher un modèle avec [`Trainer`], appeler directement [`~PreTrainedModel.push_to_hub`] sur un modèle, ou passer par l'interface web du Hub.

<Youtube id="Z1-XMy-GNLQ"/>

### Trainer

[`Trainer`] peut pusher un modèle directement sur le Hub après l'entraînement. Définissez `push_to_hub=True` dans [`TrainingArguments`] et passez-les à [`Trainer`]. Une fois l'entraînement terminé, appelez [`~transformers.Trainer.push_to_hub`] pour uploader le modèle.

[`~transformers.Trainer.push_to_hub`] ajoute automatiquement des informations utiles à la model card, comme les hyperparamètres d'entraînement et les résultats.

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

[`~utils.PushToHubMixin`] fournit les fonctionnalités nécessaires pour pusher un modèle ou un tokenizer sur le Hub.

Appelez directement [`~utils.PushToHubMixin.push_to_hub`] sur un modèle pour l'uploader sur le Hub. Cela crée un dépôt dans votre namespace, avec le nom de modèle indiqué dans [`~utils.PushToHubMixin.push_to_hub`].

```py
model.push_to_hub("my-awesome-model")
```

D'autres objets, comme un tokenizer, se pushent sur le Hub de la même manière.

```py
tokenizer.push_to_hub("my-awesome-model")
```

Votre profil Hugging Face devrait maintenant afficher le dépôt de modèle que vous venez de créer. Allez dans l'onglet **Files** pour voir tous les fichiers uploadés.

Consultez le guide [Upload files to the Hub](https://hf.co/docs/hub/how-to-upstream) pour plus d'informations sur l'envoi de fichiers vers le Hub.

### Interface web du Hub

L'interface web du Hub permet d'uploader un modèle sans écrire de code.

1. Créez un nouveau dépôt en sélectionnant [**New Model**](https://huggingface.co/new).

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png"/>
</div>

Ajoutez quelques informations sur votre modèle :

- Sélectionnez le **owner** du dépôt. Il peut s'agir de vous-même ou de n'importe quelle organisation à laquelle vous appartenez.
- Choisissez un nom pour votre modèle, qui sera aussi le nom du dépôt.
- Indiquez si votre modèle est public ou privé.
- Définissez la licence d'utilisation.

2. Cliquez sur **Create model** pour créer le dépôt du modèle.

3. Sélectionnez l'onglet **Files** et cliquez sur le bouton **Add file** pour glisser-déposer un fichier dans votre dépôt. Ajoutez un message de commit et cliquez sur **Commit changes to main** pour valider le fichier.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png"/>
</div>

## Model card

Les [model cards](https://hf.co/docs/hub/model-cards#model-cards) informent les utilisateurs des performances d'un modèle, de ses limites, de ses biais potentiels et des considérations éthiques associées. Nous vous recommandons vivement d'ajouter une model card à votre dépôt !

Une model card est un fichier `README.md` placé dans votre dépôt. Vous pouvez l'ajouter :

- en créant et en uploadant manuellement un fichier `README.md`
- en cliquant sur le bouton **Edit model card** dans le dépôt

Regardez la [model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) de Llama 3.1 pour un exemple de ce qu'on peut y mettre.

Pour en savoir plus sur les autres métadonnées disponibles dans une model card (émissions carbone, licence, lien vers un article, etc.), consultez le guide [Model Cards](https://hf.co/docs/hub/model-cards#model-cards).
