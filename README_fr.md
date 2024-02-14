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
    <img alt="Bibliothèque Hugging Face Transformers" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Construction" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/docs/transformers/index">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="Version GitHub" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Pacte des contributeurs" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
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
        <b>Français</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_de.md">Deutsch</a> |
    </p>
</h4>

<h3 align="center">
    <p>Apprentissage automatique de pointe pour JAX, PyTorch et TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers fournit des milliers de modèles pré-entraînés pour effectuer des tâches sur différentes modalités telles que le texte, la vision et l'audio.

Ces modèles peuvent être appliqués à :

* 📝 Texte, pour des tâches telles que la classification de texte, l'extraction d'informations, la réponse aux questions, le résumé, la traduction et la génération de texte, dans plus de 100 langues.
* 🖼️ Images, pour des tâches telles que la classification d'images, la détection d'objets et la segmentation.
* 🗣️ Audio, pour des tâches telles que la reconnaissance vocale et la classification audio.

Les modèles de transformer peuvent également effectuer des tâches sur **plusieurs modalités combinées**, telles que la réponse aux questions sur des tableaux, la reconnaissance optique de caractères, l'extraction d'informations à partir de documents numérisés, la classification vidéo et la réponse aux questions visuelles.

🤗 Transformers fournit des API pour télécharger et utiliser rapidement ces modèles pré-entraînés sur un texte donné, les affiner sur vos propres ensembles de données, puis les partager avec la communauté sur notre [hub de modèles](https://huggingface.co/models). En même temps, chaque module Python définissant une architecture est complètement indépendant et peut être modifié pour permettre des expériences de recherche rapides.

🤗 Transformers est soutenu par les trois bibliothèques d'apprentissage profond les plus populaires — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) et [TensorFlow](https://www.tensorflow.org/) — avec une intégration transparente entre eux. Il est facile de former vos modèles avec l'un avant de les charger pour l'inférence avec l'autre.

## Démos en ligne

Vous pouvez tester la plupart de nos modèles directement sur leurs pages du [hub de modèles](https://huggingface.co/models). Nous proposons également [l'hébergement privé de modèles, le versionning et une API d'inférence](https://huggingface.co/pricing) pour des modèles publics et privés.

Voici quelques exemples :

En traitement du langage naturel :
- [Complétion de mots masqués avec BERT](https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Reconnaissance d'entités nommées avec Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Génération de texte avec GPT-2](https://huggingface.co/gpt2?text=A+long+time+ago%2C+)
- [Inférence de langage naturel avec RoBERTa](https://huggingface.co/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Résumé avec BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Réponse aux questions avec DistilBERT](https://huggingface.co/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Traduction avec T5](https://huggingface.co/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

En vision par ordinateur :
- [Classification d'images avec ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Détection d'objets avec DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Segmentation sémantique avec SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Segmentation panoptique avec MaskFormer](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [Estimation de profondeur avec DPT](https://huggingface.co/docs/transformers/model_doc/dpt)
- [Classification vidéo avec VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Segmentation universelle avec OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

En audio :
- [Reconnaissance automatique de la parole avec Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Spotting de mots-clés avec Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Classification audio avec Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

Dans les tâches multimodales :
- [Réponses aux questions sur table avec TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Réponses aux questions visuelles avec ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Classification d'images sans étiquette avec CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
- [Réponses aux questions sur les documents avec LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Classification vidéo sans étiquette avec X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)


## 100 projets utilisant Transformers

Transformers est plus qu'une boîte à outils pour utiliser des modèles pré-entraînés : c'est une communauté de projets construits autour de lui et du Hub Hugging Face. Nous voulons que Transformers permette aux développeurs, chercheurs, étudiants, professeurs, ingénieurs et à quiconque d'imaginer et de réaliser leurs projets de rêve.

Afin de célébrer les 100 000 étoiles de transformers, nous avons décidé de mettre en avant la communauté et avons créé la page [awesome-transformers](./awesome-transformers.md) qui répertorie 100 projets incroyables construits autour de transformers.

Si vous possédez ou utilisez un projet que vous pensez devoir figurer dans la liste, veuillez ouvrir une pull request pour l'ajouter !

## Si vous recherchez un support personnalisé de la part de l'équipe Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="Programme d'accélération des experts HuggingFace" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Tour rapide

Pour utiliser immédiatement un modèle sur une entrée donnée (texte, image, audio,...), nous fournissons l'API `pipeline`. Les pipelines regroupent un modèle pré-entraîné avec la préparation des données qui a été utilisée lors de l'entraînement de ce modèle. Voici comment utiliser rapidement un pipeline pour classer des textes en positif ou négatif :

```python
>>> from transformers import pipeline

# Allouer un pipeline pour l'analyse de sentiment
>>> classifieur = pipeline('sentiment-analysis')
>>> classifieur('Nous sommes très heureux d'introduire le pipeline dans le référentiel transformers.')
[{'label': 'POSITIF', 'score': 0.9996980428695679}]
```

La deuxième ligne de code télécharge et met en cache le modèle pré-entraîné utilisé par le pipeline, tandis que la troisième l'évalue sur le texte donné. Ici, la réponse est "positive" avec une confiance de 99,97%.

De nombreuses tâches ont une pipeline pré-entraîné prêt à l'emploi, en NLP, mais aussi en vision par ordinateur et en parole. Par exemple, nous pouvons facilement extraire les objets détectés dans une image :

```python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Télécharger une image avec de jolis chats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> donnees_image = requests.get(url, stream=True).raw
>>> image = Image.open(donnees_image)

# Allouer un pipeline pour la détection d'objets
>>> detecteur_objets = pipeline('object-detection')
>>> detecteur_objets(image)
[{'score': 0.9982201457023621,
  'label': 'télécommande',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'télécommande',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'canapé',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'chat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'chat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

Ici, nous obtenons une liste d'objets détectés dans l'image, avec une boîte entourant l'objet et un score de confiance. Voici l'image originale à gauche, avec les prédictions affichées à droite :

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

Vous pouvez en savoir plus sur les tâches supportées par l'API pipeline dans [ce tutoriel](https://huggingface.co/docs/transformers/task_summary).

En plus de `pipeline`, pour télécharger et utiliser n'importe lequel des modèles pré-entraînés sur votre tâche donnée, il suffit de trois lignes de code. Voici la version PyTorch :

```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Bonjour le monde !", return_tensors="pt")
outputs = model(**inputs)
```

Et voici le code équivalent pour TensorFlow :

```python
from transformers import AutoTokenizer, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Bonjour le monde !", return_tensors="tf")
outputs = model(**inputs)
```

Le tokenizer est responsable de toutes les étapes de prétraitement que le modèle préentraîné attend et peut être appelé directement sur une seule chaîne de caractères (comme dans les exemples ci-dessus) ou sur une liste. Il produira un dictionnaire que vous pouvez utiliser dans votre code ou simplement passer directement à votre modèle en utilisant l'opérateur de déballage **.

Le modèle lui-même est un module [`nn.Module` PyTorch](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) ou un modèle [`tf.keras.Model` TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (selon votre backend) que vous pouvez utiliser comme d'habitude. [Ce tutoriel](https://huggingface.co/docs/transformers/training) explique comment intégrer un tel modèle dans une boucle d'entraînement classique PyTorch ou TensorFlow, ou comment utiliser notre API `Trainer` pour affiner rapidement sur un nouvel ensemble de données.

## Pourquoi devrais-je utiliser transformers ?

1. Des modèles de pointe faciles à utiliser :
    - Hautes performances en compréhension et génération de langage naturel, en vision par ordinateur et en tâches audio.
    - Faible barrière à l'entrée pour les éducateurs et les praticiens.
    - Peu d'abstractions visibles pour l'utilisateur avec seulement trois classes à apprendre.
    - Une API unifiée pour utiliser tous nos modèles préentraînés.

1. Coûts informatiques réduits, empreinte carbone plus petite :
    - Les chercheurs peuvent partager des modèles entraînés au lieu de toujours les réentraîner.
    - Les praticiens peuvent réduire le temps de calcul et les coûts de production.
    - Des dizaines d'architectures avec plus de 400 000 modèles préentraînés dans toutes les modalités.

1. Choisissez le bon framework pour chaque partie de la vie d'un modèle :
    - Entraînez des modèles de pointe en 3 lignes de code.
    - Trasnférer un seul modèle entre les frameworks TF2.0/PyTorch/JAX à volonté.
    - Choisissez facilement le bon framework pour l'entraînement, l'évaluation et la production.

1. Personnalisez facilement un modèle ou un exemple selon vos besoins :
    - Nous fournissons des exemples pour chaque architecture afin de reproduire les résultats publiés par ses auteurs originaux.
    - Les détails internes du modèle sont exposés de manière aussi cohérente que possible.
    - Les fichiers de modèle peuvent être utilisés indépendamment de la bibliothèque pour des expériences rapides.

## Pourquoi ne devrais-je pas utiliser transformers ?

- Cette bibliothèque n'est pas une boîte à outils modulaire de blocs de construction pour les réseaux neuronaux. Le code dans les fichiers de modèle n'est pas refactored avec des abstractions supplémentaires à dessein, afin que les chercheurs puissent itérer rapidement sur chacun des modèles sans plonger dans des abstractions/fichiers supplémentaires.
- L'API d'entraînement n'est pas destinée à fonctionner avec n'importe quel modèle, mais elle est optimisée pour fonctionner avec les modèles fournis par la bibliothèque. Pour des boucles génériques d'apprentissage automatique, vous devriez utiliser une autre bibliothèque (éventuellement, [Accelerate](https://huggingface.co/docs/accelerate)).
- Bien que nous nous efforcions de présenter autant de cas d'utilisation que possible, les scripts de notre [dossier d'exemples](https://github.com/huggingface/transformers/tree/main/examples) ne sont que cela : des exemples. Il est prévu qu'ils ne fonctionnent pas immédiatement sur votre problème spécifique et que vous devrez probablement modifier quelques lignes de code pour les adapter à vos besoins.

## Installation

### Avec pip

Ce référentiel est testé sur Python 3.8+, Flax 0.4.1+, PyTorch 1.11+ et TensorFlow 2.6+.

Vous devriez installer 🤗 Transformers dans un [environnement virtuel](https://docs.python.org/3/library/venv.html). Si vous n'êtes pas familier avec les environnements virtuels Python, consultez le [guide utilisateur](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

D'abord, créez un environnement virtuel avec la version de Python que vous allez utiliser et activez-le.

Ensuite, vous devrez installer au moins l'un de Flax, PyTorch ou TensorFlow.
Veuillez vous référer à la page d'installation de [TensorFlow](https://www.tensorflow.org/install/), de [PyTorch](https://pytorch.org/get-started/locally/#start-locally) et/ou de [Flax](https://github.com/google/flax#quick-install) et [Jax](https://github.com/google/jax#installation) pour connaître la commande d'installation spécifique à votre plateforme.

Lorsqu'un de ces backends est installé, 🤗 Transformers peut être installé avec pip comme suit :

```bash
pip install transformers
```

Si vous souhaitez jouer avec les exemples ou avez besoin de la dernière version du code et ne pouvez pas attendre une nouvelle version, vous devez [installer la bibliothèque à partir de la source](https://huggingface.co/docs/transformers/installation#installing-from-source).

### Avec conda

🤗 Transformers peut être installé avec conda comme suit :

```shell
conda install conda-forge::transformers
```

> **_NOTE:_** L'installation de `transformers` depuis le canal `huggingface` est obsolète.

Suivez les pages d'installation de Flax, PyTorch ou TensorFlow pour voir comment les installer avec conda.

> **_NOTE:_** Sur Windows, on peut vous demander d'activer le mode développeur pour bénéficier de la mise en cache. Si ce n'est pas une option pour vous, veuillez nous le faire savoir dans [cette issue](https://github.com/huggingface/huggingface_hub/issues/1062).

## Architectures de modèles

**[Tous les points de contrôle](https://huggingface.co/models)** de modèle fournis par 🤗 Transformers sont intégrés de manière transparente depuis le [hub de modèles](https://huggingface.co/models) huggingface.co, où ils sont téléchargés directement par les [utilisateurs](https://huggingface.co/users) et les [organisations](https://huggingface.co/organizations).

Nombre actuel de points de contrôle : ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)


🤗 Transformers fournit actuellement les architectures suivantes (consultez [ici](https://huggingface.co/docs/transformers/model_summary) pour un résumé global de chacune d'entre elles) :
1. **[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)** (de Google Research et du Toyota Technological Institute at Chicago) publié dans l'article [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), par Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
1. **[ALIGN](https://huggingface.co/docs/transformers/model_doc/align)** (de Google Research) publié dans l'article [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) de Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
1. **[AltCLIP](https://huggingface.co/docs/transformers/model_doc/altclip)** (de BAAI) publié dans l'article [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679) de Chen, Zhongzhi et Liu, Guang et Zhang, Bo-Wen et Ye, Fulong et Yang, Qinghong et Wu, Ledell.
1. **[Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)** (du MIT) publié dans l'article [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) de Yuan Gong, Yu-An Chung, James Glass.
1. **[Autoformer](https://huggingface.co/docs/transformers/model_doc/autoformer)** (de l'Université Tsinghua) publié dans l'article [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) de Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
1. **[Bark](https://huggingface.co/docs/transformers/model_doc/bark)** (de Suno) publié dans le référentiel [suno-ai/bark](https://github.com/suno-ai/bark) par l'équipe Suno AI.
1. **[BART](https://huggingface.co/docs/transformers/model_doc/bart)** (de Facebook) publié dans l'article [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) de Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov et Luke Zettlemoyer.
1. **[BARThez](https://huggingface.co/docs/transformers/model_doc/barthez)** (de l'École polytechnique) publié dans l'article [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://arxiv.org/abs/2010.12321) de Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
1. **[BARTpho](https://huggingface.co/docs/transformers/model_doc/bartpho)** (de VinAI Research) publié dans l'article [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese](https://arxiv.org/abs/2109.09701) de Nguyen Luong Tran, Duong Minh Le et Dat Quoc Nguyen.
1. **[BEiT](https://huggingface.co/docs/transformers/model_doc/beit)** (de Microsoft) publié dans l'article [BEiT: Pré-entraînement BERT des transformateurs d'images](https://arxiv.org/abs/2106.08254) par Hangbo Bao, Li Dong, Furu Wei.
1. **[BERT](https://huggingface.co/docs/transformers/model_doc/bert)** (de Google) publié dans l'article [BERT : Pré-entraînement de transformateurs bidirectionnels profonds pour la compréhension du langage](https://arxiv.org/abs/1810.04805) par Jacob Devlin, Ming-Wei Chang, Kenton Lee et Kristina Toutanova.
1. **[BERT For Sequence Generation](https://huggingface.co/docs/transformers/model_doc/bert-generation)** (de Google) publié dans l'article [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) parSascha Rothe, Shashi Narayan, Aliaksei Severyn.
1. **[BERTweet](https://huggingface.co/docs/transformers/model_doc/bertweet)** (de VinAI Research) publié dans l'article [BERTweet : un modèle de langage pré-entraîné pour les Tweets en anglais](https://aclanthology.org/2020.emnlp-demos.2/) par Dat Quoc Nguyen, Thanh Vu et Anh Tuan Nguyen.
1. **[BigBird-Pegasus](https://huggingface.co/docs/transformers/model_doc/bigbird_pegasus)** (de Google Research) publié dans l'article [Big Bird: Transformateurs pour des séquences plus longues](https://arxiv.org/abs/2007.14062) par Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
1. **[BigBird-RoBERTa](https://huggingface.co/docs/transformers/model_doc/big_bird)** (de Google Research) publié dans l'article [Big Bird: Transformateurs pour des séquences plus longues](https://arxiv.org/abs/2007.14062) par Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
1. **[BioGpt](https://huggingface.co/docs/transformers/model_doc/biogpt)** (de Microsoft Research AI4Science) publié dans l'article [BioGPT : transformateur génératif pré-entraîné pour la génération et l'extraction de texte biomédical](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9) par Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon et Tie-Yan Liu.
1. **[BiT](https://huggingface.co/docs/transformers/model_doc/bit)** (de Google AI) publié dans l'article [Big Transfer (BiT) : Apprentissage général de la représentation visuelle](https://arxiv.org/abs/1912.11370) par Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby.
1. **[Blenderbot](https://huggingface.co/docs/transformers/model_doc/blenderbot)** (de Facebook) publié dans l'article [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637) par Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
1. **[BlenderbotSmall](https://huggingface.co/docs/transformers/model_doc/blenderbot-small)** (de Facebook) publié dans l'article [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637) par Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
1. **[BLIP](https://huggingface.co/docs/transformers/model_doc/blip)** (de Salesforce) publié dans l'article [BLIP : Pré-entraînement de la langue et de l'image par bootstrap pour une compréhension et une génération unifiées de la vision et du langage](https://arxiv.org/abs/2201.12086) par Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.
1. **[BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)** (de Salesforce) publié dans l'article [BLIP-2 : Pré-entraînement de la langue et de l'image avec des encodeurs d'images gelés et de grands modèles de langage](https://arxiv.org/abs/2301.12597) par Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.
1. **[BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom)** (de l'atelier BigScience) publié par l'[atelier BigScience](https://bigscience.huggingface.co/).
1. **[BORT](https://huggingface.co/docs/transformers/model_doc/bort)** (d'Alexa) publié dans l'article [Optimal Subarchitecture Extraction For BERT](https://arxiv.org/abs/2010.10499) par Adrian de Wynter et Daniel J. Perry.
1. **[BridgeTower](https://huggingface.co/docs/transformers/model_doc/bridgetower)** (de l'Institut de technologie de Harbin/Microsoft Research Asia/Intel Labs) publié dans l'article [BridgeTower : Construire des ponts entre les encodeurs dans l'apprentissage de la représentation vision-langage](https://arxiv.org/abs/2206.08657) par Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan.
1. **[BROS](https://huggingface.co/docs/transformers/model_doc/bros)** (de NAVER CLOVA) publié dans l'article [BROS : un modèle de langage pré-entraîné axé sur le texte et la mise en page pour une meilleure extraction des informations clés des documents](https://arxiv.org/abs/2108.04539) par Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park.
1. **[ByT5](https://huggingface.co/docs/transformers/model_doc/byt5)** (de Google Research) publié dans l'article [ByT5 : Vers un futur sans jeton avec des modèles pré-entraînés byte-to-byte](https://arxiv.org/abs/2105.13626) par Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
1. **[CamemBERT](https://huggingface.co/docs/transformers/model_doc/camembert)** (d'Inria/Facebook/Sorbonne) publié dans l'article [CamemBERT : un modèle de langue français savoureux](https://arxiv.org/abs/1911.03894) par Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah et Benoît Sagot.
1. **[CANINE](https://huggingface.co/docs/transformers/model_doc/canine)** (de Google Research) publié dans l'article [CANINE : Pré-entraînement d'un encodeur sans tokenisation efficace pour la représentation du langage](https://arxiv.org/abs/2103.06874) par Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
1. **[Chinese-CLIP](https://huggingface.co/docs/transformers/model_doc/chinese_clip)** (d'OFA-Sys) publié dans l'article [Chinese CLIP : Pré-entraînement contrastif vision-langage en chinois](https://arxiv.org/abs/2211.01335) par An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou.
1. **[CLAP](https://huggingface.co/docs/transformers/model_doc/clap)** (de LAION-AI) publié dans l'article [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) par Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.
1. **[CLIP](https://huggingface.co/docs/transformers/model_doc/clip)** (d'OpenAI) publié dans l'article [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) par Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
1. **[CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)** (de l'Université de Göttingen) publié dans l'article [Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003) par Timo Lüddecke et Alexander Ecker.
1. **[CLVP](https://huggingface.co/docs/transformers/model_doc/clvp)** publié dans l'article [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243) par James Betker.
1. **[CodeGen](https://huggingface.co/docs/transformers/model_doc/codegen)** (de Salesforce) publié dans l'article [A Conversational Paradigm for Program Synthesis](https://arxiv.org/abs/2203.13474) par Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong.
1. **[CodeLlama](https://huggingface.co/docs/transformers/model_doc/llama_code)** (de MetaAI) publié dans l'article [Code Llama : Modèles ouverts fondamentaux pour le code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) par Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve.
1. **[Conditional DETR](https://huggingface.co/docs/transformers/model_doc/conditional_detr)** (de Microsoft Research Asia) publié dans l'article [Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152) par Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang.
1. **[ConvBERT](https://huggingface.co/docs/transformers/model_doc/convbert)** (de YituTech) publié dans l'article [ConvBERT : Amélioration de BERT avec une convolution dynamique basée sur des plages](https://arxiv.org/abs/2008.02496) par Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
1. **[ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext)** (de Facebook AI) publié dans l'article [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) par Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
1. **[ConvNeXTV2](https://huggingface.co/docs/transformers/model_doc/convnextv2)** (de Facebook AI) publié dans l'article [ConvNeXt V2 : Conception conjointe et mise à l'échelle de ConvNets avec des autoencodeurs masqués](https://arxiv.org/abs/2301.00808) par Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
1. **[CPM](https://huggingface.co/docs/transformers/model_doc/cpm)** (de l'Université de Tsinghua) publié dans l'article [CPM : Un modèle de langue chinois pré-entraîné génératif à grande échelle](https://arxiv.org/abs/2012.00413) par Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
1. **[CPM-Ant](https://huggingface.co/docs/transformers/model_doc/cpmant)** (d'OpenBMB) publié par l'[OpenBMB](https://www.openbmb.org/).
1. **[CTRL](https://huggingface.co/docs/transformers/model_doc/ctrl)** (de Salesforce) publié dans l'article [CTRL : Un modèle de langage conditionnel de type Transformer pour une génération contrôlable](https://arxiv.org/abs/1909.05858) par Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong et Richard Socher.
1. **[CvT](https://huggingface.co/docs/transformers/model_doc/cvt)** (de Microsoft) publié dans l'article [CvT : Introduction de convolutions aux transformateurs visuels](https://arxiv.org/abs/2103.15808) par Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.
1. **[Data2Vec](https://huggingface.co/docs/transformers/model_doc/data2vec)** (de Facebook) publié dans l'article [Data2Vec : Un cadre général pour l'apprentissage auto-supervisé en parole, vision et langage](https://arxiv.org/abs/2202.03555) par Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
1. **[DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta)** (de Microsoft) publié dans l'article [DeBERTa : BERT amélioré avec attention désentrelacée](https://arxiv.org/abs/2006.03654) par Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
1. **[DeBERTa-v2](https://huggingface.co/docs/transformers/model_doc/deberta-v2)** (de Microsoft) publié dans l'article [DeBERTa : BERT amélioré avec attention désentrelacée](https://arxiv.org/abs/2006.03654) par Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
1. **[Decision Transformer](https://huggingface.co/docs/transformers/model_doc/decision_transformer)** (de Berkeley/Facebook/Google) publié dans l'article [Decision Transformer : Apprentissage par renforcement via la modélisation de séquences](https://arxiv.org/abs/2106.01345) par Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
1. **[Deformable DETR](https://huggingface.co/docs/transformers/model_doc/deformable_detr)** (de SenseTime Research) publié dans l'article [Deformable DETR : Transformateurs déformables pour la détection d'objets de bout en bout](https://arxiv.org/abs/2010.04159) par Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai.
1. **[DeiT](https://huggingface.co/docs/transformers/model_doc/deit)** (de Facebook) publié dans l'article [Entraînement d'images efficace et distillation par l'attention](https://arxiv.org/abs/2012.12877) par Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
1. **[DePlot](https://huggingface.co/docs/transformers/model_doc/deplot)** (de Google AI) publié dans l'article [DePlot : Raisonnement visuel en une étape par traduction de l'intrigue en tableau](https://arxiv.org/abs/2212.10505) par Fangyu Liu, Julian Martin Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, Yasemin Altun.
1. **[Depth Anything](https://huggingface.co/docs/transformers/main/model_doc/depth_anything)** (de l'université d'Hong Kong et TikTok) publié dans l'article    [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891) by Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao.
1. **[DETA](https://huggingface.co/docs/transformers/model_doc/deta)** (de l'Université du Texas à Austin) publié dans l'article [NMS Strikes Back](https://arxiv.org/abs/2212.06137) par Jeffrey Ouyang-Zhang, Jang Hyun Cho, Xingyi Zhou, Philipp Krähenbühl.
1. **[DETR](https://huggingface.co/docs/transformers/model_doc/detr)** (de Facebook) publié dans l'article [Détection d'objets de bout en bout avec des transformateurs](https://arxiv.org/abs/2005.12872) par Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
1. **[DialoGPT](https://huggingface.co/docs/transformers/model_doc/dialogpt)** (de Microsoft Research) publié dans l'article [DialoGPT : Pré-entraînement génératif à grande échelle pour la génération de réponses conversationnelles](https://arxiv.org/abs/1911.00536) par Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
1. **[DiNAT](https://huggingface.co/docs/transformers/model_doc/dinat)** (de SHI Labs) publié dans l'article [Transformateur d'attention dilatée pour l'attention aux quartiers](https://arxiv.org/abs/2209.15001) par Ali Hassani et Humphrey Shi.
1. **[DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2)** (de Meta AI) publié dans l'article [DINOv2 : Apprentissage de fonctionnalités visuelles robustes sans supervision](https://arxiv.org/abs/2304.07193) par Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski.
1. **[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)** (de HuggingFace), publié dans l'article [DistilBERT, une version condensée de BERT : plus petit, plus rapide, moins cher et plus léger](https://arxiv.org/abs/1910.01108) par Victor Sanh, Lysandre Debut et Thomas Wolf. La même méthode a été appliquée pour compresser GPT2 en [DistilGPT2](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation), RoBERTa en [DistilRoBERTa](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation), Multilingual BERT en [DistilmBERT](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) et une version allemande de DistilBERT.
1. **[DiT](https://huggingface.co/docs/transformers/model_doc/dit)** (de Microsoft Research) publié dans l'article [DiT : Auto-pré-entraînement pour le transformateur d'images de documents](https://arxiv.org/abs/2203.02378) par Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei.
1. **[Donut](https://huggingface.co/docs/transformers/model_doc/donut)** (de NAVER), publié dans l'article [Transformation de compréhension de documents sans OCR](https://arxiv.org/abs/2111.15664) par Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park.
1. **[DPR](https://huggingface.co/docs/transformers/model_doc/dpr)** (de Facebook) publié dans l'article [Passage dense pour la recherche de réponses à des questions en domaine ouvert](https://arxiv.org/abs/2004.04906) par Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen et Wen-tau Yih.
1. **[DPT](https://huggingface.co/docs/transformers/master/model_doc/dpt)** (d'Intel Labs) publié dans l'article [Transformateurs de vision pour la prédiction dense](https://arxiv.org/abs/2103.13413) par René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
1. **[EfficientFormer](https://huggingface.co/docs/transformers/model_doc/efficientformer)** (de Snap Research) publié dans l'article [EfficientFormer : Transformateurs de vision à la vitesse de MobileNet](https://arxiv.org/abs/2206.01191) par Yanyu Li, Geng Yuan, Yang Wen, Ju Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren.
1. **[EfficientNet](https://huggingface.co/docs/transformers/model_doc/efficientnet)** (de Google Brain) publié dans l'article [EfficientNet: Repenser l'échelle des modèles pour les réseaux de neurones convolutionnels](https://arxiv.org/abs/1905.11946) par Mingxing Tan, Quoc V. Le.
1. **[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)** (de Google Research/Université Stanford) publié dans l'article [ELECTRA: Pré-entraîner les encodeurs de texte en tant que discriminateurs plutôt que des générateurs](https://arxiv.org/abs/2003.10555) par Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
1. **[EnCodec](https://huggingface.co/docs/transformers/model_doc/encodec)** (de Meta AI) publié dans l'article [Compression neuronale audio de haute fidélité](https://arxiv.org/abs/2210.13438) par Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi.
1. **[EncoderDecoder](https://huggingface.co/docs/transformers/model_doc/encoder-decoder)** (de Google Research) publié dans l'article [Exploiter des points de contrôle pré-entraînés pour les tâches de génération de séquences](https://arxiv.org/abs/1907.12461) par Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
1. **[ERNIE](https://huggingface.co/docs/transformers/model_doc/ernie)** (de Baidu) publié dans l'article [ERNIE: Intégration améliorée des représentations par la connaissance](https://arxiv.org/abs/1904.09223) par Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, Hua Wu.
1. **[ErnieM](https://huggingface.co/docs/transformers/model_doc/ernie_m)** (de Baidu) publié dans l'article [ERNIE-M: Représentation multilingue améliorée en alignant les sémantiques interlingues avec des corpus monolingues](https://arxiv.org/abs/2012.15674) par Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang.
1. **[ESM](https://huggingface.co/docs/transformers/model_doc/esm)** (de Meta AI) sont des modèles de langage de protéines de type transformateur. **ESM-1b** a été publié dans l'article [La structure et la fonction biologiques émergent de la mise à l'échelle de l'apprentissage non supervisé à 250 millions de séquences de protéines](https://www.pnas.org/content/118/15/e2016239118) par Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma et Rob Fergus. **ESM-1v** a été publié dans l'article [Les modèles de langage permettent une prédiction hors champ des effets des mutations sur la fonction des protéines](https://doi.org/10.1101/2021.07.09.450648) par Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu et Alexander Rives. **ESM-2 et ESMFold** ont été publiés avec l'article [Les modèles de langage des séquences de protéines à l'échelle de l'évolution permettent une prédiction précise de la structure](https://doi.org/10.1101/2022.07.20.500902) par Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido, Alexander Rives.
1. **[Falcon](https://huggingface.co/docs/transformers/model_doc/falcon)** (de Technology Innovation Institute) par Almazrouei, Ebtesam et Alobeidli, Hamza et Alshamsi, Abdulaziz et Cappelli, Alessandro et Cojocaru, Ruxandra et Debbah, Merouane et Goffinet, Etienne et Heslow, Daniel et Launay, Julien et Malartic, Quentin et Noune, Badreddine et Pannier, Baptiste et Penedo, Guilherme.
1. **[FastSpeech2Conformer](model_doc/fastspeech2_conformer)** (d'ESPnet) publié dans l'article [Développements récents sur la boîte à outils Espnet boostés par Conformer](https://arxiv.org/abs/2010.13956) par Pengcheng Guo, Florian Boyer, Xuankai Chang, Tomoki Hayashi, Yosuke Higuchi, Hirofumi Inaguma, Naoyuki Kamo, Chenda Li, Daniel Garcia-Romero, Jiatong Shi, Jing Shi, Shinji Watanabe, Kun Wei, Wangyou Zhang et Yuekai Zhang.
1. **[FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)** (de Google AI) publié dans le référentiel [google-research/t5x](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints) par Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le et Jason Wei
1. **[FLAN-UL2](https://huggingface.co/docs/transformers/model_doc/flan-ul2)** (de Google AI) publié dans le référentiel [google-research/t5x](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-ul2-checkpoints) par Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le et Jason Wei
1. **[FlauBERT](https://huggingface.co/docs/transformers/model_doc/flaubert)** (du CNRS) publié dans l'article [FlauBERT : Pré-entraînement de modèle de langue non supervisé pour le français](https://arxiv.org/abs/1912.05372) par Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
1. **[FLAVA](https://huggingface.co/docs/transformers/model_doc/flava)** (de Facebook AI) publié dans l'article [FLAVA : Un modèle fondamental d'alignement de la langue et de la vision](https://arxiv.org/abs/2112.04482) par Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach et Douwe Kiela.
1. **[FNet](https://huggingface.co/docs/transformers/model_doc/fnet)** (de Google Research) publié dans l'article [FNet : Mélanger les jetons avec des transformations de Fourier](https://arxiv.org/abs/2105.03824) par James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.
1. **[FocalNet](https://huggingface.co/docs/transformers/model_doc/focalnet)** (de Microsoft Research) publié dans l'article [Réseaux de modulation focale](https://arxiv.org/abs/2203.11926) par Jianwei Yang, Chunyuan Li, Xiyang Dai, Lu Yuan, Jianfeng Gao.
1. **[Funnel Transformer](https://huggingface.co/docs/transformers/model_doc/funnel)** (de l'Université Carnegie Mellon/Google Brain) publié dans l'article [Funnel-Transformer : Filtrer la redondance séquentielle pour un traitement efficace du langage](https://arxiv.org/abs/2006.03236) par Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
1. **[Fuyu](https://huggingface.co/docs/transformers/model_doc/fuyu)** (de ADEPT) Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, Sağnak Taşırlar. Publié dans l'article [billet de blog](https://www.adept.ai/blog/fuyu-8b)
1. **[GIT](https://huggingface.co/docs/transformers/model_doc/git)** (de Microsoft Research) publié dans l'article [GIT : Un transformateur génératif d'images en texte pour la vision et le langage](https://arxiv.org/abs/2205.14100) par Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, Lijuan Wang.
1. **[GLPN](https://huggingface.co/docs/transformers/model_doc/glpn)** (de la KAIST) publié dans l'article [Réseaux de chemins globaux-locaux pour l'estimation de profondeur monoculaire avec Vertical CutDepth](https://arxiv.org/abs/2201.07436) par Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim.
1. **[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)** (d'OpenAI) publié dans l'article [Améliorer la compréhension du langage par l'apprentissage préalable génératif](https://openai.com/research/language-unsupervised/) par Alec Radford, Karthik Narasimhan, Tim Salimans et Ilya Sutskever.
1. **[GPT Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)** (d'EleutherAI) publié dans le référentiel [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) par Sid Black, Stella Biderman, Leo Gao, Phil Wang et Connor Leahy.
1. **[GPT NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox)** (d'EleutherAI) publié dans l'article [GPT-NeoX-20B: Un modèle de langue autonome open source](https://arxiv.org/abs/2204.06745) par Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, Samuel Weinbach
1. **[GPT NeoX Japanese](https://huggingface.co/docs/transformers/model_doc/gpt_neox_japanese)** (de ABEJA) publié par Shinya Otani, Takayoshi Makabe, Anuj Arora et Kyo Hattori.
1. **[GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)** (d'OpenAI) a été publié dans l'article [Les modèles de langage sont des apprenants multitâches non supervisés](https://openai.com/research/better-language-models/) par Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei et Ilya Sutskever.
1. **[GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)** (d'EleutherAI) a été publié dans le dépôt [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) par Ben Wang et Aran Komatsuzaki.
1. **[GPT-Sw3](https://huggingface.co/docs/transformers/model_doc/gpt-sw3)** (d'AI-Sweden) a été publié dans l'article [Leçons apprises de GPT-SW3 : Construction du premier modèle de langage génératif à grande échelle pour le suédois](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf) par Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman, Fredrik Carlsson, Magnus Sahlgren.
1. **[GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode)** (de BigCode) a été publié dans l'article [SantaCoder: ne visez pas les étoiles !](https://arxiv.org/abs/2301.03988) par Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane Anderson, Yangtian Zi, Joel Lamy Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky, Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes, Daniel Fried, Arjun Guha, Harm de Vries, Leandro von Werra.
1. **[GPTSAN-japanese](https://huggingface.co/docs/transformers/model_doc/gptsan-japanese)** a été publié dans le dépôt [tanreinama/GPTSAN](https://github.com/tanreinama/GPTSAN/blob/main/report/model.md) par Toshiyuki Sakamoto (tanreinama).
1. **[Graphormer](https://huggingface.co/docs/transformers/model_doc/graphormer)** (de Microsoft) a été publié dans l'article [Les Transformers sont-ils vraiment inefficaces pour la représentation graphique ?](https://arxiv.org/abs/2106.05234) par Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu.
1. **[GroupViT](https://huggingface.co/docs/transformers/model_doc/groupvit)** (de l'UCSD, NVIDIA) a été publié dans l'article [GroupViT : la segmentation sémantique émerge de la supervision textuelle](https://arxiv.org/abs/2202.11094) par Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.
1. **[HerBERT](https://huggingface.co/docs/transformers/model_doc/herbert)** (d'Allegro.pl, AGH University of Science and Technology) a été publié dans l'article [KLEJ : référentiel complet pour la compréhension du langage polonais](https://www.aclweb.org/anthology/2020.acl-main.111.pdf) par Piotr Rybak, Robert Mroczkowski, Janusz Tracz, Ireneusz Gawlik.
1. **[Hubert](https://huggingface.co/docs/transformers/model_doc/hubert)** (de Facebook) a été publié dans l'article [HuBERT : Apprentissage de la représentation autonome de la parole par prédiction masquée des unités cachées](https://arxiv.org/abs/2106.07447) par Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.
1. **[I-BERT](https://huggingface.co/docs/transformers/model_doc/ibert)** (de Berkeley) a été publié dans l'article [I-BERT : Quantification entière de BERT avec des entiers uniquement](https://arxiv.org/abs/2101.01321) par Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer.
1. **[IDEFICS](https://huggingface.co/docs/transformers/model_doc/idefics)** (de HuggingFace) a été publié dans l'article [OBELICS : Un ensemble de données filtré à l'échelle du Web d'intercalation de documents texte-image](https://huggingface.co/papers/2306.16527) par Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh.
1. **[ImageGPT](https://huggingface.co/docs/transformers/model_doc/imagegpt)** (d'OpenAI) a été publié dans l'article [Pré-entraînement génératif à partir de pixels](https://openai.com/blog/image-gpt/) par Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever.
1. **[Informer](https://huggingface.co/docs/transformers/model_doc/informer)** (de l'Université de Beihang, UC Berkeley, Rutgers University, SEDD Company) a été publié dans l'article [Informer : Au-delà du Transformer efficace pour la prévision de séries temporel
1. **[InstructBLIP](https://huggingface.co/docs/transformers/model_doc/instructblip)** (de Salesforce) a été publié dans l'article [InstructBLIP : Vers des modèles vision-langage polyvalents avec un réglage d'instructions](https://arxiv.org/abs/2305.06500) de Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.
1. **[Jukebox](https://huggingface.co/docs/transformers/model_doc/jukebox)** (d'OpenAI) a été publié dans l'article [Jukebox : Un modèle génératif pour la musique](https://arxiv.org/pdf/2005.00341.pdf) de Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, Ilya Sutskever.
1. **[KOSMOS-2](https://huggingface.co/docs/transformers/model_doc/kosmos-2)** (de Microsoft Research Asia) a été publié dans l'article [Kosmos-2 : Ancrage de modèles linguistiques multimodaux à travers le monde](https://arxiv.org/abs/2306.14824) de Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei.
1. **[LayoutLM](https://huggingface.co/docs/transformers/model_doc/layoutlm)** (de Microsoft Research Asia) a été publié dans l'article [LayoutLM : Pré-entraînement de texte et de mise en page pour la compréhension d'images de documents](https://arxiv.org/abs/1912.13318) de Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
1. **[LayoutLMv2](https://huggingface.co/docs/transformers/model_doc/layoutlmv2)** (de Microsoft Research Asia) a été publié dans l'article [LayoutLMv2 : Pré-entraînement multimodal pour la compréhension visuellement riche de documents](https://arxiv.org/abs/2012.14740) de Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou.
1. **[LayoutLMv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)** (de Microsoft Research Asia) a été publié dans l'article [LayoutLMv3 : Pré-entraînement pour l'IA de documents avec un masquage de texte et d'image unifié](https://arxiv.org/abs/2204.08387) de Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
1. **[LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutxlm)** (de Microsoft Research Asia) a été publié dans l'article [LayoutXLM : Pré-entraînement multimodal pour la compréhension de documents visuellement riches et multilingues](https://arxiv.org/abs/2104.08836) de Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei.
1. **[LED](https://huggingface.co/docs/transformers/model_doc/led)** (d'AllenAI) a été publié dans l'article [Longformer : Le transformateur de documents longs](https://arxiv.org/abs/2004.05150) de Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[LeViT](https://huggingface.co/docs/transformers/model_doc/levit)** (de Meta AI) a été publié dans l'article [LeViT : Un transformateur de vision déguisé en réseau de neurones convolutionnel pour une inférence plus rapide](https://arxiv.org/abs/2104.01136) de Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze.
1. **[LiLT](https://huggingface.co/docs/transformers/model_doc/lilt)** (de l'Université de technologie du Sud de la Chine) a été publié dans l'article [LiLT : Un transformateur de mise en page simple mais efficace et indépendant de la langue pour la compréhension de documents structurés](https://arxiv.org/abs/2202.13669) de Jiapeng Wang, Lianwen Jin, Kai Ding.
1. **[LLaMA](https://huggingface.co/docs/transformers/model_doc/llama)** (de l'équipe FAIR de Meta AI) a été publié dans l'article [LLaMA : Modèles linguistiques de base ouverts et efficaces](https://arxiv.org/abs/2302.13971) de Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.
1. **[Llama2](https://huggingface.co/docs/transformers/model_doc/llama2)** (de l'équipe FAIR de Meta AI) a été publié dans l'article [Llama2 : Modèles de base ouverts et affinés pour le chat](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) de Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushka rMishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing EllenTan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom.
1. **[LLaVa](https://huggingface.co/docs/transformers/model_doc/llava)** (de Microsoft Research & University of Wisconsin-Madison) a été publié dans l'article [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) de Haotian Liu, Chunyuan Li, Yuheng Li et Yong Jae Lee.
1. **[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer)** (d'AllenAI) a été publié dans l'article [Longformer : Le transformateur de documents longs](https://arxiv.org/abs/2004.05150) de Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[LongT5](https://huggingface.co/docs/transformers/model_doc/longt5)** (de Google AI) a été publié dans l'article [LongT5 : Transformateur de texte-à-texte efficace pour de longues séquences](https://arxiv.org/abs/2112.07916) de Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung, Yinfei Yang.
1. **[LUKE](https://huggingface.co/docs/transformers/model_doc/luke)** (de Studio Ousia) a été publié dans l'article [LUKE : Représentations contextuelles profondes d'entités avec auto-attention consciente des entités](https://arxiv.org/abs/2010.01057) de Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yuji Matsumoto.
1. **[LXMERT](https://huggingface.co/docs/transformers/model_doc/lxmert)** (de l'UNC Chapel Hill) a été publié dans l'article [LXMERT : Apprentissage de représentations d'encodeur cross-modal à partir de transformateurs pour le questionnement en domaine ouvert](https://arxiv.org/abs/1908.07490) de Hao Tan et Mohit Bansal.
1. **[M-CTC-T](https://huggingface.co/docs/transformers/model_doc/mctct)** (de Facebook) a été publié dans l'article [Pseudo-Labeling For Massively Multilingual Speech Recognition](https://arxiv.org/abs/2111.00161) de Loren Lugosch, Tatiana Likhomanenko, Gabriel Synnaeve et Ronan Collobert.
1. **[M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100)** (de Facebook) a été publié dans l'article [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125) de Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin.
1. **[MADLAD-400](https://huggingface.co/docs/transformers/model_doc/madlad-400)** (de Google) a été publié dans l'article [MADLAD-400 : Un ensemble de données multilingue et de niveau document](https://arxiv.org/abs/2309.04662) de Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A. Choquette-Choo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, Orhan Firat.
1. **[MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)** Des modèles de traduction automatique formés avec les données [OPUS](http://opus.nlpl.eu/) par Jörg Tiedemann. Le [cadre Marian](https://marian-nmt.github.io/) est en cours de développement par l'équipe Microsoft Translator.
1. **[MarkupLM](https://huggingface.co/docs/transformers/model_doc/markuplm)** (de Microsoft Research Asia) a été publié dans l'article [MarkupLM : Pré-entraînement de texte et de langage de balisage pour la compréhension visuellement riche de documents](https://arxiv.org/abs/2110.08518) de Junlong Li, Yiheng Xu, Lei Cui, Furu Wei.
1. **[Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former)** (de FAIR et UIUC) a été publié dans l'article [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) de Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar.
1. **[MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer)** (de Meta et UIUC) a été publié dans l'article [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278) de Bowen Cheng, Alexander G. Schwing, Alexander Kirillov.
1. **[MatCha](https://huggingface.co/docs/transformers/model_doc/matcha)** (de Google AI) a été publié dans l'article [MatCha : Amélioration du pré-entraînement de langage visuel avec raisonnement mathématique et décomposition de graphiques](https://arxiv.org/abs/2212.09662) de Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, Julian Martin Eisenschlos.
1. **[mBART](https://huggingface.co/docs/transformers/model_doc/mbart)** (de Facebook) a été publié dans l'article [Pré-entraînement de débruitage multilingue pour la traduction automatique neuronale
1. **[mBART-50](https://huggingface.co/docs/transformers/model_doc/mbart)** (de Facebook) a été publié dans l'article [Traduction multilingue avec un pré-entraînement et un fine-tuning multilingues extensibles](https://arxiv.org/abs/2008.00401) par Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan.
1. **[MEGA](https://huggingface.co/docs/transformers/model_doc/mega)** (de Meta/USC/CMU/SJTU) a été publié dans l'article [Mega : Attention équipée d'une moyenne mobile](https://arxiv.org/abs/2209.10655) par Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May et Luke Zettlemoyer.
1. **[Megatron-BERT](https://huggingface.co/docs/transformers/model_doc/megatron-bert)** (de NVIDIA) a été publié dans l'article [Megatron-LM : Entraînement de modèles linguistiques de plusieurs milliards de paramètres en utilisant le parallélisme de modèle](https://arxiv.org/abs/1909.08053) par Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper et Bryan Catanzaro.
1. **[Megatron-GPT2](https://huggingface.co/docs/transformers/model_doc/megatron_gpt2)** (de NVIDIA) a été publié dans l'article [Megatron-LM : Entraînement de modèles linguistiques de plusieurs milliards de paramètres en utilisant le parallélisme de modèle](https://arxiv.org/abs/1909.08053) par Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper et Bryan Catanzaro.
1. **[MGP-STR](https://huggingface.co/docs/transformers/model_doc/mgp-str)** (d'Alibaba Research) a été publié dans l'article [Prédiction multi-granularité pour la reconnaissance de texte de scène](https://arxiv.org/abs/2209.03592) par Peng Wang, Cheng Da et Cong Yao.
1. **[Mistral](https://huggingface.co/docs/transformers/model_doc/mistral)** (de Mistral AI) par l'équipe [Mistral AI](https://mistral.ai) : Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.
1. **[Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral)** (de Mistral AI) par l'équipe [Mistral AI](https://mistral.ai) : Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.
1. **[mLUKE](https://huggingface.co/docs/transformers/model_doc/mluke)** (de Studio Ousia) a été publié dans l'article [mLUKE : La puissance des représentations d'entités dans les modèles linguistiques pré-entraînés multilingues](https://arxiv.org/abs/2110.08151) par Ryokan Ri, Ikuya Yamada et Yoshimasa Tsuruoka.
1. **[MMS](https://huggingface.co/docs/transformers/model_doc/mms)** (de Facebook) a été publié dans l'article [Mise à l'échelle de la technologie de la parole à plus de 1 000 langues](https://arxiv.org/abs/2305.13516) par Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, Michael Auli.
1. **[MobileBERT](https://huggingface.co/docs/transformers/model_doc/mobilebert)** (de CMU/Google Brain) a été publié dans l'article [MobileBERT : un BERT compact et agnostique pour les tâches sur les appareils à ressources limitées](https://arxiv.org/abs/2004.02984) par Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang et Denny Zhou.
1. **[MobileNetV1](https://huggingface.co/docs/transformers/model_doc/mobilenet_v1)** (de Google Inc.) a été publié dans l'article [MobileNets : Réseaux neuronaux convolutifs efficaces pour les applications de vision mobile](https://arxiv.org/abs/1704.04861) par Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam.
1. **[MobileNetV2](https://huggingface.co/docs/transformers/model_doc/mobilenet_v2)** (de Google Inc.) a été publié dans l'article [MobileNetV2 : Résidus inversés et coudes linéaires](https://arxiv.org/abs/1801.04381) par Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen.
1. **[MobileViT](https://huggingface.co/docs/transformers/model_doc/mobilevit)** (d'Apple) a été publié dans l'article [MobileViT : Vision Transformer léger, polyvalent et adapté aux mobiles](https://arxiv.org/abs/2110.02178) par Sachin Mehta et Mohammad Rastegari.
1. **[MobileViTV2](https://huggingface.co/docs/transformers/model_doc/mobilevitv2)** (d'Apple) a été publié dans l'article [Auto-attention séparable pour les Vision Transformers mobiles](https://arxiv.org/abs/2206.02680) par Sachin Mehta et Mohammad Rastegari.
1. **[MPNet](https://huggingface.co/docs/transformers/model_doc/mpnet)** (de Microsoft Research) a été publié dans l'article [MPNet : Pré-entraînement masqué et permuté pour la compréhension du langage](https://arxiv.org/abs/2004.09297) par Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.
1. **[MPT](https://huggingface.co/docs/transformers/model_doc/mpt)** (de MosaiML) a été publié avec le référentiel [llm-foundry](https://github.com/mosaicml/llm-foundry/) par l'équipe MosaiML NLP.
1. **[MRA](https://huggingface.co/docs/transformers/model_doc/mra)** (de l'Université du Wisconsin - Madison) a été publié dans l'article [Analyse multi-résolution (MRA) pour une auto-attention approximative](https://arxiv.org/abs/2207.10284) par Zhanpeng Zeng, Sourav Pal, Jeffery Kline, Glenn M Fung, Vikas Singh.
1. **[MT5](https://huggingface.co/docs/transformers/model_doc/mt5)** (de Google AI) a été publié dans l'article [mT5 : un transformateur texte-à-texte pré-entraîné massivement multilingue](https://arxiv.org/abs/2010.11934) par Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
1. **[MusicGen](https://huggingface.co/docs/transformers/model_doc/musicgen)** (de Meta) a été publié dans l'article [Génération de musique simple et contrôlable](https://arxiv.org/abs/2306.05284) par Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi et Alexandre Défossez.
1. **[MVP](https://huggingface.co/docs/transformers/model_doc/mvp)** (de RUC AI Box) a été publié dans l'article [MVP : Pré-entraînement supervisé multi-tâche pour la génération de langage naturel](https://arxiv.org/abs/2206.12131) par Tianyi Tang, Junyi Li, Wayne Xin Zhao et Ji-Rong Wen.
1. **[NAT](https://huggingface.co/docs/transformers/model_doc/nat)** (de SHI Labs) a été publié dans l'article [Transformateur d'attention de voisinage](https://arxiv.org/abs/2204.07143) par Ali Hassani, Steven Walton, Jiachen Li, Shen Li et Humphrey Shi.
1. **[Nezha](https://huggingface.co/docs/transformers/model_doc/nezha)** (du laboratoire Noah's Ark de Huawei) a été publié dans l'article [NEZHA : Représentation contextualisée neurale pour la compréhension du langage chinois](https://arxiv.org/abs/1909.00204) par Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen et Qun Liu.
1. **[NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)** (de Meta) a été publié dans l'article [No Language Left Behind : Mise à l'échelle de la traduction automatique centrée sur l'humain](https://arxiv.org/abs/2207.04672) par l'équipe NLLB.
1. **[NLLB-MOE](https://huggingface.co/docs/transformers/model_doc/nllb-moe)** (de Meta) a été publié dans l'article [No Language Left Behind : Mise à l'échelle de la traduction automatique centrée sur l'humain](https://arxiv.org/abs/2207.04672) par l'équipe NLLB.
1. **[Nougat](https://huggingface.co/docs/transformers/model_doc/nougat)** (de Meta AI) a été publié dans l'article [Nougat : Compréhension Optique Neuronale pour les Documents Académiques](https://arxiv.org/abs/2308.13418) par Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic.
1. **[Nyströmformer](https://huggingface.co/docs/transformers/model_doc/nystromformer)** (de l'Université du Wisconsin - Madison) a été publié dans l'article [Nyströmformer : Un algorithme basé sur Nyström pour approximer l'auto-attention](https://arxiv.org/abs/2102.03902) par Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh.
1. **[OneFormer](https://huggingface.co/docs/transformers/model_doc/oneformer)** (de SHI Labs) a été publié dans l'article [OneFormer : Un Transformer pour dominer la segmentation universelle d'images](https://arxiv.org/abs/2211.06220) par Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi.
1. **[OpenLlama](https://huggingface.co/docs/transformers/model_doc/open-llama)** (de [s-JoL](https://huggingface.co/s-JoL)) publié sur GitHub (maintenant supprimé).
1. **[OPT](https://huggingface.co/docs/transformers/master/model_doc/opt)** (de Meta AI) a été publié dans l'article [OPT : Modèles linguistiques Transformer pré-entraînés ouverts](https://arxiv.org/abs/2205.01068) par Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen et al.
1. **[OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)** (de Google AI) a été publié dans l'article [OWL-ViT : Détection d'objets simple à vocabulaire ouvert avec des transformateurs de vision](https://arxiv.org/abs/2205.06230) par Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf et Neil Houlsby.
1. **[OWLv2](https://huggingface.co/docs/transformers/model_doc/owlv2)** (de Google AI) a été publié dans l'article [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) par Matthias Minderer, Alexey Gritsenko, Neil Houlsby.
1. **[PatchTSMixer](https://huggingface.co/docs/transformers/model_doc/patchtsmixer)** (d'IBM Research) a été publié dans l'article [TSMixer : Modèle MLP-Mixer léger pour la prévision multivariée de séries temporelles](https://arxiv.org/pdf/2306.09364.pdf) par Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, Jayant Kalagnanam.
1. **[PatchTST](https://huggingface.co/docs/transformers/model_doc/patchtst)** (d'IBM) a été publié dans l'article [Une série temporelle vaut 64 mots : Prévision à long terme avec des Transformers](https://arxiv.org/abs/2211.14730) par Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam.
1. **[Pegasus](https://huggingface.co/docs/transformers/model_doc/pegasus)** (de Google) a été publié dans l'article [PEGASUS : Pré-entraînement avec Phrases-Écart Extraites pour la Résumé Abstrait](https://arxiv.org/abs/1912.08777) par Jingqing Zhang, Yao Zhao, Mohammad Saleh et Peter J. Liu.
1. **[PEGASUS-X](https://huggingface.co/docs/transformers/model_doc/pegasus_x)** (de Google) a été publié dans l'article [Investiguer l'Extension Efficace des Transformers pour la Longue Summarization d'Entrée](https://arxiv.org/abs/2208.04347) par Jason Phang, Yao Zhao et Peter J. Liu.
1. **[Perceiver IO](https://huggingface.co/docs/transformers/model_doc/perceiver)** (de Deepmind) a été publié dans l'article [Perceiver IO : Une architecture générale pour les entrées et sorties structurées](https://arxiv.org/abs/2107.14795) par Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals et João Carreira.
1. **[Persimmon](https://huggingface.co/docs/transformers/model_doc/persimmon)** (d'ADEPT) a été publié dans un [billet de blog](https://www.adept.ai/blog/persimmon-8b) par Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, Arushi Somani.
1. **[Phi](https://huggingface.co/docs/transformers/model_doc/phi)** (de Microsoft) a été publié avec les articles - [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) par Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee et Yuanzhi Li, [Textbooks Are All You Need II : Rapport technique phi-1.5](https://arxiv.org/abs/2309.05463) par Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar et Yin Tat Lee.
1. **[PhoBERT](https://huggingface.co/docs/transformers/model_doc/phobert)** (de VinAI Research) a été publié dans l'article [PhoBERT : Modèles linguistiques pré-entraînés pour le vietnamien](https://www.aclweb.org/anthology/2020.findings-emnlp.92/) par Dat Quoc Nguyen et Anh Tuan Nguyen.
1. **[Pix2Struct](https://huggingface.co/docs/transformers/model_doc/pix2struct)** (de Google) a été publié dans l'article [Pix2Struct : Analyse d'images d'écran en tant que pré-entraînement pour la compréhension du langage visuel](https://arxiv.org/abs/2210.03347) par Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova.
1. **[PLBart](https://huggingface.co/docs/transformers/model_doc/plbart)** (de UCLA NLP) a été publié dans l'article [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) par Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang.
1. **[PoolFormer](https://huggingface.co/docs/transformers/model_doc/poolformer)** (de Sea AI Labs) a été publié dans l'article [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) par Yu, Weihao et Luo, Mi et Zhou, Pan et Si, Chenyang et Zhou, Yichen et Wang, Xinchao et Feng, Jiashi et Yan, Shuicheng.
1. **[Pop2Piano](https://huggingface.co/docs/transformers/model_doc/pop2piano)** a été publié dans l'article [Pop2Piano : Génération de reprises de morceaux de piano basée sur l'audio pop](https://arxiv.org/abs/2211.00895) par Jongho Choi et Kyogu Lee.
1. **[ProphetNet](https://huggingface.co/docs/transformers/model_doc/prophetnet)** (de Microsoft Research) a été publié dans l'article [ProphetNet : Prédire les N-grammes futurs pour l'entraînement préalable de séquences à séquences](https://arxiv.org/abs/2001.04063) par Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang et Ming Zhou.
1. **[PVT](https://huggingface.co/docs/transformers/model_doc/pvt)** (de l'Université de Nankin, l'Université de Hong Kong, etc.) a été publié dans l'article [Pyramid Vision Transformer : Une colonne vertébrale polyvalente pour la prédiction dense sans convolutions](https://arxiv.org/pdf/2102.12122.pdf) par Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo et Ling Shao.
1. **[QDQBert](https://huggingface.co/docs/transformers/model_doc/qdqbert)** (de NVIDIA) a été publié dans l'article [Quantification entière pour l'inférence d'apprentissage profond : Principes et évaluation empirique](https://arxiv.org/abs/2004.09602) par Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev et Paulius Micikevicius.
1. **[Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)** (de l'équipe Qwen, Alibaba Group) a été publié avec le rapport technique [Qwen](https://arxiv.org/abs/2309.16609) par Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou et Tianhang Zhu.
1. **[RAG](https://huggingface.co/docs/transformers/model_doc/rag)** (de Facebook) a été publié dans l'article [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) par Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.
1. **[REALM](https://huggingface.co/docs/transformers/model_doc/realm.html)** (de Google Research) a été publié dans l'article [REALM : Pré-entraînement de modèle linguistique augmenté par la récupération](https://arxiv.org/abs/2002.08909) par Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat et Ming-Wei Chang.
1. **[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer)** (de Google Research) a été publié dans l'article [Reformer : Le transformateur efficace](https://arxiv.org/abs/2001.04451) par Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
1. **[RegNet](https://huggingface.co/docs/transformers/model_doc/regnet)** (de META Platforms) a été publié dans l'article [Designing Network Design Space](https://arxiv.org/abs/2003.13678) par Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár.
1. **[RemBERT](https://huggingface.co/docs/transformers/model_doc/rembert)** (de Google Research) a été publié dans l'article [Rethinking embedding coupling in pre-trained language models](https://arxiv.org/abs/2010.12821) par Hyung Won Chung, Thibault Févry, Henry Tsai, M. Johnson, Sebastian Ruder.
1. **[ResNet](https://huggingface.co/docs/transformers/model_doc/resnet)** (de Microsoft Research) a été publié dans l'article [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) par Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
1. **[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)** (de Facebook), publié dans l'article [RoBERTa : Une approche d'entraînement préalable BERT robuste](https://arxiv.org/abs/1907.11692) par Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
1. **[RoBERTa-PreLayerNorm](https://huggingface.co/docs/transformers/model_doc/roberta-prelayernorm)** (de Facebook) a été publié dans l'article [fairseq : Une boîte à outils rapide et extensible pour la modélisation de séquences](https://arxiv.org/abs/1904.01038) par Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli.
1. **[RoCBert](https://huggingface.co/docs/transformers/model_doc/roc_bert)** (de WeChatAI) a été publié dans l'article [RoCBert : BERT chinois robuste avec pré-entraînement contrastif multimodal](https://aclanthology.org/2022.acl-long.65.pdf) par HuiSu, WeiweiShi, XiaoyuShen, XiaoZhou, TuoJi, JiaruiFang, JieZhou.
1. **[RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer)** (de ZhuiyiTechnology), publié dans l'article [RoFormer : Transformateur amélioré avec insertion rotative de position](https://arxiv.org/abs/2104.09864) par Jianlin Su et Yu Lu et Shengfeng Pan et Bo Wen et Yunfeng Liu.
1. **[RWKV](https://huggingface.co/docs/transformers/model_doc/rwkv)** (de Bo Peng), publié sur [ce référentiel](https://github.com/BlinkDL/RWKV-LM) par Bo Peng.
1. **[RWKV5](https://huggingface.co/docs/transformers/model_doc/rwkv5)** (from Bo Peng) released with the paper [this repo](https://github.com/BlinkDL/RWKV-LM) by Bo Peng.
1. **[SeamlessM4T](https://huggingface.co/docs/transformers/model_doc/seamless_m4t)** (de Meta AI) a été publié dans l'article [SeamlessM4T — Traduction multimodale et massivement multilingue](https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf) par l'équipe de communication transparente.
1. **[SeamlessM4Tv2](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)** (de Meta AI) a été publié dans l'article [Seamless: Traduction de la parole multilingue, expressive et en continu](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) par l'équipe de communication transparente.
1. **[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)** (de NVIDIA) a été publié dans l'article [SegFormer : Conception simple et efficace pour la segmentation sémantique avec des transformateurs](https://arxiv.org/abs/2105.15203) par Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.
1. **[Segment Anything](https://huggingface.co/docs/transformers/model_doc/sam)** (de Meta AI) a été publié dans l'article [Segment Anything](https://arxiv.org/pdf/2304.02643v1.pdf) par Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alex Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick.
1. **[SEW](https://huggingface.co/docs/transformers/model_doc/sew)** (de ASAPP) a été publié dans l'article [Compromis entre performances et efficacité dans l'entraînement non supervisé pour la reconnaissance vocale](https://arxiv.org/abs/2109.06870) par Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.
1. **[SEW-D](https://huggingface.co/docs/transformers/model_doc/sew_d)** (de ASAPP) a été publié dans l'article [Compromis entre performances et efficacité dans l'entraînement non supervisé pour la reconnaissance vocale](https://arxiv.org/abs/2109.06870) par Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.
1. **[SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)** (de Google AI) a été publié dans l'article [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) par Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer.
1. **[SpeechT5](https://huggingface.co/docs/transformers/model_doc/speecht5)** (de Microsoft Research) a été publié dans l'article [SpeechT5 : Pré-entraînement unifié encodeur-décodeur pour le traitement du langage parlé](https://arxiv.org/abs/2110.07205) par Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.
1. **[SpeechToTextTransformer](https://huggingface.co/docs/transformers/model_doc/speech_to_text)** (de Facebook), publié dans l'article [fairseq S2T : Modélisation rapide de la parole à texte avec fairseq](https://arxiv.org/abs/2010.05171) par Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino.
1. **[SpeechToTextTransformer2](https://huggingface.co/docs/transformers/model_doc/speech_to_text_2)** (de Facebook), publié dans l'article [Apprentissage auto-supervisé et semi-supervisé à grande échelle pour la traduction de la parole](https://arxiv.org/abs/2104.06678) par Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.
1. **[Splinter](https://huggingface.co/docs/transformers/model_doc/splinter)** (de l'Université de Tel Aviv), publié dans l'article [Réponse à quelques questions avec peu d'exemples par la pré-sélection des spans](https://arxiv.org/abs/2101.00438) par Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy.
1. **[SqueezeBERT](https://huggingface.co/docs/transformers/model_doc/squeezebert)** (de Berkeley) a été publié dans l'article [SqueezeBERT : Que l'apprentissage automatique peut-il apprendre au traitement du langage naturel sur les réseaux neuronaux efficaces ?](https://arxiv.org/abs/2006.11316) par Forrest N. Iandola, Albert E. Shaw, Ravi Krishna et Kurt W. Keutzer.
1. **[SwiftFormer](https://huggingface.co/docs/transformers/model_doc/swiftformer)** (de MBZUAI) a été publié dans l'article [SwiftFormer : Attention additive efficace pour les applications de vision mobile en temps réel basées sur des transformateurs](https://arxiv.org/abs/2303.15446) par Abdelrahman Shaker, Muhammad Maaz, Hanoona Rasheed, Salman Khan, Ming-Hsuan Yang, Fahad Shahbaz Khan.
1. **[Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin)** (de Microsoft) a été publié dans l'article [Swin Transformer : Transformateur hiérarchique de la vision utilisant des fenêtres décalées](https://arxiv.org/abs/2103.14030) par Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
1. **[Swin Transformer V2](https://huggingface.co/docs/transformers/model_doc/swinv2)** (de Microsoft) a été publié dans l'article [Swin Transformer V2 : Augmentation de la capacité et de la résolution](https://arxiv.org/abs/2111.09883) par Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo.
1. **[Swin2SR](https://huggingface.co/docs/transformers/model_doc/swin2sr)** (de l'Université de Würzburg) a été publié dans l'article [Swin2SR : Transformateur SwinV2 pour la super-résolution et la restauration d'images compressées](https://arxiv.org/abs/2209.11345) par Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte.
1. **[SwitchTransformers](https://huggingface.co/docs/transformers/model_doc/switch_transformers)** (de Google) a été publié dans l'article [Switch Transformers : Passage à des modèles de trillions de paramètres avec une parcimonie simple et efficace](https://arxiv.org/abs/2101.03961) par William Fedus, Barret Zoph, Noam Shazeer.
1. **[T5](https://huggingface.co/docs/transformers/model_doc/t5)** (de Google AI) a été publié dans l'article [Exploration des limites de l'apprentissage par transfert avec un transformateur de texte à texte unifié](https://arxiv.org/abs/1910.10683) par Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li et Peter J. Liu.
1. **[T5v1.1](https://huggingface.co/docs/transformers/model_doc/t5v1.1)** (de Google AI) a été publié dans le dépôt [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511) par Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li et Peter J. Liu.
1. **[Table Transformer](https://huggingface.co/docs/transformers/model_doc/table-transformer)** (de Microsoft Research) a été publié dans l'article [PubTables-1M : Vers une extraction complète des tables à partir de documents non structurés](https://arxiv.org/abs/2110.00061) par Brandon Smock, Rohith Pesala, Robin Abraham.
1. **[TAPAS](https://huggingface.co/docs/transformers/model_doc/tapas)** (de Google AI) a été publié dans l'article [TAPAS : Analyse faiblement supervisée des tables via le pré-entraînement](https://arxiv.org/abs/2004.02349) par Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno et Julian Martin Eisenschlos.
1. **[TAPEX](https://huggingface.co/docs/transformers/model_doc/tapex)** (de Microsoft Research) a été publié dans l'article [TAPEX : Pré-entraînement des tables via l'apprentissage d'un exécuteur SQL neuronal](https://arxiv.org/abs/2107.07653) par Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen et Jian-Guang Lou.
1. **[Time Series Transformer](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)** (de HuggingFace).
1. **[TimeSformer](https://huggingface.co/docs/transformers/model_doc/timesformer)** (de Facebook) a été publié dans l'article [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) par Gedas Bertasius, Heng Wang, Lorenzo Torresani.
1. **[Trajectory Transformer](https://huggingface.co/docs/transformers/model_doc/trajectory_transformers)** (de l'Université de Californie à Berkeley) a été publié dans l'article [L'apprentissage par renforcement hors ligne comme un grand problème de modélisation de séquence](https://arxiv.org/abs/2106.02039) par Michael Janner, Qiyang Li, Sergey Levine.
1. **[Transformer-XL](https://huggingface.co/docs/transformers/model_doc/transfo-xl)** (de Google/CMU) a été publié dans l'article [Transformer-XL : Modèles de langage attentifs au-delà d'un contexte de longueur fixe](https://arxiv.org/abs/1901.02860) par Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
1. **[TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)** (de Microsoft), publié dans l'article [TrOCR : Reconnaissance optique de caractères basée sur un transformateur avec des modèles pré-entraînés](https://arxiv.org/abs/2109.10282) par Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei.
1. **[TVLT](https://huggingface.co/docs/transformers/model_doc/tvlt)** (de l'UNC Chapel Hill) a été publié dans l'article [TVLT : Transformer Vision-Language sans texte](https://arxiv.org/abs/2209.14156) par Zineng Tang, Jaemin Cho, Yixin Nie, Mohit Bansal.
1. **[TVP](https://huggingface.co/docs/transformers/model_doc/tvp)** (d'Intel) a été publié dans l'article [Text-Visual Prompting for Efficient 2D Temporal Video Grounding](https://arxiv.org/abs/2303.04995) par Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, Ke Ding.
1. **[UL2](https://huggingface.co/docs/transformers/model_doc/ul2)** (de Google Research) a été publié dans l'article [Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131v1) par Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, Donald Metzler.
1. **[UMT5](https://huggingface.co/docs/transformers/model_doc/umt5)** (de Google Research) a été publié dans l'article [UniMax : Échantillonnage linguistique plus équitable et plus efficace pour l'entraînement préalable multilingue à grande échelle](https://openreview.net/forum?id=kXwdL1cWOAi) par Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, Noah Constant.
1. **[UniSpeech](https://huggingface.co/docs/transformers/model_doc/unispeech)** (de Microsoft Research) a été publié dans l'article [UniSpeech : Apprentissage unifié de la représentation de la parole avec des données étiquetées et non étiquetées](https://arxiv.org/abs/2101.07597) par Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.
1. **[UniSpeechSat](https://huggingface.co/docs/transformers/model_doc/unispeech-sat)** (de Microsoft Research) a été publié dans l'article [UNISPEECH-SAT : Apprentissage universel de la représentation de la parole avec la préformation consciente du locuteur](https://arxiv.org/abs/2110.05752) par Sanyuan Chen, Yu Wu, Chengyi Wang, Zhengyang Chen, Zhuo Chen, Shujie Liu, Jian Wu, Yao Qian, Furu Wei, Jinyu Li, Xiangzhan Yu.
1. **[UnivNet](https://huggingface.co/docs/transformers/model_doc/univnet)** (de Kakao Corporation) a été publié dans l'article [UnivNet : un vocodeur neuronal avec des discriminateurs de spectrogramme multi-résolution pour la génération de formes d'onde haute fidélité](https://arxiv.org/abs/2106.07889) par Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kim et Juntae Kim.
1. **[UPerNet](https://huggingface.co/docs/transformers/model_doc/upernet)** (de l'Université de Pékin) a été publié dans l'article [Analyse perceptuelle unifiée pour la compréhension de scènes](https://arxiv.org/abs/1807.10221) par Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, Jian Sun.
1. **[VAN](https://huggingface.co/docs/transformers/model_doc/van)** (de l'Université Tsinghua et de l'Université Nankai) publié dans l'article [Visual Attention Network](https://arxiv.org/abs/2202.09741) par Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu.
1. **[VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)** (du groupe d'informatique multimédia, Université de Nankin) publié dans l'article [VideoMAE : Les autoencodeurs masqués sont des apprenants efficaces en données pour l'auto-pré-entraînement vidéo supervisé](https://arxiv.org/abs/2203.12602) par Zhan Tong, Yibing Song, Jue Wang, Limin Wang.
1. **[ViLT](https://huggingface.co/docs/transformers/model_doc/vilt)** (du NAVER AI Lab/Kakao Enterprise/Kakao Brain) publié dans l'article [ViLT : Vision-and-Language Transformer sans convolution ni supervision de région](https://arxiv.org/abs/2102.03334) par Wonjae Kim, Bokyung Son, Ildoo Kim.
1. **[VipLlava](https://huggingface.co/docs/transformers/model_doc/vipllava)** (de l'Université du Wisconsin–Madison) publié dans l'article [Rendre les grands modèles multimodaux comprenant des incitations visuelles arbitraires](https://arxiv.org/abs/2312.00784) par Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, Yong Jae Lee.
1. **[Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)** (de Google AI) publié dans l'article [Une image vaut 16x16 mots : Transformers pour la reconnaissance d'images à grande échelle](https://arxiv.org/abs/2010.11929) par Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
1. **[VisualBERT](https://huggingface.co/docs/transformers/model_doc/visual_bert)** (de UCLA NLP) publié dans l'article [VisualBERT : Une référence simple et performante pour la vision et le langage](https://arxiv.org/pdf/1908.03557) par Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.
1. **[ViT Hybrid](https://huggingface.co/docs/transformers/model_doc/vit_hybrid)** (de Google AI) publié dans l'article [Une image vaut 16x16 mots : Transformers pour la reconnaissance d'images à grande échelle](https://arxiv.org/abs/2010.11929) par Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
1. **[VitDet](https://huggingface.co/docs/transformers/model_doc/vitdet)** (de Meta AI) publié dans l'article [Exploration des transformateurs de vision plain pour la détection d'objets](https://arxiv.org/abs/2203.16527) par Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He.
1. **[ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae)** (de Meta AI) publié dans l'article [Les autoencodeurs masqués sont des apprenants évolutifs de la vision](https://arxiv.org/abs/2111.06377) par Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.
1. **[ViTMatte](https://huggingface.co/docs/transformers/model_doc/vitmatte)** (de HUST-VL) publié dans l'article [ViTMatte : Renforcer le détourage d'image avec des transformateurs de vision plain pré-entraînés](https://arxiv.org/abs/2305.15272) par Jingfeng Yao, Xinggang Wang, Shusheng Yang, Baoyuan Wang.
1. **[ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn)** (de Meta AI) publié dans l'article [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141) par Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas.
1. **[VITS](https://huggingface.co/docs/transformers/model_doc/vits)** (de Kakao Enterprise) publié dans l'article [Auto-encodeur variationnel conditionnel avec apprentissage adversarial pour la conversion texte-parole de bout en bout](https://arxiv.org/abs/2106.06103) par Jaehyeon Kim, Jungil Kong, Juhee Son.
1. **[ViViT](https://huggingface.co/docs/transformers/model_doc/vivit)** (de Google Research) publié dans l'article [ViViT : Un transformateur de vision vidéo](https://arxiv.org/abs/2103.15691) par Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid.
1. **[Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)** (de Facebook AI) publié dans l'article [wav2vec 2.0 : Un cadre pour l'apprentissage auto-supervisé des représentations de la parole](https://arxiv.org/abs/2006.11477) par Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.
1. **[Wav2Vec2-BERT](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert)** (de Meta AI) publié dans l'article [Seamless : Traduction de la parole multilingue, expressive et en continu](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) par l'équipe Seamless Communication.
1. **[Wav2Vec2-Conformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer)** (de Facebook AI) a été publié dans l'article [FAIRSEQ S2T : Modélisation rapide de la parole au texte avec FAIRSEQ](https://arxiv.org/abs/2010.05171) par Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Sravya Popuri, Dmytro Okhonko, Juan Pino.
1. **[Wav2Vec2Phoneme](https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme)** (de Facebook AI) a été publié dans l'article [Reconnaissance de phonèmes interlingues simple et efficace sans apprentissage préalable](https://arxiv.org/abs/2109.11680) par Qiantong Xu, Alexei Baevski, Michael Auli.
1. **[WavLM](https://huggingface.co/docs/transformers/model_doc/wavlm)** (de Microsoft Research) a été publié dans l'article [WavLM : Pré-entraînement auto-supervisé à grande échelle pour le traitement complet de la parole](https://arxiv.org/abs/2110.13900) par Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Furu Wei.
1. **[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)** (d'OpenAI) a été publié dans l'article [Reconnaissance robuste de la parole via une supervision faible à grande échelle](https://cdn.openai.com/papers/whisper.pdf) par Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever.
1. **[X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)** (de Microsoft Research) a été publié dans l'article [Expansion des modèles pré-entraînés langage-image pour la reconnaissance vidéo générale](https://arxiv.org/abs/2208.02816) par Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, Haibin Ling.
1. **[X-MOD](https://huggingface.co/docs/transformers/model_doc/xmod)** (de Meta AI) a été publié dans l'article [Lever le sort de la multilinguité par le pré-entraînement des transformateurs modulaires](http://dx.doi.org/10.18653/v1/2022.naacl-main.255) par Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, Mikel Artetxe.
1. **[XGLM](https://huggingface.co/docs/transformers/model_doc/xglm)** (de Facebook AI) a été publié dans l'article [Apprentissage à quelques échantillons avec des modèles de langues multilingues](https://arxiv.org/abs/2112.10668) par Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.
1. **[XLM](https://huggingface.co/docs/transformers/model_doc/xlm)** (de Facebook) a été publié dans l'article [Pré-entraînement de modèles linguistiques multilingues](https://arxiv.org/abs/1901.07291) par Guillaume Lample et Alexis Conneau.
1. **[XLM-ProphetNet](https://huggingface.co/docs/transformers/model_doc/xlm-prophetnet)** (de Microsoft Research) a été publié dans l'article [ProphetNet : Prédire l'avenir N-gramme pour le pré-entraînement séquence-séquence](https://arxiv.org/abs/2001.04063) par Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang et Ming Zhou.
1. **[XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)** (de Facebook AI), publié dans l'article [Apprentissage non supervisé de la représentation croisée-lingue à grande échelle](https://arxiv.org/abs/1911.02116) par Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer et Veselin Stoyanov.
1. **[XLM-RoBERTa-XL](https://huggingface.co/docs/transformers/model_doc/xlm-roberta-xl)** (de Facebook AI), publié dans l'article [Transformateurs à plus grande échelle pour la modélisation du langage masqué multilingue](https://arxiv.org/abs/2105.00572) par Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, Alexis Conneau.
1. **[XLM-V](https://huggingface.co/docs/transformers/model_doc/xlm-v)** (de Meta AI) a été publié dans l'article [XLM-V : Surmonter le goulot d'étranglement du vocabulaire dans les modèles de langage masqués multilingues](https://arxiv.org/abs/2301.10472) par Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, Madian Khabsa.
1. **[XLNet](https://huggingface.co/docs/transformers/model_doc/xlnet)** (de Google/CMU) a été publié dans l'article [XLNet : Préentraînement autorégressif généralisé pour la compréhension du langage](https://arxiv.org/abs/1906.08237) par Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
1. **[XLS-R](https://huggingface.co/docs/transformers/model_doc/xls_r)** (de Facebook AI) publié dans l'article [XLS-R : Apprentissage d'une représentation de la parole autonome et multilingue à grande échelle](https://arxiv.org/abs/2111.09296) par Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli.
1. **[XLSR-Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/xlsr_wav2vec2)** (de Facebook AI) publié dans l'article [Apprentissage non supervisé de représentations multilingues pour la reconnaissance vocale](https://arxiv.org/abs/2006.13979) par Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli.
1. **[YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos)** (de l'Université Huazhong des sciences et technologies) publié dans l'article [You Only Look at One Sequence : Repenser le Transformer dans la vision à travers la détection d'objets](https://arxiv.org/abs/2106.00666) par Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu.
1. **[YOSO](https://huggingface.co/docs/transformers/model_doc/yoso)** (de l'Université du Wisconsin - Madison) publié dans l'article [You Only Sample (Almost) Once : Coût linéaire Self-Attention via l'échantillonnage Bernoulli](https://arxiv.org/abs/2111.09714) par Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh.
1. Vous souhaitez contribuer avec un nouveau modèle ? Nous avons ajouté un **guide détaillé et des modèles types** pour vous guider dans le processus d'ajout d'un nouveau modèle. Vous pouvez les trouver dans le dossier [`templates`](./templates) du référentiel. Assurez-vous de consulter les [directives de contribution](./CONTRIBUTING.md) et de contacter les mainteneurs ou d'ouvrir un ticket pour recueillir des commentaires avant de commencer votre pull request.

Pour vérifier si chaque modèle a une implémentation en Flax, PyTorch ou TensorFlow, ou s'il a un tokenizer associé pris en charge par la bibliothèque 🤗 Tokenizers, consultez [ce tableau](https://huggingface.co/docs/transformers/index#supported-frameworks).

Ces implémentations ont été testées sur plusieurs ensembles de données (voir les scripts d'exemple) et devraient correspondre aux performances des implémentations originales. Vous pouvez trouver plus de détails sur les performances dans la section Exemples de la [documentation](https://github.com/huggingface/transformers/tree/main/examples).

## En savoir plus

| Section | Description |
|-|-|
| [Documentation](https://huggingface.co/docs/transformers/) | Documentation complète de l'API et tutoriels |
| [Résumé des tâches](https://huggingface.co/docs/transformers/task_summary) | Tâches prises en charge par les 🤗 Transformers |
| [Tutoriel de prétraitement](https://huggingface.co/docs/transformers/preprocessing) | Utilisation de la classe `Tokenizer` pour préparer les données pour les modèles |
| [Entraînement et ajustement fin](https://huggingface.co/docs/transformers/training) | Utilisation des modèles fournis par les 🤗 Transformers dans une boucle d'entraînement PyTorch/TensorFlow et de l'API `Trainer` |
| [Tour rapide : Scripts d'ajustement fin/d'utilisation](https://github.com/huggingface/transformers/tree/main/examples) | Scripts d'exemple pour ajuster finement les modèles sur une large gamme de tâches |
| [Partage et téléversement de modèles](https://huggingface.co/docs/transformers/model_sharing) | Téléchargez et partagez vos modèles ajustés avec la communauté |

## Citation

Nous disposons désormais d'un [article](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) que vous pouvez citer pour la bibliothèque 🤗 Transformers :
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
