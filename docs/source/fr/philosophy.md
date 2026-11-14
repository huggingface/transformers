<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Philosophie

Transformers est une bibliothèque pensée pour PyTorch. Elle propose des modèles fidèles à leurs articles de recherche, faciles à utiliser et faciles à modifier.

Un article plus long et plus approfondi, avec des exemples, des visualisations et des chronologies, est disponible [ici](https://huggingface.co/spaces/transformers-community/Transformers-tenets) et sert de référence officielle.

> [!NOTE]
> Notre philosophie évolue avec la pratique. Ce qui suit présente nos principes actuels, désormais stables.

## À qui s'adresse cette bibliothèque

- Aux chercheurs et aux enseignants qui explorent ou étendent des architectures de modèles.
- Aux praticiens qui ajustent (*fine-tuning*), évaluent ou déploient des modèles.
- Aux ingénieurs qui veulent un modèle pré-entraîné qui « fonctionne juste », avec une API prévisible.

## Ce que vous pouvez attendre

- Trois classes principales sont nécessaires pour chaque modèle : la [configuration](./main_classes/configuration), le [modèle](./main_classes/model) et une classe de prétraitement. Les [tokenizers](./main_classes/tokenizer) gèrent le texte, les [processeurs d'images](./main_classes/image_processor) gèrent les images, les [processeurs vidéo](./main_classes/video_processor) gèrent les vidéos, les [extracteurs de caractéristiques](./main_classes/feature_extractor) gèrent l'audio et les [processeurs](./main_classes/processors) gèrent les entrées multimodales.

- Toutes ces classes peuvent être initialisées de manière simple et unifiée à partir d'instances pré-entraînées, en utilisant une méthode commune, `from_pretrained()`, qui télécharge (si nécessaire), met en cache et charge l'instance de la classe concernée ainsi que les données associées (hyperparamètres des configurations, vocabulaire des tokenizers, paramètres des processeurs et poids des modèles) à partir d'un checkpoint pré-entraîné disponible sur le [Hugging Face Hub](https://huggingface.co/models) ou de votre propre checkpoint sauvegardé.

- En plus de ces trois classes de base, la bibliothèque propose deux API : [`pipeline`], pour utiliser rapidement un modèle en inférence sur une tâche donnée, et [`Trainer`], pour entraîner ou ajuster rapidement un modèle PyTorch.

## Principes fondamentaux

Les principes suivants se sont renforcés au fil du temps. Ils sont détaillés dans notre nouvel [article de blog](https://huggingface.co/spaces/transformers-community/Transformers-tenets) consacré à notre philosophie. Ils guident les décisions des mainteneurs lorsqu'ils examinent les pull requests et les contributions.

> - **Source de vérité** (*Source of Truth*). Les implémentations doivent être fidèles aux résultats officiels et au comportement prévu.
> - **Un modèle, un fichier** (*One Model, One File*). La logique principale d'inférence et d'entraînement est visible dans le fichier du modèle que lisent les utilisateurs.
> - **Le code est le produit** (*Code is the Product*). Optimiser pour la lecture et la comparaison des modifications. Préférer des noms explicites à des indirections astucieuses.
> - **Standardiser, ne pas abstraire** (*Standardize, Don't Abstract*). Garder le comportement propre à chaque modèle dans le modèle lui-même. N'utiliser des interfaces partagées que pour l'infrastructure générique.
> - **DRY\*** (se répéter quand cela aide les utilisateurs). Les fichiers de modélisation destinés aux utilisateurs restent autonomes. L'infrastructure est factorisée.
> - **API utilisateur minimale** (*Minimal User API*). Peu de chemins de code, des arguments (kwargs) prévisibles, des méthodes stables.
> - **Rétrocompatibilité** (*Backwards Compatibility*). Les interfaces publiques ne doivent pas être cassées. Les anciens artefacts du Hub doivent continuer à fonctionner.
> - **Interface publique cohérente** (*Consistent Public Surface*). Les noms, les sorties et les diagnostics optionnels sont harmonisés et testés.

## Classes principales

- Les [**classes de configuration**](./main_classes/configuration) stockent les hyperparamètres nécessaires pour construire un modèle, comme le nombre de couches et la dimension des états cachés. Vous n'avez pas toujours besoin de les instancier vous-même : lorsque vous utilisez un modèle pré-entraîné sans le modifier, la création du modèle instancie automatiquement la configuration.
- Les **classes de modèles** sont des modèles PyTorch ([torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)), couverts au moins par un [`PreTrainedModel`].

- **Transformers modulaire.** Les contributeurs écrivent un petit fichier `modular_*.py` qui déclare les composants existants qu'il réutilise. La bibliothèque le génére automatiquement en un fichier `modeling_*.py` visible, que les utilisateurs lisent et déboguent. Les mainteneurs relisent le fichier modulaire, et les utilisateurs modifient le fichier développé. Cela préserve le principe « Un modèle, un fichier » sans que le code répétitif ne diverge au fil du temps. Consultez [la documentation sur les contributions](./modular_transformers) pour en savoir plus.

- Les **classes de prétraitement** convertissent les données brutes dans un format accepté par le modèle. Un [tokenizer](./main_classes/tokenizer) stocke le vocabulaire de chaque modèle et fournit des méthodes pour encoder et décoder des chaînes de caractères en une liste d'indices de tokens embedding. Les [processeurs d'images](./main_classes/image_processor) prétraitent les entrées visuelles, les [processeurs vidéo](./main_classes/video_processor) prétraitent les entrées vidéo, les [extracteurs de caractéristiques](./main_classes/feature_extractor) prétraitent les entrées audio et les [processeurs](./main_classes/processors) prétraitent les entrées multimodales.

Toutes ces classes peuvent être instanciées à partir d'instances pré-entraînées, sauvegardées localement et partagées sur le Hub grâce à trois méthodes :

- `from_pretrained()` permet d'instancier un modèle, une configuration et une classe de prétraitement à partir d'une version pré-entraînée, soit fournie par la bibliothèque elle-même (les modèles pris en charge sont disponibles sur le [Model Hub](https://huggingface.co/models)), soit stockée localement (ou sur un serveur) par l'utilisateur.
- `save_pretrained()` permet de sauvegarder localement un modèle, une configuration et une classe de prétraitement afin de pouvoir les recharger avec `from_pretrained()`.
- `push_to_hub()` permet de partager un modèle, une configuration et une classe de prétraitement sur le Hub, afin qu'ils soient facilement accessibles à tous.
