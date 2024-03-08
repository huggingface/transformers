<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Chargement d'instances pré-entraînées avec une AutoClass

Avec autant d'architectures Transformer différentes, il peut être difficile d'en créer une pour votre ensemble de poids (aussi appelés "weights" ou "checkpoint" en anglais). Dans l'idée de créer une librairie facile, simple et flexible à utiliser, 🤗 Transformers fournit une `AutoClass` qui infère et charge automatiquement l'architecture correcte à partir d'un ensemble de poids donné. La fonction `from_pretrained()` vous permet de charger rapidement un modèle pré-entraîné pour n'importe quelle architecture afin que vous n'ayez pas à consacrer du temps et des ressources à l'entraînement d'un modèle à partir de zéro. Produire un tel code indépendant d'un ensemble de poids signifie que si votre code fonctionne pour un ensemble de poids, il fonctionnera avec un autre ensemble - tant qu'il a été entraîné pour une tâche similaire - même si l'architecture est différente.

<Tip>

Rappel, l'architecture fait référence au squelette du modèle et l'ensemble de poids contient les poids pour une architecture donnée. Par exemple, [BERT](https://huggingface.co/google-bert/bert-base-uncased) est une architecture, tandis que `google-bert/bert-base-uncased` est un ensemble de poids. Le terme modèle est général et peut signifier soit architecture soit ensemble de poids.

</Tip>

Dans ce tutoriel, vous apprendrez à:

  * Charger un tokenizer pré-entraîné.
  * Charger un processeur d'image pré-entraîné.
  * Charger un extracteur de caractéristiques pré-entraîné.
  * Charger un processeur pré-entraîné.
  * Charger un modèle pré-entraîné.

## AutoTokenizer

Quasiment toutes les tâches de traitement du langage (NLP) commencent avec un tokenizer. Un tokenizer convertit votre texte initial dans un format qui peut être traité par le modèle.

Chargez un tokenizer avec [`AutoTokenizer.from_pretrained`]:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

Puis, transformez votre texte initial comme montré ci-dessous:

```py
>>> sequence = "In a hole in the ground there lived a hobbit."
>>> print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor

Pour les tâches de vision, un processeur d'image traite l'image pour la formater correctment.

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

## AutoFeatureExtractor

Pour les tâches audio, un extracteur de caractéristiques (aussi appelés "features" en anglais) traite le signal audio pour le formater correctement.

Chargez un extracteur de caractéristiques avec [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(
...     "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
... )
```

## AutoProcessor

Les tâches multimodales nécessitent un processeur qui combine deux types d'outils de prétraitement. Par exemple, le modèle [LayoutLMV2](model_doc/layoutlmv2) nécessite un processeur d'image pour traiter les images et un tokenizer pour traiter le texte ; un processeur combine les deux.

Chargez un processeur avec [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

<frameworkcontent>
<pt>
Enfin, les classes `AutoModelFor` vous permettent de charger un modèle pré-entraîné pour une tâche donnée (voir [ici](model_doc/auto) pour une liste complète des tâches disponibles). Par exemple, chargez un modèle pour la classification de séquence avec [`AutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

Réutilisez facilement le même ensemble de poids pour charger une architecture pour une tâche différente :

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip warning={true}>

Pour les modèles PyTorch, la fonction `from_pretrained()` utilise `torch.load()` qui utilise `pickle` en interne et est connu pour être non sécurisé. En général, ne chargez jamais un modèle qui pourrait provenir d'une source non fiable, ou qui pourrait avoir été altéré. Ce risque de sécurité est partiellement atténué pour les modèles hébergés publiquement sur le Hugging Face Hub, qui sont [scannés pour les logiciels malveillants](https://huggingface.co/docs/hub/security-malware) à chaque modification. Consultez la [documentation du Hub](https://huggingface.co/docs/hub/security) pour connaître les meilleures pratiques comme la [vérification des modifications signées](https://huggingface.co/docs/hub/security-gpg#signing-commits-with-gpg) avec GPG.

Les points de contrôle TensorFlow et Flax ne sont pas concernés, et peuvent être chargés dans des architectures PyTorch en utilisant les arguments `from_tf` et `from_flax` de la fonction `from_pretrained` pour contourner ce problème.

</Tip>

En général, nous recommandons d'utiliser les classes `AutoTokenizer` et `AutoModelFor` pour charger des instances pré-entraînées de tokenizers et modèles respectivement. Cela vous permettra de charger la bonne architecture à chaque fois. Dans le prochain [tutoriel](preprocessing), vous apprenez à utiliser un tokenizer, processeur d'image, extracteur de caractéristiques et processeur pour pré-traiter un jeu de données pour le fine-tuning.
</pt>
<tf>
Enfin, les classes `TFAutoModelFor` vous permettent de charger un modèle pré-entraîné pour une tâche donnée (voir [ici](model_doc/auto) pour une liste complète des tâches disponibles). Par exemple, chargez un modèle pour la classification de séquence avec [`TFAutoModelForSequenceClassification.from_pretrained`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

Réutilisez facilement le même ensemble de poids pour charger une architecture pour une tâche différente :

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

En général, nous recommandons d'utiliser les classes `AutoTokenizer` et `TFAutoModelFor` pour charger des instances pré-entraînées de tokenizers et modèles respectivement. Cela vous permettra de charger la bonne architecture à chaque fois. Dans le prochain [tutoriel](preprocessing), vous apprenez à utiliser un tokenizer, processeur d'image, extracteur de caractéristiques et processeur pour pré-traiter un jeu de données pour le fine-tuning.
</tf>
</frameworkcontent>
