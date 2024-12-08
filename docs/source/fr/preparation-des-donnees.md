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
# Prétraitement

Avant de pouvoir entraîner un modèle sur un ensemble de données, celui-ci doit être prétraité pour correspondre au format d'entrée attendu par le modèle. Que vos données soient du texte, des images ou de l'audio, elles doivent être converties et assemblées en séries de tenseurs. 🤗 Transformers fournit un ensemble de classes de prétraitement pour vous aider à préparer vos données pour le modèle. Dans ce tutoriel, vous apprendrez à :

* Pour le texte, utiliser un [Tokenizer](./main_classes/tokenizer) pour convertir le texte en une séquence de tokens, créer une représentation numérique des tokens, et les assembler en tenseurs.
* Pour la parole et l'audio, utiliser un ["Feature Extractor"](./main_classes/feature_extractor) pour extraire des caractéristiques séquentielles des ondes audio et les convertir en tenseurs.
* Pour les entrées d'images, utiliser un ["Image Processor"](./main_classes/image_processor) pour convertir les images en tenseurs.
* Pour les entrées multimodales, utiliser un ["Processor"](./main_classes/processors) pour combiner un tokenizer et un extracteur de caractéristiques ou un processeur d'images.

<Tip>

`AutoProcessor` fonctionne **toujours** et choisit automatiquement la classe correcte pour le modèle que vous utilisez, que vous utilisiez un tokenizer, un processeur d'images, un extracteur de caractéristiques ou un processeur.

</Tip>

Avant de commencer, installez 🤗 Datasets afin de pouvoir charger quelques jeux de données pour expérimenter :

```bash
pip install datasets
```

## Traitement du Langage Naturel

<Youtube id="Yffk5aydLzg"/>

L'outil principal pour le prétraitement des données textuelles est un [tokenizer](main_classes/tokenizer). Un tokenizer divise le texte en *tokens* selon un ensemble de règles. Les tokens sont convertis en nombres puis en tenseurs, qui deviennent les entrées du modèle. Toute entrée supplémentaire requise par le modèle est ajoutée par le tokenizer.

<Tip>

Si vous prévoyez d'utiliser un modèle pré-entraîné, il est important d'utiliser le tokenizer pré-entraîné associé. Cela garantit que le texte est divisé de la même manière que le corpus de pré-entraînement, et utilise la même correspondance tokens-index (généralement appelée *vocab*) que lors du pré-entraînement.

</Tip>

Commencez par charger un tokenizer pré-entraîné avec la méthode [`AutoTokenizer.from_pretrained`]. Cela télécharge le *vocab* avec lequel un modèle a été pré-entraîné :

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

Puis passees votre texte au "*tokenizer*"

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```


Le tokenizer renvoie un dictionnaire avec trois éléments importants :

* [input_ids](glossary#input-ids) sont les indices correspondant à chaque token dans la phrase.
* [attention_mask](glossary#attention-mask) indique si un token doit être pris en compte ou non par le mécanisme d'attention.
* [token_type_ids](glossary#token-type-ids) identifie à quelle séquence appartient un token lorsqu'il y a plus d'une séquence.

Récupérez votre entrée en décodant les `input_ids` :

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

Comme vous pouvez le voir, le tokenizer a ajouté deux tokens spéciaux - `CLS` et `SEP` (classificateur et séparateur) - à la phrase. Tous les modèles n'ont pas besoin de tokens spéciaux, mais s'ils en ont besoin, le tokenizer les ajoute automatiquement pour vous.

S'il y a plusieurs phrases que vous voulez prétraiter, passez-les sous forme de liste au tokenizer :

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### Padding

Les phrases n'ont pas toujours la même longueur, ce qui peut poser problème car les entrées du modèle, représentées sous forme de tenseurs doivent avoir une forme uniforme. Le padding est une stratégie pour s'assurer que les tenseurs sont rectangulaires en ajoutant un *token de padding* spécial aux phrases plus courtes.

Définissez le paramètre `padding` à `True` pour ajouter du padding aux séquences plus courtes du lot ("batch") afin qu'elles correspondent à la séquence la plus longue :

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

Les première et troisième phrases sont maintenant complétées avec des `0` car elles sont plus courtes.

### Troncature

À l'autre extrémité du spectre, parfois une séquence peut être trop longue pour être traitée par un modèle. Dans ce cas, vous devrez tronquer la séquence à une longueur plus courte.

Définissez le paramètre `truncation` à `True` pour tronquer une séquence à la longueur maximale acceptée par le modèle :

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

<Tip>

Consultez le guide conceptuel ["Padding and truncation"](./pad_truncation) pour en savoir plus sur les différents arguments de padding et de troncature.

</Tip>

### Construction des tenseurs

Enfin, vous voulez que le tokenizer renvoie les tenseurs réels qui sont fournis au modèle.

Définissez le paramètre `return_tensors` à `pt` pour PyTorch, ou `tf` pour TensorFlow :

<frameworkcontent>
<pt>

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt>
<tf>
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>,
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>,
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
```
</tf>
</frameworkcontent>

<Tip>

Les différents pipelines prennent en charge les arguments du tokenizer dans leur `__call__()` de manière différente. Les pipelines `text-2-text-generation` ne prennent en charge (c'est-à-dire ne transmettent) que `truncation`. Les pipelines `text-generation` prennent en charge `max_length`, `truncation`, `padding` et `add_special_tokens`.

Dans les pipelines `fill-mask`, les arguments du tokenizer peuvent être passés dans l'argument `tokenizer_kwargs` (dictionnaire).

</Tip>

## Audio

Pour les tâches audio, vous aurez besoin d'un ["feature extractor"](main_classes/feature_extractor) pour préparer votre jeu de données pour le modèle. L'extracteur de caractéristiques est conçu pour extraire des caractéristiques des données audio brutes et les convertir en tenseurs.

Chargez le jeu de données [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) (consultez le tutoriel 🤗 [Datasets](https://huggingface.co/docs/datasets/load_hub) pour plus de détails sur la façon de charger un jeu de données) pour voir comment vous pouvez utiliser un extracteur de caractéristiques avec des jeux de données audio :

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

Accédez au premier élément de la colonne `audio` pour examiner l'entrée. L'appel de la colonne `audio` charge et rééchantillonne automatiquement le fichier audio :

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

Cela retourne trois éléments :

* `array` est le signal vocal chargé - et potentiellement rééchantillonné - sous forme de tableau 1D.
* `path` pointe vers l'emplacement du fichier audio.
* `sampling_rate` fait référence au nombre de points de données dans le signal vocal mesurés par seconde.

Pour ce tutoriel, vous utiliserez le modèle [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base). En examinant la fiche du modèle, vous apprendrez que Wav2Vec2 est pré-entraîné sur de l'audio vocal échantillonné à 16kHz. Il est important que le taux d'échantillonnage de vos données audio corresponde au taux d'échantillonnage du jeu de données utilisé pour pré-entraîner le modèle. Si le taux d'échantillonnage de vos données n'est pas le même, vous devez rééchantillonner vos données.

1. Utilisez la méthode [`~datasets.Dataset.cast_column`] de 🤗 Datasets pour suréchantillonner le taux d'échantillonnage à 16kHz :

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. Appelez à nouveau la colonne `audio` pour rééchantillonner le fichier audio :

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

Ensuite, chargez un extracteur de caractéristiques pour normaliser et ajouter du padding à l'entrée. Lors de l'ajout de padding aux données textuelles, un `0` est ajouté pour les séquences plus courtes. La même idée s'applique aux données audio. L'extracteur de caractéristiques ajoute un `0` - interprété comme du silence - à `array`.

Chargez l'extracteur de caractéristiques avec [`AutoFeatureExtractor.from_pretrained`] :

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

Passez le `array` audio à l'extracteur de caractéristiques. Nous recommandons également d'ajouter l'argument `sampling_rate` dans l'extracteur de caractéristiques afin de mieux déboguer les erreurs silencieuses qui pourraient survenir.

```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

Tout comme le tokenizer, vous pouvez appliquer du padding ou de la troncature pour gérer les séquences variables dans un lot. Jetez un œil à la longueur de séquence de ces deux échantillons audio :

```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

Créez une fonction pour prétraiter le jeu de données afin que les échantillons audio aient la même longueur. Spécifiez une longueur maximale d'échantillon, et l'extracteur de caractéristiques ajoutera du padding ou tronquera les séquences pour correspondre à cette longueur :

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

Appliquez la `preprocess_function` aux premiers exemples du jeu de données :

```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

Les longueurs des échantillons sont maintenant les mêmes et correspondent à la longueur maximale spécifiée. Vous pouvez maintenant passer votre jeu de données traité au modèle !

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

## Vision par ordinateur

Pour les tâches de vision par ordinateur, vous aurez besoin d'un [processeur d'images](main_classes/image_processor) pour préparer votre jeu de données pour le modèle. Le prétraitement des images consiste en plusieurs étapes qui convertissent les images en l'entrée attendue par le modèle. Ces étapes incluent, sans s'y limiter, le redimensionnement, la normalisation, la correction des canaux de couleur et la conversion des images en tenseurs.

<Tip>

Le prétraitement des images suit souvent une forme d'augmentation d'image. Le prétraitement des images et l'augmentation des images transforment tous deux les données d'image, mais ils servent des objectifs différents :

* L'augmentation d'image modifie les images d'une manière qui peut aider à prévenir le surapprentissage et augmenter la robustesse du modèle. Vous pouvez être créatif dans la façon dont vous augmentez vos données - ajustez la luminosité et les couleurs, recadrez, faites pivoter, redimensionnez, zoomez, etc. Cependant, veillez à ne pas changer le sens des images avec vos augmentations.
* Le prétraitement des images garantit que les images correspondent au format d'entrée attendu par le modèle. Lors du fine-tuning d'un modèle de vision par ordinateur, les images doivent être prétraitées exactement comme lors de l'entraînement initial du modèle.

Vous pouvez utiliser n'importe quelle bibliothèque pour l'augmentation d'image. Pour le prétraitement des images, utilisez l'`ImageProcessor` associé au modèle.

</Tip>

Chargez le jeu de données [food101](https://huggingface.co/datasets/food101) (voir le tutoriel 🤗 [Datasets](https://huggingface.co/docs/datasets/load_hub) pour plus de détails sur la façon de charger un jeu de données) pour voir comment vous pouvez utiliser un processeur d'images avec des jeux de données de vision par ordinateur :

<Tip>

Utilisez le paramètre `split` de 🤗 Datasets pour ne charger qu'un petit échantillon de la partition d'entraînement car le jeu de données est assez volumineux !

</Tip>

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

Ensuite, examinez l'image avec la fonctionnalité [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=image#datasets.Image) de 🤗 Datasets :

```py
>>> dataset[0]["image"]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/>
</div>

Chargez le processeur d'images avec [`AutoImageProcessor.from_pretrained`] :

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

Tout d'abord, ajoutons de l'augmentation d'image. Vous pouvez utiliser la bibliothèque de votre choix, mais dans ce tutoriel, nous utiliserons le module [`transforms`](https://pytorch.org/vision/stable/transforms.html) de torchvision. Si vous souhaitez utiliser une autre bibliothèque d'augmentation de données, apprenez comment faire dans les notebooks [Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb) ou [Kornia](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb).

1. Ici, nous utilisons [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) pour enchaîner quelques transformations - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) et [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html). Notez que pour le redimensionnement, nous pouvons obtenir les exigences de taille d'image à partir de l'`image_processor`. Pour certains modèles, une hauteur et une largeur exactes sont attendues, pour d'autres, seul le `shortest_edge` est défini.

```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. Le modèle accepte [`pixel_values`](model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel.forward.pixel_values) comme entrée. `ImageProcessor` peut se charger de normaliser les images et de générer les tenseurs appropriés. Créez une fonction qui combine l'augmentation d'image et le prétraitement d'image pour un lot d'images et génère `pixel_values` :

```py
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

Dans l'exemple ci-dessus, nous avons défini `do_resize=False` car nous avons déjà redimensionné les images dans la transformation d'augmentation d'image, et utilisé l'attribut `size` du `image_processor` approprié. Si vous ne redimensionnez pas les images pendant l'augmentation d'image, omettez ce paramètre. Par défaut, `ImageProcessor` gérera le redimensionnement.

Si vous souhaitez normaliser les images dans le cadre de la transformation d'augmentation, utilisez les valeurs `image_processor.image_mean` et `image_processor.image_std`.
</Tip>

3. Ensuite, utilisez [`~datasets.Dataset.set_transform`] de 🤗 Datasets pour appliquer les transformations à la volée :
```py
>>> dataset.set_transform(transforms)
```

4. Maintenant, lorsque vous accédez à l'image, vous remarquerez que le processeur d'images a ajouté `pixel_values`. Vous pouvez maintenant passer votre jeu de données traité au modèle !

```py
>>> dataset[0].keys()
```

Voici à quoi ressemble l'image après l'application des transformations. L'image a été recadrée aléatoirement et ses propriétés de couleur sont différentes.

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png"/>
</div>

<Tip>

Pour des tâches comme la détection d'objets, la segmentation sémantique, la segmentation d'instances et la segmentation panoptique, `ImageProcessor` offre des méthodes de post-traitement. Ces méthodes convertissent les sorties brutes du modèle en prédictions significatives telles que des boîtes englobantes ou des cartes de segmentation.

</Tip>

### Padding

Dans certains cas, par exemple lors du fine-tuning de [DETR](./model_doc/detr), le modèle applique une augmentation d'échelle pendant l'entraînement. Cela peut entraîner des images de tailles différentes dans un lot. Vous pouvez utiliser [`DetrImageProcessor.pad`] de [`DetrImageProcessor`] et définir une `collate_fn` personnalisée pour regrouper les images ensemble.

```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## Multimodal

Pour les tâches impliquant des entrées multimodales, vous aurez besoin d'un [processeur](main_classes/processors) pour préparer votre jeu de données pour le modèle. Un processeur couple deux objets de traitement tels qu'un tokenizer et un extracteur de caractéristiques.

Chargez le jeu de données [LJ Speech](https://huggingface.co/datasets/lj_speech) (voir le tutoriel 🤗 [Datasets](https://huggingface.co/docs/datasets/load_hub) pour plus de détails sur la façon de charger un jeu de données) pour voir comment vous pouvez utiliser un processeur pour la reconnaissance automatique de la parole (ASR) :

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

Pour l'ASR, vous vous concentrez principalement sur `audio` et `text`, donc vous pouvez supprimer les autres colonnes :

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

Maintenant, examinez les colonnes `audio` et `text` :

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

N'oubliez pas que vous devez toujours [rééchantillonner](preprocessing#audio) le taux d'échantillonnage de votre jeu de données audio pour correspondre au taux d'échantillonnage du jeu de données utilisé pour pré-entraîner un modèle !

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

Chargez un processeur avec [`AutoProcessor.from_pretrained`] :

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. Créez une fonction pour traiter les données audio contenues dans `array` en `input_values`, et tokeniser `text` en `labels`. Ce sont les entrées du modèle :

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. Appliquez la fonction `prepare_dataset` à un échantillon :

```py
>>> prepare_dataset(lj_speech[0])
```

Le processeur a maintenant ajouté `input_values` et `labels`, et le taux d'échantillonnage a également été correctement sous-échantillonné à 16kHz. Vous pouvez maintenant passer votre jeu de données traité au modèle !
