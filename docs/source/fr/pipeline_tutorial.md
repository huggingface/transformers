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

# Pipelines pour l'inférence

Le [`pipeline`] facilite l'utilisation de n'importe quel modèle sur le [Hub](https://huggingface.co/models) pour l'inférence sur n'importe quelle tâche de langage, vision par ordinateur, parole et multimodale. Même si vous n'avez pas d'expérience avec une modalité spécifique ou si vous n'êtes pas familier avec le code associé aux modèles, vous pouvez toujours les utiliser pour l'inférence avec le [`pipeline`] ! Ce tutoriel vous apprend à :

  * Utiliser un [`pipeline`] pour l'inférence.
  * Utiliser un tokenizer ou un modèle spécifique.
  * Utiliser un [`pipeline`] pour des tâches audio, de vision et multimodales.

<Tip>

Consultez la documentation du [`pipeline`] pour avoir une liste complète des tâches supportées et des paramètres disponibles.

</Tip>

## Utilisation du pipeline

Bien que chaque tâche ait un [`pipeline`] associé, il est plus simple d'utiliser l'abstraction générale [`pipeline`] qui contient tous les pipelines spécifiques aux tâches. Le [`pipeline`] charge automatiquement un modèle par défaut et une classe de prétraitement capable de faire de l'inférence pour votre tâche. Prenons l'exemple de l'utilisation du [`pipeline`] pour la reconnaissance automatique de la parole, ou la transcription de la parole en texte.

1. Commencez par créer un [`pipeline`] et spécifiez une tâche :

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition")
```

2. Passez vos données au [`pipeline`]. Pour la reconnaissance automatique de la parole, il s'agit d'un fichier audio :

```py
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}
```

Pas le résultat que vous attendiez ? Cherchez parmi les [modèles de reconnaissance automatique de la parole les plus téléchargés](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending) sur le Hub pour voir si vous pouvez obtenir une meilleure transcription.

Essayons le modèle [Whisper large-v2](https://huggingface.co/openai/whisper-large) d'OpenAI. Whisper a été publié 2 ans après Wav2Vec2, et a été entraîné avec près de 10 fois plus de données. Ainsi, il bat Wav2Vec2 sur la plupart des benchmarks. Il peut également prédire la ponctuation et la casse, ce qui n'est pas possible avec Wav2Vec2.

Voyons comment il se comporte ici :

```py
>>> transcriber = pipeline(model="openai/whisper-large-v2")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

Ce résultat semble plus précis ! Pour une comparaison approfondie entre Wav2Vec2 et Whisper, consultez le [cours sur les Transformers audio](https://huggingface.co/learn/audio-course/chapter5/asr_models). Nous vous encourageons également à consulter le Hub pour des modèles dans différentes langues, des modèles spécialisés dans certains domaines, etc. Vous pouvez vérifier et comparer les résultats des modèles directement depuis votre navigateur sur le Hub pour voir s'ils sont meilleurs ou traitent mieux les cas particuliers que les autres. Et si vous ne trouvez pas de modèle pour votre cas d'utilisation, vous pouvez toujours commencer à [entraîner](training) le vôtre !

Si vous avez plusieurs données d'entrée, vous pouvez les passer sous forme de liste :

```py
transcriber(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)
```

Les pipelines sont idéaux pour l'expérimentation car passer d'un modèle à un autre est trivial ; cependant, il est possible de les optimiser pour des cas d'utilisation plus intentifs que l'expérimentation. Consultez les guides suivants qui détaillent l'utilisation des pipelines sur des ensembles de données entiers ou dans un serveur web :

  * [Utilisation des pipelines sur un ensemble de données](#utilisation-des-pipelines-sur-un-ensemble-de-données)
  * [Utilisation des pipelines pour un serveur web](./pipeline_webserver)

## Paramètres

[`pipeline`] supporte de nombreux paramètres ; certains sont spécifiques à la tâche, d'autres sont généraux à tous les pipelines. En général, vous pouvez spécifier des paramètres n'importe où :

```py
transcriber = pipeline(model="openai/whisper-large-v2", my_parameter=1)

out = transcriber(...)  # This will use `my_parameter=1`.
out = transcriber(..., my_parameter=2)  # This will override and use `my_parameter=2`.
out = transcriber(...)  # This will go back to using `my_parameter=1`.
```

Regardons 3 paramètres importants :

### Processeur

Si vous utilisez `device=n`, le pipeline met automatiquement le modèle sur le processeur (CPU ou GPU) spécifié. Cela fonctionnera que vous utilisiez PyTorch ou Tensorflow.

```py
transcriber = pipeline(model="openai/whisper-large-v2", device=0)
```

Si le modèle est trop grand pour une seule GPU et que vous utilisez PyTorch, vous pouvez définir `device_map="auto"` pour déterminer automatiquement comment charger et stocker les poids du modèle. L'utilisation du paramètre `device_map` nécessite la librairie 🤗 [Accelerate](https://huggingface.co/docs/accelerate) :

```bash
pip install --upgrade accelerate
```

Le code suivant charge et stocke automatiquement les poids du modèle sur plusieurs processeurs :

```py
transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")
```

Notez que si `device_map="auto"` est donné, il n'est pas nécessaire d'ajouter le paramètre `device=device` lors de la création de votre `pipeline` car vous pouvez rencontrer un comportement inattendu !

### Taille de lot

Par défaut, les pipelines ne font pas d'inférence par lots pour des raisons expliquées en détail [ici](https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching). La raison est que le traitement par lots n'est pas nécessairement plus rapide, et peut même être beaucoup plus lent dans certains cas.

Mais si cela peut être utile dans votre cas d'utilisation, vous pouvez utiliser :

```py
transcriber = pipeline(model="openai/whisper-large-v2", device=0, batch_size=2)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = transcriber(audio_filenames)
```

Cet exemple exécute le pipeline sur les 4 fichiers audio fournis, mais il les passera par lots de 2 au modèle (qui est sur un GPU, où le traitement par lots est plus susceptible d'aider) sans nécessiter de code supplémentaire de votre part.
Le résultat sera le même que si vous aviez exécuté le pipeline sur chaque fichier audio individuellement. L'objectif est de vous aider à obtenir les résultats plus rapidement.

Les pipelines peuvent également atténuer certaines des complexités du traitement par lots car, pour certains pipelines, un seul élément (comme un long fichier audio) doit être divisé en plusieurs parties pour être traité par un modèle. Le pipeline effectue ce [*traitement par lots de morceaux*](./main_classes/pipelines#pipeline-chunk-batching) pour vous.

### Paramètres spécifiques à une tâche

Toutes les tâches ont des paramètres spécifiques qui permettent une flexibilité et des options supplémentaires pour vous aider à compléter cette tâche.
Par exemple, la méthode [`transformers.AutomaticSpeechRecognitionPipeline.__call__`] a un paramètre `return_timestamps` qui semble utile pour sous-titrer des vidéos :

```py
>>> transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.', 'chunks': [{'timestamp': (0.0, 11.88), 'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its'}, {'timestamp': (11.88, 12.38), 'text': ' creed.'}]}
```

Comme vous pouvez le voir, le modèle a déduit le texte et a également indiqué **quand** les différentes phrases ont été prononcées.

Il y a beaucoup de paramètres disponibles pour chaque tâche, donc consultez la documentation de l'API de chaque tâche pour voir ce que vous pouvez paramétrer !
Par exemple, le pipeline [`~transformers.AutomaticSpeechRecognitionPipeline`] a un paramètre `chunk_length_s` qui est utile quand on travaille sur des fichiers audio très longs (par exemple, sous-titrage de films entiers ou de vidéos d'une heure) qu'un modèle ne peut généralement pas gérer seul :

```python
>>> transcriber = pipeline(model="openai/whisper-large-v2", chunk_length_s=30, return_timestamps=True)
>>> transcriber("https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav")
{'text': " Chapter 16. I might have told you of the beginning of this liaison in a few lines, but I wanted you to see every step by which we came.  I, too, agree to whatever Marguerite wished, Marguerite to be unable to live apart from me. It was the day after the evening...}
```

Si vous ne trouvez pas un paramètre qui serait vraiment utile, n'hésitez pas à [le demander](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml) !

## Utilisation des pipelines sur un ensemble de données

Le pipeline peut faire de l'inférence sur un grand ensemble de données. La façon la plus simple de le faire est d'utiliser un itérateur :

```py
def data():
    for i in range(1000):
        yield f"My example {i}"


pipe = pipeline(model="gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])
```

L'itérateur `data()` renvoie chaque résultat, et le pipeline reconnaît automatiquement que les données d'entrée sont itérables et commencera à récupérer les données tout en continuant à les traiter sur le GPU (cela utilise un [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) en interne). Ceci est important car vous n'avez pas à allouer de mémoire pour l'ensemble du jeu de données et vous pouvez passer les données au GPU aussi rapidement que possible.

Comme le traitement par lots peut accélérer les choses, il peut être utile d'essayer de régler le paramètre `batch_size` ici.

La manière la plus simple d'itérer sur un ensemble de données est de simplement d'en charger un depuis 🤗 [Datasets](https://github.com/huggingface/datasets/):

```py
# KeyDataset is a util that will just output the item we're interested in.
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
```

## Utilisation des pipelines pour un serveur web

<Tip>
La création d'un moteur d'inférence est un sujet complexe qui mérite sa propre page de documentation.
</Tip>

[Link](./pipeline_webserver)

## Pipeline visuel

Utiliser un [`pipeline`] pour les tâches de vision est quasiment identique.

Specifiez votre tâche et passez votre image au classifieur. L'image peut être un lien, un chemin local ou une image encodée en base64. Par exemple, quelle espèce de chat est montrée ci-dessous ?

![pipeline-cat-chonk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)

```py
>>> from transformers import pipeline

>>> vision_classifier = pipeline(model="google/vit-base-patch16-224")
>>> preds = vision_classifier(
...     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4335, 'label': 'lynx, catamount'}, {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}, {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}, {'score': 0.0239, 'label': 'Egyptian cat'}, {'score': 0.0229, 'label': 'tiger cat'}]
```

## Pipeline textuel

Utiliser un [`pipeline`] pour les tâches de traitement du langage est quasiment identique.

```py
>>> from transformers import pipeline

>>> # This model is a `zero-shot-classification` model.
>>> # It will classify text, except you are free to choose any label you might imagine
>>> classifier = pipeline(model="facebook/bart-large-mnli")
>>> classifier(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
```

## Pipeline multimodale

Le [`pipeline`] supporte plus d'une modalité. Par exemple, une tâche de question-réponse visuelle combine texte et image. N'hésitez pas à utiliser n'importe quel lien d'image que vous aimez et une question que vous voulez poser sur l'image. L'image peut être un URL ou un emplacement de fichier local.

Par exemple, si vous utilisez cette [image de facture](https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png) :

```py
>>> from transformers import pipeline

>>> vqa = pipeline(model="impira/layoutlm-document-qa")
>>> vqa(
...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
...     question="What is the invoice number?",
... )
[{'score': 0.42515, 'answer': 'us-001', 'start': 16, 'end': 16}]
```

<Tip>

Pour exécuter l'exemple ci-dessus, [`pytesseract`](https://pypi.org/project/pytesseract/) doit être installé en plus de 🤗 Transformers :

```bash
sudo apt install -y tesseract-ocr
pip install pytesseract
```

</Tip>

## Utiliser un `pipeline` avec de grands modèles avec 🤗 `accelerate`

Vous pouvez facilement utiliser `pipeline` avec de grands modèles en utilisant 🤗 `accelerate` ! Assurez-vous d'abord d'avoir installé `accelerate` avec `pip install accelerate`.

Commencez par charger votre modèle en utilisant `device_map="auto"` ! Nous utilisons `facebook/opt-1.3b` pour cette exemple.

```py
# pip install accelerate
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

Vous pouvez également charger des modèles en 8 bits si vous installez `bitsandbytes` et ajoutez l'argument `load_in_8bit=True`

```py
# pip install accelerate bitsandbytes
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
```

Notez que vous pouvez remplacer l'ensemble de poids ("checkpoint" en anglais) par n'importe quel modèle Hugging Face qui prend en charge le chargement de grands modèles tels que BLOOM !
