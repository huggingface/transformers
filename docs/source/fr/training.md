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

# Fine-tuning

Le fine-tuning (ou *réglage fin*) consiste à poursuivre l'entraînement d'un grand modèle pré-entraîné sur un jeu de données plus petit, propre à une tâche ou à un domaine. Par exemple, un fine-tuning sur un jeu de données d'exemples de code aide le modèle à mieux programmer. Le fine-tuning est identique au pré-entraînement, à ceci près que vous ne partez pas de poids aléatoires. Il demande aussi beaucoup moins de puissance de calcul, de données et de temps.

Le tutoriel ci-dessous vous guide pas à pas dans le fine-tuning d'un grand modèle de langage avec [`Trainer`].

Connectez-vous à votre compte Hugging Face avec votre jeton d'utilisateur pour pouvoir publier sur le Hub le modèle que vous aurez affiné.

```py
from huggingface_hub import login

login()
```

## Tokenisation

Chargez un jeu de données et [tokenisez](./fast_tokenizers) la colonne de texte sur laquelle le modèle s'entraîne (`horoscope` dans le jeu de données ci-dessous).

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Le tokenizer produit les entrées du modèle, `input_ids` et `attention_mask`. La méthode `forward` du modèle n'accepte que `input_ids` et `attention_mask` : utilisez donc `remove_columns` pour supprimer les colonnes telles que `horoscope` après la tokenisation.

- Définissez `truncation=True` ainsi qu'une valeur `max_length` pour tronquer les séquences trop longues à la longueur maximale indiquée.
- Utilisez la méthode [`~datasets.train_test_split`] pour créer une partition de test qui servira à évaluer le modèle.

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)
```

Un *data collator* assemble les échantillons du jeu de données en lots (*batches*) que le modèle pourra traiter. [`DataCollatorForLanguageModeling`] complète *dynamiquement* chaque lot jusqu'à la longueur de la plus longue séquence de ce lot, au lieu de compléter toutes les séquences du jeu de données à une même longueur. Cela économise du calcul et de la mémoire en évitant de traiter des tokens de remplissage inutiles.

- Définissez `mlm=False` pour éviter de masquer des tokens de façon aléatoire.

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

## Chargement d'un modèle

Chargez un ensemble de poids pré-entraîné (aussi appelé *checkpoint* en anglais) à affiner. Consultez le guide [Chargement des modèles](./models) pour plus de détails à ce sujet.

- Définissez `dtype="auto"` pour charger les poids dans le type de données sous lequel ils ont été enregistrés. Sans cela, PyTorch les charge en `torch.float32`, ce qui double l'utilisation de la mémoire si les poids sont initialement en `torch.bfloat16`.

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## Configuration de l'entraînement

[`TrainingArguments`] rassemble toutes les options permettant de personnaliser un entraînement. Seuls les arguments les plus courants sont présentés ici. Les autres ont des valeurs par défaut raisonnables ou ne concernent que des cas particuliers, comme l'entraînement distribué. Consultez la documentation de l'API [`TrainingArguments`] pour la liste complète des arguments.

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs` et `per_device_train_batch_size` contrôlent la durée de l'entraînement et la taille des lots. `learning_rate` définit le taux d'apprentissage initial de l'optimiseur.

</hfoption>
<hfoption id="training optimizations">

- Définissez `bf16=True` pour un entraînement rapide en précision mixte si votre matériel le permet (GPU Ampere et plus récents). Sinon, repliez-vous sur `fp16=True` sur du matériel plus ancien.
- `gradient_accumulation_steps` simule une taille de lot effective plus grande en accumulant les gradients sur plusieurs passes avant de mettre à jour les poids.
- `gradient_checkpointing` échange du calcul contre de la mémoire en recalculant les activations intermédiaires pendant la passe arrière plutôt qu'en les conservant.

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy` et `save_strategy` déterminent à quel moment évaluer le modèle pendant l'entraînement et à quel moment enregistrer un point de sauvegarde.
- `load_best_model_at_end` charge le meilleur point de sauvegarde une fois l'entraînement terminé. Cet argument nécessite que `eval_strategy` soit défini.

</hfoption>
<hfoption id="logging">

- `logging_steps` contrôle la fréquence à laquelle la perte est mise à jour et renvoyée pendant l'entraînement.

</hfoption>
</hfoptions>

```py
training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

## Entraînement

Créez une instance de [`Trainer`] avec tous les composants nécessaires, puis appelez [`~Trainer.train`] pour lancer l'entraînement.

```py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.push_to_hub()
```

[`~Trainer.push_to_hub`] envoie sur le Hub les poids affinés, la configuration de génération, le tokenizer et la configuration du modèle.

## Pour aller plus loin

- Consultez le guide [Fonctionnalités du Trainer](./trainer_recipes) pour des exemples minimaux et fonctionnels des fonctionnalités courantes du Trainer : fonctions de perte personnalisées, évaluation économe en mémoire, points de sauvegarde, et bien d'autres.
- Consultez le guide [Redéfinir les méthodes du Trainer](./trainer_customize) pour apprendre à surcharger les méthodes de [`Trainer`] afin de prendre en charge des fonctionnalités nouvelles et personnalisées.
- Consultez le guide [Callbacks](./trainer_callbacks) pour apprendre à vous brancher sur les événements de l'entraînement : journalisation, arrêt anticipé et autres comportements personnalisés.
- Consultez le guide [Data collators](./data_collators) pour apprendre à personnaliser la façon dont les échantillons sont assemblés en lots.
- Parcourez [transformers/examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), les [notebooks](./notebooks) ou la section **Resources > Task Recipes** pour d'autres exemples d'entraînement sur des tâches variées : texte, audio, vision et multimodal.
