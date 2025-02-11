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

# Entraîner avec un script

En plus des [notebooks](./notebooks) de 🤗 Transformers, il existe également des exemples de scripts démontrant comment entraîner un modèle pour une tâche avec [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow) ou [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax).


Vous trouverez également des scripts que nous avons utilisé dans nos [projets de recherche](https://github.com/huggingface/transformers/tree/main/examples/research_projects) et des [exemples "legacy"](https://github.com/huggingface/transformers/tree/main/examples/legacy) qui sont des contributions de la communauté. Ces scripts ne sont pas activement maintenus et nécessitent une version spécifique de 🤗 Transformers qui sera probablement incompatible avec la dernière version de la librairie.

Les exemples de scripts ne sont pas censés fonctionner immédiatement pour chaque problème, et il se peut que vous ayez besoin d'adapter le script au problème que vous essayez de résoudre. Pour vous aider dans cette tâche, la plupart des scripts exposent entièrement la manière dont les données sont prétraitées, vous permettant de les modifier selon vos besoins.

Pour toute fonctionnalité que vous souhaitez implémenter dans un script d'exemple, veuillez en discuter sur le [forum](https://discuss.huggingface.co/) ou dans une [issue](https://github.com/huggingface/transformers/issues) avant de soumettre une Pull Request. Bien que nous acceptions les corrections de bugs, il est peu probable que nous fusionnions une Pull Request (opération "merge" dans Git) ajoutant plus de fonctionnalités au détriment de la lisibilité.

Ce guide vous montrera comment exécuter un script d'entraînement de résumé en exemple avec [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) et [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization). Tous les exemples sont censés fonctionner avec les deux frameworks, sauf indication contraire.

## Configuration

Pour exécuter avec succès la dernière version des scripts d'exemple, vous devez **installer 🤗 Transformers à partir du code source** dans un nouvel environnement virtuel :

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

Pour les versions plus anciennes des exemples de scripts, cliquez sur le bouton ci-dessous :

<details>
  <summary>Exemples pour les anciennes versions de Transformers 🤗</summary>
	<ul>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>
		<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>
	</ul>
</details>

Ensuite, changez votre clone actuel de  🤗 Transformers pour une version spécifique, comme par exemple v3.5.1 :

```bash
git checkout tags/v3.5.1
```

Après avoir configuré la bonne version de la librairie, accédez au dossier d'exemple de votre choix et installez les prérequis spécifiques à l'exemple.

```bash
pip install -r requirements.txt
```

## Exécuter un script

<frameworkcontent>
<pt>

Le script d'exemple télécharge et prétraite un jeu de données à partir de la bibliothèque 🤗 [Datasets](https://huggingface.co/docs/datasets/). Ensuite, le script affine un ensemble de données à l'aide de [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) sur une architecture qui prend en charge la tâche de résumé. L'exemple suivant montre comment ajuster le modèle [T5-small](https://huggingface.co/google-t5/t5-small) sur les données [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail). Le modèle T5 nécessite un argument supplémentaire `source_prefix` en raison de la façon dont il a été entraîné. Cette invite permet à T5 de savoir qu'il s'agit d'une tâche de résumé.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>

Le script d'exemple télécharge et prétraite un jeu de données à partir de la bibliothèque  🤗 [Datasets](https://huggingface.co/docs/datasets/). Ensuite, le script ajuste un modèle à l'aide de Keras sur une architecture qui prend en charge la tâche de résumé. L'exemple suivant montre comment ajuster le modèle [T5-small](https://huggingface.co/google-t5/t5-small) sur le jeu de données [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail). Le modèle T5 nécessite un argument supplémentaire source_prefix en raison de la façon dont il a été entraîné. Cette invite permet à T5 de savoir qu'il s'agit d'une tâche de résumé.

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## Entraînement distribué et précision mixte

[Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) prend en charge l'entraînement distribué et la précision mixte, ce qui signifie que vous pouvez également les utiliser dans un script. Pour activer ces deux fonctionnalités :

- Ajoutez l'argument fp16 pour activer la précision mixte.
- Définissez le nombre de GPU à utiliser avec l'argument `nproc_per_node`.

```bash
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

Les scripts TensorFlow utilisent une Strategie en Miroir [`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) pour l'entraînement distribué, et vous n'avez pas besoin d'ajouter d'arguments supplémentaires au script d'entraînement. Le script TensorFlow utilisera plusieurs GPU par défaut s'ils sont disponibles.

## Exécuter un script sur un TPU 

<frameworkcontent>
<pt>

Les unités de traitement de tenseurs (UTT) (TPU) sont spécialement conçues pour accélérer les performances. PyTorch prend en charge les TPU avec le compilateur de deep learning [XLA](https://www.tensorflow.org/xla). Pour utiliser un TPU, lancez le script xla_spawn.py et utilisez l'argument num_cores pour définir le nombre de cœurs TPU que vous souhaitez utilise

```bash
python xla_spawn.py --num_cores 8 \
    summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```
</pt>
<tf>
Les scripts TensorFlow utilisent une [`TPUStrategy`](https://www.tensorflow.org/guide/distributed_training#tpustrategy) pour l'entraînement sur TPU. Pour utiliser un TPU, passez le nom de la ressource TPU à l'argument tpu.

```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval
```
</tf>
</frameworkcontent>

## Exécuter un script avec 🤗 Accelerate 

🤗 [Accelerate](https://huggingface.co/docs/accelerate) est une bibliothèque uniquement pour PyTorch qui offre une méthode unifiée pour entraîner un modèle sur plusieurs types de configurations (CPU uniquement, plusieurs GPU, TPU) tout en maintenant une visibilité complète sur la boucle d'entraînement PyTorch. Assurez-vous que vous avez installé 🤗 Accelerate si ce n'est pas déjà le cas.

> Note : Comme Accelerate est en développement rapide, la version git d'accelerate doit être installée pour exécuter les scripts.
```bash
pip install git+https://github.com/huggingface/accelerate
```

Au lieu du script `run_summarization.py`, vous devez utiliser le script `run_summarization_no_trainer.py`. Les scripts compatibles avec 🤗 Accelerate auront un fichier `task_no_trainer.py` dans le dossier. Commencez par exécuter la commande suivante pour créer et enregistrer un fichier de configuration.

```bash
accelerate config
```

Testez votre configuration pour vous assurer qu'elle est correctement configurée :

```bash
accelerate test
```

Maintenant, vous êtes prêt à lancer l'entraînement :

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
```

## Utiliser un jeu de données personnalisé 

Le script de résumé prend en charge les jeux de données personnalisés tant qu'ils sont au format CSV ou JSON Line. Lorsque vous utilisez votre propre jeu de données, vous devez spécifier plusieurs arguments supplémentaires :

- `train_file` et `validation_file` spécifient le chemin vers vos fichiers d'entraînement et de validation.
- `text_column` est le texte d'entrée à résumer.
- `summary_column` est le texte cible à produire.

Un exemple de script de résumé utilisant un ensemble de données personnalisé ressemblerait à ceci :

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```

## Tester un script
Il est souvent judicieux d'exécuter votre script sur un plus petit nombre d'exemples de jeu de données pour s'assurer que tout fonctionne comme prévu avant de s'engager sur un jeu de données complet qui pourrait prendre des heures à traiter. Utilisez les arguments suivants pour tronquer le jeu de données à un nombre maximal d'échantillons :

- `max_train_samples`
- `max_eval_samples`
- `max_predict_samples`

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

Tous les scripts d'exemple ne prennent pas en charge l'argument `max_predict_samples`. Si vous n'êtes pas sûr que votre script prenne en charge cet argument, ajoutez l'argument `-h` pour vérifier.

```bash
examples/pytorch/summarization/run_summarization.py -h
```

## Reprendre l'entraînement à partir d'un point de contrôle

Une autre option utile est de reprendre l'entraînement à partir d'un point de contrôle précédent. Cela vous permettra de reprendre là où vous vous étiez arrêté sans recommencer si votre entraînement est interrompu. Il existe deux méthodes pour reprendre l'entraînement à partir d'un point de contrôle.

La première méthode utilise l'argument `output_dir previous_output_dir` pour reprendre l'entraînement à partir du dernier point de contrôle stocké dans `output_dir`. Dans ce cas, vous devez supprimer l'argument `overwrite_output_dir`.

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --output_dir previous_output_dir \
    --predict_with_generate
```

La seconde méthode utilise l'argument `resume_from_checkpoint path_to_specific_checkpoint` pour reprendre l'entraînement à partir d'un dossier de point de contrôle spécifique.

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate
```

## Partage ton modèle

Tous les scripts peuvent télécharger votre modèle final sur le Model Hub. Assurez-vous que vous êtes connecté à Hugging Face avant de commencer :

```bash
huggingface-cli login
```

Ensuite, ajoutez l'argument `push_to_hub` au script. Cet argument créera un dépôt avec votre nom d'utilisateur Hugging Face et le nom du dossier spécifié dans `output_dir`.


Pour donner un nom spécifique à votre dépôt, utilisez l'argument `push_to_hub_model_id` pour l'ajouter. Le dépôt sera automatiquement listé sous votre namespace. 

L'exemple suivant montre comment télécharger un modèle avec un nom de dépôt spécifique :

```bash
python examples/pytorch/summarization/run_summarization.py
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```