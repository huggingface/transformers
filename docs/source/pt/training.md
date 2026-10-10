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

# Fine-tuning de um modelo pré-treinado

[[open-in-colab]]

O uso de um modelo pré-treinado tem importantes vantagens. Redução do custo computacional, a pegada de carbono, e te
permite utilizar modelos de última geração sem ter que treinar um novo desde o início.
O 🤗 Transformers proporciona acesso a milhares de modelos pré-treinados numa ampla gama de tarefas.
Quando utilizar um modelo pré-treinado, treine-o com um dataset específico para a sua tarefa.
Isto é chamado de fine-tuning, uma técnica de treinamento incrivelmente poderosa. Neste tutorial faremos o fine-tuning
de um modelo pré-treinado com um framework de Deep Learning da sua escolha:

* Fine-tuning de um modelo pré-treinado com o 🤗 Transformers [`Trainer`].
* Fine-tuning de um modelo pré-treinado no TensorFlow com o Keras.
* Fine-tuning de um modelo pré-treinado em PyTorch nativo.

<a id='data-processing'></a>

## Preparando um dataset

<Youtube id="_BZearw7f0w"/>

Antes de aplicar o fine-tuning a um modelo pré-treinado, baixe um dataset e prepare-o para o treinamento.
O tutorial anterior ensinará a processar os dados para o treinamento, e então poderá ter a oportunidade de testar
esse novo conhecimento em algo prático.

Comece carregando o dataset [Yelp Reviews](https://huggingface.co/datasets/Yelp/yelp_review_full):

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset[100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

Como já sabe, é necessário ter um tokenizador para processar o texto e incluir uma estratégia de padding e truncamento,
para manejar qualquer tamanho varíavel de sequência. Para processar o seu dataset em apenas um passo, utilize o método de
🤗 Datasets [`map`](https://huggingface.co/docs/datasets/process#map) para aplicar uma função de preprocessamento sobre
todo o dataset.

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

Se desejar, é possível criar um subconjunto menor do dataset completo para aplicar o fine-tuning e assim reduzir o tempo necessário.

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Fine-tuning com o `Trainer`

<Youtube id="nvBXf7s7vTI"/>

O 🤗 Transformers proporciona uma classe [`Trainer`] otimizada para o treinamento de modelos de 🤗 Transformers,
facilitando os primeiros passos do treinamento sem a necessidade de escrever manualmente o seu próprio ciclo.
A API do [`Trainer`] suporta um grande conjunto de opções de treinamento e funcionalidades, como o logging,
o gradient accumulation e o mixed precision.

Comece carregando seu modelo e especifique o número de labels de previsão.
A partir do [Card Dataset](https://huggingface.co/datasets/Yelp/yelp_review_full#data-fields) do Yelp Reveiw, que ja
sabemos ter 5 labels usamos o seguinte código:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

<Tip>

    Você verá um alerta sobre alguns pesos pré-treinados que não estão sendo utilizados e que alguns pesos estão
    sendo inicializados aleatoriamente. Não se preocupe, essa mensagem é completamente normal.
    O header/cabeçário pré-treinado do modelo BERT é descartado e substitui-se por um header de classificação
    inicializado aleatoriamente. Assim, pode aplicar o fine-tuning a este novo header do modelo em sua tarefa
    de classificação de sequências fazendo um transfer learning do modelo pré-treinado.

</Tip>

### Hiperparâmetros de treinamento

Em seguida, crie uma classe [`TrainingArguments`] que contenha todos os hiperparâmetros que possam ser ajustados, assim
como os indicadores para ativar as diferentes opções de treinamento. Para este tutorial, você pode começar o treinamento
usando os [hiperparámetros](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) padrão,
porém, sinta-se livre para experimentar com eles e encontrar uma configuração ótima.

Especifique onde salvar os checkpoints do treinamento:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### Métricas

O [`Trainer`] não avalia automaticamente o rendimento do modelo durante o treinamento. Será necessário passar ao
[`Trainer`] uma função para calcular e fazer um diagnóstico sobre as métricas. A biblioteca 🤗 Datasets proporciona
uma função de [`accuracy`](https://huggingface.co/metrics/accuracy) simples que pode ser carregada com a função
`load_metric` (ver este [tutorial](https://huggingface.co/docs/datasets/metrics) para mais informações):

```py
>>> import numpy as np
>>> from datasets import load_metric

>>> metric = load_metric("accuracy")
```

Defina a função `compute` dentro de `metric` para calcular a precisão das suas predições.
Antes de passar as suas predições ao `compute`, é necessário converter as predições à logits (lembre-se que
todos os modelos de 🤗 Transformers retornam logits).

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

Se quiser controlar as suas métricas de avaliação durante o fine-tuning, especifique o parâmetro `eval_strategy`
nos seus argumentos de treinamento para que o modelo considere a métrica de avaliação ao final de cada época:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

### Trainer

Crie um objeto [`Trainer`] com o seu modelo, argumentos de treinamento, conjuntos de dados de treinamento e de teste, e a sua função de avaliação:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Em seguida, aplique o fine-tuning a seu modelo chamado [`~transformers.Trainer.train`]:

```py
>>> trainer.train()
```

<a id='keras'></a>

## Fine-tuning com Keras

<Youtube id="rnTGBy2ax1c"/>

Os modelos de 🤗 Transformers também permitem realizar o treinamento com o TensorFlow com a API do Keras.
Contudo, será necessário fazer algumas mudanças antes de realizar o fine-tuning.

### Conversão do dataset ao formato do TensorFlow

O [`DefaultDataCollator`] junta os tensores em um batch para que o modelo possa ser treinado em cima deles.
Assegure-se de especificar os `return_tensors` para retornar os tensores do TensorFlow:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```

<Tip>

    O [`Trainer`] utiliza [`DataCollatorWithPadding`] por padrão, então você não precisa especificar explicitamente um
    colador de dados (data collator).

</Tip>

Em seguida, converta os datasets tokenizados em datasets do TensorFlow com o método
[`to_tf_dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.to_tf_dataset).
Especifique suas entradas em `columns` e seu rótulo em `label_cols`:

```py
>>> tf_train_dataset = small_train_dataset.to_tf_dataset(
...     columns=["attention_mask", "input_ids", "token_type_ids"],
...     label_cols="labels",
...     shuffle=True,
...     collate_fn=data_collator,
...     batch_size=8,
... )

>>> tf_validation_dataset = small_eval_dataset.to_tf_dataset(
...     columns=["attention_mask", "input_ids", "token_type_ids"],
...     label_cols="labels",
...     shuffle=False,
...     collate_fn=data_collator,
...     batch_size=8,
... )
```

### Compilação e ajustes

Carregue um modelo do TensorFlow com o número esperado de rótulos:

```py
>>> import tensorflow as tf
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

A seguir, compile e ajuste o fine-tuning a seu modelo com [`fit`](https://keras.io/api/models/model_training_apis/) como
faria com qualquer outro modelo do Keras:

```py
>>> model.compile(
...     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
...     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
...     metrics=tf.metrics.SparseCategoricalAccuracy(),
... )

>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)
```

<a id='pytorch_native'></a>

## Fine-tune em PyTorch nativo

<Youtube id="Dh9CL8fyG80"/>

O [`Trainer`] se encarrega do ciclo de treinamento e permite aplicar o fine-tuning a um modelo em uma linha de código apenas.
Para os usuários que preferirem escrever o seu próprio ciclo de treinamento, também é possível aplicar o fine-tuning a um
modelo de 🤗 Transformers em PyTorch nativo.

Neste momento, talvez ocorra a necessidade de reinicar seu notebook ou executar a seguinte linha de código para liberar
memória:

```py
del model
del pytorch_model
del trainer
torch.cuda.empty_cache()
```

Em sequência, faremos um post-processing manual do `tokenized_dataset` e assim prepará-lo para o treinamento.

1. Apague a coluna de `text` porque o modelo não aceita texto cru como entrada:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. Troque o nome da coluna `label` para `labels`, pois o modelo espera um argumento de mesmo nome:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. Defina o formato do dataset para retornar tensores do PyTorch no lugar de listas:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

Em sequência, crie um subconjunto menor do dataset, como foi mostrado anteriormente, para acelerá-lo o fine-tuning.

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

Crie um `DataLoader` para os seus datasets de treinamento e de teste para poder iterar sobre batches de dados:

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

Carregue seu modelo com o número de labels esperados:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

### Otimização e configuração do Learning Rate

Crie um otimizador e um learning rate para aplicar o fine-tuning ao modelo.
Iremos utilizar o otimizador [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) do PyTorch:

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

Defina o learning rate do [`Trainer`]:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

Por último, especifique o `device` do ambiente para utilizar uma GPU se tiver acesso à alguma. Caso contrário, o treinamento
em uma CPU pode acabar levando várias horas em vez de minutos.

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

    Se necessário, você pode obter o acesso gratuito a uma GPU na núvem por meio de um notebook no
    [Colaboratory](https://colab.research.google.com/) ou [SageMaker StudioLab](https://studiolab.sagemaker.aws/)
    se não tiver esse recurso de forma local.

</Tip>

Perfeito, agora estamos prontos para começar o treinamento! 🥳

### Ciclo de treinamento

Para visualizar melhor o processo de treinamento, utilize a biblioteca [tqdm](https://tqdm.github.io/) para adicionar
uma barra de progresso sobre o número de passos percorridos no treinamento atual:

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### Métricas

Da mesma forma que é necessário adicionar uma função de avaliação ao [`Trainer`], é necessário fazer o mesmo quando
escrevendo o próprio ciclo de treinamento. Contudo, em vez de calcular e retornar a métrica final de cada época,
você deverá adicionar todos os batches com [`add_batch`](https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=add_batch#datasets.Metric.add_batch)
e calcular a métrica apenas no final.

```py
>>> metric = load_metric("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```

<a id='additional-resources'></a>

## Recursos adicionais

Para mais exemplos de fine-tuning acesse:

- [🤗 Transformers Notebooks](notebooks) contém vários notebooks sobre como aplicar o fine-tuning a um modelo
para tarefas específicas no PyTorch e TensorFlow.
