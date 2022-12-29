<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Classificação de tokens

<Youtube id="wVHdVlPScxA"/>

A classificação de tokens atribui um rótulo a tokens individuais em uma frase. Uma das tarefas de classificação de tokens mais comuns é o Reconhecimento de Entidade Nomeada, também chamada de NER (sigla em inglês para Named Entity Recognition). O NER tenta encontrar um rótulo para cada entidade em uma frase, como uma pessoa, local ou organização.

Este guia mostrará como realizar o fine-tuning do [DistilBERT](https://huggingface.co/distilbert-base-uncased) no conjunto de dados [WNUT 17](https://huggingface.co/datasets/wnut_17) para detectar novas entidades.

<Tip>

Consulte a [página de tarefas de classificação de tokens](https://huggingface.co/tasks/token-classification) para obter mais informações sobre outras formas de classificação de tokens e seus modelos, conjuntos de dados e métricas associadas.

</Tip>

## Carregando o conjunto de dados WNUT 17

Carregue o conjunto de dados WNUT 17 da biblioteca 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("wnut_17")
```

E dê uma olhada em um exemplo:

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

Cada número em `ner_tags` representa uma entidade. Converta o número em um rótulo para obter mais informações:

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
```

O `ner_tag` descreve uma entidade, como uma organização, local ou pessoa. A letra que prefixa cada `ner_tag` indica a posição do token da entidade:

- `B-` indica o início de uma entidade.
- `I-` indica que um token está contido dentro da mesma entidade (por exemplo, o token `State` pode fazer parte de uma entidade como `Empire State Building`).
- `0` indica que o token não corresponde a nenhuma entidade.

## Pré-processamento

<Youtube id="iY2AZYdZAr0"/>

Carregue o tokenizer do DistilBERT para processar os `tokens`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

Como a entrada já foi dividida em palavras, defina `is_split_into_words=True` para tokenizar as palavras em subpalavras:

```py
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

Ao adicionar os tokens especiais `[CLS]` e `[SEP]` e a tokenização de subpalavras uma incompatibilidade é gerada entre a entrada e os rótulos. Uma única palavra correspondente a um único rótulo pode ser dividida em duas subpalavras. Você precisará realinhar os tokens e os rótulos da seguinte forma:

1. Mapeie todos os tokens para a palavra correspondente com o método [`word_ids`](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.Encoding.word_ids).
2. Atribuindo o rótulo `-100` aos tokens especiais `[CLS]` e `[SEP]` para que a função de loss do PyTorch ignore eles.
3. Rotular apenas o primeiro token de uma determinada palavra. Atribuindo `-100` a outros subtokens da mesma palavra.

Aqui está como você pode criar uma função para realinhar os tokens e rótulos e truncar sequências para não serem maiores que o comprimento máximo de entrada do DistilBERT:

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # Set the special tokens to -100.
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

Use a função [`map`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map) do 🤗 Datasets para tokenizar e alinhar os rótulos em todo o conjunto de dados. Você pode acelerar a função `map` configurando `batched=True` para processar vários elementos do conjunto de dados de uma só vez:

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

Use o [`DataCollatorForTokenClassification`] para criar um batch de exemplos. Ele também *preencherá dinamicamente* seu texto e rótulos para o comprimento do elemento mais longo em seu batch, para que tenham um comprimento uniforme. Embora seja possível preencher seu texto na função `tokenizer` configurando `padding=True`, o preenchimento dinâmico é mais eficiente.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## Treinamento

<frameworkcontent>
<pt>
Carregue o DistilBERT com o [`AutoModelForTokenClassification`] junto com o número de rótulos esperados:

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=14)
```

<Tip>

Se você não estiver familiarizado com o fine-tuning de um modelo com o [`Trainer`], dê uma olhada no tutorial básico [aqui](../training#finetune-with-trainer)!

</Tip>

Nesse ponto, restam apenas três passos:

1. Definir seus hiperparâmetros de treinamento em [`TrainingArguments`].
2. Passar os argumentos de treinamento para o [`Trainer`] junto com o modelo, conjunto de dados, tokenizador e o data collator.
3. Chamar a função [`~Trainer.train`] para executar o fine-tuning do seu modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="./results",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_wnut["train"],
...     eval_dataset=tokenized_wnut["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```
</pt>
<tf>
Para executar o fine-tuning de um modelo no TensorFlow, comece convertendo seu conjunto de dados para o formato `tf.data.Dataset` com [`to_tf_dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.to_tf_dataset). Nessa execução você deverá especificar as entradas e rótulos (no parâmetro `columns`), se deseja embaralhar o conjunto de dados, o tamanho do batch e o data collator:

```py
>>> tf_train_set = tokenized_wnut["train"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = tokenized_wnut["validation"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

<Tip>

Se você não estiver familiarizado com o fine-tuning de um modelo com o Keras, dê uma olhada no tutorial básico [aqui](training#finetune-with-keras)!

</Tip>

Configure o otimizador e alguns hiperparâmetros de treinamento:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 3
>>> num_train_steps = (len(tokenized_wnut["train"]) // batch_size) * num_train_epochs
>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=2e-5,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=0.01,
...     num_warmup_steps=0,
... )
```

Carregue o DistilBERT com o [`TFAutoModelForTokenClassification`] junto com o número de rótulos esperados:

```py
>>> from transformers import TFAutoModelForTokenClassification

>>> model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
```

Configure o modelo para treinamento com o método [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

Chame o método [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) para executar o fine-tuning do modelo:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3)
```
</tf>
</frameworkcontent>

<Tip>

Para obter um exemplo mais aprofundado de como executar o fine-tuning de um modelo para classificação de tokens, dê uma olhada nesse [notebook utilizando PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb) ou nesse [notebook utilizando TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

</Tip>