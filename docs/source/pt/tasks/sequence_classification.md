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

# Classificação de texto

<Youtube id="leNG9fN9FQU"/>

A classificação de texto é uma tarefa comum de NLP que atribui um rótulo ou classe a um texto. Existem muitas aplicações práticas de classificação de texto amplamente utilizadas em produção por algumas das maiores empresas da atualidade. Uma das formas mais populares de classificação de texto é a análise de sentimento, que atribui um rótulo como positivo, negativo ou neutro a um texto.

Este guia mostrará como realizar o fine-tuning do [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) no conjunto de dados [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) para determinar se a crítica de filme é positiva ou negativa.

<Tip>

Consulte a [página de tarefas de classificação de texto](https://huggingface.co/tasks/text-classification) para obter mais informações sobre outras formas de classificação de texto e seus modelos, conjuntos de dados e métricas associados.

</Tip>

## Carregue o conjunto de dados IMDb

Carregue o conjunto de dados IMDb utilizando a biblioteca 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> imdb = load_dataset("stanfordnlp/imdb")
```

Em seguida, dê uma olhada em um exemplo:

```py
>>> imdb["test"][0]
{
    "label": 0,
    "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
}
```

Existem dois campos neste dataset:

- `text`: uma string contendo o texto da crítica do filme.
- `label`: um valor que pode ser `0` para uma crítica negativa ou `1` para uma crítica positiva.

## Pré-processamento dos dados

Carregue o tokenizador do DistilBERT para processar o campo `text`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

Crie uma função de pré-processamento para tokenizar o campo `text` e truncar as sequências para que não sejam maiores que o comprimento máximo de entrada do DistilBERT:

```py
>>> def preprocess_function(examples):
...     return tokenizer(examples["text"], truncation=True)
```

Use a função [`map`](https://huggingface.co/docs/datasets/process#map) do 🤗 Datasets para aplicar a função de pré-processamento em todo o conjunto de dados. Você pode acelerar a função `map` definindo `batched=True` para processar vários elementos do conjunto de dados de uma só vez:

```py
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

Use o [`DataCollatorWithPadding`] para criar um batch de exemplos. Ele também *preencherá dinamicamente* seu texto até o comprimento do elemento mais longo em seu batch, para que os exemplos do batch tenham um comprimento uniforme. Embora seja possível preencher seu texto com a função `tokenizer` definindo `padding=True`, o preenchimento dinâmico utilizando um data collator é mais eficiente.

```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## Train

Carregue o DistilBERT com [`AutoModelForSequenceClassification`] junto com o número de rótulos esperados:

```py
>>> from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

>>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)
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
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=5,
...     weight_decay=0.01,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_imdb["train"],
...     eval_dataset=tokenized_imdb["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

<Tip>

O [`Trainer`] aplicará o preenchimento dinâmico por padrão quando você definir o argumento `tokenizer` dele. Nesse caso, você não precisa especificar um data collator explicitamente.

</Tip>

<Tip>

Para obter um exemplo mais aprofundado de como executar o fine-tuning de um modelo para classificação de texto, dê uma olhada nesse [notebook utilizando PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) ou nesse [notebook utilizando TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

</Tip>
