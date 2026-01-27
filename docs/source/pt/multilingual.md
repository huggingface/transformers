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

# Modelos multilinguísticos para inferência

[[open-in-colab]]

Existem vários modelos multilinguísticos no 🤗 Transformers e seus usos para inferência diferem dos modelos monolíngues.
No entanto, nem *todos* os usos dos modelos multilíngues são tão diferentes.
Alguns modelos, como o [google-bert/bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased),
podem ser usados como se fossem monolíngues. Este guia irá te ajudar a usar modelos multilíngues cujo uso difere
para o propósito de inferência.

## XLM

O XLM tem dez checkpoints diferentes dos quais apenas um é monolíngue.
Os nove checkpoints restantes do modelo são subdivididos em duas categorias:
checkpoints que usam de language embeddings e os que não.

### XLM com language embeddings

Os seguintes modelos de XLM usam language embeddings para especificar a linguagem utilizada para a inferência.

- `FacebookAI/xlm-mlm-ende-1024` (Masked language modeling, English-German)
- `FacebookAI/xlm-mlm-enfr-1024` (Masked language modeling, English-French)
- `FacebookAI/xlm-mlm-enro-1024` (Masked language modeling, English-Romanian)
- `FacebookAI/xlm-mlm-xnli15-1024` (Masked language modeling, XNLI languages)
- `FacebookAI/xlm-mlm-tlm-xnli15-1024` (Masked language modeling + translation, XNLI languages)
- `FacebookAI/xlm-clm-enfr-1024` (Causal language modeling, English-French)
- `FacebookAI/xlm-clm-ende-1024` (Causal language modeling, English-German)

Os language embeddings são representados por um tensor de mesma dimensão que os `input_ids` passados ao modelo.
Os valores destes tensores dependem do idioma utilizado e se identificam pelos atributos `lang2id` e `id2lang` do tokenizador.

Neste exemplo, carregamos o checkpoint `FacebookAI/xlm-clm-enfr-1024`(Causal language modeling, English-French):

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
```

O atributo `lang2id` do tokenizador mostra os idiomas deste modelo e seus ids:

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

Em seguida, cria-se um input de exemplo:

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1
```

Estabelece-se o id do idioma, por exemplo `"en"`, e utiliza-se o mesmo para definir a language embedding.
A language embedding é um tensor preenchido com `0`, que é o id de idioma para o inglês.
Este tensor deve ser do mesmo tamanho que os `input_ids`.

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # We reshape it to be of size (batch_size, sequence_length)
>>> langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)
```

Agora você pode passar os `input_ids` e a language embedding ao modelo:

```py
>>> outputs = model(input_ids, langs=langs)
```

O script [run_generation.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation/run_generation.py) pode gerar um texto com language embeddings utilizando os checkpoints `xlm-clm`.

### XLM sem language embeddings

Os seguintes modelos XLM não requerem o uso de language embeddings durante a inferência:

- `FacebookAI/xlm-mlm-17-1280` (Modelagem de linguagem com máscara, 17 idiomas)
- `FacebookAI/xlm-mlm-100-1280` (Modelagem de linguagem com máscara, 100 idiomas)

Estes modelos são utilizados para representações genéricas de frase diferentemente dos checkpoints XLM anteriores.

## BERT

Os seguintes modelos do BERT podem ser utilizados para tarefas multilinguísticas:

- `google-bert/bert-base-multilingual-uncased` (Modelagem de linguagem com máscara + Previsão de frases, 102 idiomas)
- `google-bert/bert-base-multilingual-cased` (Modelagem de linguagem com máscara + Previsão de frases, 104 idiomas)

Estes modelos não requerem language embeddings durante a inferência. Devem identificar a linguagem a partir
do contexto e realizar a inferência em sequência.

## XLM-RoBERTa

Os seguintes modelos do XLM-RoBERTa podem ser utilizados para tarefas multilinguísticas:

- `FacebookAI/xlm-roberta-base` (Modelagem de linguagem com máscara, 100 idiomas)
- `FacebookAI/xlm-roberta-large` Modelagem de linguagem com máscara, 100 idiomas)

O XLM-RoBERTa foi treinado com 2,5 TB de dados do CommonCrawl recém-criados e testados em 100 idiomas.
Proporciona fortes vantagens sobre os modelos multilinguísticos publicados anteriormente como o mBERT e o XLM em tarefas
subsequentes como a classificação, a rotulagem de sequências e à respostas a perguntas.

## M2M100

Os seguintes modelos de M2M100 podem ser utilizados para traduções multilinguísticas:

- `facebook/m2m100_418M` (Tradução)
- `facebook/m2m100_1.2B` (Tradução)

Neste exemplo, o checkpoint `facebook/m2m100_418M` é carregado para traduzir do mandarim ao inglês. É possível
estabelecer o idioma de origem no tokenizador:

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

Tokenização do texto:

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

O M2M100 força o id do idioma de destino como o primeiro token gerado para traduzir ao idioma de destino.
É definido o `forced_bos_token_id` como `en` no método `generate` para traduzir ao inglês.

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

Os seguintes modelos do MBart podem ser utilizados para tradução multilinguística:

- `facebook/mbart-large-50-one-to-many-mmt` (Tradução automática multilinguística de um a vários, 50 idiomas)
- `facebook/mbart-large-50-many-to-many-mmt` (Tradução automática multilinguística de vários a vários, 50 idiomas)
- `facebook/mbart-large-50-many-to-one-mmt` (Tradução automática multilinguística vários a um, 50 idiomas)
- `facebook/mbart-large-50` (Tradução multilinguística, 50 idiomas)
- `facebook/mbart-large-cc25`

Neste exemplo, carrega-se o checkpoint `facebook/mbart-large-50-many-to-many-mmt` para traduzir do finlandês ao inglês.
Pode-se definir o idioma de origem no tokenizador:

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

Tokenizando o texto:

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

O MBart força o id do idioma de destino como o primeiro token gerado para traduzir ao idioma de destino.
É definido o `forced_bos_token_id` como `en` no método `generate` para traduzir ao inglês.

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id("en_XX"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

Se estiver usando o checkpoint `facebook/mbart-large-50-many-to-one-mmt` não será necessário forçar o id do idioma de destino
como sendo o primeiro token generado, caso contrário a usagem é a mesma.
