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

# Modelos multilingües para inferencia

[[open-in-colab]]

Existen varios modelos multilingües en 🤗 Transformers y su uso para inferencia difiere de los modelos monolingües. Sin embargo, no *todos* los usos de los modelos multilingües son diferentes. Algunos modelos, como [google-bert/bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased), pueden utilizarse igual que un modelo monolingüe. Esta guía te enseñará cómo utilizar modelos multilingües cuyo uso difiere en la inferencia.

## XLM

XLM tiene diez checkpoints diferentes de los cuales solo uno es monolingüe. Los nueve checkpoints restantes del modelo pueden dividirse en dos categorías: los checkpoints que utilizan language embeddings y los que no.

### XLM con language embeddings

Los siguientes modelos XLM usan language embeddings para especificar el lenguaje utilizado en la inferencia:

- `FacebookAI/xlm-mlm-ende-1024` (Masked language modeling, English-German)
- `FacebookAI/xlm-mlm-enfr-1024` (Masked language modeling, English-French)
- `FacebookAI/xlm-mlm-enro-1024` (Masked language modeling, English-Romanian)
- `FacebookAI/xlm-mlm-xnli15-1024` (Masked language modeling, XNLI languages)
- `FacebookAI/xlm-mlm-tlm-xnli15-1024` (Masked language modeling + translation, XNLI languages)
- `FacebookAI/xlm-clm-enfr-1024` (Causal language modeling, English-French)
- `FacebookAI/xlm-clm-ende-1024` (Causal language modeling, English-German)

Los language embeddings son representados como un tensor de la mismas dimensiones que los `input_ids` pasados al modelo. Los valores de estos tensores dependen del idioma utilizado y se identifican mediante los atributos `lang2id` y `id2lang` del tokenizador.

En este ejemplo, carga el checkpoint `FacebookAI/xlm-clm-enfr-1024` (Causal language modeling, English-French):

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("FacebookAI/xlm-clm-enfr-1024")
```

El atributo `lang2id` del tokenizador muestra los idiomas de este modelo y sus ids:

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

A continuación, crea un input de ejemplo:

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1
```

Establece el id del idioma, por ejemplo `"en"`, y utilízalo para definir el language embedding. El language embedding es un tensor lleno de `0` ya que es el id del idioma para inglés. Este tensor debe ser del mismo tamaño que `input_ids`. 

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # We reshape it to be of size (batch_size, sequence_length)
>>> langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)
```

Ahora puedes pasar los `input_ids` y el language embedding al modelo:

```py
>>> outputs = model(input_ids, langs=langs)
```

El script [run_generation.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation/run_generation.py) puede generar texto con language embeddings utilizando los checkpoints `xlm-clm`.

### XLM sin language embeddings

Los siguientes modelos XLM no requieren language embeddings durante la inferencia:

- `FacebookAI/xlm-mlm-17-1280` (modelado de lenguaje enmascarado, 17 idiomas)
- `FacebookAI/xlm-mlm-100-1280` (modelado de lenguaje enmascarado, 100 idiomas)

Estos modelos se utilizan para representaciones genéricas de frases a diferencia de los anteriores checkpoints XLM.

## BERT

Los siguientes modelos de BERT pueden utilizarse para tareas multilingües:

- `google-bert/bert-base-multilingual-uncased` (modelado de lenguaje enmascarado + predicción de la siguiente oración, 102 idiomas)
- `google-bert/bert-base-multilingual-cased` (modelado de lenguaje enmascarado + predicción de la siguiente oración, 104 idiomas)

Estos modelos no requieren language embeddings durante la inferencia. Deben identificar la lengua a partir del
contexto e inferir en consecuencia.

## XLM-RoBERTa

Los siguientes modelos de XLM-RoBERTa pueden utilizarse para tareas multilingües:

- `FacebookAI/xlm-roberta-base` (modelado de lenguaje enmascarado, 100 idiomas)
- `FacebookAI/xlm-roberta-large` (Modelado de lenguaje enmascarado, 100 idiomas)

XLM-RoBERTa se entrenó con 2,5 TB de datos CommonCrawl recién creados y depurados en 100 idiomas. Proporciona fuertes ventajas sobre los modelos multilingües publicados anteriormente como mBERT o XLM en tareas posteriores como la clasificación, el etiquetado de secuencias y la respuesta a preguntas.

## M2M100

Los siguientes modelos de M2M100 pueden utilizarse para traducción multilingüe:

- `facebook/m2m100_418M` (traducción)
- `facebook/m2m100_1.2B` (traducción)

En este ejemplo, carga el checkpoint `facebook/m2m100_418M` para traducir del chino al inglés. Puedes establecer el idioma de origen en el tokenizador:

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

Tokeniza el texto:

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

M2M100 fuerza el id del idioma de destino como el primer token generado para traducir al idioma de destino.. Establece el `forced_bos_token_id` a `en` en el método `generate` para traducir al inglés:

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

Los siguientes modelos de MBart pueden utilizarse para traducción multilingüe:

- `facebook/mbart-large-50-one-to-many-mmt` (traducción automática multilingüe de uno a muchos, 50 idiomas)
- `facebook/mbart-large-50-many-to-many-mmt` (traducción automática multilingüe de muchos a muchos, 50 idiomas)
- `facebook/mbart-large-50-many-to-one-mmt` (traducción automática multilingüe muchos a uno, 50 idiomas)
- `facebook/mbart-large-50` (traducción multilingüe, 50 idiomas)
- `facebook/mbart-large-cc25`

En este ejemplo, carga el checkpoint `facebook/mbart-large-50-many-to-many-mmt` para traducir del finlandés al inglés. Puedes establecer el idioma de origen en el tokenizador:

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

Tokeniza el texto:

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

MBart fuerza el id del idioma de destino como el primer token generado para traducirlo. Establece el `forced_bos_token_id` a `en` en el método `generate` para traducir al inglés:

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id("en_XX"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

Si estás usando el checkpoint `facebook/mbart-large-50-many-to-one-mmt` no necesitas forzar el id del idioma de destino como el primer token generado, de lo contrario el uso es el mismo.
