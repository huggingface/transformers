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

# Usa los tokenizadores de 🤗 Tokenizers

[`PreTrainedTokenizerFast`] depende de la biblioteca [🤗 Tokenizers](https://huggingface.co/docs/tokenizers). Los tokenizadores obtenidos desde la biblioteca 🤗 Tokenizers pueden ser
cargados de forma muy sencilla en los 🤗 Transformers.

Antes de entrar en detalles, comencemos creando un tokenizador dummy en unas cuantas líneas:

```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> from tokenizers.pre_tokenizers import Whitespace

>>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
>>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

>>> tokenizer.pre_tokenizer = Whitespace()
>>> files = [...]
>>> tokenizer.train(files, trainer)
```

Ahora tenemos un tokenizador entrenado en los archivos que definimos. Lo podemos seguir utilizando en ese entorno de ejecución (runtime en inglés), o puedes guardarlo
en un archivo JSON para reutilizarlo en un futuro.

## Cargando directamente desde el objeto tokenizador

Veamos cómo utilizar este objeto tokenizador en la biblioteca 🤗 Transformers. La clase
[`PreTrainedTokenizerFast`] permite una instanciación fácil, al aceptar el objeto
*tokenizer* instanciado como argumento:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

Este objeto ya puede ser utilizado con todos los métodos compartidos por los tokenizadores de 🤗 Transformers! Visita la [página sobre tokenizadores
](main_classes/tokenizer) para más información.

## Cargando desde un archivo JSON

Para cargar un tokenizador desde un archivo JSON, comencemos por guardar nuestro tokenizador:

```python
>>> tokenizer.save("tokenizer.json")
```

La localización (path en inglés) donde este archivo es guardado puede ser incluida en el método de inicialización de [`PreTrainedTokenizerFast`]
utilizando el parámetro `tokenizer_file`:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

Este objeto ya puede ser utilizado con todos los métodos compartidos por los tokenizadores de 🤗 Transformers! Visita la [página sobre tokenizadores
](main_classes/tokenizer) para más información.
