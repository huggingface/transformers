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

# Relleno y truncamiento

Las entradas agrupadas por lotes (batched) suelen tener longitudes diferentes, por lo que no se pueden convertir en tensores de tamaño fijo. El relleno (también conocido como "Padding") y el truncamiento (conocido como "Truncation") son estrategias para abordar este problema y crear tensores rectangulares a partir de lotes de longitudes variables. El Padding agrega un **padding token** especial para garantizar que las secuencias más cortas tengan la misma longitud que la secuencia más larga en un lote o la longitud máxima aceptada por el modelo. Truncation funciona en la otra dirección al truncar secuencias largas.

En la mayoría de los casos, es bastante eficaz rellenar su lote hasta la longitud de la secuencia más larga y truncar hasta la longitud máxima que un modelo puede aceptar. Sin embargo, la API admite más estrategias si las necesitas. Los tres argumentos que necesitas son: `padding`, `truncation` y `max_length`.

El argumento `padding` controla el relleno. Puede ser un booleano o una cadena:

  - `True` o `'longest'`: rellena hasta la longitud de la secuencia más larga en el lote (no se aplica relleno si solo proporcionas una única secuencia).
  - `'max_length'`: rellena hasta una longitud especificada por el argumento `max_length` o la longitud máxima aceptada
    por el modelo si no se proporciona `max_length` (`max_length=None`). El Padding se aplicará incluso si solo proporcionas una única secuencia.
  - `False` o `'do_not_pad'`: no se aplica relleno. Este es el comportamiento predeterminado.

El argumento `truncation` controla el truncamiento. Puede ser un booleano o una cadena:

  - `True` o `'longest_first'`: trunca hasta una longitud máxima especificada por el argumento `max_length` o
    la longitud máxima aceptada por el modelo si no se proporciona `max_length` (`max_length=None`). Esto
    truncará token por token, eliminando un token de la secuencia más larga en el par hasta alcanzar la longitud adecuada.
  - `'only_second'`: trunca hasta una longitud máxima especificada por el argumento `max_length` o la longitud máxima
    aceptada por el modelo si no se proporciona `max_length` (`max_length=None`). Esto solo truncará
    la segunda oración de un par si se proporciona un par de secuencias (o un lote de pares de secuencias).
  - `'only_first'`: trunca hasta una longitud máxima especificada por el argumento `max_length` o la longitud máxima
    aceptada por el modelo si no se proporciona `max_length` (`max_length=None`). Esto solo truncará
    la primera oración de un par si se proporciona un par de secuencias (o un lote de pares de secuencias).
  - `False` o `'do_not_truncate'`: no se aplica truncamiento. Este es el comportamiento predeterminado.

The `max_length` argument controls the length of the padding and truncation. It can be an integer or `None`, in which case it will default to the maximum length the model can accept. If the model has no specific maximum input length, truncation or padding to `max_length` is deactivated.

The following table summarizes the recommended way to setup padding and truncation. If you use pairs of input sequences in any of the following examples, you can replace `truncation=True` by a `STRATEGY` selected in
`['only_first', 'only_second', 'longest_first']`, i.e. `truncation='only_second'` or `truncation='longest_first'` to control how both sequences in the pair are truncated as detailed before.

| Truncation                           | Padding                           | Instruction                                                                                 |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------|
| no truncation                        | no padding                        | `tokenizer(batch_sentences)`                                                           |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True)` or                                          |
|                                      |                                   | `tokenizer(batch_sentences, padding='longest')`                                        |
|                                      | padding to max model input length | `tokenizer(batch_sentences, padding='max_length')`                                     |
|                                      | padding to specific length        | `tokenizer(batch_sentences, padding='max_length', max_length=42)`                      |
|                                      | padding to a multiple of a value  | `tokenizer(batch_sentences, padding=True, pad_to_multiple_of=8)                        |
| truncation to max model input length | no padding                        | `tokenizer(batch_sentences, truncation=True)` or                                       |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY)`                                      |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True, truncation=True)` or                         |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY)`                        |
|                                      | padding to max model input length | `tokenizer(batch_sentences, padding='max_length', truncation=True)` or                 |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY)`                |
|                                      | padding to specific length        | Not possible                                                                                |
| truncation to specific length        | no padding                        | `tokenizer(batch_sentences, truncation=True, max_length=42)` or                        |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY, max_length=42)`                       |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` or          |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY, max_length=42)`         |
|                                      | padding to max model input length | Not possible                                                                                |
|                                      | padding to specific length        | `tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` or  |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY, max_length=42)` |
