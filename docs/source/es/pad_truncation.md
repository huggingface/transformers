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

Las entradas agrupadas por lotes (batched) suelen tener longitudes diferentes, por lo que no se pueden convertir en tensores de tamaño fijo. El relleno (también conocido como "Padding") y el truncamiento (conocido como "Truncation") son estrategias para abordar este problema y crear tensores rectangulares a partir de lotes de longitudes variables. El relleno agrega un **padding token** especial para garantizar que las secuencias más cortas tengan la misma longitud que la secuencia más larga en un lote o la longitud máxima aceptada por el modelo. El truncamiento funciona en la otra dirección al truncar secuencias largas.

En la mayoría de los casos, es bastante eficaz rellenar el lote hasta la longitud de la secuencia más larga y truncar hasta la longitud máxima que un modelo puede aceptar. Sin embargo, la API admite más estrategias si las necesitas. Los tres argumentos que necesitas son: `padding`, `truncation` y `max_length`.

El argumento `padding` controla el relleno. Puede ser un booleano o una cadena:

  - `True` o `'longest'`: rellena hasta la longitud de la secuencia más larga en el lote (no se aplica relleno si solo proporcionas una única secuencia).
  - `'max_length'`: rellena hasta una longitud especificada por el argumento `max_length` o la longitud máxima aceptada
    por el modelo si no se proporciona `max_length` (`max_length=None`). El relleno se aplicará incluso si solo proporcionas una única secuencia.
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

El argumento `max_length` controla la longitud del relleno y del truncamiento. Puede ser un número entero o `None`, en cuyo caso se establecerá automáticamente en la longitud máxima que el modelo puede aceptar. Si el modelo no tiene una longitud máxima de entrada específica, se desactiva el truncamiento o el relleno hasta `max_length`.

La siguiente tabla resume la forma recomendada de configurar el relleno y el truncamiento. Si usas pares de secuencias de entrada en alguno de los siguientes ejemplos, puedes reemplazar `truncation=True` por una `ESTRATEGIA` seleccionada en
`['only_first', 'only_second', 'longest_first']`, es decir, `truncation='only_second'` o `truncation='longest_first'` para controlar cómo se truncan ambas secuencias en el par, como se detalló anteriormente.

| Truncation                           | Padding                             | Instrucción                                                                                 |
|-----------------------------------------|--------------------------------------|---------------------------------------------------------------------------------------------|
| sin truncamiento                       | sin relleno                          | `tokenizer(batch_sentences)`                                                           |
|                                         | relleno hasta la longitud máxima del lote | `tokenizer(batch_sentences, padding=True)` o                                          |
|                                         |                                      | `tokenizer(batch_sentences, padding='longest')`                                        |
|                                         | relleno hasta la longitud máxima del modelo | `tokenizer(batch_sentences, padding='max_length')`                                     |
|                                         | relleno hasta una longitud específica | `tokenizer(batch_sentences, padding='max_length', max_length=42)`                      |
|                                         | relleno hasta un múltiplo de un valor | `tokenizer(batch_sentences, padding=True, pad_to_multiple_of=8)`                        |
| truncamiento hasta la longitud máxima del modelo | sin relleno                          | `tokenizer(batch_sentences, truncation=True)` o                                       |
|                                         |                                      | `tokenizer(batch_sentences, truncation=ESTRATEGIA)`                                      |
|                                         | relleno hasta la longitud máxima del lote | `tokenizer(batch_sentences, padding=True, truncation=True)` o                         |
|                                         |                                      | `tokenizer(batch_sentences, padding=True, truncation=ESTRATEGIA)`                        |
|                                         | relleno hasta la longitud máxima del modelo | `tokenizer(batch_sentences, padding='max_length', truncation=True)` o                 |
|                                         |                                      | `tokenizer(batch_sentences, padding='max_length', truncation=ESTRATEGIA)`                |
|                                         | relleno hasta una longitud específica | No es posible                                                                                |
| truncamiento hasta una longitud específica | sin relleno                          | `tokenizer(batch_sentences, truncation=True, max_length=42)` o                        |
|                                         |                                      | `tokenizer(batch_sentences, truncation=ESTRATEGIA, max_length=42)`                       |
|                                         | relleno hasta la longitud máxima del lote | `tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` o          |
|                                         |                                      | `tokenizer(batch_sentences, padding=True, truncation=ESTRATEGIA, max_length=42)`         |
|                                         | relleno hasta la longitud máxima del modelo | No es posible                                                                                |
|                                         | relleno hasta una longitud específica | `tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` o  |
|                                         |                                      | `tokenizer(batch_sentences, padding='max_length', truncation=ESTRATEGIA, max_length=42)` |
