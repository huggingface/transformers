<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Mecanismos de atención

La mayoría de los modelos transformers utilizan atención completa, en el sentido de que la matriz de atención es cuadrada. Esto puede ser un gran cuello de botella computacional cuando tienes textos largos. `Longformer` y `reformer` son modelos que intentan ser más eficientes y utilizan una versión dispersa de la matriz de atención para acelerar el entrenamiento.

## Atención LSH

[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer) utiliza atención LSH. En el softmax(QK^t), solo los elementos más grandes (en la dimensión softmax) de la matriz QK^t van a dar contribuciones útiles. Entonces, para cada consulta q en Q, podemos considerar solo las claves k en K que estén cerca de q. Se utiliza una función hash para determinar si q y k están cerca. La máscara de atención se modifica para enmascarar el token actual (excepto en la primera posición), porque dará una consulta y una clave iguales (entonces muy similares entre sí). Dado que el hash puede ser un poco aleatorio, en la práctica se utilizan varias funciones hash (determinadas por un parámetro n_rounds) y luego se promedian juntas.

## Atención local

[Longformer](https://huggingface.co/docs/transformers/model_doc/longformer) utiliza atención local: a menudo, el contexto local (por ejemplo, ¿cuáles son los dos tokens a la izquierda y a la derecha?) es suficiente para tomar acción para un token dado. Además, apilando capas de atención que tienen una ventana pequeña, la última capa tendrá un campo receptivo mayor que solamente los tokens en la ventana, lo que les permite construir una representación de toda la oración.

Algunos tokens de entrada preseleccionados también reciben atención global: para esos pocos tokens, la matriz de atención puede acceder a todos los tokens y este proceso es simétrico: todos los demás tokens tienen acceso a esos tokens específicos (además de los que están en su ventana local). Esto se muestra en la Figura 2d del artículo, el cual se puede apreciar un ejemplo de una máscara de atención:

<div class="flex justify-center">
    <img scale="50 %" align="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local_attention_mask.png"/>
</div>

El uso de dichas matrices de atención con menos parámetros permite que el modelo tenga entradas con una longitud de secuencia mayor.

## Otros trucos

### Codificación posicional axial

[Reformer](https://huggingface.co/docs/transformers/model_doc/reformer) utiliza codificación posicional axial: en los modelos transformers tradicionales, la codificación posicional E es una matriz de tamaño \\(l\\) por \\(d\\), donde \\(l\\) es la longitud de la secuencia y \\(d\\) es la dimensión del estado oculto. Si tienes textos muy extensos, esta matriz puede ser enorme y ocupar demasiado espacio en la GPU. Para aliviar eso, las codificaciones posicionales axiales consisten en factorizar esa gran matriz E en dos matrices más pequeñas E1 y E2, con dimensiones \\(l_{1} \times d_{1}\\) y \\(l_{2} \times d_{2}\\), tal que \\(l_{1} \times l_{2} = l\\) y \\(d_{1} + d_{2} = d\\) (con el producto de las longitudes, esto termina siendo mucho más pequeño). La incrustación (embedding) para el paso de tiempo \\(j\\) en E se obtiene concatenando las incrustaciones para el paso de tiempo \\(j \% l1\\) en E1 y \\(j // l1\\) en E2.
