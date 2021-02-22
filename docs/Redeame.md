Generando la documentación
Para generar la documentación, primero debe compilarla. Se necesitan varios paquetes para construir el documento, puede instalarlos con el siguiente comando, en la raíz del repositorio de código:

pip install -e " . [docs] "
NOTA

Solo necesita generar la documentación para inspeccionarla localmente (si está planeando cambios y desea verificar cómo se ven antes de comprometerse, por ejemplo). No es necesario que confirme la documentación creada.

Paquetes instalados
Aquí hay una descripción general de todos los paquetes instalados. Si ejecutó el comando anterior instalando todos los paquetes desde requirements.txt, no necesita ejecutar los siguientes comandos.

Construirlo requiere el paquete sphinxque puede instalar usando:

instalación de pip -U sphinx
También necesitaría el tema personalizado instalado por Read The Docs . Puedes instalarlo usando el siguiente comando:

pip instalar sphinx_rtd_theme
El tercer paquete necesario es el recommonmarkpaquete para aceptar Markdown y texto reestructurado:

pip install recommonmark
Construyendo la documentación
Una vez que haya configurado sphinx, puede compilar la documentación ejecutando el siguiente comando en la /docscarpeta:

hacer html
Se _build/htmldebería haber creado una carpeta llamada . Ahora puede abrir el archivo _build/html/index.htmlen su navegador.

NOTA

Si está agregando / eliminando elementos del árbol toc o de cualquier elemento estructural, se recomienda limpiar el directorio de compilación antes de reconstruir. Ejecute el siguiente comando para limpiar y construir:

hacer limpio && hacer html
Debería crear la aplicación estática que estará disponible en /docs/_build/html

Agregar un nuevo elemento al árbol (toc-tree)
Los archivos aceptados son reStructuredText (.rst) y Markdown (.md). Cree un archivo con su extensión y colóquelo en el directorio de origen. Luego puede vincularlo al árbol toc poniendo el nombre del archivo sin la extensión.

Obtenga una vista previa de la documentación en una solicitud de extracción
Una vez que haya realizado su solicitud de extracción, puede verificar cómo se verá la documentación después de fusionarse siguiendo estos pasos:

Mire los cheques en la parte inferior de la página de conversación de su RP (es posible que deba hacer clic en "mostrar todos los cheques" para expandirlos).
Haga clic en "detalles" junto al ci/circleci: build_doccheque.
En la nueva ventana, haga clic en la pestaña "Artefactos".
Busque el archivo "docs / _build / html / index.html" (o cualquier página específica que desee consultar) y haga clic en él para obtener una vista previa.
Redacción de documentación: especificación
La huggingface/transformersdocumentación sigue el estilo de documentación de Google . Está escrito principalmente en ReStructuredText ( documentación simple de Sphinx , documentación completa de Sourceforge ).

Agregar un nuevo tutorial
La adición de un nuevo tutorial o sección se realiza en dos pasos:

Agregue un nuevo archivo debajo ./source. Este archivo puede ser ReStructuredText (.rst) o Markdown (.md).
Vincula ese archivo en ./source/index.rstel árbol toc correcto.
Asegúrese de poner su nuevo archivo en la sección adecuada. Es poco probable que vaya en la primera sección ( Comenzar ), por lo que dependiendo de los objetivos previstos (principiantes, usuarios más avanzados o investigadores) debería ir en la sección dos, tres o cuatro.

Agregar un nuevo modelo
Al agregar un nuevo modelo:

Cree un archivo xxx.rsten ./source/model_doc(no dude en copiar un archivo existente como plantilla).
Vincula ese archivo en ./source/index.rstel model_docárbol de toc.
Escriba una breve descripción general del modelo:
Resumen con artículos y autores
Resumen de papel
Consejos y trucos y cómo usarlo mejor
Agregue las clases que deberían estar vinculadas en el modelo. Esto generalmente incluye la configuración, el tokenizador y todos los modelos de esa clase (el modelo base, junto con los modelos con cabezales adicionales), tanto en PyTorch como en TensorFlow. El orden es generalmente:
Configuración,
Tokenizer
Modelo base de PyTorch
Modelos de cabezales PyTorch
Modelo base de TensorFlow
Modelos de cabezal TensorFlow
Estas clases deben agregarse utilizando la sintaxis RST. Por lo general, de la siguiente manera:

XXXConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XXXConfig
    :members:
Esto incluirá todos los métodos públicos de la configuración documentados. Si por alguna razón desea que un método no se muestre en la documentación, puede hacerlo especificando qué métodos deben estar en los documentos:

XXXTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XXXTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary

Escribir documentación fuente
Los valores que deben introducirse codedeben estar rodeados de comillas dobles: `` como tal '' o escribirse como un objeto usando la sintaxis: obj::: obj: `como tal`. Tenga en cuenta que los nombres de los argumentos y los objetos como Verdadero, Ninguno o cualquier cadena deben introducirse normalmente code.

Al mencionar una clase, se recomienda usar la sintaxis: class: ya que la clase mencionada será automáticamente vinculada por Sphinx:: class: `~ transformers.XXXClass`

Cuando se menciona una función, se recomienda utilizar la sintaxis: func: ya que la función mencionada será automáticamente vinculada por Sphinx:: func: `~ transformers.function`.

Al mencionar un método, se recomienda utilizar la sintaxis: meth: ya que el método mencionado será automáticamente vinculado por Sphinx:: meth: `~ transformers.XXXClass.method`.

Los enlaces deben hacerse como tal (tenga en cuenta el doble subrayado al final): `texto para el enlace <./ local-link-or-global-link # loc>` __

Definición de argumentos en un método
Los argumentos deben definirse con el Args:prefijo, seguido de un retorno de línea y una sangría. El argumento debe ir seguido de su tipo, con su forma si es un tensor y un retorno de línea. Es necesaria otra sangría antes de escribir la descripción del argumento.

Aquí hay un ejemplo que muestra todo hasta ahora:

    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.AlbertTokenizer`.
            See :meth:`~transformers.PreTrainedTokenizer.encode` and
            :meth:`~transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
Para argumentos opcionales o argumentos con valores predeterminados, seguimos la siguiente sintaxis: imagina que tenemos una función con la siguiente firma:

def my_function(x: str = None, a: float = 1):
entonces su documentación debería verse así:

    Args:
        x (:obj:`str`, `optional`):
            This argument controls ...
        a (:obj:`float`, `optional`, defaults to 1):
            This argument is used to ...
Tenga en cuenta que siempre omitimos "por defecto: obj:` None` "cuando None es el valor predeterminado para cualquier argumento. También tenga en cuenta que incluso si la primera línea que describe su tipo de argumento y su valor predeterminado se vuelve larga, no puede dividirla en varias líneas. Sin embargo, puede escribir tantas líneas como desee en la descripción con sangría (consulte el ejemplo anterior con input_ids).

Escribir un bloque de código de varias líneas
Los bloques de código de varias líneas pueden resultar útiles para mostrar ejemplos. Se hacen así:

Example::

    # first line of code
    # second line
    # etc
La Examplecadena al principio se puede reemplazar por cualquier cosa siempre que haya dos puntos y coma a continuación.

Seguimos la sintaxis de doctest para que los ejemplos prueben automáticamente que los resultados se mantienen consistentes con la biblioteca.

Escribir un bloque de retorno
Los argumentos deben definirse con el Args:prefijo, seguido de un retorno de línea y una sangría. La primera línea debe ser el tipo de retorno, seguida de un retorno de línea. No es necesario sangrar más los elementos que forman la declaración.

A continuación, se muestra un ejemplo de retorno de tupla, que comprende varios objetos:

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
A continuación, se muestra un ejemplo de una devolución de valor único:

    Returns:
        :obj:`List[int]`: A list of integers in the range [0, 1] --- 1 for a special token, 0 for a sequence token.
