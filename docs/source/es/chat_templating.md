<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# Plantillas para Modelos de Chat

## Introducción

Un caso de uso cada vez más común para LLMs es **el chat**. En un contexto de chat, en lugar de continuar una única cadena de texto (como es el caso con un modelo de lenguaje estándar), el modelo continúa una conversación que consta de uno o más **mensajes**, cada uno de los cuales incluye un **rol**, como "usuario" o "asistente", así como el texto del mensaje.
Al igual que con la tokenización, diferentes modelos esperan formatos de entrada muy diferentes para el chat. Esta es la razón por la que agregamos las plantillas de chat como una característica. Las plantillas de chat son parte del tokenizador. Especifican cómo convertir conversaciones, representadas como listas de mensajes, en una única cadena tokenizable en el formato que el modelo espera.
Vamos a hacer esto con un ejemplo concreto utilizando el modelo `BlenderBot`. BlenderBot tiene una plantilla predeterminada extremadamente simple, que principalmente solo agrega espacios en blanco entre rondas de diálogo:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...    {"role": "user", "content": "Hello, how are you?"},
...    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...    {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"

```
Observa cómo todo el chat se condensa en una sola cadena. Si usamos `tokenize=True`, que es la configuración predeterminada, esa cadena también será tokenizada para nosotros. Sin embargo, para ver una plantilla más compleja en acción, usemos el modelo `mistralai/Mistral-7B-Instruct-v0.1`

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

Ten en cuenta que esta vez, el tokenizador ha añadido los tokens de control [INST] y [/INST] para indicar el inicio y el final de los mensajes de usuario (¡pero no de los mensajes del asistente!). Mistral-instruct fue entrenado con estos tokens, pero BlenderBot no lo fue.

## ¿Cómo uso las plantillas de chat?

Como puedes ver en el ejemplo anterior, las plantillas de chat son fáciles de usar. Simplemente construye una lista de mensajes, con claves de `rol` y `contenido`, y luego pásala al método [`~PreTrainedTokenizer.apply_chat_template`]. Una vez que hagas eso, ¡obtendrás una salida lista para usar! Al utilizar plantillas de chat como entrada para la generación de modelos, también es una buena idea usar `add_generation_prompt=True` para agregar una [indicación de generación](#¿Qué-son-los-"generation-prompts"?).

Aquí tienes un ejemplo de cómo preparar la entrada para `model.generate()` utilizando el modelo de asistente `Zephyr`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```

Esto generará una cadena en el formato de entrada que Zephyr espera.

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
```

Ahora que nuestra entrada está formateada correctamente para Zephyr, podemos usar el modelo para generar una respuesta a la pregunta del usuario:

```python
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))

```
Esto producirá:

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

¡Arr, al final resultó ser fácil!

## ¿Existe un pipeline automatizado para chats?

Sí, lo hay! Nuestros canales de generación de texto admiten entradas de chat, cual facilita más facíl utilizar los modelos de chat. En el pasado, solíamos utilizar una clase dedicada "ConversationalPipeline", pero ahora ha quedado obsoleta y su funcionalidad se ha fusionado en [`TextGenerationPipeline`]. Este pipeline está diseñado para facilitar el uso de modelos de chat. Intentemos el ejemplo de `Zephyr` de nuevo, pero esta vez utilizando el pipeline:

```python
from transformers import pipeline

pipe = pipeline("conversational", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response
```

```text
{'role': 'assistant', 'content': "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."}
```


La canalización se encargará de todos los detalles de la tokenización y de llamar a `apply_chat_template` por ti. Una vez que el modelo tenga una plantilla de chat, ¡todo lo que necesitas hacer es inicializar el pipeline y pasarle la lista de mensajes!

# ¿Qué son los "generation prompts"?

Puede que hayas notado que el método `apply_chat_template` tiene un argumento `add_generation_prompt`. Este argumento indica a la plantilla que agregue tokens que indiquen el inicio de una respuesta del bot. Por ejemplo, considera el siguiente chat:

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

Así es cómo se verá esto sin un "generation prompt", usando la plantilla ChatML que vimos en el ejemplo de Zephyr:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

Y así es como se ve **con** un "generation prompt":

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
"""
```

Ten en cuenta que esta vez, hemos agregado los tokens que indican el inicio de una respuesta del bot. Esto asegura que cuando el modelo genere texto, escribirá una respuesta del bot en lugar de hacer algo inesperado, como continuar el mensaje del usuario. Recuerda, los modelos de chat siguen siendo solo modelos de lenguaje: están entrenados para continuar texto, ¡y el chat es solo un tipo especial de texto para ellos! Necesitas guiarlos con los tokens de control apropiados para que sepan lo que se supone que deben estar haciendo.

No todos los modelos requieren "generation prompts". Algunos modelos, como BlenderBot y LLaMA, no tienen ningún token especial antes de las respuestas del bot. En estos casos, el argumento `add_generation_prompt` no tendrá ningún efecto. El efecto exacto que tiene `add_generation_prompt` dependerá de la plantilla que se esté utilizando.

## ¿Puedo usar plantillas de chat en el entrenamiento?

¡Sí! Recomendamos que apliques la plantilla de chat como un paso de preprocesamiento para tu conjunto de datos. Después de esto, simplemente puedes continuar como cualquier otra tarea de entrenamiento de modelos de lenguaje. Durante el entrenamiento, generalmente deberías establecer `add_generation_prompt=False`, porque los tokens añadidos para solicitar una respuesta del asistente no serán útiles durante el entrenamiento. Veamos un ejemplo:

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```

Y obtenemos:

```text
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

Desde aquí, simplemente continúa el entrenamiento como lo harías con una tarea estándar de modelado de lenguaje, utilizando la columna `formatted_chat`.

## Avanzado: ¿Cómo funcionan las plantillas de chat?

La plantilla de chat para un modelo se almacena en el atributo `tokenizer.chat_template`. Si no se establece ninguna plantilla de chat, se utiliza en su lugar la plantilla predeterminada para esa clase de modelo. Echemos un vistazo a la plantilla para `BlenderBot`:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

¡Es un poco intimidante! Vamos a agregar algunas líneas nuevas y sangria para que sea más legible. Ten en cuenta que la primera línea nueva después de cada bloque, así como cualquier espacio en blanco anterior a un bloque, se ignoran de forma predeterminada, utilizando las banderas `trim_blocks` y `lstrip_blocks` de Jinja. Sin embargo, ¡ten cuidado! Aunque el espacio en blanco inicial en cada línea se elimina, los espacios entre bloques en la misma línea no. ¡Te recomendamos encarecidamente que verifiques que tu plantilla no esté imprimiendo espacios adicionales donde no debería estarlo!

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ ' ' }}
    {% endif %}
    {{ message['content'] }}
    {% if not loop.last %}
        {{ '  ' }}
    {% endif %}
{% endfor %}
{{ eos_token }}
```

Si nunca has visto uno de estos antes, esto es una [plantilla de Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/). Jinja es un lenguaje de plantillas que te permite escribir código simple que genera texto. En muchos aspectos, el código y la sintaxis se asemejan a Python. En Python puro, esta plantilla se vería algo así:

```python
for idx, message in enumerate(messages):
    if message['role'] == 'user':
        print(' ')
    print(message['content'])
    if not idx == len(messages) - 1:  # Check for the last message in the conversation
        print('  ')
print(eos_token)
```

Efectivamente, la plantilla hace tres cosas:
1. Para cada mensaje, si el mensaje es un mensaje de usuario, añade un espacio en blanco antes de él, de lo contrario no imprime nada.
2. Añade el contenido del mensaje.
3. Si el mensaje no es el último mensaje, añade dos espacios después de él. Después del último mensaje, imprime el token EOS.

Esta es una plantilla bastante simple: no añade ningún token de control y no admite mensajes "del sistema", que son una forma común de dar al modelo directivas sobre cómo debe comportarse en la conversación posterior. ¡Pero Jinja te brinda mucha flexibilidad para hacer esas cosas! Veamos una plantilla de Jinja que pueda formatear las entradas de manera similar a la forma en que LLaMA las formatea (nota que la plantilla real de LLaMA incluye el manejo de mensajes del sistema predeterminados y el manejo de mensajes del sistema ligeramente diferentes en general; ¡no uses esta en tu código real!)

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + message['content'] + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

Si observas esto por un momento, puedas ver lo que esta plantilla está haciendo: añade tokens específicos basados en el "rol" de cada mensaje, que representa quién lo envió. Los mensajes de usuario, asistente y sistema son claramente distinguibles para el modelo debido a los tokens en los que están envueltos.

## Avanzado: Añadiendo y editando plantillas de chat

### ¿Cómo creo una plantilla de chat?

Simple, solo escribe una plantilla de Jinja y establece `tokenizer.chat_template`. ¡Puede resultarte más fácil comenzar con una plantilla existente de otro modelo y simplemente editarla según tus necesidades! Por ejemplo, podríamos tomar la plantilla de LLaMA de arriba y añadir "[ASST]" y "[/ASST]" a los mensajes del asistente:

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {% endif %}
{% endfor %}
```

Ahora, simplemente establece el atributo `tokenizer.chat_template`. ¡La próxima vez que uses [`~PreTrainedTokenizer.apply_chat_template`], se utilizará tu nueva plantilla! Este atributo se guardará en el archivo tokenizer_config.json, por lo que puedes usar [`~utils.PushToHubMixin.push_to_hub`] para cargar tu nueva plantilla en el Hub y asegurarte de que todos estén utilizando la plantilla correcta para tu modelo.

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
tokenizer.push_to_hub("model_name")  # Upload your new template to the Hub!
```

El método [`~PreTrainedTokenizer.apply_chat_template`], que utiliza tu plantilla de chat, es llamado por la clase [`TextGenerationPipeline`], así que una vez que configures la plantilla de chat correcta, tu modelo se volverá automáticamente compatible con [`TextGenerationPipeline`].

<Tip>

Si estás ajustando finamente un modelo para chat, además de establecer una plantilla de chat, probablemente deberías agregar cualquier nuevo token de control de chat como los tokens especiales en el tokenizador. Los tokens especiales nunca se dividen, asegurando que tus tokens de control siempre se manejen como tokens únicos en lugar de ser tokenizados en piezas. También deberías establecer el atributo `eos_token` del tokenizador con el token que marca el final de las generaciones del asistente en tu plantilla. Esto asegurará que las herramientas de generación de texto puedan determinar correctamente cuándo detener la generación de texto.

</Tip>

### ¿Qué plantilla debería usar?

Cuando establezcas la plantilla para un modelo que ya ha sido entrenado para chat, debes asegurarte de que la plantilla coincida exactamente con el formato de mensajes que el modelo vio durante el entrenamiento, o de lo contrario es probable que experimentes degradación del rendimiento. Esto es cierto incluso si estás entrenando aún más el modelo; probablemente obtendrás el mejor rendimiento si mantienes constantes los tokens de chat. Esto es muy análogo a la tokenización: generalmente obtienes el mejor rendimiento para la inferencia o el ajuste fino cuando coincides precisamente con la tokenización utilizada durante el entrenamiento.

Si estás entrenando un modelo desde cero o ajustando finamente un modelo de lenguaje base para chat, por otro lado, ¡tienes mucha libertad para elegir una plantilla apropiada! Los LLM son lo suficientemente inteligentes como para aprender a manejar muchos formatos de entrada diferentes. Nuestra plantilla predeterminada para modelos que no tienen una plantilla específica de clase sigue el formato ChatML, y esta es una buena elección flexible para muchos casos de uso. Se ve así:

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```

Si te gusta esta plantilla, aquí está en forma de una sola línea, lista para copiar en tu código. La versión de una sola línea también incluye un práctico soporte para [prompts de generación](#¿Qué-son-los-"generation-prompts"?), ¡pero ten en cuenta que no añade tokens de BOS o EOS! Si tu modelo espera esos tokens, no se agregarán automáticamente por `apply_chat_template`, en otras palabras, el texto será tokenizado con `add_special_tokens=False`. Esto es para evitar posibles conflictos entre la plantilla y la lógica de `add_special_tokens`. ¡Si tu modelo espera tokens especiales, asegúrate de añadirlos a la plantilla!

```python
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

Esta plantilla envuelve cada mensaje en tokens `<|im_start|>` y `<|im_end|>`, y simplemente escribe el rol como una cadena, lo que permite flexibilidad en los roles con los que entrenas. La salida se ve así:

```text
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

Los roles "usuario", "sistema" y "asistente" son los estándar para chat, y recomendamos usarlos cuando tenga sentido, particularmente si deseas que tu modelo funcione bien con [`TextGenerationPipeline`]. Sin embargo, no estás limitado a estos roles: la plantilla es extremadamente flexible y cualquier cadena puede ser un rol.

### ¡Quiero añadir algunas plantillas de chat! ¿Cómo debo empezar?

Si tienes algún modelo de chat, debes establecer su atributo `tokenizer.chat_template` y probarlo usando [`~PreTrainedTokenizer.apply_chat_template`], luego subir el tokenizador actualizado al Hub. Esto se aplica incluso si no eres el propietario del modelo: si estás usando un modelo con una plantilla de chat vacía o que todavía está utilizando la plantilla predeterminada de clase, por favor abre una solicitud de extracción [pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) al repositorio del modelo para que este atributo se pueda establecer correctamente.

Una vez que se establece el atributo, ¡eso es todo, has terminado! `tokenizer.apply_chat_template` ahora funcionará correctamente para ese modelo, ¡lo que significa que también es compatible automáticamente en lugares como `TextGenerationPipeline`!

Al asegurarnos de que los modelos tengan este atributo, podemos garantizar que toda la comunidad pueda utilizar todo el poder de los modelos de código abierto. Los desajustes de formato han estado acechando el campo y dañando silenciosamente el rendimiento durante demasiado tiempo: ¡es hora de ponerles fin!

## Avanzado: Consejos para escribir plantillas

Si no estás familiarizado con Jinja, generalmente encontramos que la forma más fácil de escribir una plantilla de chat es primero escribir un script de Python corto que formatee los mensajes como desees, y luego convertir ese script en una plantilla.

Recuerda que el manejador de plantillas recibirá el historial de conversación como una variable llamada mensajes. Cada mensaje es un diccionario con dos claves, `role` y `content`. Podrás acceder a los `mensajes` en tu plantilla tal como lo harías en Python, lo que significa que puedes recorrerlo con `{% for message in messages %}` o acceder a mensajes individuales con, por ejemplo, `{{ messages[0] }}`.

También puedes usar los siguientes consejos para convertir tu código a Jinja:

### Bucles For 

Los bucles For en Jinja se ven así:

```
{% for message in messages %}
{{ message['content'] }}
{% endfor %}
```

Ten en cuenta que todo lo que esté dentro del {{bloque de expresión}} se imprimirá en la salida. Puedes usar operadores como `+` para combinar cadenas dentro de bloques de expresión.

### Declaraciones if

Las declaraciones if en Jinja se ven así:

```
{% if message['role'] == 'user' %}
{{ message['content'] }}
{% endif %}
```

Observa cómo donde Python utiliza espacios en blanco para marcar el inicio y el final de los bloques `for` e `if`, Jinja requiere que los termines explícitamente con `{% endfor %}` y `{% endif %}`.

### Variables especiales

Dentro de tu plantilla, tendrás acceso a la lista de `mensajes`, pero también puedes acceder a varias otras variables especiales. Estas incluyen tokens especiales como `bos_token` y `eos_token`, así como la variable `add_generation_prompt` que discutimos anteriormente. También puedes usar la variable `loop` para acceder a información sobre la iteración actual del bucle, por ejemplo, usando `{% if loop.last %}` para verificar si el mensaje actual es el último mensaje en la conversación. Aquí tienes un ejemplo que combina estas ideas para agregar un prompt de generación al final de la conversación si add_generation_prompt es `True`:

```
{% if loop.last and add_generation_prompt %}
{{ bos_token + 'Assistant:\n' }}
{% endif %}
```

### Notas sobre los espacios en blanco

Hemos intentado que Jinja ignore los espacios en blanco fuera de las {{expresiones}} tanto como sea posible. Sin embargo, ten en cuenta que Jinja es un motor de plantillas de propósito general y puede tratar el espacio en blanco entre bloques en la misma línea como significativo e imprimirlo en la salida. ¡Te recomendamos **encarecidamente** que verifiques que tu plantilla no esté imprimiendo espacios adicionales donde no debería antes de subirla!
