<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Cómo funciona el caché

Imagina que estás conversando con alguien y que, en lugar de recordar lo que dijo anteriormente, esa persona tuviera que empezar desde cero cada vez que respondes. Esto sería lento e ineficiente, ¿verdad?

Puedes extender esta analogía a los modelos transformers. La generación de modelos autorregresivos puede ser lenta porque realiza una predicción de un token a la vez. Cada nueva predicción depende de todo el contexto previo.

Para predecir el token número 1000, el modelo necesita la información de los 999 tokens anteriores. Esta información se representa como multiplicaciones de matrices sobre las representaciones de los tokens.

Para predecir el token número 1001, necesitas la misma información de los 999 tokens anteriores, además de cualquier información del token número 1000. ¡Son muchas multiplicaciones de matrices que el modelo tiene que calcular una y otra vez para cada token!

Un caché de pares clave-valor (KV, por sus siglas en inglés) elimina esta ineficiencia almacenando los pares KV derivados de las capas de atención de los tokens procesados previamente. Los pares KV almacenados se recuperan del caché y se reutilizan para los tokens posteriores, evitando la necesidad de volver a calcularlos.

> [!WARNING]
> El caché solo debe usarse para **inferencia**. Puede provocar errores inesperados si se habilita durante el entrenamiento.

Para entender mejor cómo y por qué funciona el caché, veamos más de cerca la estructura de las matrices de atención.

## Matrices de atención

La **atención de producto escalar escalada** (scaled dot-product attention) se calcula como se muestra a continuación para un batch de tamaño `b`, un número de cabezas de atención `h`, una longitud de secuencia hasta el momento `T`, y una dimensión por cabeza de atención `d_head`.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_{\text{head}}}} \times \text{mask} \right) V
$$

Las matrices de consulta (`Q`), clave (`K`) y valor (`V`) son proyecciones de los embeddings de entrada con forma `(b, h, T, d_head)`.

Para la atención causal, la máscara evita que el modelo atienda a tokens futuros. Una vez que un token es procesado, su representación nunca cambia con respecto a los tokens futuros, lo que significa que $ K_{\text{past}} $ y $ V_{\text{past}} $ pueden almacenarse en el caché y reutilizarse para calcular la representación del último token.

$$
\text{Attention}(q_t, [\underbrace{k_1, k_2, \dots, k_{t-1}}_{\text{cached}}, k_{t}], [\underbrace{v_1, v_2, \dots, v_{t-1}}_{\text{cached}}, v_{t}])
$$

En tiempo de inferencia, solo necesitas la consulta del último token para calcular la representación $ x_t $ que predice el siguiente token $ t+1 $. En cada paso, los nuevos vectores de clave y valor se **almacenan** en el caché y se **añaden** a las claves y valores pasados.

$$
K_{\text{cache}} \leftarrow \text{concat}(K_{\text{past}}, k_t), \quad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{past}}, v_t)
$$

La atención se calcula de forma independiente en cada capa del modelo, y el caché se gestiona por capa.

Consulta la siguiente tabla para comparar cómo el caché mejora la eficiencia.

| sin caché | con caché |
|---|---|
| en cada paso, se recalculan todas las `K` y `V` anteriores | en cada paso, solo se calculan la `K` y la `V` actuales |
| el costo de atención por paso es **cuadrático** con la longitud de secuencia | el costo de atención por paso es **lineal** con la longitud de secuencia (la memoria crece linealmente, pero el cómputo por token se mantiene bajo) |

## Clase Cache

Una interfaz básica de caché KV toma un tensor de clave y un tensor de valor para el token actual y devuelve los tensores `K` y `V` actualizados. Esto lo gestiona internamente el método `forward` del modelo.

```py
new_K, new_V = cache.update(k_t, v_t, layer_idx)
attn_output = attn_layer_idx_fn(q_t, new_K, new_V)
```

Cuando usas la clase [`Cache`] de Transformers, el módulo de autoatención realiza varios pasos críticos para integrar la información pasada y la presente.

1. El módulo de atención concatena los pares KV actuales con los pares KV pasados almacenados en el caché. Esto crea pesos de atención con forma `(new_tokens_length, past_kv_length + new_tokens_length)`. Los pares KV actuales y pasados se combinan esencialmente para calcular las puntuaciones de atención, asegurando que el modelo tenga en cuenta tanto el contexto previo como la entrada actual.

2. Cuando el método `forward` se llama de forma iterativa, es crucial que la forma de la máscara de atención coincida con la longitud combinada de los pares KV pasados y actuales. La máscara de atención debe tener la forma `(batch_size, past_kv_length + new_tokens_length)`. Esto suele gestionarse internamente en [`~GenerationMixin.generate`], pero si quieres implementar tu propio bucle de generación con [`Cache`], ¡tenlo en cuenta! La máscara de atención debe contener los valores de los tokens pasados y actuales.

## Implementación del almacenamiento del caché

Los cachés se estructuran como una lista de capas, donde cada capa contiene un caché de claves y un caché de valores. Los cachés de claves y valores son tensores con la forma `[batch_size, num_heads, seq_len, head_dim]`.

Las capas pueden ser de distintos tipos (por ejemplo, `DynamicLayer`, `StaticLayer`, `StaticSlidingWindowLayer`), lo que cambia principalmente cómo se gestiona la longitud de secuencia y cómo se actualiza el caché.

El más simple es `DynamicLayer`, que crece a medida que se procesan más tokens. La dimensión de longitud de secuencia (`seq_len`) aumenta con cada token nuevo:

```py
cache.layers[idx].keys = torch.cat([cache.layers[idx].keys, key_states], dim=-2)
cache.layers[idx].values = torch.cat([cache.layers[idx].values, value_states], dim=-2)
```

Otros tipos de capa, como `StaticLayer` y `StaticSlidingWindowLayer`, tienen una longitud de secuencia fija que se establece cuando se crea el caché. Esto los hace compatibles con `torch.compile`. En el caso de `StaticSlidingWindowLayer`, los tokens existentes se desplazan fuera del caché cuando se añade un token nuevo.

El siguiente ejemplo demuestra cómo crear un bucle de generación con [`DynamicCache`]. Como se ha comentado, la máscara de atención es una concatenación de los valores de los tokens pasados y actuales.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from accelerate import Accelerator

device = Accelerator().device

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

past_key_values = DynamicCache(config=model.config)
messages = [{"role": "user", "content": "Hello, what's your name."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

generated_ids = inputs.input_ids
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    # Muestrea codiciosamente el siguiente token
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # Prepara las entradas para el siguiente paso de generación dejando los tokens sin procesar; en nuestro caso
    # solo tenemos un token nuevo, y expandimos la máscara de atención para el nuevo token, como se explicó arriba
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
```
