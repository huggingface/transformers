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

# Conceptos básicos de chat

Los modelos de chat son modelos conversacionales a los que puedes enviar un mensaje y recibir una respuesta. La mayoría de los modelos de lenguaje desde mediados de 2023 en adelante son modelos de chat y pueden denominarse modelos "instruct" o "instruction-tuned". Los modelos que no soportan chat suelen llamarse modelos "base" o "preentrenados".

Los modelos más grandes y recientes son generalmente más capaces, pero los modelos especializados en ciertos dominios (texto médico, legal, idiomas distintos al inglés, etc.) a menudo pueden superar a estos modelos más grandes. Prueba tablas de clasificación como [OpenLLM](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard) y [LMSys Chatbot Arena](https://chat.lmsys.org/?leaderboard) para ayudarte a identificar el mejor modelo para tu caso de uso.

Esta guía te muestra cómo cargar rápidamente modelos de chat en Transformers desde la línea de comandos, cómo construir y formatear una conversación, y cómo chatear usando el [`TextGenerationPipeline`].

## CLI de chat

Después de [instalar Transformers](./installation), puedes chatear con un modelo directamente desde la línea de comandos. El siguiente comando inicia una sesión interactiva con un modelo, con algunos comandos base listados al inicio de la sesión.

> Para los siguientes comandos, asegúrate de que [`transformers serve` esté ejecutándose](https://huggingface.co/docs/transformers/main/en/serving).

```bash
transformers chat Qwen/Qwen2.5-0.5B-Instruct
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers-chat-cli.png"/>
</div>

Puedes iniciar el CLI con flags arbitrarios de `generate`, con el formato `arg_1=valor_1 arg_2=valor_2 ...`

```bash
transformers chat Qwen/Qwen2.5-0.5B-Instruct do_sample=False max_new_tokens=10
```

Para una lista completa de opciones, ejecuta el siguiente comando.

```bash
transformers chat -h
```

El chat está implementado sobre [AutoClass](./model_doc/auto), usando herramientas de [generación de texto](./llm_tutorial) y [chat](./chat_templating). Usa el CLI `transformers serve` internamente ([docs](./serve-cli/serving)).

## TextGenerationPipeline

[`TextGenerationPipeline`] es una clase de alto nivel para generación de texto con un "modo chat". El modo chat se activa cuando se detecta un modelo conversacional y el prompt de chat está [correctamente formateado](./llm_tutorial#wrong-prompt-format).

Los modelos de chat aceptan una lista de mensajes (el historial de chat) como entrada. Cada mensaje es un diccionario con las claves `role` y `content`.
Para iniciar el chat, agrega un solo mensaje `user`. Opcionalmente puedes incluir un mensaje `system` para dar al modelo instrucciones sobre cómo comportarse.

```py
chat = [
    {"role": "system", "content": "Eres un asistente científico útil."},
    {"role": "user", "content": "Oye, ¿puedes explicarme la gravedad?"}
]
```

Crea el [`TextGenerationPipeline`] y pásale `chat`. Para modelos grandes, configurar [device_map="auto"](./models#big-model-inference) ayuda a cargar el modelo más rápido y lo coloca automáticamente en el dispositivo más rápido disponible.

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="auto", device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Si esto funciona correctamente, ¡deberías ver una respuesta del modelo! Si quieres continuar la conversación, necesitas actualizar el historial de chat con la respuesta del modelo. Puedes hacerlo agregando el texto a `chat` (usando el rol `assistant`), o leyendo `response[0]["generated_text"]`, que contiene el historial completo del chat, incluyendo la respuesta más reciente.

Una vez que tienes la respuesta del modelo, puedes continuar la conversación agregando un nuevo mensaje `user` al historial de chat.

```py
chat = response[0]["generated_text"]
chat.append(
    {"role": "user", "content": "¡Wow! ¿Pero se puede reconciliar con la mecánica cuántica?"}
)
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Repitiendo este proceso, puedes continuar la conversación tanto como quieras, al menos hasta que el modelo se quede sin ventana de contexto o te quedes sin memoria.

## Rendimiento y uso de memoria

Transformers carga los modelos en precisión completa `float32` por defecto, y para un modelo de 8B, ¡esto requiere ~32GB de memoria! Usa el argumento `torch_dtype="auto"`, que generalmente usa `bfloat16` para modelos que fueron entrenados con él, para reducir tu uso de memoria.

> [!TIP]
> Consulta la documentación de [Cuantización](./quantization/overview) para más información sobre los diferentes backends de cuantización disponibles.

Para reducir el uso de memoria aún más, puedes cuantizar el modelo a 8 bits o 4 bits con [bitsandbytes](https://hf.co/docs/bitsandbytes/index). Crea un [`BitsAndBytesConfig`] con la configuración de cuantización deseada y pásalo al parámetro `model_kwargs` del pipeline. El siguiente ejemplo cuantiza un modelo a 8 bits.

```py
from transformers import pipeline, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", model_kwargs={"quantization_config": quantization_config})
```

En general, el tamaño del modelo y el rendimiento están directamente correlacionados. Los modelos más grandes son más lentos además de requerir más memoria porque cada parámetro activo debe leerse de la memoria por cada token generado.
Este es un cuello de botella para la generación de texto con LLMs y las principales opciones para mejorar la velocidad de generación son cuantizar el modelo o usar hardware con mayor ancho de banda de memoria. Agregar más poder de cómputo no ayuda significativamente.

También puedes probar técnicas como la [decodificación especulativa](./generation_strategies#speculative-decoding), donde un modelo más pequeño genera tokens candidatos que son verificados por el modelo más grande. Si los tokens candidatos son correctos, el modelo más grande puede generar más de un token a la vez. Esto alivia significativamente el cuello de botella de ancho de banda y mejora la velocidad de generación.

> [!TIP]
Los modelos Mixture-of-Expert (MoE) como [Mixtral](./model_doc/mixtral), [Qwen2MoE](./model_doc/qwen2_moe) y [GPT-OSS](./model_doc/gpt-oss) tienen muchos parámetros, pero solo "activan" una pequeña fracción de ellos para generar cada token. Como resultado, los modelos MoE generalmente tienen requisitos de ancho de banda de memoria mucho menores y pueden ser más rápidos que un LLM regular del mismo tamaño. Sin embargo, técnicas como la decodificación especulativa son inefectivas con modelos MoE porque más parámetros se activan con cada nuevo token especulado.
