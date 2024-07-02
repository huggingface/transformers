<!--Copyright 2023 The HuggingFace Team. All rights reserved.
Licensed under la Licencia Apache, Version 2.0 (la "Licencia"); no puede usar este archivo excepto en cumplimiento con
la Licencia. Puede obtener una copia de la Licencia en
http://www.apache.org/licenses/LICENSE-2.0
A menos que la ley aplicable lo requiera o se acuerde por escrito, el software distribuido bajo la Licencia se distribuye
"COMO ESTA", SIN GARANTIAS NI CONDICIONES DE NINGUN TIPO, ya sean expresas o implicitas. Consulte la Licencia para conocer
el lenguaje especifico que rige los permisos y limitaciones bajo la Licencia.
⚠️ Note que este archivo esta en Markdown pero contiene sintaxis especifica para nuestro generador de documentacion (similar a MDX) que puede no ser 
renderizada correctamente en tu visor de Markdown.
-->

# Cargar adaptadores con 🤗 PEFT

[[open-in-colab]]

Los metodos de afinamiento eficiente en parametros (PEFT, por sus siglas en ingles) congelan los parametros del modelo preentrenado durante el afinamiento y agregan un pequeno numero de parametros entrenables (los adaptadores) encima. Los adaptadores estan entrenados para aprender informacion especifica de una tarea. Este enfoque ha demostrado ser muy eficiente en memoria, con un menor uso de recursos computacionales, al mismo tiempo que produce resultados comparables a los de un modelo completamente ajustado.

Los adaptadores entrenados con PEFT tambien suelen ser un orden de magnitud mas pequenos que el modelo completo, lo que facilita compartirlos, almacenarlos y cargarlos.

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">Los pesos del adaptador para un modelo OPTForCausalLM almacenado en el Hub son solo de ~6MB en comparacion con el tamano completo de los pesos del modelo, que puede ser de ~700MB.</figcaption>
</div>

Si esta interesado en aprender mas sobre la biblioteca de 🤗 PEFT, consulte la documentacion.

## Configuración

Vamos a empezar instalando 🤗 PEFT:

```bash
pip install peft
```

Si quiere probar las nuevas caracteristicas, es posible que le interese instalar la biblioteca desde la fuente:

```bash
pip install git+https://github.com/huggingface/peft.git
```
## Modelos PEFT admitidos 

🤗 Transformers admite nativamente algunos metodos de PEFT.  De esta manera puedes cargar los pesos del adaptador almacenados localmente o almacenados en el Hub y facilmente ejecutar o entrenar los pesos con unas cuantas lineas de codigo.  Se cuenta con soporte para los siguientes metodos:


- [Low Rank Adapters](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [AdaLoRA](https://arxiv.org/abs/2303.10512)

Si desea utilizar otros metodos PEFT, como el aprendizaje de indicaciones (prompt learning) o el ajuste de indicaciones (prompt tuning), o aprender mas sobre la biblioteca de 🤗 PEFT en general, consulte la [documentación](https://huggingface.co/docs/peft/index).

## Cargar un adaptador PEFT


Para cargar y utilizar un modelo adaptador PEFT desde 🤗 Transformers, asegurese de que el repositorio en el Hub o el directorio local contenga un archivo `adapter_config.json` y los pesos del adaptador, como se muestra en la imagen del ejemplo anterior. Luego, puede cargar el modelo adaptador PEFT usando la clase `AutoModelFor`. Por ejemplo, para cargar un modelo adaptador PEFT para modelado de lenguaje causal:

1. Especifique el ID del modelo PEFT
2. Paselo a la clase [`AutoModelForCausalLM`]

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```

<Tip>

Puede cargar un adaptador PEFT con una clase AutoModelFor o con la clase del modelo base, como OPTForCausalLM o LlamaForCausalLM.

</Tip>

Tambien puede cargar un adaptador PEFT llamando al metodo `load_adapter`:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

Consulte la seccion de documentacion de API (#transformers.integrations.PeftAdapterMixin) a continuacion para mas detalles.

##Cargar en 8 bits o 4 bits

La integracion `bitsandbytes` soporta tipos de datos de precisión de 8 bits y 4 bits, los cuales son utiles para cargar modelos grandes porque ahorran memoria (consulta la [guia de integracion bitsandbytes](./quantization#bitsandbytes-integration) para aprender mas). Anada los parametros `load_in_8bit` o `load_in_4bit` a [`~PreTrainedModel.from_pretrained`] y establezca `device_map="auto"` para distribuir efectivamente el modelo en su hardware:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

## Agregar un nuevo adaptador

Puede usar [`~peft.PeftModel.add_adapter`] para agregar un nuevo adaptador a un modelo con un adaptador existente siempre y cuando el nuevo adaptador sea del mismo tipo que el actual. Por ejemplo, si tiene un adaptador LoRA existente conectado a un modelo:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import LoraConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```

Agregar un nuevo adaptador: 

```py
# attach new adapter with same config
model.add_adapter(lora_config, adapter_name="adapter_2")
```

Ahora puede usar [`~peft.PeftModel.set_adapter`] para configurar cual adaptador usar:

```py
# use adapter_1
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))

# use adapter_2
model.set_adapter("adapter_2")
output_enabled = model.generate(**inputs)
print(tokenizer.decode(output_enabled[0], skip_special_tokens=True))
```

## Activación y desactivación de los adaptadores 

Una vez que haya agregado un adaptador a un modelo, puede activar o desactivar el modulo del adaptador. Para activar el modulo:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)

# to initiate with random weights
peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_adapters()
output = model.generate(**inputs)
```

Para desactivar el modulo del adaptador: 

```py
model.disable_adapters()
output = model.generate(**inputs)
```

## Como entrenar un adaptador PEFT

Los adaptadores PEFT son compatibles con la clase [`Trainer`] para que puedas entrenar el adaptador para tu caso de uso específico. Solo requiere agregar unas cuantas lineas mas de codigo.  Por ejemplo, para entrenar un adaptador de LoRA:  

<Tip>

Si no esta familiarizado con el afinamiento de un modelo con [`Trainer`], revise el tutorial [Afinamiento de un modelo preentrenado] (training)

</Tip>

1. Defina la configuracion de su adaptador con el tipo de tarea y los hiperparametros (consulte [`~peft.LoraConfig`] para obtener mas detalles sobre lo que hacen los hiperparametros).

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

2. Agregue una daptador al modelo.

```py
model.add_adapter(peft_config)
```

3. ¡Ahora puede pasar el modelo a [`Trainer`]!

```py
trainer = Trainer(model=model, ...)
trainer.train()
```

Para guardar su adaptador entrenado y volver a cargarlo: 

```py
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```

## Agregue capas entrenables adicionales a un adaptador PEFT

Tambien puede ajustar adaptadores entranbles adicionales sobre un modelo que tiene adaptadores conectados pasando `modules_to_save` en su configuracion PEFT. Por ejemplo, si tambien quieres ajustar el `lm_head` encima de un  modelo adaptadpr LoRA:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import LoraConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    modules_to_save=["lm_head"],
)

model.add_adapter(lora_config)
```

## Documentacion API

[[autodoc]] integrations.PeftAdapterMixin
    - load_adapter
    - add_adapter
    - set_adapter
    - disable_adapters
    - enable_adapters
    - active_adapters
    - get_adapter_state_dict




<!--
TODO: (@younesbelkada @stevhliu)
-   Link to PEFT docs for further details
-   Trainer  
-   8-bit / 4-bit examples ?
-->
