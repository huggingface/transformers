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

# Generación de resúmenes

<Youtube id="yHnr5Dk2zCI"/>

La generación de resúmenes (summarization, en inglés) crea una versión más corta de un documento o un artículo que resume toda su información importante. Junto con la traducción, es un ejemplo de una tarea que puede ser formulada como una tarea secuencia a secuencia. La generación de resúmenes puede ser:

- Extractiva: Extrae la información más relevante de un documento.
- Abstractiva: Genera un texto nuevo que captura la información más importante.

Esta guía te mostrará cómo puedes hacer fine-tuning del modelo [T5](https://huggingface.co/google-t5/t5-small) sobre el subset de proyectos de ley del estado de California, dentro del dataset [BillSum](https://huggingface.co/datasets/billsum) para hacer generación de resúmenes abstractiva.

<Tip>

Consulta la [página de la tarea](https://huggingface.co/tasks/summarization) de generación de resúmenes para obtener más información sobre sus modelos, datasets y métricas asociadas.

</Tip>

## Carga el dataset BillSum

Carga el dataset BillSum de la biblioteca 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

Divide el dataset en un set de train y un set de test:

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

A continuación, observa un ejemplo:

```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

El campo `text` es el input y el campo `summary` es el objetivo.

## Preprocesa

Carga el tokenizador T5 para procesar `text` y `summary`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
```

La función de preprocesamiento necesita:

1. Agregar un prefijo al input; una clave para que T5 sepa que se trata de una tarea de generación de resúmenes. Algunos modelos capaces de realizar múltiples tareas de NLP requieren una clave que indique la tarea específica.
2. Usar el argumento `text_target` para tokenizar etiquetas.
3. Truncar secuencias para que no sean más largas que la longitud máxima fijada por el parámetro `max_length`.

```py
>>> prefix = "summarize: "


>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

Usa la función [`~datasets.Dataset.map`] de 🤗 Datasets para aplicar la función de preprocesamiento sobre el dataset en su totalidad. Puedes acelerar la función `map` configurando el argumento `batched=True` para procesar múltiples elementos del dataset a la vez:

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

Usa [`DataCollatorForSeq2Seq`] para crear un lote de ejemplos. Esto también *rellenará dinámicamente* tu texto y etiquetas a la dimensión del elemento más largo del lote para que tengan un largo uniforme. Si bien es posible rellenar tu texto en la función `tokenizer` mediante el argumento `padding=True`, el rellenado dinámico es más eficiente.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
```
</tf>
</frameworkcontent>

## Entrenamiento

<frameworkcontent>
<pt>
Carga T5 con [`AutoModelForSeq2SeqLM`]:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
```

<Tip>

Para familiarizarte con el proceso para realizar fine-tuning sobre un modelo con [`Trainer`], ¡mira el tutorial básico [aquí](../training#finetune-with-trainer)!

</Tip>

En este punto, solo faltan tres pasos:

1. Definir tus hiperparámetros de entrenamiento en [`Seq2SeqTrainingArguments`].
2. Pasarle los argumentos de entrenamiento a [`Seq2SeqTrainer`] junto con el modelo, dataset y data collator.
3. Llamar [`~Trainer.train`] para realizar el fine-tuning sobre tu modelo.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="./results",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=1,
...     fp16=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```
</pt>
<tf>
Para hacer fine-tuning de un modelo en TensorFlow, comienza por convertir tus datasets al formato `tf.data.Dataset` con [`~datasets.Dataset.to_tf_dataset`]. Especifica los inputs y etiquetas en `columns`, el tamaño de lote, el data collator, y si es necesario mezclar el dataset:

```py
>>> tf_train_set = tokenized_billsum["train"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = tokenized_billsum["test"].to_tf_dataset(
...     columns=["attention_mask", "input_ids", "labels"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

<Tip>

Para familiarizarte con el fine-tuning con Keras, ¡mira el tutorial básico [aquí](training#finetune-with-keras)!

</Tip>

Crea la función optimizadora, establece la tasa de aprendizaje y algunos hiperparámetros de entrenamiento:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

Carga T5 con [`TFAutoModelForSeq2SeqLM`]:

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
```

Configura el modelo para entrenamiento con [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> model.compile(optimizer=optimizer)
```

Llama a [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) para realizar el fine-tuning del modelo:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3)
```
</tf>
</frameworkcontent>

<Tip>

Para un ejemplo con mayor profundidad de cómo hacer fine-tuning a un modelo para generación de resúmenes, revisa la
[notebook en PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
o a la [notebook en TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb).

</Tip>
