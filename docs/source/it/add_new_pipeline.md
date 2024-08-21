<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Come creare una pipeline personalizzata?

In questa guida, scopriremo come creare una pipeline personalizzata e condividerla sull' [Hub](https://hf.co/models) o aggiungerla nella libreria
Transformers.

Innanzitutto, è necessario decidere gli input grezzi che la pipeline sarà in grado di accettare. Possono essere strings, raw bytes,
dictionaries o qualsiasi cosa sia l'input desiderato più probabile. Cerca di mantenere questi input il più possibile in Python
in quanto facilita la compatibilità (anche con altri linguaggi tramite JSON). Questi saranno gli `inputs` della
pipeline (`preprocess`).

Poi definire gli `outputs`. Stessa strategia degli `inputs`. Più è seplice e meglio è. Questi saranno gli output del metodo
`postprocess`.

Si parte ereditando la classe base `Pipeline`. con i 4 metodi che bisogna implementare `preprocess`,
`_forward`, `postprocess` e `_sanitize_parameters`.


```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

La struttura di questa suddivisione consiste nel supportare in modo relativamente continuo CPU/GPU, supportando allo stesso tempo l'esecuzione di
pre/postelaborazione sulla CPU su thread diversi.

`preprocess` prenderà gli input originariamente definiti e li trasformerà in qualcosa di alimentabile dal modello. Potrebbe
contenere più informazioni e di solito è un `Dict`.

`_forward` è il dettaglio dell'implementazione e non è destinato a essere chiamato direttamente. `forward` è il metodo preferito per assicurarsi che tutto funzioni correttamente perchè contiene delle slavaguardie. Se qualcosa è
è collegato a un modello reale, appartiene al metodo `_forward`, tutto il resto è nel preprocess/postprocess.

`postprocess` prende l'otput di `_forward` e lo trasforma nell'output finale che era stato deciso in precedenza.

`_sanitize_parameters` esiste per consentire agli utenti di passare i parametri ogni volta che desiderano sia a inizialization time `pipeline(...., maybe_arg=4)` che al call time `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`.

`_sanitize_parameters` ritorna 3 dicts di kwargs che vengono passati direttamente a `preprocess`,
`_forward` e `postprocess`. Non riempire nulla se il chiamante non ha chiamato con alcun parametro aggiuntivo. Questo
consente di mantenere gli argomenti predefiniti nella definizione della funzione, che è sempre più "naturale".

Un esempio classico potrebbe essere l'argomento `top_k` nel post processing dei classification tasks.

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

In order to achieve that, we'll update our `postprocess` method with a default parameter to `5`. and edit
`_sanitize_parameters` to allow this new parameter.


```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

Cercare di mantenere gli input/output molto semplici e idealmente serializzabili in JSON, in quanto ciò rende l'uso della pipeline molto facile
senza richiedere agli utenti di comprendere nuovi tipi di oggetti. È anche relativamente comune supportare molti tipi di argomenti
per facilitarne l'uso (ad esempio file audio, possono essere nomi di file, URL o byte puri).

## Aggiungilo alla lista dei tasks supportati

Per registrar il tuo `new-task` alla lista dei tasks supportati, devi aggiungerlo al `PIPELINE_REGISTRY`:

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

Puoi specificare il modello di default che desideri, in questo caso dovrebbe essere accompagnato da una revisione specifica (che può essere il nome di un branch o l'hash di un commit, in questo caso abbiamo preso `"abcdef"`) e anche dal type:

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## Condividi la tua pipeline sull'Hub

Per condividere la tua pipeline personalizzata sull'Hub, devi solo salvare il codice della tua sottoclasse `Pipeline` in un file
python. Per esempio, supponiamo di voler utilizzare una pipeline personalizzata per la classificazione delle coppie di frasi come la seguente:

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

L'implementazione è agnostica al framework, e lavorerà sia con modelli PyTorch che con TensorFlow. Se l'abbiamo salvato in un file chiamato `pair_classification.py`, può essere successivamente importato e registrato in questo modo:

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

Una volta fatto, possiamo usarla con un modello pretrained. L'istanza `sgugger/finetuned-bert-mrpc` è stata
fine-tuned sul dataset MRPC, che classifica le coppie di frasi come parafrasi o no.

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

Successivamente possiamo condividerlo sull'Hub usando il metodo `push_to_hub`

```py
classifier.push_to_hub("test-dynamic-pipeline")
```

Questo codice copierà il file dove è stato definitp `PairClassificationPipeline` all'interno della cartella `"test-dynamic-pipeline"`,
insieme al salvataggio del modello e del tokenizer della pipeline, prima di pushare il tutto nel repository
`{your_username}/test-dynamic-pipeline`. Dopodiché chiunque potrà utilizzarlo, purché fornisca l'opzione
`trust_remote_code=True`:

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## Aggiungere la pipeline a Transformers

Se vuoi contribuire con la tua pipeline a Transformers, dovrai aggiungere un modulo nel sottomodulo `pipelines`
con il codice della tua pipeline, quindi aggiungilo all'elenco dei tasks definiti in `pipelines/__init__.py`.

Poi hai bisogno di aggiungere i test. Crea un nuovo file `tests/test_pipelines_MY_PIPELINE.py` con esempi ed altri test.

La funzione `run_pipeline_test` sarà molto generica e su piccoli modelli casuali su ogni possibile
architettura, come definito da `model_mapping` e `tf_model_mapping`.

Questo è molto importante per testare la compatibilità futura, nel senso che se qualcuno aggiunge un nuovo modello di
`XXXForQuestionAnswering` allora il test della pipeline tenterà di essere eseguito su di esso. Poiché i modelli sono casuali, è
è impossibile controllare i valori effettivi, per questo esiste un aiuto `ANY` che tenterà solamente di far corrispondere l'output della pipeline TYPE.

Hai anche *bisogno* di implementare 2 (idealmente 4) test.

- `test_small_model_pt` : Definire 1 piccolo modello per questa pipeline (non importa se i risultati non hanno senso)
  e testare i risultati della pipeline. I risultati dovrebbero essere gli stessi di `test_small_model_tf`.
- `test_small_model_tf` : Definire 1 piccolo modello per questa pipeline (non importa se i risultati non hanno senso)
  e testare i risultati della pipeline. I risultati dovrebbero essere gli stessi di `test_small_model_pt`.
- `test_large_model_pt` (`optional`): Testare la pipeline su una pipeline reale in cui i risultati dovrebbero avere
  senso. Questi test sono lenti e dovrebbero essere contrassegnati come tali. In questo caso l'obiettivo è mostrare la pipeline e assicurarsi che non ci siano  derive nelle versioni future
- `test_large_model_tf` (`optional`): Testare la pipeline su una pipeline reale in cui i risultati dovrebbero avere
  senso. Questi test sono lenti e dovrebbero essere contrassegnati come tali. In questo caso l'obiettivo è mostrare la pipeline e assicurarsi
  che non ci siano derive nelle versioni future