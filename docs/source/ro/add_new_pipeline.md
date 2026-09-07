<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Adăugarea unui nou pipeline

Fă-ți propriul [`Pipeline`] creând o subclasă a acestuia și implementând câteva metode. Partajează codul cu comunitatea pe [Hub](https://hf.co) și înregistrează pipeline-ul în Transformers astfel încât toată lumea să îl poată folosi rapid și ușor.

Acest ghid te va ghida prin procesul de adăugare a unui nou pipeline în Transformers.

## Alegeri de design

La minimum, trebuie doar să oferi [`Pipeline`] un input potrivit pentru un task. Acesta este și punctul de la care ar trebui să începi atunci când îți proiectezi pipeline-ul.

Decide ce tipuri de input poate accepta [`Pipeline`]. Pot fi string-uri, raw bytes, dicționare și așa mai departe. Încearcă să păstrezi input-urile în Python pur acolo unde este posibil deoarece este mai compatibil. Apoi, decide ce output ar trebui să returneze [`Pipeline`]. Din nou, păstrarea output-ului în Python este opțiunea cea mai simplă și cea mai bună deoarece este mai ușor de lucrat cu el.

Păstrarea input-urilor și output-urilor simple și, ideal, serializabile în JSON, le face mai ușor utilizatorilor să ruleze [`Pipeline`]-ul tău fără a fi nevoie să învețe tipuri noi de obiecte. De asemenea, este obișnuit să suporți multe tipuri diferite de input pentru o ușurință de utilizare și mai mare. De exemplu, faptul că un fișier audio poate fi acceptat dintr-un nume de fișier, un URL sau raw bytes îi oferă utilizatorului mai multă flexibilitate în modul în care furnizează datele audio.

## Crearea unui pipeline

După ce ai decis un input și un output, poți începe să implementezi [`Pipeline`]. Pipeline-ul tău ar trebui să moștenească din clasa de bază [`Pipeline`] și să includă 4 metode.

```py
from transformers import Pipeline

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):

    def preprocess(self, inputs, args=2):

    def _forward(self, model_inputs):

    def postprocess(self, model_outputs):
```

1. `preprocess` preia input-urile și le transformă în formatul de input potrivit pentru model.

```py
def preprocess(self, inputs, maybe_arg=2):
    model_input = Tensor(inputs["input_ids"])
    return {"model_input": model_input}
```

2. `_forward` nu ar trebui apelată direct. `forward` este metoda preferată deoarece include mecanisme de protecție pentru a se asigura că totul funcționează corect pe device-ul așteptat. Orice este legat de model aparține în `_forward`, iar tot restul aparține fie în `preprocess`, fie în `postprocess`.

```py
def _forward(self, model_inputs):
    outputs = self.model(**model_inputs)
    return outputs
```

3. `postprocess` generează output-ul final din output-ul modelului din `_forward`.

```py
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    return best_class
```

4. `_sanitize_parameters` le permite utilizatorilor să transmită parametri suplimentari către [`Pipeline`]. Acest lucru poate avea loc în timpul inițializării sau atunci când [`Pipeline`] este apelat. `_sanitize_parameters` returnează 3 dicts de argumente keyword suplimentare care sunt transmise direct către `preprocess`, `_forward` și `postprocess`. Nu adăuga nimic dacă un utilizator nu a apelat pipeline-ul cu parametri suplimentari. Acest lucru păstrează argumentele implicite în definiția funcției, ceea ce este întotdeauna mai natural.

De exemplu, adaugă un parametru `top_k` în `postprocess` pentru a returna primele 5 cele mai probabile clase. Apoi, în `_sanitize_parameters`, verifică dacă utilizatorul a transmis `top_k` și adaugă-l la `postprocess_kwargs`.

```py
def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

Acum pipeline-ul poate returna cele mai probabile etichete dacă un utilizator alege acest lucru.

```py
from transformers import pipeline

pipeline = pipeline("my-task")
# returnează cele mai probabile 3 etichete
pipeline("This is the best meal I've ever had", top_k=3)
# returnează implicit cele mai probabile 5 etichete
pipeline("This is the best meal I've ever had")
```

## Înregistrarea unui pipeline

Înregistrează noul task pe care îl suportă pipeline-ul tău în `PIPELINE_REGISTRY`. Registry-ul definește:

- Clasa de model Pytorch suportată cu `pt_model`
- un model implicit care ar trebui să provină dintr-o anumită revizie (branch sau commit hash) unde modelul funcționează conform așteptărilor, cu `default`
- input-ul așteptat cu `type`

```py
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome-model", "branch-name")},
    type="text",
)
```

## Partajarea pipeline-ului tău

Partajează pipeline-ul tău cu comunitatea pe [Hub](https://hf.co) sau îl poți adăuga direct în Transformers.

Este mai rapid să încarci codul pipeline-ului tău pe Hub deoarece nu necesită o revizuire din partea echipei Transformers. Adăugarea pipeline-ului în Transformers poate fi mai lentă deoarece necesită o revizuire și trebuie să adaugi teste pentru a te asigura că [`Pipeline`]-ul tău funcționează.

### Încărcarea pe Hub

Adaugă codul pipeline-ului tău pe Hub într-un fișier Python.

De exemplu, un pipeline personalizat pentru clasificarea perechilor de propoziții ar putea arăta ca în codul de mai jos.

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

Salvează codul într-un fișier numit `pair_classification.py` și importă-l și înregistrează-l așa cum se arată mai jos.

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

Funcția [register_pipeline](https://github.com/huggingface/transformers/blob/9feae5fb0164e89d4998e5776897c16f7330d3df/src/transformers/pipelines/base.py#L1387) înregistrează detaliile pipeline-ului (tipul de task, clasa de pipeline, backend-urile suportate) în fișierul `config.json` al unui model.

```json
  "custom_pipelines": {
    "pair-classification": {
      "impl": "pair_classification.PairClassificationPipeline",
      "pt": [
        "AutoModelForSequenceClassification"
      ],
    }
  },
```

Apelează [`~Pipeline.push_to_hub`] pentru a face push pipeline-ului către Hub. Fișierul Python care conține codul este copiat pe Hub, iar modelul și tokenizer-ul pipeline-ului sunt, de asemenea, salvate și trimise (push) către Hub. Pipeline-ul tău ar trebui să fie acum disponibil pe Hub în namespace-ul tău.

```py
from transformers import pipeline

pipeline = pipeline(task="pair-classification", model="sgugger/finetuned-bert-mrpc")
pipeline.push_to_hub("pair-classification-pipeline")
```

Pentru a folosi pipeline-ul, adaugă `trust_remote_code=True` atunci când încarci pipeline-ul.

```py
from transformers import pipeline

pipeline = pipeline(task="pair-classification", trust_remote_code=True)
```

### Adăugarea în Transformers

Adăugarea unui pipeline personalizat în Transformers necesită adăugarea de teste pentru a te asigura că totul funcționează conform așteptărilor și solicitarea unei revizuiri din partea echipei Transformers.

Adaugă codul pipeline-ului tău ca un nou modul în submodulul [pipelines](https://github.com/huggingface/transformers/tree/main/src/transformers/pipelines) și adaugă-l la lista de task-uri definite în [pipelines/__init__.py](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py).

Apoi, adaugă un nou test pentru pipeline în [transformers/tests/pipelines](https://github.com/huggingface/transformers/tree/main/tests/pipelines). Poți să te uiți la celelalte teste pentru exemple despre cum să-ți testezi pipeline-ul.

Funcția [run_pipeline_test](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L186) ar trebui să fie foarte generică și să ruleze pe modelele definite în [model_mapping](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L48). Acest lucru este important pentru testarea compatibilității viitoare cu modele noi.

Vei observa, de asemenea, că `ANY` este folosit pe tot parcursul funcției [run_pipeline_test](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L186). Modelele sunt aleatorii, așa că nu poți verifica valorile efective. Folosirea lui `ANY` permite în schimb testului să se potrivească cu output-ul de tipul pipeline-ului.

În cele din urmă, ar trebui să implementezi și următoarele 4 teste.

1. [test_small_model_pt](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L59), folosește un model mic pentru aceste pipeline-uri pentru a te asigura că returnează output-urile corecte. Rezultatele nu trebuie să aibă sens. Fiecare pipeline ar trebui să returneze același rezultat.
1. [test_large_model_pt](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_zero_shot_image_classification.py#L187), folosește un model realist pentru aceste pipeline-uri pentru a te asigura că returnează rezultate semnificative. Aceste teste sunt lente și ar trebui marcate ca fiind lente (slow).
