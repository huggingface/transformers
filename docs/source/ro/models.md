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

# Încărcarea modelelor

Transformers oferă multe modele pre-antrenate gata de utilizare cu o singură linie de cod. Acestea necesită o clasă de model și metoda [`~PreTrainedModel.from_pretrained`].

Apelează [`~PreTrainedModel.from_pretrained`] pentru a descărca și încărca weights și configurația unui model stocate pe [Hub-ul Hugging Face](https://hf.co/models).

> [!TIP]
> Metoda [`~PreTrainedModel.from_pretrained`] încarcă weights stocate în formatul de fișier [safetensors](https://hf.co/docs/safetensors/index) dacă sunt disponibile. În mod tradițional, weights modelelor PyTorch sunt serializate cu utilitarul [pickle](https://docs.python.org/3/library/pickle.html) care este cunoscut ca nesecurizat. Fișierele safetensors sunt mai sigure și mai rapid de încărcat.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
```

Acest ghid explică cum sunt încărcate modelele, diferitele moduri de a încărca un model, cum să depășești problemele de memorie pentru modele foarte mari și cum să încarci modele personalizate.

## Modele și configurații

Toate modelele au un fișier `configuration.py` cu atribute specifice precum numărul de hidden layers, dimensiunea vocabularului, funcția de activare și altele. Vei găsi și un fișier `modeling.py` care definește layers și operațiile matematice din interiorul fiecărui strat. Fișierul `modeling.py` preia atributele modelului din `configuration.py` și construiește modelul corespunzător. În acest moment, ai un model cu weights aleatorii care trebuie antrenat pentru a produce rezultate semnificative.

<!-- insert diagram of model and configuration -->

> [!TIP]
> O *arhitectură* se referă la scheletul modelului, iar un *checkpoint* se referă la weights pentru o anumită arhitectură. De exemplu, [BERT] este o arhitectură, în timp ce [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) este un checkpoint. Vei vedea termenul *model* utilizat interschimbabil cu arhitectura și checkpoint-ul.

Există două tipuri generale de modele pe care le poți încărca:

1. Un model de bază, precum [`AutoModel`] sau [`LlamaModel`], care returnează hidden states.
2. Un model cu un *head* specific atașat, precum [`AutoModelForCausalLM`] sau [`LlamaForCausalLM`], pentru efectuarea unor task-uri specifice.

## Clasele de model

Pentru a obține un model pre-antrenat, trebuie să încarci weights în model. Acest lucru se face apelând [`~PreTrainedModel.from_pretrained`] care acceptă weights de pe Hub-ul Hugging Face sau dintr-un folder local.

Există două clase de model: clasa [AutoModel] și o clasă specifică modelului.

<hfoptions id="model-classes">
<hfoption id="AutoModel">

<Youtube id="AhChOFRegn4"/>

Clasa [AutoModel] este o modalitate convenabilă de a încărca o arhitectură fără a fi nevoie să cunoști numele exact al clasei de model, deoarece există multe modele disponibile. Selectează automat clasa de model corectă pe baza fișierului de configurație. Trebuie să știi doar task-ul și checkpoint-ul pe care vrei să le utilizezi.

Comută ușor între modele sau task-uri, atât timp cât arhitectura este suportată pentru un task dat.

De exemplu, același model poate fi utilizat pentru task-uri separate.

```py
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

# utilizează același API pentru 3 task-uri diferite
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForQuestionAnswering.from_pretrained("meta-llama/Llama-2-7b-hf")
```

În alte cazuri, ai putea testa rapid mai multe modele diferite pentru un task.

```py
from transformers import AutoModelForCausalLM

# utilizează același API pentru 3 modele diferite
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
```

</hfoption>
<hfoption id="model-specific class">

Clasa [AutoModel] este construită pe baza claselor specifice modelului. Toate clasele de model care suportă un task specific sunt mapate la clasa de task `AutoModelFor` corespunzătoare.

Dacă știi deja ce clasă de model vrei să utilizezi, poți folosi direct clasa specifică.

```py
from transformers import LlamaModel, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
```

</hfoption>
</hfoptions>

## Modele mari

Modelele mari pre-antrenate necesită multă memorie pentru a fi încărcate. Procesul de încărcare implică:

1. crearea unui model cu weights aleatoare
2. încărcarea weights pre-antrenate
3. plasarea weights pre-antrenate pe model

Ai nevoie de suficientă memorie pentru a reține două copii ale weights modelului (aleatoare și pre-antrenate), ceea ce poate să nu fie posibil în funcție de hardware-ul tău. În medii de antrenare distribuită, aceasta este și mai dificilă deoarece fiecare proces încarcă un model pre-antrenat.

Transformers reduce unele dintre aceste provocări legate de memorie prin inițializare rapidă, checkpoint-uri sharded, funcția [Big Model Inference](https://hf.co/docs/accelerate/usage_guides/big_modeling) din Accelerate și suportul pentru tipuri de date cu mai puțini biți.

### Checkpoint-uri sharded

[`~PreTrainedModel.save_pretrained`] shardează automat checkpoint-urile mai mari de 50GB. Aceasta menține numărul de shards redus pentru modelele mari și simplifică gestionarea fișierelor.

Parametrii se încarcă în paralel și vârful de memorie depinde doar de dimensiunea modelului. Utilizează `max_shard_size` în [`~PreTrainedModel.save_pretrained`] pentru a seta dimensiunea maximă a checkpoint-ului înainte de sharding.

> [!NOTE]
> Utilizarea memoriei pentru modelele care necesită conversie dinamică de weights depinde de dimensiunea modelului și de dimensiunea celor mai mari parametri dintr-o singură conversie. Aceasta se aplică de obicei modelelor mixture-of-experts (MoE) unde utilizarea memoriei este dimensiunea modelului plus numărul de experți de pe un strat. Consultă ghidul [încărcarea dinamică a modelelor](./weightconverter#încărcarea-rapidă-și-eficientă-a-modelelor) pentru a afla mai multe despre cum sunt încărcate modelele.

[`~PreTrainedModel.save_pretrained`] creează și un fișier index care mapează numele parametrilor la fișierele lor de shard. Index-ul conține două chei, `metadata` și `weight_map`.

```py
import json

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="50GB")
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
        index = json.load(f)

print(index.keys())
```

`metadata` stochează dimensiunea totală a modelului.

```py
index["metadata"]
{'total_size': 28966928384}
```

`weight_map` mapează fiecare parametru la fișierul său de shard.

```py
index["weight_map"]
{'lm_head.weight': 'model-00006-of-00006.safetensors',
 'model.embed_tokens.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.input_layernorm.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.mlp.down_proj.weight': 'model-00001-of-00006.safetensors',
 ...
}
```

### Big Model Inference

> [!TIP]
> Asigură-te că ai Accelerate v0.9.0 și PyTorch v1.9.0 sau o versiune mai nouă instalate pentru a utiliza această funcție!

<Youtube id="MWCSGj9jEAo"/>

[`~PreTrainedModel.from_pretrained`] este potențat cu funcția [Big Model Inference](https://hf.co/docs/accelerate/usage_guides/big_modeling) din Accelerate.

Big Model Inference creează un *schelet de model* pe dispozitivul [meta](https://pytorch.org/docs/main/meta.html) PyTorch. Dispozitivul meta nu stochează date reale, doar metadata.

Weights inițializate aleatoriu sunt create doar când sunt încărcate weights pre-antrenate, pentru a evita menținerea în memorie a două copii ale modelului în același timp. Utilizarea maximă a memoriei este doar dimensiunea modelului.

> [!TIP]
> Află mai multe despre plasarea pe dispozitive în [Designing a device map](https://hf.co/docs/accelerate/v0.33.0/en/concept_guides/big_model_inference#designing-a-device-map).

A doua funcție a Big Model Inference se referă la modul în care weights sunt încărcate și distribuite în scheletul modelului. Weights modelului sunt distribuite pe toate dispozitivele disponibile, începând cu cel mai rapid dispozitiv (de obicei GPU) și descărcând ulterior weights rămase pe dispozitive mai lente (CPU și hard disk).

Ambele funcții combinate reduc utilizarea memoriei și timpii de încărcare pentru modelele mari pre-antrenate.

Setează [device_map](https://github.com/huggingface/transformers/blob/026a173a64372e9602a16523b8fae9de4b0ff428/src/transformers/modeling_utils.py#L3061) la `"auto"` pentru a activa Big Model Inference.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")
```

Poți, de asemenea, să atribui manual straturi unui dispozitiv în `device_map`. Ar trebui să mapeze toți parametrii modelului la un dispozitiv, dar nu trebuie să detaliezi unde merg toate sub-modulele unui strat dacă întregul strat se află pe același dispozitiv.

Accesează atributul `hf_device_map` pentru a vedea cum este distribuit un model pe dispozitive.

```py
device_map = {"model.layers.1": 0, "model.layers.14": 1, "model.layers.31": "cpu", "lm_head": "disk"}
model.hf_device_map
```

### Tipul de date al modelului

Argumentul `dtype` controlează [dtype-ul](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) PyTorch utilizat pentru a instanția weights modelului. În mod implicit, Transformers încarcă weights cu valoarea `dtype` sau a moștenitorului `torch_dtype` din `config.json`. Dacă `config.json` nu include niciuna dintre aceste valori, Transformers utilizează dtype-ul primului weight în virgulă mobilă din checkpoint.

Suprascrie valoarea implicită pasând un dtype specific.

```py
import torch
from transformers import AutoModelForCausalLM

# specific dtype
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", dtype=torch.float16)
```

[`AutoConfig`] acceptă de asemenea `dtype` pentru modelele instanțiate de la zero.

```py
import torch
from transformers import AutoConfig, AutoModel

my_config = AutoConfig.from_pretrained("google/gemma-2b", dtype=torch.float16)
model = AutoModel.from_config(my_config)
```

## Modele personalizate

Modelele personalizate se construiesc pe clasele de configurație și modelare ale Transformers, suportă API-ul [AutoClass](#autoclass) și sunt încărcate cu [`~PreTrainedModel.from_pretrained`]. Diferența este că codul de modelare *nu* provine din Transformers.

Fii atent la încărcarea unui model personalizat. Deși Hub-ul include [scanare de malware](https://hf.co/docs/hub/security-malware#malware-scanning) pentru fiecare repository, trebuie totuși să fii atent să nu execuți accidental cod malițios.

Setează `trust_remote_code=True` în [`~PreTrainedModel.from_pretrained`] pentru a încărca un model personalizat.

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

Ca un nivel suplimentar de securitate, încarcă un model personalizat dintr-o revizuire specifică pentru a evita încărcarea codului de model care ar putea fi schimbat. Hash-ul commit-ului poate fi copiat din [istoricul commit-urilor](https://hf.co/sgugger/custom-resnet50d/commits/main) modelului.

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

Consultă ghidul [Personalizarea modelelor](./custom_models) pentru mai multe informații.
