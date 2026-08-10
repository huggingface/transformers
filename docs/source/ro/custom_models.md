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

# Personalizarea modelelor

Modelele Transformers sunt concepute pentru a fi personalizabile. Codul unui model este complet conținut în subfolderul [model](https://github.com/huggingface/transformers/tree/main/src/transformers/models) al repository-ului Transformers. Fiecare folder conține un fișier `modeling.py` și un fișier `configuration.py`. Copiază aceste fișiere pentru a începe personalizarea unui model.

> [!TIP]
> Poate fi mai ușor să începi de la zero dacă creezi un model complet nou. Pentru modele similare cu unul existent în Transformers, este mai rapid să reutilizezi sau să subclasezi aceeași configurație și clasă de model.

Acest ghid îți va arăta cum să personalizezi un model ResNet, să activezi suportul [AutoClass](./models#autoclass) și să-l partajezi pe Hub.

## Configurație

O configurație, furnizată de clasa de bază [`PreTrainedConfig`], conține toate informațiile necesare pentru a construi un model. Acesta este locul unde vei configura atributele modelului ResNet personalizat. Atribute diferite oferă tipuri diferite de modele ResNet.

Regulile principale pentru personalizarea unei configurații sunt:

1. O configurație personalizată trebuie să subclaseze [`PreTrainedConfig`]. Aceasta asigură că un model personalizat are toate funcționalitățile unui model Transformers, precum [`~PreTrainedConfig.from_pretrained`], [`~PreTrainedConfig.save_pretrained`] și [`~PreTrainedConfig.push_to_hub`].
2. `__init__`-ul [`PreTrainedConfig`] trebuie să accepte orice `kwargs` și acestea trebuie pasate `__init__`-ului superclasei. [`PreTrainedConfig`] are mai multe câmpuri decât cele setate în configurația ta personalizată, astfel că atunci când încarci o configurație cu [`~PreTrainedConfig.from_pretrained`], acele câmpuri trebuie acceptate de configurația ta și pasate superclasei.

> [!TIP]
> Este util să verifici validitatea unora dintre parametri. În exemplul de mai jos, se implementează o verificare pentru a te asigura că `block_type` și `stem_type` aparțin uneia dintre valorile predefinite.
>
> Adaugă `model_type` la clasa de configurație pentru a activa suportul [AutoClass](./models#autoclass).

```py
from transformers import PreTrainedConfig
from typing import List

class ResnetConfig(PreTrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: list[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

Salvează configurația într-un fișier JSON în folderul modelului personalizat, `custom-resnet`, cu [`~PreTrainedConfig.save_pretrained`].

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

## Model

Cu configurația ResNet personalizată, poți acum crea și personaliza modelul. Modelul subclasează clasa de bază [`PreTrainedModel`]. Ca și [`PreTrainedConfig`], moștenirea din [`PreTrainedModel`] și inițializarea superclasei cu configurația extinde funcționalitățile Transformers, precum salvarea și încărcarea, la modelul personalizat.

Modelele Transformers urmează convenția de a accepta un obiect `config` în metoda `__init__`. Acesta pasează întregul `config` sublayers modelului, în loc să rupă obiectul `config` în mai multe argumente care sunt pasate individual către sublayers.

Scrierea modelelor în acest mod produce cod mai simplu cu o sursă clară de adevăr pentru orice hyperparameters. De asemenea, face mai ușoară reutilizarea codului din alte modele Transformers.

Vei crea două modele ResNet: un model ResNet de bază care returnează hidden states și un model ResNet cu un head de clasificare a imaginilor.

<hfoptions id="resnet">
<hfoption id="ResnetModel">

Definește o mapare între tipurile de blocuri și clase. Tot restul este creat pasând clasa de configurație clasei modelului ResNet.

> [!TIP]
> Adaugă `config_class` la clasa de model pentru a activa suportul [AutoClass](#autoclass-support).

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig

BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}

class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

</hfoption>
<hfoption id="ResnetModelForImageClassification">

Metoda `forward` trebuie rescrisă pentru a calcula loss-ul pentru fiecare logit dacă sunt disponibile labels. Altfel, clasa modelului ResNet este aceeași.

> [!TIP]
> Adaugă `config_class` la clasa de model pentru a activa suportul [AutoClass](#autoclass-support).

```py
import torch

class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

</hfoption>
</hfoptions>

Un model poate returna orice format de ieșire. Returnarea unui dicționar (ca `ResnetModelForImageClassification`) cu losses când sunt disponibile labels face modelul personalizat compatibil cu [`Trainer`]. Pentru alte formate de ieșire, vei avea nevoie de propriul loop de antrenare sau de o bibliotecă diferită pentru antrenare.

Instanțiază clasa modelului personalizat cu configurația.

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

În acest moment, poți încărca weights pre-antrenate în model sau îl poți antrena de la zero. În acest ghid, vei încărca weights pre-antrenate.

Încarcă weights pre-antrenate din biblioteca [timm](https://hf.co/docs/timm/index), apoi transferă acele weights modelului personalizat cu [load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict).

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

## AutoClass

API-ul [AutoClass](./models#clasele-de-model) este o scurtătură pentru a încărca automat arhitectura corectă pentru un model dat. Este convenabil să activezi aceasta pentru utilizatorii care încarcă modelul tău personalizat.

Asigură-te că ai atributul `model_type` (trebuie să fie diferit de tipurile de modele existente) în clasa de configurație și atributul `config_class` în clasa de model. Folosește metoda [`~AutoConfig.register`] pentru a adăuga configurația și modelul personalizat la API-ul [AutoClass](./models#clasele-de-model).

> [!TIP]
> Primul argument al [`AutoConfig.register`] trebuie să corespundă atributului `model_type` din clasa de configurație personalizată, iar primul argument al [`AutoModel.register`] trebuie să corespundă `config_class`-ului clasei modelului personalizat.

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

Codul modelului tău personalizat este acum compatibil cu API-ul [AutoClass](./models#autoclass). Utilizatorii pot încărca modelul cu clasele [AutoModel] sau [`AutoModelForImageClassification`].

## Publicare pe Hub

Publică un model personalizat pe [Hub](https://hf.co/models) pentru a permite altor utilizatori să îl încarce și utilizeze cu ușurință.

Asigură-te că directorul modelului este structurat corect, după cum se arată mai jos. Directorul ar trebui să conțină:

- `modeling.py`: Conține codul pentru `ResnetModel` și `ResnetModelForImageClassification`. Acest fișier poate utiliza importuri relative la alte fișiere atât timp cât se află în același director.

> [!WARNING]
> Când copiezi un fișier de model Transformers, înlocuiește toate importurile relative din partea de sus a fișierului `modeling.py` pentru a importa din Transformers.

- `configuration.py`: Conține codul pentru `ResnetConfig`.
- `__init__.py`: Poate fi gol; acest fișier permite utilizarea `resnet_model` ca modul Python.

```bash
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

Pentru a partaja modelul, importă modelul ResNet și configurația.

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

Copiază codul din fișierele de model și configurație. Pentru a te asigura că obiectele AutoClass sunt salvate cu [`~PreTrainedModel.save_pretrained`], apelează metoda [`~PreTrainedConfig.register_for_auto_class`]. Aceasta modifică fișierul JSON de configurație pentru a include obiectele AutoClass și maparea.

Pentru un model, alege clasa `AutoModelFor` corespunzătoare pe baza task-ului.

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

Pentru a mapa mai mult de un task la model, editează `auto_map` direct în fișierul JSON de configurație.

```json
"auto_map": {
    "AutoConfig": "<your-repo-name>--<config-name>",
    "AutoModel": "<your-repo-name>--<config-name>",
    "AutoModelFor<Task>": "<your-repo-name>--<config-name>",    
},
```

Creează configurația și modelul și încarcă weights pre-antrenate în acesta.

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

Modelul este acum gata să fie publicat pe Hub. Autentifică-te în contul tău Hugging Face din linia de comandă sau notebook.

<hfoptions id="push">
<hfoption id="huggingface-CLI">

```bash
hf auth login
```

</hfoption>
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
</hfoptions>

Apelează [`~PreTrainedModel.push_to_hub`] pe model pentru a-l publica pe Hub.

```py
resnet50d.push_to_hub("custom-resnet50d")
```

Weights pre-antrenate, configurația și fișierele `modeling.py` și `configuration.py` ar trebui acum să fie toate publicate pe Hub într-un [repository](https://hf.co/sgugger/custom-resnet50d) sub namespace-ul tău.

Deoarece un model personalizat nu utilizează același cod de modelare ca un model Transformers, trebuie să adaugi `trust_remote_code=True` în [`~PreTrainedModel.from_pretrained`] pentru a-l încărca. Consultă secțiunea [modele personalizate](./models#modele-personalizate) pentru mai multe informații.
