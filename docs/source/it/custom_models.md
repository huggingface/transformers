<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Condividere modelli personalizzati
La libreria 🤗 Transformers è studiata per essere facilmente estendibile. Il codice di ogni modello è interamente 
situato in una sottocartella del repository senza alcuna astrazione, perciò puoi facilmente copiare il file di un 
modello e modificarlo in base ai tuoi bisogni.

Se stai scrivendo un nuovo modello, potrebbe essere più semplice iniziare da zero. In questo tutorial, ti mostreremo
come scrivere un modello personalizzato e la sua configurazione in modo che possa essere utilizzato all’interno di
Transformers, e come condividerlo con la community (assieme al relativo codice) così che tutte le persone possano usarlo, anche
se non presente nella libreria 🤗 Transformers.

Illustriamo tutto questo su un modello ResNet, avvolgendo la classe ResNet della 
[libreria timm](https://github.com/rwightman/pytorch-image-models) in un [`PreTrainedModel`].

## Scrivere una configurazione personalizzata
Prima di iniziare a lavorare al modello, scriviamone la configurazione. La configurazione di un modello è un oggetto
che contiene tutte le informazioni necessarie per la build del modello. Come vedremo nella prossima sezione, il 
modello può soltanto essere inizializzato tramite `config`, per cui dovremo rendere tale oggetto più completo possibile.

Nel nostro esempio, prenderemo un paio di argomenti della classe ResNet che potremmo voler modificare. 
Configurazioni differenti ci daranno quindi i differenti possibili tipi di ResNet. Salveremo poi questi argomenti, 
dopo averne controllato la validità.

```python
from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
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

Le tre cose più importanti da ricordare quando scrivi le tue configurazioni sono le seguenti:
- Devi ereditare da `Pretrainedconfig`,
- Il metodo `__init__` del tuo `Pretrainedconfig` deve accettare i kwargs,
- I `kwargs` devono essere passati alla superclass `__init__`

L’eredità è importante per assicurarsi di ottenere tutte le funzionalità della libreria 🤗 transformers, 
mentre gli altri due vincoli derivano dal fatto che un `Pretrainedconfig` ha più campi di quelli che stai settando. 
Quando ricarichi una config da un metodo `from_pretrained`, questi campi devono essere accettati dalla tua config e
poi inviati alla superclasse.

Definire un `model_type` per la tua configurazione (qua `model_type = “resnet”`) non è obbligatorio, a meno che tu
non voglia registrare il modello con le classi Auto (vedi l'ultima sezione).

Una volta completato, puoi facilmente creare e salvare la tua configurazione come faresti con ogni altra configurazione
di modelli della libreria. Ecco come possiamo creare la config di un resnet50d e salvarlo:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

Questo salverà un file chiamato `config.json` all'interno della cartella `custom-resnet`. Potrai poi ricaricare la tua
config con il metodo `from_pretrained`.

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

Puoi anche usare qualunque altro metodo della classe [`PretrainedConfig`], come [`~PretrainedConfig.push_to_hub`]
per caricare direttamente la tua configurazione nell'hub.

## Scrivere un modello personalizzato

Ora che abbiamo la nostra configurazione ResNet, possiamo continuare a scrivere il modello. In realtà, ne scriveremo
due: uno che estrae le features nascoste da una batch di immagini (come [`BertModel`]) e uno che è utilizzabile per 
la classificazione di immagini (come [`BertModelForSequenceClassification`]).

Come abbiamo menzionato in precedenza, scriveremo soltanto un wrapper del modello, per mantenerlo semplice ai fini di 
questo esempio. L'unica cosa che dobbiamo fare prima di scrivere questa classe è una mappatura fra i tipi di blocco e 
le vere classi dei blocchi. Successivamente il modello è definito tramite la configurazione, passando tutto quanto alla
classe `ResNet`.

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

Per il modello che classificherà le immagini, cambiamo soltanto il metodo forward:

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

Nota come, in entrambi i casi, ereditiamo da `PreTrainedModel` e chiamiamo l'inizializzazione della superclasse 
con il metodo `config` (un po' come quando scrivi un normale `torch.nn.Module`). La riga che imposta la  `config_class`
non è obbligatoria, a meno che tu non voglia registrare il modello con le classi Auto (vedi l'ultima sezione).

<Tip>

Se il tuo modello è molto simile a un modello all'interno della libreria, puoi ri-usare la stessa configurazione di quel modello.

</Tip>

Puoi fare in modo che il tuo modello restituisca in output qualunque cosa tu voglia, ma far restituire un dizionario 
come abbiamo fatto per `ResnetModelForImageClassification`, con la funzione di perdita inclusa quando vengono passate le labels,
renderà il tuo modello direttamente utilizzabile all'interno della classe [`Trainer`]. Utilizzare altri formati di output va bene
se hai in progetto di utilizzare un tuo loop di allenamento, o se utilizzerai un'altra libreria per l'addestramento.

Ora che abbiamo la classe del nostro modello, creiamone uno:

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

Ribadiamo, puoi usare qualunque metodo dei [`PreTrainedModel`], come [`~PreTrainedModel.save_pretrained`] o
[`~PreTrainedModel.push_to_hub`]. Utilizzeremo quest'ultimo nella prossima sezione, e vedremo come caricare i pesi del
modello assieme al codice del modello stesso. Ma prima, carichiamo alcuni pesi pre-allenati all'interno del nostro modello.

Nel tuo caso specifico, probabilmente allenerai il tuo modello sui tuoi dati. Per velocizzare in questo tutorial, 
utilizzeremo la versione pre-allenata del resnet50d. Dato che il nostro modello è soltanto un wrapper attorno a quel modello,
sarà facile trasferirne i pesi:

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

Vediamo adesso come assicurarci che quando facciamo [`~PreTrainedModel.save_pretrained`] o [`~PreTrainedModel.push_to_hub`], 
il codice del modello venga salvato.

## Inviare il codice all'Hub

<Tip warning={true}>

Questa API è sperimentale e potrebbe avere alcuni cambiamenti nei prossimi rilasci.

</Tip>

Innanzitutto, assicurati che il tuo modello sia completamente definito in un file `.py`. Può sfruttare import relativi
ad altri file, purchè questi siano nella stessa directory (non supportiamo ancora sotto-moduli per questa funzionalità).
Per questo esempio, definiremo un file `modeling_resnet.py` e un file `configuration_resnet.py` in una cartella dell'attuale
working directory chiamata `resnet_model`. Il file configuration contiene il codice per `ResnetConfig` e il file modeling 
contiene il codice di `ResnetModel` e `ResnetModelForImageClassification`.

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

Il file `__init__.py` può essere vuoto, serve solo perchè Python capisca che `resnet_model` può essere utilizzato come un modulo.

<Tip warning={true}>

Se stai copiando i file relativi alla modellazione della libreria, dovrai sostituire tutti gli import relativi in cima al file con import del 
    pacchetto `transformers`.

</Tip>

Nota che puoi ri-utilizzare (o usare come sottoclassi) un modello/configurazione esistente.

Per condividere il tuo modello con la community, segui questi passi: prima importa il modello ResNet e la sua configurazione 
dai nuovi file creati:

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

Dopodichè dovrai dire alla libreria che vuoi copiare i file con il codice di quegli oggetti quando utilizzi il metodo
`save_pretrained` e registrarli in modo corretto con una Auto classe (specialmente per i modelli). Utilizza semplicemente:

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

Nota che non c'è bisogno di specificare una Auto classe per la configurazione (c'è solo una Auto classe per le configurazioni,
[`AutoConfig`], ma è diversa per i modelli). Il tuo modello personalizato potrebbe essere utilizzato per diverse tasks, 
per cui devi specificare quale delle classi Auto è quella corretta per il tuo modello.

Successivamente, creiamo i modelli e la config come abbiamo fatto in precedenza:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

Adesso, per inviare il modello all'Hub, assicurati di aver effettuato l'accesso. Lancia dal tuo terminale:

```bash
huggingface-cli login
```

O da un notebook:

```py
from huggingface_hub import notebook_login

notebook_login()
```

Potrai poi inviare il tutto sul tuo profilo (o di un'organizzazione di cui fai parte) in questo modo:

```py
resnet50d.push_to_hub("custom-resnet50d")
```

Oltre ai pesi del modello e alla configurazione in formato json, questo ha anche copiato i file `.py` modeling e
configuration all'interno della cartella `custom-resnet50d` e ha caricato i risultati sull'Hub. Puoi controllare
i risultati in questa [model repo](https://huggingface.co/sgugger/custom-resnet50d).

Puoi controllare il tutorial di condivisione [tutorial di condivisione](model_sharing) per più informazioni sul 
metodo con cui inviare all'Hub.

## Usare un modello con codice personalizzato

Puoi usare ogni configurazione, modello o tokenizer con file di codice personalizzati nella sua repository 
con le classi Auto e il metodo `from_pretrained`. Tutti i files e il codice caricati sull'Hub sono scansionati da malware
(fai riferimento alla documentazione [Hub security](https://huggingface.co/docs/hub/security#malware-scanning) per più informazioni),
ma dovresti comunque assicurarti dell'affidabilità del codice e dell'autore per evitare di eseguire codice dannoso sulla tua macchina. 
Imposta `trust_remote_code=True` per usare un modello con codice personalizzato:

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

Inoltre, raccomandiamo fortemente di passare un hash del commit come `revision` per assicurarti che le autrici o gli autori del modello 
non abbiano modificato il codice con alcune nuove righe dannose (a meno che non ti fidi completamente della fonte):

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

Nota che quando cerchi la storia dei commit della repo del modello sull'Hub, c'è un bottone con cui facilmente copiare il 
commit hash di ciascun commit.

## Registrare un modello con codice personalizzato nelle classi Auto

Se stai scrivendo una libreria che estende 🤗 Transformers, potresti voler estendere le classi Auto per includere il tuo modello.
Questo è diverso dall'inviare codice nell'Hub: gli utenti dovranno importare la tua libreria per ottenere il modello personalizzato
(anzichè scaricare automaticamente il modello dall'Hub).

Finchè il tuo file di configurazione ha un attributo `model_type` diverso dai model types esistenti, e finchè le tue 
classi modello hanno i corretti attributi `config_class`, potrai semplicemente aggiungerli alle classi Auto come segue:

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

Nota che il primo argomento utilizzato quando registri la configurazione di un modello personalizzato con [`AutoConfig`] 
deve corrispondere al `model_type` della tua configurazione personalizzata, ed il primo argomento utilizzato quando 
registri i tuoi modelli personalizzati in una qualunque classe Auto del modello deve corrispondere alla `config_class`
di quei modelli.
