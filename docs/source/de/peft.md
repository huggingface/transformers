<!--Copyright 2023 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Adapter mit 🤗 PEFT laden

[[open-in-colab]]

Die [Parameter-Efficient Fine Tuning (PEFT)](https://huggingface.co/blog/peft) Methoden frieren die vorab trainierten Modellparameter während der Feinabstimmung ein und fügen eine kleine Anzahl trainierbarer Parameter (die Adapter) hinzu. Die Adapter werden trainiert, um aufgabenspezifische Informationen zu lernen. Es hat sich gezeigt, dass dieser Ansatz sehr speichereffizient ist und weniger Rechenleistung beansprucht, während die Ergebnisse mit denen eines vollständig feinabgestimmten Modells vergleichbar sind. 

Adapter, die mit PEFT trainiert wurden, sind in der Regel um eine Größenordnung kleiner als das vollständige Modell, so dass sie bequem gemeinsam genutzt, gespeichert und geladen werden können.

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">Die Adaptergewichte für ein OPTForCausalLM-Modell, die auf dem Hub gespeichert sind, sind nur ~6MB groß, verglichen mit der vollen Größe der Modellgewichte, die ~700MB betragen können.</figcaption>
</div>

Wenn Sie mehr über die 🤗 PEFT-Bibliothek erfahren möchten, sehen Sie sich die [Dokumentation](https://huggingface.co/docs/peft/index) an.

## Setup

Starten Sie mit der Installation von 🤗 PEFT:

```bash
pip install peft
```

Wenn Sie die brandneuen Funktionen ausprobieren möchten, sollten Sie die Bibliothek aus dem Quellcode installieren:

```bash
pip install git+https://github.com/huggingface/peft.git
```

## Unterstützte PEFT-Modelle

Transformers unterstützt nativ einige PEFT-Methoden, d.h. Sie können lokal oder auf dem Hub gespeicherte Adaptergewichte laden und sie mit wenigen Zeilen Code einfach ausführen oder trainieren. Die folgenden Methoden werden unterstützt:

- [Low Rank Adapters](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [AdaLoRA](https://arxiv.org/abs/2303.10512)

Wenn Sie andere PEFT-Methoden, wie z.B. Prompt Learning oder Prompt Tuning, verwenden möchten, oder über die 🤗 PEFT-Bibliothek im Allgemeinen, lesen Sie bitte die [Dokumentation](https://huggingface.co/docs/peft/index).


## Laden Sie einen PEFT-Adapter

Um ein PEFT-Adaptermodell von 🤗 Transformers zu laden und zu verwenden, stellen Sie sicher, dass das Hub-Repository oder das lokale Verzeichnis eine `adapter_config.json`-Datei und die Adaptergewichte enthält, wie im obigen Beispielbild gezeigt. Dann können Sie das PEFT-Adaptermodell mit der Klasse `AutoModelFor` laden. Um zum Beispiel ein PEFT-Adaptermodell für die kausale Sprachmodellierung zu laden:

1. Geben Sie die PEFT-Modell-ID an.
2. übergeben Sie es an die Klasse [`AutoModelForCausalLM`].

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```

<Tip>

Sie können einen PEFT-Adapter entweder mit einer `AutoModelFor`-Klasse oder der Basismodellklasse wie `OPTForCausalLM` oder `LlamaForCausalLM` laden.

</Tip>

Sie können einen PEFT-Adapter auch laden, indem Sie die Methode `load_adapter` aufrufen:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

## Laden in 8bit oder 4bit

Die `bitsandbytes`-Integration unterstützt Datentypen mit 8bit und 4bit Genauigkeit, was für das Laden großer Modelle nützlich ist, weil es Speicher spart (lesen Sie den `bitsandbytes`-Integrations [guide](./quantization#bitsandbytes-integration), um mehr zu erfahren). Fügen Sie die Parameter `load_in_8bit` oder `load_in_4bit` zu [`~PreTrainedModel.from_pretrained`] hinzu und setzen Sie `device_map="auto"`, um das Modell effektiv auf Ihre Hardware zu verteilen:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", load_in_8bit=True)
```

## Einen neuen Adapter hinzufügen

Sie können [`~peft.PeftModel.add_adapter`] verwenden, um einen neuen Adapter zu einem Modell mit einem bestehenden Adapter hinzuzufügen, solange der neue Adapter vom gleichen Typ ist wie der aktuelle Adapter. Wenn Sie zum Beispiel einen bestehenden LoRA-Adapter an ein Modell angehängt haben:

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```

Um einen neuen Adapter hinzuzufügen:

```py
# attach new adapter with same config
model.add_adapter(lora_config, adapter_name="adapter_2")
```

Jetzt können Sie mit [`~peft.PeftModel.set_adapter`] festlegen, welcher Adapter verwendet werden soll:

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

## Aktivieren und Deaktivieren von Adaptern

Sobald Sie einen Adapter zu einem Modell hinzugefügt haben, können Sie das Adaptermodul aktivieren oder deaktivieren. So aktivieren Sie das Adaptermodul:

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

So deaktivieren Sie das Adaptermodul:

```py
model.disable_adapters()
output = model.generate(**inputs)
```

## PEFT-Adapter trainieren

PEFT-Adapter werden von der Klasse [`Trainer`] unterstützt, so dass Sie einen Adapter für Ihren speziellen Anwendungsfall trainieren können. Dazu müssen Sie nur ein paar weitere Codezeilen hinzufügen. Zum Beispiel, um einen LoRA-Adapter zu trainieren:

<Tip>

Wenn Sie mit der Feinabstimmung eines Modells mit [`Trainer`] noch nicht vertraut sind, werfen Sie einen Blick auf das Tutorial [Feinabstimmung eines vortrainierten Modells](Training).

</Tip>

1. Definieren Sie Ihre Adapterkonfiguration mit dem Aufgabentyp und den Hyperparametern (siehe [`~peft.LoraConfig`] für weitere Details darüber, was die Hyperparameter tun).

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

2. Fügen Sie dem Modell einen Adapter hinzu.

```py
model.add_adapter(peft_config)
```

3. Jetzt können Sie das Modell an [`Trainer`] übergeben!

```py
trainer = Trainer(model=model, ...)
trainer.train()
```

So speichern Sie Ihren trainierten Adapter und laden ihn wieder:

```py
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```

<!--
TODO: (@younesbelkada @stevhliu)
-   Link to PEFT docs for further details
-   Trainer  
-   8-bit / 4-bit examples ?
-->
