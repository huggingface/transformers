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

# Pornire rapidă

[[open-in-colab]]

Transformers este construit astfel încât să fie rapid și ușor de folosit pentru ca toată lumea să poată să învețe și să construiască utilizând modele transformers.

Numărul de abstractizări orientate către utilizator este limitat la trei clase pentru inițializarea unui model și două API-uri pentru antrenare sau inferență. Acest ghid de pornire rapidă te introduce în funcțiile de bază ale Transformers și îți arată cum să:

- încarci un model pre-antrenat
- rulezi inferență cu [`Pipeline`]
- ajustezi un model cu [`Trainer`]

## Configurare

Pentru început, recomandăm să creezi un [cont](https://hf.co/join) Hugging Face. Un cont îți permite să rulezi și să accesezi modele, seturi de date și [Spaces](https://hf.co/spaces) pe [Hub-ul](https://hf.co/docs/hub/index) Hugging Face, o platformă colaborativă pentru construire și descoperire.

Creează un [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) și autentifică-te în contul tău.

<hfoptions id="authenticate">
<hfoption id="notebook">

Introdu User Access Token-ul în [`~huggingface_hub.notebook_login`] la autentificare.

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
<hfoption id="CLI">

Asigură-te că package-ul [huggingface_hub[cli]](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) este instalat și rulează comanda de mai jos. Introdu User Access Token-ul la autentificare.

```bash
hf auth login
```

</hfoption>
</hfoptions>

Instalează PyTorch.

```bash
!pip install torch
```

După, instalează o versiune la zi a Transformers și biblioteci adiționale din ecosistemul Hugging Face pentru a accesa seturi de date, modele de vision, și pentru a evalua și optimiza antrenarea modelelor mari.

```bash
!pip install -U transformers datasets evaluate accelerate timm
```

## Modele pre-antrenate

Fiecare model pre-antrenat moștenește din 3 clase de bază.

| **Clasă** | **Descriere** |
|---|---|
| [`PreTrainedConfig`] | Un fișier care specifică atributele modelului, precum numărul de attention heads și dimensiunea vocabularului. |
| [`PreTrainedModel`] | Un model (sau o arhitectură) definit(ă) de atributele modelului din fișierul de configurație. Un model pre-antrenat returnează doar raw hidden states. Pentru un task specific, folosește model head-ul potrivit pentru a transforma raw hidden states într-un rezultat relevant (spre exemplu, [`LlamaModel`] versus [`LlamaForCausalLM`]). |
| Preprocessor | O clasă pentru transformarea din raw inputs (text, imagini, audio, multimodale) în numerical inputs pentru model. Spre exemplu, [`PreTrainedTokenizer`] transformă text în tensori și [`ImageProcessingMixin`] transformă pixeli în tensori. |

Recomandăm să utilizezi API-ul [AutoClass](./model_doc/auto) pentru încărcarea modelelor și a preprocesatoarelor pentru că alege automat arhitectura potrivită pentru fiecare task și framework-ul de machine learning în baza numelui sau path-ului pentru model weights și a fișierului de configurație.

Folosește [`~PreTrainedModel.from_pretrained`] pentru a încărca weights și fișierul de configurație de pe Hub în model și clasa de preprocesare.

Când încarci un model, configurează parametrii de mai jos pentru ca modelul să fie încărcat optim.

- `device_map="auto"` alocă automat model weights celui mai rapid dispozitiv.
- `dtype="auto"` inițializează din start model weights cu tipul de date în care acestea sunt salvate, ceea ce poate evita încărcarea lor de două ori (PyTorch utilizează implicit tipul `torch.float32`).

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

Tokenizează textul și returnează tensori PyTorch cu Tokenizatorul. Mută modelul pe un accelerator (dacă unul este disponibil) pentru a accelera inferența.

```py
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
```

Modelul este pregătit pentru antrenare sau inferență.

Pentru inferență, introdu input-urile tokenizate în [`~GenerationMixin.generate`] pentru a genera text. Decodează token id-urile înapoi în text utilizând [`~PreTrainedTokenizerBase.batch_decode`].

```py
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
```

> [!TIP]    
> Vezi secțiunea [Trainer](#trainer-api) ca să înveți cum să ajustezi un model.

## Pipeline

Clasa [`Pipeline`] este cel mai convenabil mod de a face inferență cu un model pre-antrenat. Suportă multe task-uri, precum generarea de text, segmentarea de imagini, recunoașterea vocală automată, răspunsuri la întrebări din documente și multe altele.

> [!TIP]
> Vezi referința API-ului [Pipeline](./main_classes/pipelines) pentru o listă completă de task-uri disponibile.

Creează un obiect [`Pipeline`] și selectează un task. [`Pipeline`] descarcă și salvează în cache un model pentru un task dat. Introdu numele modelului în parametrul `model` pentru a alege un model specific.

<hfoptions id="pipeline-tasks">
<hfoption id="text generation">

Utilizează [`Accelerator`] pentru a detecta automat un accelerator disponibil pentru inferență.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device=device)
```

Introdu text inițial în [`Pipeline`] pentru a genera mai mult text.

```py
pipeline("The secret to baking a good cake is ", max_length=50)
[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we’ve created the best buttercream frosting reci'}]
```

</hfoption>
<hfoption id="image segmentation">

Utilizează [`Accelerator`] pentru a detecta automat un accelerator disponibil pentru inferență.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device=device)
```

Introdu o imagine - un URL sau un path local - în [`Pipeline`].

<div class="flex justify-center">
   <img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"/>
</div>

```py
segments = pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
segments[0]["label"]
'bird'
segments[1]["label"]
'bird'
```

</hfoption>
<hfoption id="automatic speech recognition">

Utilizează [`Accelerator`] pentru a detecta automat un accelerator disponibil pentru inferență.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
```

Introdu un fișier audio în [`Pipeline`].

```py
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

</hfoption>
</hfoptions>

## Trainer

[`Trainer`] este un loop complet de antrenare și evaluare pentru modelele PyTorch. Abstractizează mult din codul boilerplate întâlnit la scrierea manuală a unui loop de antrenare, ca să poți începe antrenarea mai rapid și ca să te focusezi pe designul antrenării. Ai nevoie doar de un model, un set de date, un preprocesator și un data collator ca să construiești batch-uri de date din set.

Utilizează clasa [`TrainingArguments`] pentru a personaliza procesul de antrenare. Dispune de multe opțiuni pentru antrenare, evaluare și multe altele. Experimentează cu hyperparameters de antrenare și funcții precum mărimea batch-ului, rata de învățare, precizia mixtă, torch.compile și multe altele, pentru necesitățile tale la antrenare. Poți folosi și parametrii impliciți de antrenare pentru a produce rapid un baseline.

Încarcă un model, un set de date și un tokenizer pentru antrenare.

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")
```

Creează o funcție pentru a tokeniza textul și a-l transforma în tensori PyTorch. Aplică această funcție întregului set de date cu metoda [`~datasets.Dataset.map`].

```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

Încarcă un data collator pentru a crea batch-uri de date și asociază-i tokenizer-ul.

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

După, configurează [`TrainingArguments`] cu features și hyperparameters de antrenare.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

În final, introdu toate aceste componente în [`Trainer`] și apelează funcția [`~Trainer.train`] pentru a începe.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

Distribuie-ți modelul și tokenizer-ul pe Hub cu [`~Trainer.push_to_hub`].

```py
trainer.push_to_hub()
```

Felicitări, ai terminat de antrenat primul tău model cu Transformers!

## Următorii pași

Acum că știi mai multe despre Transformers și despre ce are de oferit, continuă să explorezi și să înveți ceea ce te interesează.

- **Clase de bază**: Învață mai multe despre clasele de configurație, model și procesare. Asta te va ajuta să creezi și personalizezi modele, să procesezi diferite tipuri de input (audio, imagini, multimodale) și cum să-ți partajezi modelul.
- **Inferență**: Explorează [`Pipeline`] în detaliu, inferența si conversarea cu LLMs, agenți și cum să optimizezi inferența cu framework-ul de machine learning și hardware-ul tău.
- **Antrenare**: Studiază [`Trainer`] în detaliu, antrenarea distribuită și optimizarea antrenării pe diferite configurații hardware.
- **Quantization**: Redu cerințele de memorie și stocare cu quantization și mărește viteza de inferență prin reprezentarea model weights în mai puțini biți.
- **Resurse**: Cauți ghiduri complete pentru antrenarea și inferența cu un model pentru un task specific? Consultă ghidurile pentru task-uri!
