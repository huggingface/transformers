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

# Pipeline

[`Pipeline`] este un API de inferență simplu, dar puternic, disponibil imediat pentru o varietate de task-uri de machine learning cu orice model din Hugging Face [Hub](https://hf.co/models).

Adaptează [`Pipeline`] la task-ul tău cu parametri specifici task-ului, precum adăugarea de timestamp-uri la un pipeline de automatic speech recognition (ASR) pentru transcrierea notițelor de ședință. [`Pipeline`] suportă GPU-uri, Apple Silicon și weights cu precizie redusă (half-precision) pentru a accelera inferența și a economisi memorie.

<Youtube id="tiZFewofSLM"/>

Transformers are două clase de pipeline, un [`Pipeline`] generic și multe pipeline-uri individuale specifice task-ului precum [`TextGenerationPipeline`]. Încarcă aceste pipeline-uri individuale setând identificatorul task-ului în parametrul `task` din [`Pipeline`]. Poți găsi identificatorul task-ului pentru fiecare pipeline în documentația lor API.

Fiecare task este configurat să folosească un model preantrenat și un preprocesor implicit, dar acest lucru poate fi suprascris cu parametrul `model` dacă vrei să folosești un alt model.

De exemplu, pentru a folosi [`TextGenerationPipeline`] cu [Gemma 2], setează `task="text-generation"` și `model="google/gemma-2-2b"`.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

Când ai mai mult de un input, transmite-le ca listă.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

Acest ghid îți va prezenta [`Pipeline`], îți va demonstra funcționalitățile sale și îți va arăta cum să configurezi diferiții săi parametri.

## Task-uri

[`Pipeline`] este compatibil cu multe task-uri de machine learning din diferite modalități. Transmite un input potrivit către pipeline și acesta se va ocupa de restul.

Iată câteva exemple despre cum să folosești [`Pipeline`] pentru diferite task-uri și modalități.

<hfoptions id="tasks">
<hfoption id="automatic speech recognition">

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</hfoption>
<hfoption id="image classification">

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```

</hfoption>
<hfoption id="visual question answering">

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</hfoption>
</hfoptions>

## Parametri

La minimum, [`Pipeline`] necesită doar un identificator de task, un model și input-ul potrivit. Dar există mulți parametri disponibili pentru a configura pipeline-ul, de la parametri specifici task-ului până la optimizarea performanței.

Această secțiune îți prezintă câțiva dintre cei mai importanți parametri.

### Device

[`Pipeline`] este compatibil cu multe tipuri de hardware, inclusiv GPU-uri, CPU-uri, Apple Silicon și altele. Configurează tipul de hardware cu parametrul `device`. Implicit, [`Pipeline`] rulează pe un CPU, ceea ce este indicat de `device=-1`.

<hfoptions id="device">
<hfoption id="GPU">

Pentru a rula [`Pipeline`] pe un GPU, setează `device` la id-ul de device CUDA asociat. De exemplu, `device=0` rulează pe primul GPU.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

De asemenea, poți lăsa [Accelerate](https://hf.co/docs/accelerate/index), o bibliotecă pentru antrenare distribuită, să aleagă automat cum să încarce și să stocheze weights modelului pe device-ul potrivit. Acest lucru este util în special dacă ai mai multe device-uri. Accelerate încarcă și stochează weights modelului mai întâi pe cel mai rapid device, iar apoi mută weights pe alte device-uri (CPU, hard drive) după cum este nevoie. Setează `device_map="auto"` pentru a lăsa Accelerate să aleagă device-ul.

> [!TIP]
> Asigură-te că ai instalat [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install).
>
> ```py
> !pip install -U accelerate
> ```

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
<hfoption id="Apple silicon">

Pentru a rula [`Pipeline`] pe Apple Silicon, setează `device="mps"`.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
</hfoptions>

### Inferență în batch-uri

[`Pipeline`] poate procesa și batch-uri de input-uri cu parametrul `batch_size`. Inferența în batch-uri poate îmbunătăți viteza, în special pe un GPU, dar acest lucru nu este garantat. Alte variabile precum hardware-ul, datele și modelul în sine pot afecta dacă inferența în batch-uri îmbunătățește viteza. Din acest motiv, inferența în batch-uri este dezactivată implicit.

În exemplul de mai jos, când există 4 input-uri și `batch_size` este setat la 2, [`Pipeline`] transmite modelului câte un batch de 2 input-uri pe rând.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device, batch_size=2)
pipeline(["the secret to baking a really good cake is", "a baguette is", "paris is the", "hotdogs are"])
[[{'generated_text': 'the secret to baking a really good cake is to use a good cake mix.\n\ni’'}],
 [{'generated_text': 'a baguette is'}],
 [{'generated_text': 'paris is the most beautiful city in the world.\n\ni’ve been to paris 3'}],
 [{'generated_text': 'hotdogs are a staple of the american diet. they are a great source of protein and can'}]]
```

Un alt caz de utilizare bun pentru inferența în batch-uri este transmiterea de date în flux (streaming) în [`Pipeline`].

```py
from transformers import pipeline
from accelerate import Accelerator
from transformers.pipelines.pt_utils import KeyDataset
import datasets

device = Accelerator().device

# KeyDataset este un utilitar care returnează elementul din dict-ul returnat de dataset
dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Ține cont de următoarele reguli generale pentru a determina dacă inferența în batch-uri poate ajuta la îmbunătățirea performanței.

1. Singura modalitate de a ști sigur este să măsori performanța pe modelul, datele și hardware-ul tău.
2. Nu folosi inferența în batch-uri dacă ești constrâns de latență (de exemplu, un produs de inferență live).
3. Nu folosi inferența în batch-uri dacă folosești un CPU.
4. Nu folosi inferența în batch-uri dacă nu cunoști `sequence_length` al datelor tale. Măsoară performanța, crește iterativ `sequence_length` și include verificări out-of-memory (OOM) pentru a te recupera din eșecuri.
5. Folosește inferența în batch-uri dacă `sequence_length` este regulat și continuă să îl crești până când ajungi la o eroare OOM. Cu cât GPU-ul este mai mare, cu atât inferența în batch-uri este mai utilă.
6. Asigură-te că poți gestiona erorile OOM dacă decizi să folosești inferența în batch-uri.

### Parametri specifici task-ului

[`Pipeline`] acceptă orice parametri care sunt suportați de fiecare pipeline individual de task. Asigură-te că verifici fiecare pipeline individual de task pentru a vedea ce tip de parametri sunt disponibili. Dacă nu găsești un parametru util pentru cazul tău de utilizare, nu ezita să deschizi un [issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml) pe GitHub pentru a-l solicita!

Exemplele de mai jos demonstrează unii dintre parametrii specifici task-ului disponibili.

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

Transmite parametrul `return_timestamps="word"` către [`Pipeline`] pentru a returna momentul în care a fost rostit fiecare cuvânt.

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamp="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)},
  {'text': ' that', 'timestamp': (1.92, 3.7)},
  {'text': ' one', 'timestamp': (3.7, 3.88)},
  {'text': ' day', 'timestamp': (3.88, 4.24)},
  {'text': ' this', 'timestamp': (4.24, 5.82)},
  {'text': ' nation', 'timestamp': (5.82, 6.78)},
  {'text': ' will', 'timestamp': (6.78, 7.36)},
  {'text': ' rise', 'timestamp': (7.36, 7.88)},
  {'text': ' up', 'timestamp': (7.88, 8.46)},
  {'text': ' and', 'timestamp': (8.46, 9.2)},
  {'text': ' live', 'timestamp': (9.2, 10.34)},
  {'text': ' out', 'timestamp': (10.34, 10.58)},
  {'text': ' the', 'timestamp': (10.58, 10.8)},
  {'text': ' true', 'timestamp': (10.8, 11.04)},
  {'text': ' meaning', 'timestamp': (11.04, 11.4)},
  {'text': ' of', 'timestamp': (11.4, 11.64)},
  {'text': ' its', 'timestamp': (11.64, 11.8)},
  {'text': ' creed.', 'timestamp': (11.8, 12.3)}]}
```

</hfoption>
<hfoption id="text generation">

Transmite `return_full_text=False` către [`Pipeline`] pentru a returna doar textul generat în loc de textul complet (prompt-ul și textul generat).

[`~TextGenerationPipeline.__call__`] suportă, de asemenea, argumente keyword suplimentare din metoda [`~GenerationMixin.generate`]. Pentru a returna mai mult de o secvență generată, setează `num_return_sequences` la o valoare mai mare decât 1.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
pipeline("the secret to baking a good cake is", num_return_sequences=4, return_full_text=False)
[{'generated_text': ' how easy it is for me to do it with my hands. You must not go nuts, or the cake is going to fall out.'},
 {'generated_text': ' to prepare the cake before baking. The key is to find the right type of icing to use and that icing makes an amazing frosting cake.\n\nFor a good icing cake, we give you the basics'},
 {'generated_text': " to remember to soak it in enough water and don't worry about it sticking to the wall. In the meantime, you could remove the top of the cake and let it dry out with a paper towel.\n"},
 {'generated_text': ' the best time to turn off the oven and let it stand 30 minutes. After 30 minutes, stir and bake a cake in a pan until fully moist.\n\nRemove the cake from the heat for about 12'}]
```

</hfoption>
</hfoptions>

## Chunk batching

Există unele situații în care trebuie să procesezi datele în chunk-uri.

- pentru unele tipuri de date, un singur input (de exemplu, un fișier audio foarte lung) poate fi nevoie să fie împărțit în mai multe părți (chunk-uri) înainte de a putea fi procesat
- pentru unele task-uri, precum clasificarea zero-shot sau question answering, un singur input poate necesita mai multe forward pass-uri, ceea ce poate cauza probleme cu parametrul `batch_size`

Clasa [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) este concepută pentru a gestiona aceste cazuri de utilizare. Ambele clase de pipeline sunt folosite în același mod, dar deoarece [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) poate gestiona automat batching-ul, nu trebuie să-ți faci griji cu privire la numărul de forward pass-uri pe care le declanșează input-urile tale. În schimb, poți optimiza `batch_size` independent de input-uri.

Exemplul de mai jos arată cum diferă de [`Pipeline`].

```py
# ChunkPipeline
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)

# Pipeline
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

## Seturi de date mari

Pentru inferență cu seturi de date mari, poți itera direct peste setul de date în sine. Acest lucru evită alocarea imediată de memorie pentru întregul set de date și nu trebuie să-ți faci griji cu privire la crearea de batch-uri manual. Încearcă [Inferența în batch-uri](#inferență-în-batch-uri) cu parametrul `batch_size` pentru a vedea dacă îmbunătățește performanța.

```py
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from accelerate import Accelerator
from datasets import load_dataset

device = Accelerator().device

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Alte modalități de a rula inferența pe seturi de date mari cu [`Pipeline`] includ folosirea unui iterator sau generator.

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## Modele mari

[Accelerate](https://hf.co/docs/accelerate/index) activează câteva optimizări pentru rularea modelelor mari cu [`Pipeline`]. Asigură-te mai întâi că Accelerate este instalat.

```py
!pip install -U accelerate
```

Setarea `device_map="auto"` este utilă pentru a distribui automat modelul mai întâi pe cele mai rapide device-uri (GPU-uri) înainte de a-l trimite către alte device-uri mai lente, dacă sunt disponibile (CPU, hard drive).

[`Pipeline`] suportă weights cu precizie redusă (torch.float16), care pot fi semnificativ mai rapide și economisesc memorie. Pierderea de performanță este neglijabilă pentru majoritatea modelelor, în special pentru cele mai mari. Dacă hardware-ul tău suportă acest lucru, poți activa în schimb torch.bfloat16 pentru un interval mai mare.

> [!TIP]
> Input-urile sunt convertite intern la torch.float16 și funcționează doar pentru modele cu un backend PyTorch.

În cele din urmă, [`Pipeline`] acceptă și modele cuantizate pentru a reduce și mai mult utilizarea memoriei. Asigură-te mai întâi că ai instalată biblioteca [bitsandbytes](https://hf.co/docs/bitsandbytes/installation), iar apoi adaugă `quantization_config` în `model_kwargs` în pipeline.

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```
