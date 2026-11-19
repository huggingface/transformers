<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Contribuția legacy la modele

> [!TIP]
> Încearcă mai întâi să adaugi modele noi cu abordarea mai [modulară](./modular_transformers). Asta face contribuirea cu un model la Transformers semnificativ mai ușoară!

Multe din modelele din Transformers sunt contribuite de developeri și cercetători. Ca proiect open-source, suntem investiți în a da putere comunității să adauge independent mai multe modele.

Când adaugi un model la Transformers, vei învăța:

- mai multe despre cele mai bune practici open-source
- despre arhitectura unui model
- despre principiile de design ale Transformers
- cum să testezi eficient modele mari
- cum să folosești utilitare Python precum [Black](https://black.readthedocs.io/en/stable/) și [Ruff](https://docs.astral.sh/ruff/) ca să creezi cod curat și lizibil

Este un proces provocator, dar satisfăcător.

Acest ghid te va conduce prin adăugarea unui model BrandNewLlama PyTorch de exemplu în Transformers. Înainte să începi, e o idee bună să te familiarizezi cu librăria.

## Prezentare generală Transformers

Transformers este o librărie cu opinii proprii, cu propria filozofie și alegeri de design. Aceste alegeri ne ajută să scalăm și să menținem Transformers sustenabil.

> [!TIP]
> Află mai multe despre principiile noastre de design în documentul [Philosophy].

Câteva din aceste alegeri de design sunt:

- compoziție > over-abstraction
- codul duplicat nu e întotdeauna rău dacă îmbunătățește semnificativ lizibilitatea și accesibilitatea
- fișierele de model sunt self-contained și tot codul necesar al modelului se găsește în fișierul `modeling_mymodel.py`

Aceste alegeri de design sunt importante *pentru toată lumea* care interacționează cu modelul. Este mai ușor de citit, înțeles și modificat.

Această secțiune descrie cum interacționează clasele de model și configurare și stilul de cod al Transformers.

### Model și configurare

Toate modelele Transformers moștenesc dintr-o clasă de bază [`PreTrainedModel`] și [`PreTrainedConfig`]. Configurarea este blueprint-ul modelului.

Nu există niciodată mai mult de două niveluri de abstractizare pentru niciun model, ca să menținem codul lizibil. Modelul de exemplu de aici, BrandNewLlama, moștenește din `BrandNewLlamaPreTrainedModel` și [`PreTrainedModel`]. Este important că un model nou depinde doar de [`PreTrainedModel`] ca să poată folosi metodele [`~PreTrainedModel.from_pretrained`] și [`~PreTrainedModel.save_pretrained`].

Alte funcții importante precum metoda forward sunt definite în fișierul `modeling.py`.

Head-urile specifice de model (de exemplu, clasificarea de secvențe sau modelarea limbajului) ar trebui să apeleze modelul de bază în forward pass, nu să moștenească din el, ca să menținem abstractizarea scăzută.

Modelele noi necesită o configurare, de exemplu `BrandNewLlamaConfig`, care este stocată ca atribut al [`PreTrainedModel`].

```py
model = BrandNewLlamaModel.from_pretrained("username/brand_new_llama")
model.config
```

[`PreTrainedConfig`] furnizează metodele [`~PreTrainedConfig.from_pretrained`] și [`~PreTrainedConfig.save_pretrained`].

Când folosești [`PreTrainedModel.save_pretrained`], apelează automat [`PreTrainedConfig.save_pretrained`] ca atât modelul, cât și configurarea să fie salvate împreună.

Un model este salvat într-un fișier `model.safetensors` și o configurare este salvată într-un fișier `config.json`.

### Stilul codului

Transformers preferă cod curat și lizibil față de un stil mai abstractizat. Câteva din alegerile de stil includ:

- Codul ar trebui să fie accesibil utilizatorilor non-anglofoni. Alege nume de variabile descriptive și evită abrevierile. De exemplu, "activation" este preferată față de "act". Numele de variabile cu o singură literă sunt puternic descurajate, cu excepția unui index într-un for loop.

- Codul explicit este preferat — chiar dacă e mai lung — față de codul mai scurt.

- Evită subclasarea [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html). Subclasează în schimb [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) ca codul să poată fi debugged rapid cu print statements sau breakpoints.

- Semnăturile funcțiilor ar trebui să aibă adnotări de tip. Altfel, folosește nume bune de variabile ca să fie mai ușor de înțeles.

## Issue pentru adăugarea unui model nou

Deschide un issue [New model addition](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&template=new-model-addition.yml) ca să adaugi un model specific.

> [!TIP]
> Filtrează după label-ul [New model](https://github.com/huggingface/transformers/labels/New%20model) pe GitHub ca să vizualizezi și să adaugi cereri de modele existente.

Acum este un moment bun să te familiarizezi cu BrandNewLlama. Este util să citești lucrarea de cercetare a modelului ca să înțelegi designul tehnic și implementarea. Nu trebuie neapărat să te îngrijorezi prea mult de detaliile teoretice. Concentrează-te pe cele practice. Folosește întrebările de mai jos ca să ghidezi lectura.

- Ce tip de model este BrandNewLlama? Este un model encoder, decoder sau encoder-decoder?
- Pentru ce task-uri poate fi folosit BrandNewLlama?
- Ce îl face pe BrandNewLlama diferit de alte modele?
- Ce modele din Transformers sunt cele mai similare cu BrandNewLlama?
- Ce tokenizer folosește BrandNewLlama?

Pe lângă a afla mai multe despre model, folosește sfaturile de mai jos ca să adaugi un model mai repede.

> [!TIP]
> Fiecare contributor are un stil și workflow unic pentru adăugarea modelelor în Transformers. Ca exemplu, aruncă o privire la cum a fost adăugat [Gemma](https://github.com/huggingface/transformers/pull/29167).

- Nu reinventa roata! Ia-ți timp să explorezi modelele și tokenizatoarele existente ca să vezi ce poți copia și reutiliza. [Grep](https://www.gnu.org/software/grep/) și [ripgrep](https://github.com/BurntSushi/ripgrep) sunt instrumente excelente pentru asta.
- Aceasta este mai degrabă o provocare de inginerie decât una de știință. Concentrează-te pe aspectele mai practice (configurarea unui mediu de debugging eficient, de exemplu) în loc de aspectele teoretice ale modelului.
- Nu fi timid să ceri ajutor! Suntem aici să te sprijinim. 🤗

## Mediu de dev

Dă click pe butonul **Fork** din repository-ul [Transformers](https://github.com/huggingface/transformers) ca să creezi propria copie pe care să lucrezi. Clonează repository-ul pe discul local și adaugă repository-ul de bază ca remote.

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

Creează un mediu virtual și efectuează o [instalare editabilă](./installation#instalare-editabilă) a librăriei cu dependencies "dev" sau de development.

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

Din cauza numărului de dependencies opționale pe măsură ce Transformers crește, această comandă poate eșua. În acest caz, instalează dependencies "quality". Asigură-te și că ai instalat un framework de deep learning.

```bash
pip install -e ".[quality]"
```

Întoarce-te la directorul părinte și clonează și instalează repository-ul original BrandNewLlama.

```bash
git clone https://github.com/org_that_created_brand_new_llama_org/brand_new_llama.git
cd brand_new_bert
pip install -e .
```

Întoarce-te la clona ta de Transformers ca să începi portarea BrandNewLlama.

```bash
cd transformers
```

Există două medii posibile de debugging pentru rularea modelului original: un notebook ([Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) sau [Jupyter](https://jupyter.org/)) sau un script Python local.

> [!WARNING]
> Nu recomandăm configurarea unui mediu GPU ca să rulezi modelul original, deoarece poate fi costisitor. Lucrează mai întâi într-un mediu CPU ca să verifici că modelul funcționează în Transformers. Odată ce funcționează, poți verifica pe un GPU.

Notebook-urile sunt excelente pentru executarea codului celulă cu celulă, ceea ce poate ajuta la separarea componentelor logice. Poate accelera și ciclurile de debugging deoarece rezultatele intermediare pot fi stocate. Poți partaja notebook-uri și când lucrezi cu alți contributori.

Dezavantajul este că dacă nu ești obișnuit cu ele, poate dura ceva timp să te acomodezi.

> [!TIP]
> Dacă arhitectura modelului este identică cu un model existent, sari înainte la adăugarea unui [script de conversie](#conversia-checkpoint-urilor), deoarece poți reutiliza arhitectura modelului existent.

Rulează comanda de mai jos ca să pornești și completezi chestionarul cu informații de bază despre noul model. Această comandă pornește procesul generând automat cod de model pe care va trebui să îl adaptezi.

```bash
transformers add-new-model-like
```

## Crearea unui pull request

Înainte să începi adaptarea codului, creează un pull request ca să urmărești progresul și să primești feedback de la echipa Transformers. Intitulează pull request-ul **[WIP] Add BrandNewLlama** ca să fie clar că este o lucrare în desfășurare.

Creează un branch cu un nume descriptiv din branch-ul tău main.

```bash
git checkout -b add_brand_new_bert
```

Fă commit la cod, apoi fetch și rebase pe branch-ul main.

```bash
git add .
git commit
git fetch upstream
git rebase upstream/main
```

Fă push la orice modificări pe branch-ul tău și dă click pe **Compare & pull request** ca să deschizi un pull request pe GitHub. Deschide pull request-ul ca *draft* ca să indici că este o lucrare în desfășurare.

```bash
git push -u origin a-descriptive-name-for-my-changes
```

Include membrii relevanți ai echipei Hugging Face adăugând handle-urile lor GitHub în pull request pentru întrebări, feedback, comentarii și review-uri. Direcționează membrii echipei la părți specifice din cod dând click pe tab-ul **Files changed**, apoi dând click pe **+** la stânga numărului de linie ca să adaugi un comentariu. Când o întrebare sau problemă este rezolvată, dă click pe **Resolve** ca să indici că problema este rezolvată. Asta menține conversația organizată și curată.

Amintește-ți să faci periodic commit și push la munca ta și să actualizezi cu branch-ul main curent.

```bash
git fetch upstream
git merge upstream/main
```

## Checkpoint-ul original

Ia-ți timp să lucrezi mai întâi la implementarea modelului original ca să înțelegi cum funcționează.

Aceasta poate fi dificilă dacă repository-ul modelului original nu are documentație sau dacă codebase-ul este complex. Dar ar trebui să folosești asta ca motivație pentru a implementa modelul în Transformers. Contribuția ta îl face mai accesibil și prietenos pentru toată lumea!

Orientează-te cu repository-ul original făcând următoarele.

- Localizează weights preantrenate.
- Dă-ți seama cum să încarci weights preantrenate în model.
- Dă-ți seama cum să rulezi tokenizer-ul independent de model.
- Urmărește un forward pass ca să înțelegi ce clase și funcții sunt necesare. Acestea sunt probabil singurele clase și funcții pe care va trebui să le implementezi.
- Localizează toate componentele importante (clasa model, subclasele modelului, layerul de self-attention etc.) ale modelului.
- Dă-ți seama cum să dai debug modelul în repository-ul original. Adaugă print statements, folosește debuggers interactive precum [ipdb](https://github.com/gotcha/ipdb) sau un IDE eficient precum [PyCharm](https://www.jetbrains.com/pycharm/).

Ultimul punct este deosebit de important deoarece vei avea nevoie de o înțelegere profundă a ce se întâmplă în interiorul modelului original înainte de a-l reimplementa în Transformers. Nu ezita să deschizi issues și pull request-uri în repository-ul original dacă întâmpini probleme.

Un prim pas bun este să încarci un checkpoint preantrenat *mic* și să încerci să reproduci un singur forward pass cu un vector de exemplu de întregi ca inputuri. De exemplu, în pseudocod, ar putea arăta astfel:

```py
model = BrandNewLlamaModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = [0, 4, 5, 2, 3, 7, 9]  # vector de input ids
original_output = model.generate(input_ids)
```

### Debugging

Dacă întâmpini probleme, va trebui să alegi una din următoarele strategii de debugging în funcție de codebase-ul modelului original.

<hfoptions id="debug-strategy">
<hfoption id="sub-components">

Această strategie se bazează pe descompunerea modelului original în sub-componente mai mici, cum ar fi atunci când codul poate fi rulat ușor în modul eager. Deși mai dificilă, există câteva avantaje ale acestei abordări.

1. Este mai ușor mai târziu să compari modelul original cu implementarea ta. Poți verifica automat că fiecare componentă individuală corespunde componentei corespunzătoare din implementarea Transformers. Asta e mai bine decât să te bazezi pe o comparație vizuală bazată pe print statements.
2. Este mai ușor să portezi componente individuale în loc de întregul model.
3. Este mai ușor să înțelegi cum funcționează un model descompunându-l în părți mai mici.
4. Este mai ușor să previi regresiile la o etapă ulterioară când modifici codul, datorită testelor componentă cu componentă.

> [!TIP]
> Consultă [integration checks](https://gist.github.com/LysandreJik/db4c948f6b4483960de5cbac598ad4ed) ELECTRA pentru un exemplu bun de cum să descompui un model în componente mai mici.

</hfoption>
<hfoption id="model and tokenizer">

Această strategie este viabilă când codebase-ul original este prea complex, permite doar rularea componentelor intermediare în modul compilat sau dacă este prea consumatoare de timp (poate chiar imposibil) să separi modelul în sub-componente mai mici.

De exemplu, implementarea MeshTensorFlow a lui [T5](https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow) este prea complexă și nu oferă o modalitate simplă de a descompune modelul în sub-componentele sale. În această situație, va trebui să te bazezi pe verificarea print statements.

</hfoption>
</hfoptions>

Oricare strategie alegi, este recomandat să dai debug mai întâi pentru layers inițiale și după pentru layers finale. Recuperează ieșirea, fie cu print statements fie cu funcții de sub-componentă, a layers următoare în această ordine.

1. input ids pasați modelului
2. word embeddings
3. inputul primului layer Transformer
4. ieșirea primului layer Transformer
5. ieșirea layers Transformer n-1 următoare
6. ieșirea întregului model

Input ids ar trebui să fie doar un array de întregi precum `input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]`.

Ieșirile layers-urilor constau adesea din array-uri float multidimensionale.

```py
[[
 [-0.1465, -0.6501,  0.1993,  ...,  0.1451,  0.3430,  0.6024],
 [-0.4417, -0.5920,  0.3450,  ..., -0.3062,  0.6182,  0.7132],
 [-0.5009, -0.7122,  0.4548,  ..., -0.3662,  0.6091,  0.7648],
 ...,
 [-0.5613, -0.6332,  0.4324,  ..., -0.3792,  0.7372,  0.9288],
 [-0.5416, -0.6345,  0.4180,  ..., -0.3564,  0.6992,  0.9191],
 [-0.5334, -0.6403,  0.4271,  ..., -0.3339,  0.6533,  0.8694]]],
```

Fiecare ieșire de model Transformers ar trebui să aibă o precizie sau toleranță de eroare de *1e-3*. Asta compensează orice diferențe de ieșire care apar din folosirea unui framework de librărie diferit. Compară ieșirile intermediare ale modelului original cu implementarea Transformers ca să te asiguri că sunt aproape identice. A avea un mediu de debugging *eficient* este crucial pentru acest pas.

Iată câteva sfaturi pentru un mediu de debugging eficient.

- Pentru a da debug rezultatelor intermediare, depinde de framework-ul de machine learning pe care îl folosește repository-ul modelului original. Pentru PyTorch, ar trebui să scrii un script care să descompună modelul original în sub-componente mai mici ca să recuperezi valorile intermediare.

- Este mai rapid să dai debug cu un checkpoint preantrenat mai mic față de un checkpoint mai mare unde forward pass-ul durează mai mult de 10 secunde. Dacă sunt disponibile doar checkpoint-uri mari, creează un model dummy cu weights inițializate aleatoriu și salvează acele weights ca să le compari cu implementarea Transformers.

- Găsește cea mai ușoară modalitate de a apela forward pass-ul modelului. Ideal, această funcție (poate fi numită `predict`, `evaluate`, `forward` sau `__call__`) ar trebui să apeleze forward pass-ul doar *o singură dată*. Este mai dificil să dai debug unei funcții care apelează forward pass-ul de mai multe ori.

- Separă tokenizarea de forward pass. Localizează unde un input string este schimbat în input ids în forward pass și pornește de aici. Poate fi nevoie să creezi un script mic sau să modifici codul original ca să introduci direct input ids în loc de un input string.

- Asigură-te că modelul *nu* este în modul de antrenare. Asta poate produce ieșiri aleatorii din cauza mai multor layers de dropout dintr-un model. Forward pass-ul din mediul tău de debugging ar trebui să fie *determinist* ca layers-urile de dropout să nu fie folosite.

Odată ce poți rula checkpoint-ul original, ești gata să începi adaptarea codului de model pentru Transformers.

## Adaptarea codului de model

Comanda `transformers add-new-model-like` ar trebui să fi generat un fișier de model și de configurare.

- `src/transformers/models/brand_new_llama/modeling_brand_new_llama.py`
- `src/transformers/models/brand_new_llama/configuration_brand_new_llama.py`

Codul generat automat din fișierul `modeling.py` are aceeași arhitectură ca Llama dacă ai răspuns că este un model decoder-only sau va avea aceeași arhitectură ca BART dacă ai răspuns că este un model encoder-decoder. Codul generat este doar un punct de plecare. Bazat pe cercetarea ta despre noul model, va trebui să implementezi acele modificări specifice adaptând codul generat. Asta poate implica modificări ale layerului de self-attention, ordinea layerului de normalizare și altele.

### Inițializarea modelului

La acest punct, codul tău nu trebuie să fie curat sau chiar complet corect. Este mai eficient să creezi rapid un prim draft și să îl îmbunătățești iterativ. Cel mai important lucru este că modelul tău poate fi instanțiat din Transformers. Comanda de mai jos creează un model din configurare cu weights aleatorii, verificând că metoda `__init__` funcționează.

```py
from transformers import BrandNewLlama, BrandNewLlamaConfig
model = BrandNewLlama(BrandNewLlamaConfig())
```

Inițializarea aleatorie are loc în metoda `_init_weights` a `BrandNewLlamaPreTrainedModel`. Toate modulele leaf sunt inițializate în funcție de variabilele de configurare.

```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        module.weight.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.zero_()
        module.weight.fill_(1.0)
```

Schema de inițializare poate arăta diferit dacă trebuie s-o adaptezi la modelul tău. De exemplu, [`Wav2Vec2ForPreTraining`] inițializează [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) în ultimele sale două layers liniare.

Flag-ul `_is_hf_initialized` se asigură că submodulul este inițializat doar o singură dată. Setarea `module.project_q` și `module.project_hid` la `True` asigură că inițializarea personalizată nu este suprascrisă mai târziu. Funcția `_init_weights` nu va fi aplicată la aceste module.

```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, Wav2Vec2ForPreTraining):
        module.project_hid.reset_parameters()
        module.project_q.reset_parameters()
        module.project_hid._is_hf_initialized = True
        module.project_q._is_hf_initialized = True
    elif isinstance(module, nn.Linear):
        module.weight.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.zero_()
```

### Conversia checkpoint-urilor la Transformers

Checkpoint-ul original trebuie convertit într-un checkpoint compatibil cu Transformers.

> [!TIP]
> Încearcă să cauți un script de conversie existent pe care să îl copiezi, adaptezi și reutilizezi pentru modelul tău!
>
> - Dacă portezi un model din TensorFlow la PyTorch, un bun punct de plecare poate fi [scriptul de conversie](https://github.com/huggingface/transformers/blob/7acfa95afb8194f8f9c1f4d2c6028224dbed35a2/src/transformers/models/bert/modeling_bert.py#L91) BERT.
> - Dacă portezi un model din PyTorch la PyTorch, un bun punct de plecare poate fi [scriptul de conversie](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py) BART.

Asigură-te că **toate** weights necesare sunt inițializate și printează toate weights de checkpoint care nu au fost folosite pentru inițializare pentru a te asigura că modelul a fost convertit corect.

Poți întâlni instrucțiuni de formă greșită sau atribuiri de nume greșite în timpul conversiei. Asta se datorează cel mai probabil parametrilor incorecți în `BrandNewLlamaConfig`, arhitecturii greșite, unui bug în metoda `init` a implementării tale sau trebuie să transpoziționezi unul dintre weights checkpoint-ului.

Continuă să iterezi pe secțiunea [Adaptarea codului de model](#adaptarea-codului-de-model) până când toate weights de checkpoint se încarcă corect. Odată ce poți încărca un checkpoint în modelul tău, salvează-l într-un folder. Acesta ar trebui să conțină un fișier `model.safetensors` și un fișier `config.json`.

```py
model.save_pretrained("/path/to/converted/checkpoint/folder")
```

Ca să ajuți cu conversia, secțiunea următoare descrie pe scurt cum PyTorch stochează și definește weights și numele layer-urilor.

#### Weights și numele layer-urilor PyTorch

Este util să creezi un model PyTorch de bază ca să înțelegi cum sunt definite numele layers-urilor și cum sunt inițializate weights.

```py
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 10)
        self.intermediate = nn.Linear(10, 10)
        self.layer_norm = nn.LayerNorm(10)
```

Numele layers-urilor PyTorch sunt definite de numele atributului de clasă al layerului (`dense`, `intermediate`, `layer_norm`). Creează o instanță a `SimpleModel` ca să umpli toate layers-urile cu weights aleatorii.

```py
model = SimpleModel()
print(model)
SimpleModel(
  (dense): Linear(in_features=10, out_features=10, bias=True)
  (intermediate): Linear(in_features=10, out_features=10, bias=True)
  (layer_norm): LayerNorm((10,), eps=1e-05, elementwise_affine=True)
)
```

Valorile weights unui layer specific sunt inițializate aleatoriu.

```py
print(model.dense.weight.data)
tensor([[-0.0818,  0.2207, -0.0749, -0.0030,  0.0045, -0.1569, -0.1598,  0.0212,
         -0.2077,  0.2157],
        ...
        [-0.1492, -0.1616,  0.1057,  0.1950, -0.2807, -0.2710, -0.1586,  0.0739,
          0.2220,  0.2358]]).
```

În scriptul de conversie, weights aleatorii ar trebui înlocuite cu weights exacte din layer-ul corespunzător în checkpoint-ul original.

```py
# recuperează weights layer-ului potrivit cu algoritm recursiv
layer_name = "dense"
pretrained_weight = array_of_dense_layer

model_pointer = getattr(model, "dense")
model_pointer.weight.data = torch.from_numpy(pretrained_weight)
```

Verifică că weights inițializate aleatoriu și weights corespunzătoare din checkpoint-ul preantrenat au **forma** și **numele** identice. Adaugă instrucțiuni assert pentru formă și printează numele de weights din checkpoint.

```py
assert (
    model_pointer.weight.shape == pretrained_weight.shape
), f"Pointer shape of random weight {model_pointer.shape} and array shape of checkpoint weight {pretrained_weight.shape} mismatched"

logger.info(f"Initialize PyTorch weight {layer_name} from {pretrained_weight.name}")
```

Când forma sau numele nu corespund, poate că ai atribuit un weight de checkpoint greșit unui layer inițializat aleatoriu. O formă greșită poate fi cauzată de parametrii `BrandNewLlama` ce nu corespund exact cu parametrii modelului original. Ar putea fi și că implementarea layerului PyTorch necesită ca weights să fie transpoziționate mai întâi.

### Implementarea forward pass-ului

Forward pass-ul ar trebui implementat următor dacă modelul se încarcă corect. Ia câteva inputuri și returnează ieșirea modelului.

```py
model = BrandNewLlamaModel.from_pretrained("/path/to/converted/checkpoint/folder")
input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]
output = model.generate(input_ids).last_hidden_states
```

Nu te descuraja dacă forward pass-ul tău nu este identic cu ieșirea modelului original sau dacă returnează o eroare. Verifică că forward pass-ul nu aruncă erori. Asta se datorează adesea că dimensiunile sunt greșite sau că se folosește tipul de date greșit ([torch.long](https://pytorch.org/docs/stable/generated/torch.Tensor.long.html) în loc de [torch.float32](https://pytorch.org/docs/stable/tensors.html)).

Ieșirea ta ar trebui să aibă o precizie de *1e-3*. Asigură-te că formele de ieșire și valorile de ieșire sunt identice. Motive comune pentru care ieșirile nu sunt identice includ:

- Câteva layers nu au fost adăugate (un layer de activare sau o conexiune reziduală).
- Matricea de word embedding nu este legată.
- Se folosesc embeddings poziționale greșite deoarece implementarea originală include un offset.
- Dropout este aplicat în timpul forward pass-ului. Remediază această eroare asigurându-te că `model.training` este `False` și pasând `self.training` la [torch.nn.functional.dropout](https://pytorch.org/docs/stable/nn.functional.html?highlight=dropout#torch.nn.functional.dropout).

Compară forward pass-ul modelului original cu implementarea ta ca să verifici dacă există diferențe. Ideal, dai debug și printezi ieșirile intermediare ale ambelor implementări ale forward pass-ului ca să identifici unde implementarea originală diferă de a ta.

1. Asigură-te că `input_ids` hardcoded în ambele implementări sunt identice.
2. Verifică că ieșirile primei transformări ale `input_ids` (de obicei word embeddings) sunt identice și lucrează prin layers până la ultimul.

Orice diferență între cele două implementări ar trebui să pointeze la bug-ul din implementarea ta.

Una din cele mai bune strategii este să adaugi multe print statements în aceleași poziții în ambele implementări, și apoi să le elimini succesiv când produc valori identice pentru ieșirile intermediare.

Când ambele implementări produc aceeași ieșire, verifică că ieșirile sunt în cadrul unei precizii de *1e-3*.

```py
torch.allclose(original_output, output, atol=1e-3)
```

Acesta este de obicei cel mai dificil pas al procesului. Felicitări dacă ai ajuns până aici!

Și dacă ești blocat sau te lupți cu acest pas, nu ezita să ceri ajutor pe pull request-ul tău.

### Adăugarea testelor de model

Deși modelul funcționează, mai trebuie să adaugi teste ca să te asiguri că este compatibil cu Transformers. Testele sunt importante pentru că ajută utilizatorii să înțeleagă munca ta uitându-se la teste specifice și pentru că previn spargerea modelului în viitor dacă se fac modificări.

[Cookiecutter](https://cookiecutter.readthedocs.io/en/stable/) ar trebui să fi adăugat un fișier de test pentru modelul tău. Rulează fișierul de test de mai jos ca să te asiguri că toate testele comune trec.

```bash
pytest tests/models/brand_new_llama/test_modeling_brand_new_llama.py
```

Testele de integrare ar trebui adăugate mai întâi pentru că servesc același scop ca scripturile de debugging pe care le-ai folosit mai devreme ca să implementezi noul model în Transformers. Un șablon al acelor teste de model, `BrandNewLlamaModelIntegrationTests`, a fost adăugat de Cookiecutter și ar trebui completat. Ca să te asiguri că trece, rulează comanda de mai jos.

<hfoptions id="integration-test">
<hfoption id="macOS">

```bash
RUN_SLOW=1 pytest -sv tests/models/brand_new_llama/test_modeling_brand_new_llama.py::BrandNewLlamaModelIntegrationTests
```

</hfoption>
<hfoption id="Windows">

```bash
SET RUN_SLOW=1 pytest -sv tests/models/brand_new_llama/test_modeling_brand_new_llama.py::BrandNewLlamaModelIntegrationTests
```

</hfoption>
</hfoptions>

Toate features-urile unice ale lui BrandNewLlama ar trebui testate într-un test separat sub `BrandNewLlamaModelTester/BrandNewLlamaModelTest`. Acest test este adesea trecut cu vederea, dar este extrem de important pentru că:

- ajută la transferul cunoștințelor pe care le-ai dobândit în timpul procesului către comunitate, arătând cum funcționează features-urile noi ale modelului
- contributorii viitori pot testa rapid modificările la model rulând aceste teste speciale

## Implementarea tokenizer-ului

> [!TIP]
> Recomandăm adăugarea unui tokenizer rapid ([`PreTrainedTokenizerFast`]) ca să oferi utilizatorilor cele mai bune performanțe. Nu ezita să dai tag lui [@ArthurZucker](https://github.com/ArthurZucker) sau [@itazap](https://github.com/itazap) în PR-ul tău pentru ajutor cu adăugarea [`PreTrainedTokenizerFast`].

Cu modelul rezolvat, e timpul să te concentrezi pe tokenizer. Tokenizer-ul ar trebui să fie identic sau foarte similar cu un tokenizer existent în Transformers.

Găsește și încarcă fișierul original de tokenizer în implementarea ta. Creează un script în repository-ul original care primește un string și returnează `input_ids`. Pseudocodul ar trebui să arate similar cu codul de mai jos.

```py
input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
model = BrandNewLlamaModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = model.tokenize(input_str)
```

Poate fi nevoie să cauți în repository-ul original ca să găsești funcția corectă de tokenizer sau să modifici tokenizer-ul existent din clona ta a repository-ului original ca să returneze doar `input_ids`. Scriptul pentru tokenizer-ul tău ar trebui să arate similar cu cel de mai jos.

```py
from transformers import BrandNewLlamaTokenizer

input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
tokenizer = BrandNewLlamaTokenizer.from_pretrained("/path/to/tokenizer/folder/")
input_ids = tokenizer(input_str).input_ids
```

Când ambele implementări au aceleași `input_ids`, adaugă un fișier de test pentru tokenizer. Acesta este analog cu fișierele de test de modelare. Fișierele de test pentru tokenizer ar trebui să conțină câteva teste de integrare hardcoded.

## Implementarea procesorului de imagini

> [!TIP]
> Procesoarele de imagini folosesc acum o arhitectură bazată pe backend. Backend-ul implicit este [`TorchvisionBackend`], care folosește librăria [torchvision](https://pytorch.org/vision/stable/index.html) și poate efectua procesarea imaginilor pe GPU. Un backend alternativ PIL/NumPy ([`PilBackend`]) este și el furnizat. Ambele backend-uri sunt importate din `image_processing_backends`. Nu ezita să dai tag lui [@yonigozlan](https://github.com/yonigozlan) pentru ajutor.

Deși acest exemplu nu include un procesor de imagini, poate fi nevoie să implementezi unul dacă modelul tău necesită inputuri de imagini. Procesorul de imagini este responsabil de convertirea imaginilor într-un format potrivit pentru modelul tău. Înainte de a implementa unul nou, verifică dacă un procesor de imagini existent din librăria Transformers poate fi reutilizat, deoarece multe modele partajează tehnici similare de procesare a imaginilor. Reține că poți folosi și [modular](./modular_transformers) pentru procesoarele de imagini ca să reutilizezi componente existente.

Dacă ai nevoie să implementezi un procesor de imagini nou, fiecare model are două fișiere de procesor:

- `image_processing_<model>.py`: procesorul **implicit** cu backend torchvision (`<Model>ImageProcessor`), moștenind din [`TorchvisionBackend`]. Acesta înlocuiește vechiul procesor "fast".
- `image_processing_pil_<model>.py`: procesorul alternativ PIL/NumPy (`<Model>ImageProcessorPil`), moștenind din [`PilBackend`]. Acesta înlocuiește vechiul procesor "slow".

Fișierul cu backend torchvision definește și orice clasă de kwargs personalizate pe care fișierul PIL o importă. Ambele fișiere folosesc decoratorul `@auto_docstring` — nu adăuga docstring-uri de clasă manuale. Consultă [IMAGE_PROCESSOR_REFACTORING_GUIDE.md](https://github.com/huggingface/transformers/blob/main/IMAGE_PROCESSOR_REFACTORING_GUIDE.md) pentru un walkthrough pas cu pas și exemple complete.

Adaugă teste pentru procesorul de imagini în `tests/models/your_model_name/test_image_processing_your_model_name.py`. Aceste teste ar trebui să fie similare cu cele pentru alte procesoare de imagini și să verifice că procesorul de imagini gestionează corect inputurile de imagini. Dacă procesorul tău de imagini include features sau metode de procesare unice, asigură-te că adaugi teste specifice pentru acelea.

## Implementarea procesorului

Dacă modelul tău acceptă modalități multiple, precum text și imagini, trebuie să adaugi un procesor. Procesorul centralizează preprocesarea diferitelor modalități înainte de a le pasa modelului.

Procesorul ar trebui să apeleze procesoarele specifice modalității corespunzătoare în interiorul funcției `__call__` ca să gestioneze corect fiecare tip de input. Asigură-te că verifici procesoarele existente din librărie ca să înțelegi structura așteptată. Transformers folosește convenția de mai jos în semnătura funcției `__call__`.

```python
def __call__(
    self,
    images: ImageInput = None,
    text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
    audio=None,
    videos=None,
    **kwargs: Unpack[YourModelProcessorKwargs],
) -> BatchFeature:
    ...
```

`YourModelProcessorKwargs` este un `TypedDict` care include toate argumentele tipice de procesare și orice argumente suplimentare pe care un procesor specific le poate necesita.

Adaugă teste pentru procesor în `tests/models/your_model_name/test_processor_your_model_name.py`. Aceste teste ar trebui să fie similare cu cele pentru alte procesoare și să verifice că procesorul gestionează corect diferitele modalități.

## Teste de integrare

Acum că ai un model și tokenizer, adaugă teste de integrare end-to-end pentru model și tokenizer în `tests/models/brand_new_llama/test_modeling_brand_new_llama.py`.

Testul ar trebui să furnizeze un exemplu semnificativ text-la-text ca să arate că modelul funcționează conform așteptărilor. De exemplu, poți include o pereche de traducere sursă-țintă, o pereche articol-rezumat sau o pereche întrebare-răspuns.

Dacă checkpoint-ul nu a fost fine-tuned pe un task downstream, atunci testele de model sunt suficiente.

În final, încearcă să te asiguri că testele pot rula pe un GPU adăugând instrucțiuni `.to(self.device)` la tensorii interni ai modelului. Dacă nu ai acces la un GPU, ne putem ocupa de asta.

## Adăugarea documentației

Modelul tău este util doar dacă utilizatorii știu cum să îl folosească. De aceea este important să adaugi documentație și docstring-uri. Cookiecutter a adăugat un fișier șablon, `docs/source/model_doc/brand_new_llama.md`, pe care îl poți completa cu informații despre modelul tău.

Aceasta este de obicei prima interacțiune a unui utilizator cu un model, deci documentația ar trebui să fie clară și concisă. Este adesea foarte util să adaugi exemple de cum ar trebui folosit modelul.

Asigură-te că docstring-urile sunt adăugate la `src/transformers/models/brand_new_llama/modeling_brand_new_llama.py` și includ toate inputurile și ieșirile necesare. Consultă [ghidul](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification) nostru de scriere a documentației și docstring-urilor.

## Refactorizare

E timpul să faci ordine și să te asiguri că stilul codului este consistent cu restul librăriei. Rulează comanda de mai jos ca să remediezi automat stilurile incorecte.

```bash
make style
```

Ca să verifici că stilul codului trece verificările de calitate, rulează comanda de mai jos.

```bash
make check-repo
```

Pot exista alte teste sau verificări eșuate (docstring lipsă sau denumire incorectă) pe pull request-ul tău din cauza testelor stricte de design ale Transformers. Putem ajuta cu aceste probleme dacă ești blocat.

După ce te-ai asigurat că codul rulează corect, poate vrei să îl refactorizezi ca să fie mai lizibil sau mai curat.

## Încărcarea pe Hub

Convertește și încarcă toate checkpoint-urile pe [Hub](https://hf.co/models). Adaugă un model card pentru a oferi mai multă transparență și context despre model. Model card-ul ar trebui să evidențieze caracteristici specifice ale unui checkpoint, cum a fost antrenat modelul și exemple de cod pentru cum să îl folosești.

> [!TIP]
> În multe cazuri, adăugarea unui notebook interactiv pe care utilizatorii îl pot rula este o modalitate excelentă de a arăta cum să folosești modelul pentru inferență sau să îl fine-tunezi pe un task downstream. Deși nu este obligatoriu, includerea unui notebook poate duce la o adoptare mai mare a modelului.

Ar trebui să consulți și echipa Transformers ca să decideți un nume potrivit pentru model și să obții drepturile de acces necesare pentru a încărca modelul.

Folosește metoda [`~PreTrainedModel.push_to_hub`] ca să încarci modelul.

```py
brand_new_bert.push_to_hub("brand_new_llama")
```

Consultă ghidul [Partajare](./model_sharing) pentru mai multe informații despre încărcarea modelelor pe Hub.

## Integrarea PR-ului

Ești în sfârșit gata să integrezi pull request-ul și să adaugi oficial modelul în Transformers! Asigură-te că toate testele trec și că toate comentariile și feedback-ul au fost adresate.

Felicitări pentru adăugarea unui nou model în Transformers! 🥳

Aceasta este o contribuție foarte semnificativă. Munca ta face Transformers mai accesibil pentru developeri și cercetători din întreaga lume. Ar trebui să fii mândru de contribuția ta și să îți împartășești realizarea cu comunitatea!

## Cronologia adăugării de modele

Există patru cronologii pentru adăugările de modele în funcție de contributorul modelului și cererea comunității pentru o arhitectură.

- **integrare day-0**: Dacă plănuiești să ai un release Transformers-first, aceasta este o opțiune excelentă deoarece putem asigura că documentația este clară și putem optimiza modelul tău cât mai mult (quantization, FlashAttention, KV-cache etc.). Putem și să te ajutăm să adaugi modelul, să oferim review-uri timpurii și să ne asigurăm că funcționează conform așteptărilor.

  Contactează transformers@huggingface.co cu câteva zile (preferabil săptămâni) în avans, mai ales dacă o arhitectură este deosebit de nouă, ca să asiguri integrarea modelului. Vom lucra împreună pe un fork privat al Transformers până când checkpoint-ul și release-ul tău sunt gata.

- **integrare în aceeași săptămână**: Modelele cu cereri/cerere semnificativă sunt de obicei adăugate în aceeași săptămână dacă autorul modelului nu ia legătura.

  Folosește [issue tracker-ul](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&projects=&template=new-model-addition.yml) ca să soliciți un model specific de adăugat. Cu cât mai multă activitate pe issue, cu atât mai rapid și mai probabil îl vom integra.

- **integrare post-release**: Modelele fără cereri/cerere populară sau dacă nu avem banda necesară să le integrăm sunt adăugate post-release.

  Aceasta este o bună oportunitate dacă ești interesat să contribui un model la Transformers. Aruncă o privire la issues deschise cu tag-ul ["New model"](https://github.com/huggingface/transformers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+model%22). Nu ezita să încerci mai întâi modelele cele mai solicitate ca să multiplici impactul contribuției tale. Vom fi acolo să te ajutăm la fiecare pas!

- **Release Hub-first**: Feature-ul [remote-code](./models#modele-personalizate) al Transformers permite proiectelor bazate pe Transformers să fie partajate direct pe Hub. Aceasta este o bună opțiune dacă nu ai banda necesară să adaugi un model direct în Transformers.

  Dacă un model devine foarte popular, atunci este foarte probabil că îl vom integra noi înșine în Transformers ca să îi asigurăm un suport mai bun (documentație, întreținere, optimizare etc.). Un release Hub-first este cel mai puțin restrictiv mod de a adăuga un model.

## Vezi și

- [Regulile de structură a modelului](./modeling_rules) — reguli statice aplicate pe toate fișierele `modeling_*.py` și `configuration_*.py`. Rulează `make typing` ca să le verifici înainte de a deschide un PR.
- [Verificările pentru pull request](./pr_checks) — referință completă pentru ce verificări CI rulează pe PR-ul tău și cum să le treci.
