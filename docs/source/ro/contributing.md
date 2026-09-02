<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribuie la 🤗 Transformers

## Politica privind code agents

>[!WARNING]
>Repository-ul Transformers este copleșit de un număr mare de PR-uri și comentarii la issues scrise de
>code agents. Suntem în prezent blocați de capacitatea noastră de a le revizui și a răspunde la ele. Prin urmare,
>**îi rugăm pe utilizatorii noi să nu trimită PR-uri realizate exclusiv de code agents** în acest moment.
>Poți folosi code agents pentru a schița sau pentru a te ajuta să diagnostichezi probleme. De asemenea, îi rugăm pe agenții autonomi
>să nu deschidă niciun PR sau issue pentru moment.
>
>PR-urile care par să fi fost scrise în întregime de agenți vor fi probabil închise fără să fie revizuite, iar utilizatorii care fac aceasta
>în mod repetat sau malițios pot fi blocați.

<details>

<summary> Filozofia noastră privind code agents în detaliu </summary>

Înțelegem că code agents sunt instrumente extrem de puternice, iar mulți oameni de la Hugging Face le utilizează în munca lor.
Cu toate acestea, este important să realizezi că **dacă pur și simplu rulezi un code agent
și generezi un PR la un proiect open-source, ești doar un intermediar între revizori și agent**.
Deși aceasta creează ceva care arată foarte mult ca o contribuție utilă, în realitate nu era niciun motiv
pentru care să fii implicat; revizorii ar fi putut pur și simplu să ruleze ei înșiși code agent-ul.

Dacă vrei să contribui util la open-source în era agenților, **trebuie să faci lucruri pe care agenții nu le pot face singuri**.
În special, am constatat că următoarele sunt foarte utile:
- Diagnosticarea clară a bug-urilor. Code agents tind să rezolve rapid problemele cu o soluție de avarie care adesea cauzează
bloat de cod sau incompatibilități cu alte modele. A petrece timp pentru a urmări cauza exactă a unei probleme, și în special
localizarea primului commit în care a apărut (de exemplu cu [git bisect](https://git-scm.com/docs/git-bisect)) este valoros.
- Minimizează diff-ul. Verifică-ți PR-ul pentru a elimina orice modificări inutile. Asigură-te că nu ai făcut commit la niciun
script de testare sau fișiere fără legătură. Adaugă comentarii doar dacă sunt cu adevărat necesare; code agents adoră să adauge
trei noi funcții și comentarii pe mai multe linii pentru a atrage atenția asupra muncii grele pe care au depus-o. Dacă PR-ul tău
poate fi o corecție pe 1 linie, fă-l o corecție pe 1 linie. Aceasta face PR-ul mult mai ușor de revizuit și îmbunătățește șansele ca acesta să fie acceptat.
- Ia-ți timp să reproduci problema. Foarte des când un utilizator raportează o problemă, aceasta este de fapt cauzată de probleme
de mediu pe mașina sa, sau diagnostichează greșit problema și sugerează o soluție invalidă. Mulți code agents au prea multă
încredere în comentariile utilizatorilor, ceea ce rezultă în soluții proaste, uneori pentru probleme care
nu există! Scrierea unui script simplu de reproducere și rularea lui pentru a te asigura că vezi problema este valoroasă.
- Compară cu alte modele. Repository-ul Transformers este foarte mare, iar multe modele fac lucruri similare. Când
corectezi un bug, este valoros să verifici dacă bug-ul există și în alte modele. Dacă PR-ul tău spune
"rezolvat folosind aceeași abordare ca (alt model)", cu un link la codul relevant, aceasta este foarte utilă pentru maintaineri,
deoarece ne spune că corectura este probabil corectă și compatibilă cu restul codebase-ului. Code agents privesc adesea
codul "în mod restrâns" și fac o corecție care determină modelele să divergă de la restul codebase-ului.
- Evită PR-urile mici sau de tip "busywork". În trecut le acceptam, dar dat fiind actualul aflux, pur și simplu nu
avem timp pentru mici modificări de stil sau corectarea typo-urilor din comentarii. Poți oferi valoare dincolo de un code
agent pur și simplu prin a avea bun gust în privința a ceea ce este cu adevărat important.
- Verifică testele local și în CI. Înainte de a deschide un PR, rulează `make fix-repo` și folosește `utils/tests_fetcher.py` pentru
a vedea o listă de teste care acoperă fișierele pe care le-ai modificat în branch-ul PR-ului tău. Rulează acele teste local și asigură-te
că trec înainte de a deschide un PR. După ce deschizi PR-ul, verifică că CI-ul este verde și rezolvă orice probleme înainte
de a contacta pe cineva pentru revizuire! Aceasta reduce mult spam-ul de notificări, ceea ce menține maintainerii sănătoși.

Te rugăm să ții cont că aceasta este o eră interesantă, în schimbare rapidă, dar provocatoare pentru dezvoltarea open-source, și într-adevăr
pentru industria software în ansamblu. Vom actualiza probabil rapid aceste ghiduri pe măsură ce învățăm mai multe despre
gestionarea eficientă a code agents. Ai răbdare cu noi dacă revizuirile sunt mai lente decât de obicei, sau dacă unele
PR-uri sunt închise fără revizuire!

</details>

## Bine ai venit în comunitatea 🤗 Transformers!

Toată lumea este binevenită să contribuie, iar noi apreciem contribuția fiecăruia. Contribuțiile de cod
nu sunt singura modalitate de a ajuta comunitatea. Răspunsurile la întrebări, ajutorarea altora
și îmbunătățirea documentației sunt de asemenea valori imense.

Ne ajutați și dacă răspândiți vestea! Menționați biblioteca în postările de blog
despre proiectele extraordinare pe care le-a făcut posibile, postați pe Twitter de fiecare dată când v-a
ajutat, sau pur și simplu acordați o ⭐️ repository-ului pentru a spune mulțumesc.

Indiferent cum alegi să contribui, te rugăm să fii atent și să respecți
[codul nostru de conduită](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md).

**Acest ghid a fost puternic inspirat de minunatul [ghid scikit-learn pentru contribuții](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

## Modalități de a contribui

Există mai multe moduri în care poți contribui la 🤗 Transformers:

* Remediază problemele existente în codul curent.
* Trimite issues legate de bug-uri sau funcții noi dorite.
* Implementează modele noi.
* Contribuie la exemple sau la documentație.

Dacă nu știi de unde să începi, există o listă specială [Good First
Issue](https://github.com/huggingface/transformers/contribute). Aceasta îți va oferi o listă de
issues deschise prietenoase pentru începători și te va ajuta să începi să contribui la open-source. Cel mai bun mod de a face aceasta este să deschizi un Pull Request și să îl legi de issue-ul la care vrei să lucrezi. Încercăm să acordăm prioritate PR-urilor deschise deoarece putem urmări cu ușurință progresul corecturii, iar dacă contribuitorul nu mai are timp, altcineva poate prelua PR-ul.

Pentru ceva puțin mai provocator, poți arunca și o privire la lista [Good Second Issue](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue). În general, dacă simți că știi ce faci, mergi înainte și te vom ajuta să ajungi acolo! 🚀

> Toate contribuțiile sunt la fel de valoroase pentru comunitate. 🥰

## Remedierea problemelor existente

Dacă observi o problemă în codul existent și ai o soluție în minte, nu ezita să [începi să contribui](#crearea-unui-pull-request) și deschide un Pull Request!

## Trimiterea unui issue legat de bug sau a unei cereri de funcție

Fă tot posibilul să urmezi aceste ghiduri când trimiți un issue legat de bug sau o cerere de funcție. Aceasta ne va face mai ușor să revenim la tine rapid și cu feedback bun.

### Ai găsit un bug?

Biblioteca 🤗 Transformers este robustă și fiabilă datorită utilizatorilor care raportează problemele pe care le întâmpină.

Înainte de a raporta un issue, am aprecia cu adevărat dacă ai putea **să te asiguri că bug-ul nu a fost deja raportat** (folosește bara de căutare pe GitHub la Issues). Issue-ul tău ar trebui să fie legat și de bug-uri din bibliotecă în sine, nu din codul tău. Dacă nu ești sigur dacă bug-ul este în codul tău sau în bibliotecă, te rugăm să întrebi mai întâi în [forum](https://discuss.huggingface.co/) sau pe [Discord-ul](https://discord.com/invite/hugging-face-879548962464493619) nostru. Aceasta ne ajută să răspundem mai rapid la problemele legate de bibliotecă față de întrebările generale.

> [!TIP]
> Avem un [bot de documentație](https://huggingface.co/spaces/huggingchat/hf-docs-chat) și te încurajăm să adresezi toate întrebările acolo. Există întotdeauna posibilitatea ca bug-ul tău să poată fi rezolvat cu un simplu flag 👾🔫

Odată ce ai confirmat că bug-ul nu a fost deja raportat, te rugăm să incluzi următoarele informații în issue-ul tău pentru ca să îl putem rezolva rapid:

* **Tipul și versiunea OS**-ului tău, și versiunile **Python** și **PyTorch** când este cazul.
* Un snippet de cod scurt, independent, care ne permite să reproducem bug-ul în
  mai puțin de 30s.
* Traceback-ul *complet* dacă este aruncată o excepție.
* Atașează orice alte informații suplimentare, precum screenshots, pe care crezi că ar putea ajuta.

Pentru a obține automat versiunile OS și software, rulează următoarea comandă:

```bash
transformers env
```

Poți rula, de asemenea, aceeași comandă din rădăcina repository-ului:

```bash
python src/transformers/commands/transformers_cli.py env
```

### Vrei o funcție nouă?

Dacă există o funcție nouă pe care ai dori să o vezi în 🤗 Transformers, te rugăm să deschizi un issue și să descrii:

1. Care este *motivația* din spatele acestei funcții? Este legată de o problemă sau frustrare cu biblioteca? Este o funcție legată de ceva de care ai nevoie pentru un proiect? Este ceva la care ai lucrat și crezi că ar putea beneficia comunitatea?

   Indiferent ce este, am dori să aflăm despre aceasta!

2. Descrie funcția solicitată cu cât mai multe detalii posibil. Cu cât poți să ne spui mai multe despre aceasta, cu atât mai bine te putem ajuta.
3. Furnizează un *snippet de cod* care demonstrează utilizarea funcției.
4. Dacă funcția este legată de un articol, te rugăm să incluzi un link.

Dacă issue-ul tău este bine scris, suntem deja la 80% din drum până la momentul în care îl creezi.

Am adăugat [template-uri](https://github.com/huggingface/transformers/tree/main/templates) pentru a te ajuta să începi cu issue-ul tău.

## Vrei să implementezi un model nou?

Modele noi sunt lansate constant și dacă vrei să implementezi un model nou, te rugăm să furnizezi următoarele informații:

* O scurtă descriere a modelului și un link la articol.
* Link la implementare dacă este open-source.
* Link la model weights dacă sunt disponibile.

Dacă ești dispus să contribui modelul tu însuți, anunță-ne pentru ca să te putem ajuta să îl adaugi la 🤗 Transformers!

Avem un ghid tehnic pentru [cum să adaugi un model la 🤗 Transformers](https://huggingface.co/docs/transformers/modular_transformers).

### Lista de verificare pentru contribuții de modele viziune-limbaj

Dacă vrei să contribui cu un **model viziune-limbaj** (sau orice model multimodal care procesează imagini/videoclipuri), te rugăm să urmezi această listă de verificare. Maintainerii o vor folosi pentru a revizui PR-ul tău, iar completarea acestor pași va crește semnificativ probabilitatea ca PR-ul tău să fie merged rapid.

**Listă de verificare obligatorie pentru toate contribuțiile de modele viziune-limbaj:**

☐ **1. Implementează un fișier modular**

Toate modelele noi ar trebui să folosească pattern-ul de arhitectură modulară. Creează un fișier `modular_<model_name>.py` folosind convertorul de modele modular:

- Folosește CLI-ul, [`transformers add-new-model-like`](https://github.com/huggingface/transformers/blob/main/src/transformers/cli/add_new_model_like.py) pentru a genera un schelet modular și a începe
- Tot codul ar trebui să fie în fișierul modular dacă este posibil. Modelarea trebuie să fie în acesta, este preferat ca și configurația să fie în acesta. [Ghidul modular](https://huggingface.co/docs/transformers/modular_transformers#implementing-a-modular-file) arată o modalitate rapidă de a configura un fișier modular.
- Reutilizează pattern-urile existente din modele similare pe cât posibil
- Poți face modelul compatibil cu motoare de inferență precum vLLM sau SGLang și activa integrarea fără efort. Consultă cerințele specifice pentru implementarea modelului în ["Transformers modeling backend"](https://huggingface.co/docs/transformers/transformers_as_backend#multimodal-models)

Pentru a verifica că fișierul tău modular este corect, rulează:

```bash
python utils/modular_model_converter.py <model_name>
```

Aceasta va genera fișierele separate (`modeling_*.py`, `configuration_*.py`, etc.) din fișierul tău modular. CI-ul va impune că aceste fișiere generate corespund fișierului tău modular.

☐ **2. Adaugă procesoare de imagini (pentru modele de imagini)**

Dacă modelul tău procesează imagini, implementează atât un procesor suportat de torchvision (implicit, accelerat GPU) cât și un procesor suportat de PIL (alternativa):

- Procesorul backend torchvision (`<Model>ImageProcessor`) moștenește din `TorchvisionBackend` și se află în `image_processing_<model>.py`
- Procesorul backend PIL (`<Model>ImageProcessorPil`) moștenește din `PilBackend` și se află în `image_processing_pil_<model>.py`
- Ambele sunt importate din `image_processing_backends`; clasa PIL kwargs este definită în fișierul torchvision și importată de fișierul PIL
- Consultă ghidul detaliat în [IMAGE_PROCESSOR_REFACTORING_GUIDE.md](https://github.com/huggingface/transformers/blob/main/IMAGE_PROCESSOR_REFACTORING_GUIDE.md)
- Exemple: `CLIPImageProcessor` / `CLIPImageProcessorPil`, `DonutImageProcessor` / `DonutImageProcessorPil`

☐ **3. Creează un script de conversie a weights**

Adaugă un script `convert_<model_name>_to_hf.py` care convertește weights originale ale modelului în formatul HuggingFace:

- Scriptul ar trebui să gestioneze încărcarea checkpoint-ului, maparea cheilor și salvarea în format HF
- Include exemple de utilizare și documentație în script
- Exemple: [`convert_llava_onevision_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/convert_llava_onevision_weights_to_hf.py), [`convert_idefics2_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/convert_idefics2_weights_to_hf.py)

☐ **4. Adaugă teste de integrare cu potrivire exactă a output-urilor**

Cel puțin, adaugă o clasă `IntegrationTest` care testează generarea end-to-end (procesare și modelare) cu potrivire **exactă** a output-urilor:

- Pentru modele generative: testează că textul generat corespunde exact output-ului așteptat
- Pentru modele non-generative: testează că logit-urile de output corespund valorilor așteptate
- Testele ar trebui să folosească checkpoint-uri reale (încarcă în 4-bit sau jumătate de precizie dacă checkpoint-ul este prea mare pentru CI runnerii noștri) și input-uri reale
- Pattern exemplu:

```python
class MyModelIntegrationTest(unittest.TestCase):
    @slow
    def test_model_integration(self):
        model = MyModelForConditionalGeneration.from_pretrained("org/model-name")
        processor = AutoProcessor.from_pretrained("org/model-name")

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_TEXT = "exact expected output"
        self.assertEqual(processor.decode(output[0]), EXPECTED_TEXT)
```

Consultă `tests/models/llava_onevision/test_modeling_llava_onevision.py` pentru exemple complete.

☐ **5. Actualizează documentația**

Adaugă sau actualizează documentația modelului:

- Creează dacă CLI-ul nu a creat `docs/source/en/model_doc/<model_name>.md` cu exemple de utilizare
- Include descrierea modelului, link la articol și utilizarea de bază cu `Pipeline` și `AutoModel`
- Adaugă modelul la fișierele TOC corespunzătoare

☐ **6. Caută pattern-uri reutilizabile**

Biblioteca are 400+ modele cu multe pattern-uri stabilite:

- Caută modele similare (e.g., alte modele viziune-limbaj)
- Reutilizează mecanisme de attention, implementări de layers și pattern-uri de procesare
- Verifică modele precum LLaVA, Idefics2, Fuyu pentru pattern-uri viziune-limbaj
- Folosește decoratorii furnizați precum (`auto_docstring`, `can_return_tuple`, `capture_outputs`, `merge_with_config_defaults` și `_can_record_outputs`) unde este relevant.
- Nu reinventa roata

☐ **7. Rulează verificările de calitate și citește output-ul**

Înainte de a trimite PR-ul tău, instalează dependencies pentru calitate și rulează suita completă de verificări:

```bash
pip install -e ".[quality]"
make style
```

**Important**: Ia-ți timp să citești output-ul `make style`. Acesta va:

- Linta și formata automat codul tău
- Rula verificări de consistență (importuri, docstrings, etc.)
- Afișa orice probleme rămase care necesită corecturi manuale

Toate verificările trebuie să treacă înainte ca PR-ul tău să poată fi merged.

**Dacă această listă de verificare este completă, PR-ul tău are o probabilitate foarte mare de a fi merged!** Urmarea acestor pași face munca maintainerilor mult mai ușoară și va reduce numărul de iterații de revizuire, ducând munca ta importantă acolo mai repede.

#### Listă de verificare copiabilă pentru maintaineri

Iată o versiune condensată pe care maintainerii o pot copia în PR-uri:

```markdown
## Multimodal Model Addition Checklist

Please ensure your PR completes all following items. See the [full checklist](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#vision-language-model-contribution-checklist) for details.

- [ ] **Modular file**: `modular_<model_name>.py` implemented and verified with `python utils/modular_model_converter.py <model_name>`
- [ ] **Image processors**: Torchvision backend (`<Model>ImageProcessor` from `TorchvisionBackend`) and PIL backend (`<Model>ImageProcessorPil` from `PilBackend`) both implemented (see [IMAGE_PROCESSOR_REFACTORING_GUIDE.md](https://github.com/huggingface/transformers/blob/main/IMAGE_PROCESSOR_REFACTORING_GUIDE.md))
- [ ] **Conversion script**: `convert_<model_name>_to_hf.py` added with usage examples
- [ ] **Integration tests**: End-to-end tests with exact output matching (text or logits)
- [ ] **Documentation**: Model docs added/updated in `docs/source/en/model_doc/`
- [ ] **Pattern reuse**: Verified against similar models (LLaVA, Idefics2, etc.)
- [ ] **Quality checks**: `make style` passes with no errors

```

## Vrei să adaugi documentație?

Căutăm mereu îmbunătățiri ale documentației care să o facă mai clară și mai exactă. Te rugăm să ne anunți cum poate fi îmbunătățită documentația, precum typo-uri și orice conținut care lipsește, este neclar sau inexact. Vom fi bucuroși să facem modificările sau să te ajutăm să faci o contribuție dacă ești interesat!

Pentru mai multe detalii despre cum să generezi, construiești și scrii documentația, aruncă o privire la [README-ul](https://github.com/huggingface/transformers/tree/main/docs) documentației.

## Programare cu AI agents

Acest repository păstrează configurația AI agent în `.ai/` și expune fișierele locale de agent prin symlinks.

Skills-urile pot fi expuse agenților rulând `make codex` sau `make claude`

Cursor citește `AGENTS.md` și citește skills-urile din căile Claude sau Codex, deci configurarea repository-ului
pentru Claude sau Codex va funcționa pentru Claude.

## Crearea unui Pull Request

Înainte de a scrie orice cod, îți recomandăm cu tărie să cauți prin PR-urile sau issues existente pentru a te asigura că nimeni nu lucrează deja la același lucru. Dacă nu ești sigur, este întotdeauna o idee bună să deschizi un issue pentru a obține feedback.

Vei avea nevoie de competențe de bază în `git` pentru a contribui la
🤗 Transformers. Deși `git` nu este cel mai ușor instrument de utilizat, are cel mai detaliat
manual. Tastează `git --help` într-un shell și bucură-te! Dacă preferi cărțile, [Pro
Git](https://git-scm.com/book/en/v2) este o referință foarte bună.

Vei avea nevoie de **[Python 3.9](https://github.com/huggingface/transformers/blob/main/setup.py#L449)** sau o versiune mai nouă pentru a contribui la 🤗 Transformers. Urmează pașii de mai jos pentru a începe să contribui:

1. Fă fork la [repository](https://github.com/huggingface/transformers) dând click pe butonul
   **[Fork](https://github.com/huggingface/transformers/fork)** pe pagina repository-ului. Aceasta creează o copie a codului
   sub contul tău de utilizator GitHub.

2. Clonează fork-ul pe discul tău local și adaugă repository-ul de bază ca remote:

   ```bash
   git clone git@github.com:<your Github handle>/transformers.git
   cd transformers
   git remote add upstream https://github.com/huggingface/transformers.git
   ```

3. Creează un nou branch pentru a-ți păstra modificările de dezvoltare:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Nu** lucra pe branch-ul `main`!

4. Configurează un mediu de dezvoltare rulând următoarea comandă într-un virtual environment:

   ```bash
   pip install -e ".[dev]"
   ```

   Dacă 🤗 Transformers era deja instalat în virtual environment, elimină-l
   cu `pip uninstall transformers` înainte de a-l reinstala în modul editabil
   cu flag-ul `-e`.

   În funcție de OS-ul tău, și deoarece numărul de dependencies opționale ale Transformers crește, ai putea întâmpina o eroare cu această comandă. În acest caz, instalează PyTorch și execută:

   ```bash
   pip install -e ".[quality]"
   ```

   ceea ce ar trebui să fie suficient pentru majoritatea cazurilor de utilizare.

5. Dezvoltă funcțiile în branch-ul tău.

   Pe măsură ce lucrezi la codul tău, ar trebui să te asiguri că suita de teste
   trece. Rulează testele afectate de modificările tale astfel:

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   Pentru mai multe informații despre teste, consultă
   ghidul [Testing](https://huggingface.co/docs/transformers/testing).

   🤗 Transformers se bazează pe `black` și `ruff` pentru a formata codul sursă
   în mod consistent. După ce faci modificări, aplică corecturi automate de stil și verificări de cod
   care nu pot fi automatizate dintr-o singură mișcare cu:

   ```bash
   make style
   ```

   🤗 Transformers folosește și `ruff` și câteva scripturi personalizate pentru a verifica greșelile de codare. Controalele de
   calitate sunt rulate de CI, dar poți rula aceleași verificări cu:

   ```bash
   make check-repo
   ```

   Pentru a afla mai multe despre aceste verificări și cum să rezolvi orice probleme cu ele, consultă
   ghidul [Checks on a Pull Request](https://huggingface.co/docs/transformers/pr_checks).

   Dacă modifici documente din directorul `docs/source`, asigură-te că documentația poate fi în continuare construită. Această verificare va rula și în CI când deschizi un pull request. Pentru a rula o verificare locală
   instalează [doc-builder-ul](https://github.com/huggingface/doc-builder).

   ```bash
   pip install ".[docs]"
   ```

   Rulează următoarea comandă din root-ul repository-ului:

   ```bash
   doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
   ```

   Aceasta va construi documentația în folderul `~/tmp/test-build` unde poți inspecta fișierele
   Markdown generate cu editorul tău preferat. Poți, de asemenea, previzualiza documentele pe GitHub când deschizi un pull request.

   Dacă adaugi sau editezi exemple rulabile în documentele Markdown, marchează gardurile Python cu `runnable` sau
   `runnable:<label>` și rulează-le local cu `pytest`:

   ```bash
   pytest -q docs/source/en/my_page.md
   pytest -q docs/source/en/
   ```

   Pentru sintaxa completă rulabilă, inclusiv blocuri de continuare, `# pytest-decorator:` și
   `# doc-builder: hide`, consultă
   [ghidul doc-builder pentru blocuri de cod rulabile](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md).

   Odată ce ești mulțumit de modificările tale, adaugă fișierele modificate cu `git add` și
   înregistrează modificările tale local cu `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Te rugăm să îți amintești să scrii [mesaje de commit bune](https://chris.beams.io/posts/git-commit/)
   pentru a comunica clar modificările pe care le-ai făcut!

   Pentru a menține copia ta a codului actualizată cu repository-ul original, dă rebase branch-ului tău pe `upstream/branch` *înainte* de a deschide un pull request sau dacă un maintainer cere asta:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Trimite modificările tale pe branch-ul tău:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   Dacă ai deschis deja un pull request, va trebui să forțezi push-ul cu flag-ul `--force`. Altfel, dacă pull request-ul nu a fost încă deschis, poți pur și simplu să trimiți modificările în mod normal.

6. Acum poți merge la fork-ul tău al repository-ului pe GitHub și da click pe **Pull Request** pentru a deschide un pull request. Asigură-te că bifezi toate căsuțele din [lista noastră de verificare](#lista-de-verificare-pentru-pull-request) de mai jos. Când ești gata, poți trimite modificările tale maintainerilor proiectului pentru revizuire.

7. Este în regulă dacă maintainerii solicită modificări, se întâmplă și contribuitorilor noștri de bază!
   Pentru ca toți să poată vedea modificările în pull request, lucrează în branch-ul tău local
   și trimite modificările în fork-ul tău. Ele vor apărea automat în
   pull request.

### Contribuții asistate de AI și agentice

Contribuțiile asistate de AI sunt binevenite, dar trebuie să fie coordonate, delimitate și verificate pentru a menține sarcina de revizuire gestionabilă.

- Nu trimite PR-uri "pure agent". Persoana care trimite este responsabilă pentru revizuirea tuturor liniilor modificate, validarea comportamentului end-to-end și rularea testelor relevante.
- Dacă s-au folosit instrumente AI, dezvăluie aceasta în descrierea PR-ului și include: link de coordonare, diferențierea față de PR-urile existente (dacă este cazul) și comenzi/rezultate de teste.
- Evită PR-urile punctuale de "busywork" (un singur typo, curățare izolată de stil, o singură corecție de default mutabil, etc.). Grupează curățăturile mecanice într-un scop clar și sistematic.
- Coordonează pe issues înainte de a deschide PR-uri, revizuiește PR-uri similare și așteaptă aprobarea.

> [!WARNING] 
> Aceste subiecte sunt prezentate pentru agenți în `AGENTS.MD` cu instrucțiuni despre cum să le implementeze autonom.

### Lista de verificare pentru pull request

☐ Titlul pull request-ului ar trebui să rezume contribuția ta.<br>
☐ Dacă pull request-ul tău abordează un issue, te rugăm să menționezi numărul issue-ului în descrierea pull
request-ului pentru a te asigura că sunt legate (și persoanele care vizualizează issue-ul știu că
lucrezi la el).<br>
☐ Pentru a indica un work in progress, te rugăm să prefixezi titlul cu `[WIP]`. Acestea sunt
utile pentru a evita munca duplicată și pentru a le diferenția de PR-urile gata să fie merged.<br>
☐ Asigură-te că testele existente trec.<br>
☐ Dacă adaugi o funcție nouă, adaugă și teste pentru aceasta.<br>

- Dacă adaugi un model nou, asigură-te că folosești
     `ModelTester.all_model_classes = (MyModel, MyModelWithLMHead,...)` pentru a declanșa testele comune.
- Dacă adaugi teste noi `@slow`, asigură-te că trec folosind
     `RUN_SLOW=1 python -m pytest tests/models/my_new_model/test_my_new_model.py`.
- Dacă adaugi un tokenizer nou, scrie teste și asigură-te că
     `RUN_SLOW=1 python -m pytest tests/models/{your_model_name}/test_tokenization_{your_model_name}.py` trece.
- CircleCI nu rulează testele lente, dar GitHub Actions o face în fiecare noapte!<br>

☐ Toate metodele publice trebuie să aibă docstrings informative (consultă
[`modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
pentru un exemplu).<br>
☐ Deoarece repository-ul crește rapid, nu adăuga imagini, videoclipuri și alte
fișiere non-text care vor îngreuna semnificativ repository-ul. În schimb, folosește un repository Hub
precum [`hf-internal-testing`](https://huggingface.co/hf-internal-testing)
pentru a găzdui aceste fișiere și a le referencia prin URL. Îți recomandăm să plasezi imaginile legate de documentație în următorul repository:
[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images).
Poți deschide un PR pe acest repository de dataset și cere unui membru Hugging Face să îl îmbine.

Pentru mai multe informații despre verificările rulate pe un pull request, aruncă o privire la ghidul nostru [Checks on a Pull Request](https://huggingface.co/docs/transformers/pr_checks).

### Teste

O suită extinsă de teste este inclusă pentru a testa comportamentul bibliotecii și mai multe exemple. Testele de bibliotecă se găsesc în
folder-ul [tests](https://github.com/huggingface/transformers/tree/main/tests) și testele de exemple în
folder-ul [examples](https://github.com/huggingface/transformers/tree/main/examples).

Preferăm `pytest` și `pytest-xdist` deoarece este mai rapid. Din root-ul
repository-ului, specifică un *path către un subfolder sau un fișier de test* pentru a rula testul:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
```

Similar, pentru directorul `examples`, specifică o *cale către un subfolder sau fișier de test* pentru a rula testul. De exemplu, următoarea comandă testează subfolderul de clasificare text din directorul PyTorch `examples`:

```bash
pip install -r examples/xxx/requirements.txt  # necesar doar prima dată
python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

De fapt, acesta este modul în care comenzile noastre `make test` și `make test-examples` sunt implementate (fără a include `pip install`)!

Poți, de asemenea, să specifici un set mai mic de teste pentru a testa doar funcția la care lucrezi.

În mod implicit, testele lente sunt omise, dar poți seta variabila de mediu `RUN_SLOW` la
`yes` pentru a le rula. Aceasta va descărca mulți gigabytes de modele, deci asigură-te că
ai suficient spațiu pe disc, o conexiune bună la internet sau multă răbdare!

<Tip warning={true}>

Ține minte să specifici un *path către un subfolder sau un fișier de test* pentru a rula testul. Altfel, vei rula toate testele din folderul `tests` sau `examples`, ceea ce va dura foarte mult timp!

</Tip>

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/models/my_new_model
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/text-classification
```

Ca și testele lente, există și alte variabile de mediu disponibile care nu sunt activate implicit în timpul testării:

- `RUN_CUSTOM_TOKENIZERS`: Activează testele pentru tokenizere personalizate.

Mai multe variabile de mediu și informații suplimentare se găsesc în [testing_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py).

🤗 Transformers folosește `pytest` doar ca runner de teste. Nu folosește nicio funcție specifică
`pytest` în suita de teste în sine.

Aceasta înseamnă că `unittest` este suportat în totalitate. Iată cum să rulezi teste cu
`unittest`:

```bash
python -m unittest discover -s tests -t . -v
python -m unittest discover -s examples -t examples -v
```

### Ghid de stil

Pentru documentstrings, 🤗 Transformers urmează [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Consultă [ghidul nostru de scriere a documentației](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification)
pentru mai multe informații.

### Development pe Windows

Pe Windows (cu excepția cazului în care lucrezi în [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) sau WSL), trebuie să configurezi git pentru a transforma terminările de linie Windows `CRLF` în terminările de linie Linux `LF`:

```bash
git config core.autocrlf input
```

O modalitate de a rula comanda `make` pe Windows este cu MSYS2:

1. [Descarcă MSYS2](https://www.msys2.org/), și presupunem că este instalat în `C:\msys64`.
2. Deschide linia de comandă `C:\msys64\msys2.exe` (ar trebui să fie disponibilă din meniul **Start**).
3. Rulează în shell: `pacman -Syu` și instalează `make` cu `pacman -S make`.
4. Adaugă `C:\msys64\usr\bin` la variabila de mediu PATH.

Poți acum folosi `make` din orice terminal (PowerShell, cmd.exe, etc.)! 🎉

### Sincronizarea unui repository fork cu upstream main (repository-ul Hugging Face)

Când actualizezi branch-ul main al unui repository fork, te rugăm să urmezi acești pași pentru a evita notificarea repository-ului upstream care adaugă note de referință la fiecare PR upstream și trimite notificări inutile developerilor implicați în aceste PR-uri.

1. Când este posibil, evită sincronizarea cu upstream-ul prin folosirea unui branch și PR pe repository-ul fork. În schimb, dă merge direct în branch-ul main al fork-ului.
2. Dacă un PR este absolut necesar, folosește pașii următori după ce faci checkout pe branch-ul tău:

   ```bash
   git checkout -b your-branch-for-syncing
   git pull --squash --no-commit upstream main
   git commit -m '<mesajul tău fără referințe Github>'
   git push --set-upstream origin your-branch-for-syncing
   ```
