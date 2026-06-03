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

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Verificările pentru pull request

Când deschizi un pull request, CI-ul Hugging Face rulează mai multe verificări care trebuie să treacă înainte ca PR-ul tău să poată fi integrat.

- [Remedierea CI-ului](#remedierea-ci-ului) listează comenzile de rulat local ca să treci verificările.
- Secțiunile care urmează descriu ce validează fiecare verificare, cum ar fi [calitatea codului](#calitatea-codului) și [consistența repository-ului](#consistența-repository-ului).
- [Teste](#teste) descrie ce teste sunt rulate, categoriile de teste și testele slow.

## Remedierea CI-ului

În cele mai multe cazuri, `make style` este suficient pentru a trece verificarea de calitate a codului, care este cel mai comun eșec.

```bash
make style
```

Pentru eșecuri la consistența repository-ului, copii sau fișiere auto-generate, rulează `make fix-repo`. Remediază stilul, copiile, docstring-urile și fișierele auto-generate într-o singură trecere.

```bash
make fix-repo
```

Pentru o modificare mai mare, secvența de trei comenzi de mai jos acoperă fiecare verificare. Este mai grea, dar prinde tot înainte să dai push.

```bash
make fix-repo   # auto-remediază tot ce poate fi auto-remediat
make typing     # verifică tipurile și structura modelului, remediază manual orice erori
make check-repo # verifică că toate verificările trec, remediază ce mai rămâne
```

`make typing` prinde erori de tip și violări de structură a modelului pe care le remediezi manual. `make check-repo` face o trecere finală read-only ca să confirmi că totul este pregătit.

> [!NOTE]
> CI-ul este uneori instabil. Dacă o verificare eșuează pe ceva fără legătură cu modificarea ta, contactează un maintainer ca să o reruleze.

## Calitatea codului

Verificarea calității codului acoperă formatarea, importurile, type checking-ul și regulile de structură a modelului. Corespunde cu `make fix-repo` și `make typing`.

`make style` (inclus în `make fix-repo`) auto-remediază linting-ul și formatarea [Ruff](https://docs.astral.sh/ruff/), ordinea importurilor din `__init__.py` și consistența auto-mapărilor.

`make typing` efectuează type checking cu [ty](https://docs.astral.sh/ty/) și validează regulile TRansFormers (TRF), care acoperă convențiile de denumire ale claselor de config și semnăturile `forward()`. Erorile de tip și violările TRF raportează un număr de regulă specific și trebuie remediate manual. Regulile se găsesc în repository-ul [mlinter](https://github.com/huggingface/transformers-mlinter). Rulează `python -m utils.mlinter --list-rules` ca să vezi fiecare regulă TRF, sau `python -m utils.mlinter --rule TRFXXX` ca să vezi documentația completă pentru o regulă specifică.

Dacă o regulă TRF necesită o excepție, alege una din aceste opțiuni (vezi [Suprimarea violărilor](./modeling_rules#suprimarea-violărilor) pentru mai multe detalii).

- Adaugă numele modelului tău în lista `allowlist_models` pentru regula relevantă din `utils/mlinter/rules.toml`. Folosește asta când întregul fișier de model necesită o excepție.
- Adaugă `# trf-ignore: TRFXXX` pe aceeași linie cu construcția semnalată, sau pe linia imediat de deasupra. Folosește asta când doar o construcție semnalată necesită o excepție.

## Consistența repository-ului

Verificarea consistenței repository-ului este similară cu `make check-repo`, cu excepția că se oprește la primul eșec. Menține repository-ul intern consistent în categoriile de mai jos: obiectele publice rămân importabile, codul copiat rămâne sincronizat cu sursa sa, iar fișierele auto-generate (dummy-uri, doctests, metadate) reflectă starea curentă a codului. Pentru modelele noi, verifică și că fiecare clasă de model nouă este înregistrată în auto-mapări.

| Categorie | Ce validează | Auto-remediat? |
|---|---|---|
| Fișiere init | Fiecare obiect public nou trebuie să apară atât în `_import_structure` (lazy loading), cât și în blocul `if TYPE_CHECKING` (importuri pentru type checker) din `__init__.py` | Manual |
| Copii și modular | Blocurile `# Copied from` corespund sursei lor și fișierele generate din modular sunt actualizate | `make fix-repo` |
| Docstring-uri și docs | Docstring-urile argumentelor corespund semnăturilor funcțiilor și cuprinsului documentației | `make fix-repo` |
| Fișiere auto-generate | Dummy-uri, pipeline typing, lista doctest, metadate, tabel de dependențe | `make fix-repo` |
| Validarea config | Clasele de config au checkpoint-uri valide în docstring-uri și atributele de config corespund fișierului de modelare | Manual |

## Teste

CI rulează un subset țintit de teste bazat pe ce modifică PR-ul tău. CI rulează testele într-o ordine ușor randomizată cu [pytest-random-order](https://github.com/jbasko/pytest-random-order) ca să prindă testele cuplate. Rularea printează seed-ul random la început ca să poți repeta aceeași ordine cu `--random-order-seed=<seed>`.

Dacă un test trece local pe GPU, dar eșuează în CI, setează `TRANSFORMERS_TEST_DEVICE="cpu"` ca să verifici dacă poți reproduce eșecul pe CPU.

```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/models/my_model/ -v
```

Secțiunile de mai jos explică cum funcționează selecția testelor, ce job-uri rulează și cum să gestionezi testele slow.

### Selecția testelor

CI nu rulează suita completă de teste la fiecare PR. CI sare peste testele decorate cu `@slow` la fiecare PR, iar un maintainer le declanșează pe GPU odată ce PR-ul este în review.

`utils/tests_fetcher.py` urmărește dependențele de import din fișierele tale modificate ca să identifice testele afectate și le rulează doar pe acelea. Prinde și regresiile în alte modele când atingi utilitare partajate. Fetcher-ul printează ce fișiere s-au modificat și ce teste sunt impactate, apoi scrie lista în `tests_torch_test_list.txt`.

Folosește fetcher-ul ca să replici exact ce rulează CI.

```bash
python utils/tests_fetcher.py
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_preparation/tests_torch_test_list.txt)
```

> [!TIP]
> Modificarea fișierelor de bază precum `modeling_utils.py` sau `generation/utils.py` declanșează toate testele de model, nu doar subsetul afectat.

Poți și să rulezi fiecare test pentru modelul tău necondiționat. O rulare directă este o verificare locală mai rapidă, dar nu va prinde regresiile în alte modele cauzate de cod partajat pe care l-ai atins.

```bash
pytest tests/models/my_model/ -v
```

### Categoriile de job-uri de test

Testele sunt împărțite pe job-uri CI paralele, iar fiecare job preia fișiere după pattern de cale. Job-urile relevante pentru un PR de model sunt:

- `tests_torch`: teste de modelare (`tests/models/*/test_modeling_*.py`)
- `tests_tokenization`: teste pentru tokenizer (`tests/models/*/test_tokenization_*.py`)
- `tests_processors`: teste pentru procesatoare și feature extractors (`tests/models/*/test_(processing|image_processing|feature_extractor)_*.py`)
- `tests_generate`: teste de generare
- `pipelines_torch`: teste de pipeline
- `tests_training_ci`: teste pentru loop-ul de antrenare
- `tests_tensor_parallel_ci`: teste de tensor parallelism

### Testele slow

Rulările regulate CI sar peste testele decorate cu `@slow`. Acestea descarcă checkpoint-uri reale sau necesită resurse de calcul semnificative, deci rulează pe instanțe GPU, iar maintainerii le declanșează odată ce PR-ul tău este în review.

Testele slow rulează pe un NVIDIA A10, iar rezultatele numerice pot varia ușor între hardware-ul CI și mașina ta locală. Maintainerii ajustează de obicei acele valori în teste dacă e necesar când se adaugă un model nou.

Rulează testele slow local cu comanda de mai jos.

```bash
RUN_SLOW=1 python -m pytest tests/models/my_model/ -v
```

### Build-ul documentației

Un job `build_pr_documentation` construiește și generează un preview al documentației. Un bot postează un link de preview în PR-ul tău, iar verificarea trebuie să treacă înainte de integrare. Cele mai multe eșecuri sunt o intrare lipsă în `toctree`. Ca să construiești documentația local, vezi [README.md](https://github.com/huggingface/transformers/tree/main/docs) din folder-ul docs.

## Sintaxa `# Copied from`

> [!WARNING]
> Pentru modele noi, preferă întotdeauna [workflow-ul modular](modular_transformers) (`modular_*.py`) față de `# Copied from`. Evită `# Copied from` ori de câte ori e posibil.

Mecanismul `# Copied from` menține codul copiat sincronizat cu sursa sa. Când `make fix-repo` rulează, verifică fiecare bloc `# Copied from` și îl actualizează să corespundă originalului, deci editările din interiorul unui bloc `# Copied from` sunt suprascrise. Editează sursa în schimb și lasă `make fix-repo` să propage modificarea.

Formele de bază ale `# Copied from` includ următoarele.

```py
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput

# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta

# Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification with Bert->MobileBert all-casing
```

Sintaxa `with model->newModel` aplică înlocuiri de string-uri după copiere. Separă înlocuirile multiple cu virgule, aplicate de la stânga la dreapta. Opțiunea `all-casing` înlocuiește toate variantele de scriere deodată (`Bert`, `bert`, `BERT` devin `MobileBert`, `mobilebert`, `MOBILEBERT`).
