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

# Controlli su una Pull Request

Quando apri una pull request sui 🤗 Transformers, vengono eseguiti un discreto numero di controlli per assicurarsi che la patch che stai aggiungendo non stia rompendo qualcosa di esistente. Questi controlli sono di quattro tipi:
- test regolari
- costruzione della documentazione
- stile del codice e della documentazione
- coerenza generale del repository

In questo documento, cercheremo di spiegare quali sono i vari controlli e le loro ragioni, oltre a spiegare come eseguire il debug locale se uno di essi fallisce sulla tua PR.

Nota che tutti richiedono un'installazione dev:

```bash
pip install transformers[dev]
```

o un'installazione modificabile:

```bash
pip install -e .[dev]
```

all'interno del repo Transformers.

## Tests

Tutti i job che iniziano con `ci/circleci: run_tests_` eseguono parti della suite di test dei Transformers. Ognuno di questi job si concentra su una parte della libreria in un determinato ambiente: per esempio `ci/circleci: run_tests_pipelines_tf` esegue il test delle pipeline in un ambiente in cui è installato solo TensorFlow.

Nota che per evitare di eseguire i test quando non ci sono cambiamenti reali nei moduli che si stanno testando, ogni volta viene eseguita solo una parte della suite di test: viene eseguita una utility per determinare le differenze nella libreria tra prima e dopo la PR (ciò che GitHub mostra nella scheda "Files changes") e sceglie i test che sono stati impattati dalla diff. Questa utility può essere eseguita localmente con:

```bash
python utils/tests_fetcher.py
```

dalla root del repo Transformers. Di seguito ciò che farà:

1. Controlla per ogni file nel diff se le modifiche sono nel codice o solo nei commenti o nelle docstrings. Vengono mantenuti solo i file con modifiche reali al codice.
2. Costruisce una mappa interna che fornisce per ogni file del codice sorgente della libreria tutti i file su cui ha un impatto ricorsivo. Si dice che il modulo A ha un impatto sul modulo B se il modulo B importa il modulo A. Per l'impatto ricorsivo, abbiamo bisogno di una catena di moduli che va dal modulo A al modulo B in cui ogni modulo importa il precedente.
3. Applica questa mappa ai file raccolti nel passaggio 1, si ottiene l'elenco dei file del modello interessati dalla PR.
4. Mappa ciascuno di questi file con i corrispondenti file di test e ottiene l'elenco dei test da eseguire.

Quando esegui lo script in locale, dovresti ottenere la stampa dei risultati dei passi 1, 3 e 4 e quindi sapere quali test sono stati eseguiti. Lo script creerà anche un file chiamato `test_list.txt` che contiene l'elenco dei test da eseguire e che puoi eseguire localmente con il seguente comando:

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

Nel caso in cui qualcosa sia sfuggito, l'intera suite di test viene eseguita quotidianamente.

## Build della documentazione

Il job `ci/circleci: build_doc` esegue una build della documentazione per assicurarsi che tutto sia a posto una volta che la PR è stata unita. Se questo passaggio fallisce, puoi controllare localmente entrando nella cartella `docs` del repo Transformers e digitare

```bash
make html
```

Sphinx non è noto per i suoi messaggi di errore chiari, quindi potrebbe essere necessario che provi alcune cose per trovare davvero la fonte dell'errore.

## Stile del codice e della documentazione

La formattazione del codice viene applicata a tutti i file sorgenti, agli esempi e ai test usando `black` e `isort`. Abbiamo anche uno strumento personalizzato che si occupa della formattazione delle docstring e dei file `rst` (`utils/style_doc.py`), così come dell'ordine dei lazy imports eseguiti nei file `__init__.py` dei Transformers (`utils/custom_init_isort.py`). Tutto questo può essere lanciato eseguendo

```bash
make style
```

I controlli della CI sono applicati all'interno del controllo `ci/circleci: check_code_quality`. Esegue anche `flake8`, che dà un'occhiata di base al codice e si lamenta se trova una variabile non definita o non utilizzata. Per eseguire questo controllo localmente, usare

```bash
make quality
```

Questa operazione può richiedere molto tempo, quindi per eseguire la stessa operazione solo sui file modificati nel branch corrente, eseguire

```bash
make fixup
```

Quest'ultimo comando eseguirà anche tutti i controlli aggiuntivi per la consistenza del repository. Diamogli un'occhiata.

## Coerenza del repository

All'interno sono raggruppati tutti i test per assicurarsi che la tua PR lasci il repository in un buono stato ed è eseguito dal controllo `ci/circleci: check_repository_consistency`. Puoi eseguire localmente questo controllo eseguendo quanto segue:

```bash
make repo-consistency
```

Questo verifica che:

- Tutti gli oggetti aggiunti all'init sono documentati (eseguito da `utils/check_repo.py`)
- Tutti i file `__init__.py` hanno lo stesso contenuto nelle loro due sezioni (eseguito da `utils/check_inits.py`)
- Tutto il codice identificato come copia da un altro modulo è coerente con l'originale (eseguito da `utils/check_copies.py`)
- Le traduzioni dei README e l'indice della documentazione hanno lo stesso elenco di modelli del README principale (eseguito da `utils/check_copies.py`)
- Le tabelle autogenerate nella documentazione sono aggiornate (eseguito da `utils/check_table.py`)
- La libreria ha tutti gli oggetti disponibili anche se non tutte le dipendenze opzionali sono installate (eseguito da `utils/check_dummies.py`)

Se questo controllo fallisce, le prime due voci richiedono una correzione manuale, mentre le ultime quattro possono essere corrette automaticamente per te eseguendo il comando

```bash
make fix-copies
```

Ulteriori controlli riguardano le PR che aggiungono nuovi modelli, principalmente che:

- Tutti i modelli aggiunti sono in un Auto-mapping (eseguita da `utils/check_repo.py`)
<!-- TODO Sylvain, add a check that makes sure the common tests are implemented.-->
- Tutti i modelli sono testati correttamente (eseguito da `utils/check_repo.py`)

<!-- TODO Sylvain, add the following
- All models are added to the main README, inside the main doc
- All checkpoints used actually exist on the Hub

-->