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

# Inferență pe web server

Un web server este un sistem care așteaptă cereri și le servește pe măsură ce sosesc. Acest lucru înseamnă că poți folosi [`Pipeline`] ca motor de inferență pe un web server, deoarece poți folosi un iterator (similar cu modul în care ai [itera peste un set de date](./pipeline_tutorial#seturi-de-date-mari)) pentru a gestiona fiecare cerere primită.

Totuși, proiectarea unui web server cu [`Pipeline`] este aparte deoarece sunt fundamental diferite. Web server-ele sunt multiplexate (multithreaded, async etc.) pentru a gestiona simultan mai multe cereri. [`Pipeline`] și modelul său de bază, pe de altă parte, nu sunt concepute pentru paralelism deoarece consumă multă memorie. Cel mai bine este să oferi unui [`Pipeline`] toate resursele disponibile atunci când rulează sau pentru un job intensiv din punct de vedere computațional.

Acest ghid arată cum să ocolești această diferență folosind un web server pentru a gestiona sarcina mai ușoară de primire și trimitere a cererilor și având un singur thread pentru a gestiona sarcina mai grea de rulare a [`Pipeline`].

## Crearea unui server

[Starlette](https://www.starlette.io/) este un framework ușor pentru construirea de web server-e. Poți folosi orice alt framework dorești, dar s-ar putea să fie nevoie să faci câteva modificări la codul de mai jos.

Înainte de a începe, asigură-te că Starlette și [uvicorn](http://www.uvicorn.org/) sunt instalate.

```py
!pip install starlette uvicorn
```

Acum poți crea un web server simplu într-un fișier `server.py`. Cheia este să încarci modelul **o singură dată** pentru a preveni ca niște copii inutile ale acestuia să consume memorie.

Creează un pipeline pentru a completa token-ul mascat, `[MASK]`.

```py
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio

async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)

async def server_loop(q):
    pipe = pipeline(task="fill-mask",model="google-bert/bert-base-uncased")
    while True:
        (string, response_q) = await q.get()
        out = pipe(string)
        await response_q.put(out)

app = Starlette(
    routes=[
        Route("/", homepage, methods=["POST"]),
    ],
)

@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
```

Pornește server-ul cu următoarea comandă.

```bash
uvicorn server:app
```

Interoghează server-ul cu o cerere POST.

```bash
curl -X POST -d "Paris is the [MASK] of France." http://localhost:8000/
```

Acest lucru ar trebui să returneze output-ul de mai jos.

```bash
[{'score': 0.9969332218170166,
  'token': 3007,
  'token_str': 'capital',
  'sequence': 'paris is the capital of france.'},
 {'score': 0.0005914849461987615,
  'token': 2540,
  'token_str': 'heart',
  'sequence': 'paris is the heart of france.'},
 {'score': 0.00043787318281829357,
  'token': 2415,
  'token_str': 'center',
  'sequence': 'paris is the center of france.'},
 {'score': 0.0003378340043127537,
  'token': 2803,
  'token_str': 'centre',
  'sequence': 'paris is the centre of france.'},
 {'score': 0.00026995912776328623,
  'token': 2103,
  'token_str': 'city',
  'sequence': 'paris is the city of france.'}]
```

## Punerea cererilor în queue

Mecanismul de punere în queue al server-ului poate fi folosit pentru unele aplicații interesante precum dynamic batching. Dynamic batching acumulează mai întâi mai multe cereri înainte de a le procesa cu [`Pipeline`].

Exemplul de mai jos este scris în pseudocod pentru lizibilitate mai degrabă decât pentru performanță, în special, vei observa că:

1. Nu există o limită pentru dimensiunea batch-ului.
2. Timeout-ul este resetat la fiecare preluare din queue, așa că ai putea ajunge să aștepți mult mai mult decât valoarea `timeout` înainte de a procesa o cerere. Acest lucru ar întârzia și prima cerere de inferență cu acea perioadă de timp. Web server-ul așteaptă mereu 1s chiar dacă queue-ul este gol, ceea ce este ineficient, deoarece acel timp poate fi folosit pentru a porni inferența. Totuși, ar putea avea sens dacă batching-ul este esențial pentru cazul tău de utilizare.

    Ar fi mai bine să ai un singur deadline de 1s, în loc să îl resetezi la fiecare preluare, așa cum se arată mai jos.

```py
async def server_loop(q):
    pipe = pipeline(task="fill-mask", model="google-bert/bert-base-uncased")
    while True:
        (string, rq) = await q.get()
        strings = []
        queues = []
        strings.append(string)
        queues.append(rq)
        while True:
            try:
                (string, rq) = await asyncio.wait_for(q.get(), timeout=1)
            except asyncio.exceptions.TimeoutError:
                break
            strings.append(string)
            queues.append(rq)
        outs = pipe(strings, batch_size=len(strings))
        for rq, out in zip(queues, outs):
            await rq.put(out)
```

## Verificarea erorilor

Sunt multe lucruri care pot merge prost în producție. Ai putea rămâne fără memorie (out-of-memory), fără spațiu, ai putea eșua la încărcarea unui model, ai putea avea o configurație incorectă a modelului, o interogare incorectă și multe altele.

Adăugarea de instrucțiuni `try...except` este utilă pentru a returna aceste erori utilizatorului în scop de debugging. Ține cont că acest lucru ar putea fi un risc de securitate dacă nu ar trebui să dezvălui anumite informații.

## Circuit breaking

Încearcă să returnezi o eroare 503 sau 504 atunci când server-ul este suprasolicitat, în loc să forțezi un utilizator să aștepte la nesfârșit.

Este relativ simplu să implementezi aceste tipuri de erori deoarece este vorba doar de un singur queue. Aruncă o privire la dimensiunea queue-ului pentru a determina când să începi să returnezi erori înainte ca server-ul tău să cedeze sub sarcină.

## Blocarea thread-ului principal

PyTorch nu este conștient de async, așa că procesarea va bloca rularea thread-ului principal.

Din acest motiv, este mai bine să rulezi PyTorch pe propriul său thread sau proces separat. Când inferența unei singure cereri este deosebit de lungă (mai mult de 1s), acest lucru este și mai important deoarece înseamnă că fiecare interogare din timpul inferenței trebuie să aștepte 1s înainte de a primi măcar o eroare.

## Dynamic batching

Dynamic batching poate fi foarte eficient atunci când este folosit în contextul potrivit, dar nu este necesar când transmiți doar o cerere pe rând (vezi [inferența în batch-uri](./pipeline_tutorial#inferență-în-batch-uri) pentru mai multe detalii).
