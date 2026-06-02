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

# Algoritmi de tokenization

<Youtube id="zHvTiHr506c"/>

Transformers suportă trei algoritmi de tokenizare la nivel de subword: Byte pair encoding (BPE), Unigram și WordPiece. Aceștia împart textul în unități între cuvinte și caractere, menținând vocabularul compact, captând în același timp bucăți cu sens. Cuvintele comune rămân intacte ca token-uri unice, iar cuvintele rare sau necunoscute se descompun în subwords.

De exemplu, `annoyingly` ar putea fi împărțit în `["annoying", "ly"]` sau `["annoy", "ing", "ly"]` în funcție de vocabular. Împărțirea în subwords permite modelului să reprezinte cuvinte nevăzute din subwords cunoscute.

> [!TIP]
> Tokenizarea la nivel de subword este deosebit de utilă pentru limbi ca turca, unde poți forma cuvinte lungi și complexe înlănțuind subword-uri.

## Byte pair encoding (BPE)

<Youtube id="HEikzVL-lZU"/>

[Byte pair encoding](https://huggingface.co/papers/1508.07909) (BPE) este cel mai popular algoritm de tokenizare din Transformers, folosit de modele ca [Llama], [Gemma], [Qwen2] și altele.

1. Un pre-tokenizator împarte textul pe spații sau alte reguli, producând un set de cuvinte unice și frecvențele lor.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

2. Algoritmul BPE creează un vocabular de bază, `["b", "g", "h", "n", "p", "s", "u"]`, din toate caracterele.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

3. BPE pornește cu caractere individuale și îmbină iterativ perechea adiacentă cea mai frecventă. `"u"` și `"g"` apar împreună cel mai des în `"hug"`, `"pug"` și `"hugs"`, deci BPE le îmbină în `"ug"` și îl adaugă în vocabular.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

4. Următoarea pereche cea mai comună este `"u"` și `"n"`, care apar în `"pun"` și `"bun"`, deci se îmbină în `"un"`.

```text
("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

5. Vocabularul este acum `["b", "g", "h", "n", "p", "s", "u", "ug", "un"]`. BPE continuă să învețe reguli de îmbinare până atinge dimensiunea țintă a vocabularului, egală cu dimensiunea vocabularului de bază plus numărul de îmbinări. [GPT] folosește BPE cu un vocabular de 40.478 (478 token-uri de bază + 40.000 îmbinări).

Orice caracter care nu se află în vocabularul de bază se mapează la un token necunoscut ca `"<unk>"`. În practică, vocabularul de bază acoperă toate caracterele văzute în antrenare, deci token-urile necunoscute sunt rare.

### BPE la nivel de byte

Includerea tuturor caracterelor Unicode ar face vocabularul de bază enorm. BPE la nivel de byte folosește în schimb 256 de valori de byte ca vocabular de bază, asigurând că orice cuvânt poate fi tokenizat fără token-ul `"<unk>"`. [GPT-2] folosește BPE la nivel de byte cu un vocabular de 50.257 (256 token-uri de byte + 50.000 îmbinări + un token special de sfârșit de text).

## Unigram

<Youtube id="TGZfZVuF9Yc"/>

[Unigram](https://huggingface.co/papers/1804.10959) este al doilea algoritm de tokenizare ca popularitate din Transformers, folosit de modele precum [T5], [BigBird], [Pegasus] și altele.

1. Unigram pornește cu un set mare de subword-uri candidate, iar fiecare candidat primește un scor de probabilitate bazat pe frecvența sa.

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "pu", "bu", "gs", "hug", "pug", "pun", "bun", "ugs", "hugs"]
```

2. Unigram evaluează cât de bine tokenizează vocabularul curent datele de antrenare la fiecare pas.
3. Pentru fiecare token, Unigram măsoară cât de mult ar crește pierderea globală dacă token-ul ar fi eliminat. De exemplu, eliminarea `"pu"` afectează puțin pierderea pentru că `"pug"` și `"pun"` pot fi tokenizate în continuare ca `["p", "ug"]` și `["p", "un"]`.

    Dar eliminarea `"ug"` ar crește semnificativ pierderea pentru că `"hug"`, `"pug"` și `"hugs"` depind toate de el.

4. Unigram elimină token-urile cu cea mai mică creștere a pierderii, de obicei primele 10-20% de jos. Caracterele de bază rămân mereu ca orice cuvânt să poată fi tokenizat. Token-uri ca `"bu"`, `"pu"`, `"gs"`, `"pug"` și `"bun"` sunt eliminate pentru că au contribuit cel mai puțin la probabilitatea globală.

```text
["b", "g", "h", "n", "p", "s", "u", "hu", "ug", "un", "hug", "pun", "ugs", "hugs"]
```

5. Pașii 2-4 se repetă până vocabularul atinge dimensiunea țintă.

În inferență, Unigram poate tokeniza un cuvânt în mai multe moduri. `"hugs"` ar putea deveni `["hug", "s"]`, `["h", "ug", "s"]` sau `["h", "u", "g", "s"]`. Unigram alege tokenizarea cu cea mai mare probabilitate. Spre deosebire de BPE, care este determinist și bazat pe reguli de îmbinare, Unigram este probabilistic și poate eșantiona tokenizări diferite în antrenare.

## SentencePiece

[SentencePiece](https://huggingface.co/papers/1808.06226) este o bibliotecă de tokenizare care aplică BPE sau Unigram direct pe text brut. BPE și Unigram standard presupun că spațiile separă cuvintele, ceea ce nu funcționează pentru limbi ca chineza și japoneza care nu folosesc spații.

1. SentencePiece tratează textul de intrare ca un flux brut de bytes sau caractere și include caracterul spațiu, reprezentat ca `"▁"`, în vocabular.

```text
("▁hug", 10), ("▁pug", 5), ("▁pun", 12), ("▁bun", 4), ("▁hugs", 5)
```

2. SentencePiece aplică apoi BPE sau Unigram pe text.

La decodare, SentencePiece concatenează toate token-urile și înlocuiește `"▁"` cu un spațiu.

## WordPiece

<Youtube id="qpv6ms_t_1A"/>

[WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) este algoritmul de tokenizare pentru modelele din familia BERT, cum ar fi [DistilBERT] și [Electra].

Este similar cu [BPE](#byte-pair-encoding-bpe) și îmbină iterativ perechi de jos în sus, dar diferă în modul în care selectează perechile.

```text
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

WordPiece îmbină perechile care maximizează probabilitatea datelor de antrenare.

```text
score("u", "g") = frequency("ug") / (frequency("u") × frequency("g"))
```

| pereche | frecvență | scor |
|---|---|---|
| `"u"` + `"g"` | 20 | 20 / (36 × 20) = 0.028 |
| `"u"` + `"n"` | 16 | 16 / (36 × 16) = 0.028 |
| `"h"` + `"u"` | 15 | 15 / (15 × 36) = 0.028 |
| `"g"` + `"s"` | 5 | 5 / (20 × 5) = 0.050 |

Scorul favorizează îmbinarea `"g"` și `"s"` unde token-ul combinat apare mai des decât s-ar aștepta din frecvențele individuale ale token-urilor. BPE îmbină pur și simplu perechea care apare cel mai des. WordPiece măsoară cât de *informativă* este fiecare îmbinare. Două token-uri care apar împreună mult mai des decât prevede probabilitatea sunt îmbinate primele.

## Tokenization la nivel de cuvânt

<Youtube id="nhJxYji1aho"/>

Tokenization-ul la nivel de cuvânt împarte textul în token-uri după spații, punctuație sau reguli specifice limbii.

```text
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

Dimensiunea vocabularului devine extrem de mare pentru că fiecare cuvânt unic necesită propriul token, inclusiv toate variantele (`"love"`, `"loving"`, `"loved"`, `"lovingly"`). Matricea de embeddings rezultată este enormă, crescând memoria și consumul de calcul. Cuvintele care nu se află în vocabular se mapează la un token `"<unk>"`, deci modelul nu poate gestiona cuvinte noi.

## Tokenization la nivel de caracter

Tokenization-ul la nivel de caracter împarte textul în caractere individuale.

```text
["D", "o", "n", "'", "t", "y", "o", "u", "l", "o", "v", "e"]
```

Vocabularul este mic și orice cuvânt poate fi reprezentat, deci nu există problema `"<unk>"`. Dar secvențele devin mult mai lungi. Un caracter ca `"l"` poartă mult mai puțin sens decât `"love"`, deci performanța scade.

## Resurse

- [Capitolul 6](https://huggingface.co/learn/llm-course/chapter6/1) din cursul LLM te învață cum să antrenezi un tokenizer de la zero și explică diferențele dintre algoritmii BPE, Unigram și WordPiece.
