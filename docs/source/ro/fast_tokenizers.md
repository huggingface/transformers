<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tokenizere

Un tokenizer convertește text în tensori, care sunt input-urile unui model. Acesta normalizează și împarte textul, aplică algoritmul de tokenization, adaugă token-uri speciale și decodifică id-urile de output înapoi în text.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("Sphinx of black quartz, judge my vow.", return_tensors="pt")
{
    'input_ids': tensor([[     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

Acest ghid acoperă încărcarea, encodarea, decodarea, procesarea în batch și backend-urile disponibile pentru tokenizere.

## Încarcă un tokenizer

Încarcă un tokenizer cu clasa [`AutoTokenizer`] sau cu o clasă de tokenizer specifică modelului.

<hfoptions id="tokenizers">
<hfoption id="AutoTokenizer">

[`AutoTokenizer.from_pretrained`] citește config-ul modelului, rezolvă clasa corectă de tokenizer și returnează o instanță a ei. Nu trebuie să știi clasa de tokenizer dinainte. Majoritatea tokenizere-lor se rezolvă la o subclasă a [`TokenizersBackend`], un tokenizer rapid bazat pe Rust din biblioteca [Tokenizers](https://huggingface.co/docs/tokenizers/index).

Încărcarea cu [`AutoTokenizer`] este abordarea recomandată.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
```

</hfoption>
<hfoption id="model-specific tokenizer">

O clasă de tokenization specifică modelului este un [`TokenizersBackend`] pre-configurat care folosește exact configurația de tokenization (normalizator, pre-tokenizator, convenții pentru token-uri speciale etc.) cu care a fost antrenat modelul.

Folosește o clasă specifică modelului ca să inițializezi un tokenizer gol pentru antrenare sau ca să pasezi argumente specifice modelului precum `vocab` sau `merges` (vezi ghidul [Personalizarea tokenizere-lor](./custom_tokenizers) ca să afli cum). Un tokenizer gol este minimal și conține doar token-urile speciale ale modelului, cum ar fi `<pad>`, `<eos>` sau `<bos>`.

```py
from transformers import GemmaTokenizer

tokenizer = GemmaTokenizer()
corpus = [
    ["Sphinx of black quartz, judge my vow."],
    ["Pack my box with five dozen liquor jugs."],
    ["How vexingly quick daft zebras jump!"],
]
new_tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=1000)
```

</hfoption>
</hfoptions>

## Encodare și decodare

Metoda [`TokenizersBackend.__call__`] encodează text sau un batch de text în `input_ids`, `attention_mask` și alte input-uri pentru model. Controlează și padding-ul, trunchierea și inserarea token-urilor speciale.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("Sphinx of black quartz, judge my vow.", return_tensors="pt")
{
    'input_ids': tensor([[     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

[`TokenizersBackend.encode`] este similar, dar returnează doar `input_ids`.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.encode("Sphinx of black quartz, judge my vow.")
[2, 235277, 82913, 576, 2656, 30407, 235269, 11490, 970, 29871, 235265]
```

[`TokenizersBackend.decode`] convertește o secvență sau un batch de `input_ids` tokenizate înapoi în text.

```py
tokenizer.decode(outputs["input_ids"])
['<bos>Sphinx of black quartz, judge my vow.']
```

[`TokenizersBackend.decode`] păstrează spațierea exactă a operației de tokenization. Setează `clean_up_tokenization_spaces` ca să elimini spațiile de dinaintea punctuației și `skip_special_tokens` ca să scoți token-urile speciale din output.

```py
tokenizer.decode(outputs["input_ids"], skip_special_tokens=True)
['Sphinx of black quartz, judge my vow.']
``` 

## Token-uri speciale

Token-urile speciale marchează limitele structurale dintr-o secvență, cum ar fi începutul secvenței sau pozițiile de padding. Fiecare model definește propriul set de token-uri speciale. Tokenizer-ul le adaugă când îl apelezi.

```py
tokenizer.encode("Sphinx of black quartz, judge my vow.")
[2, 235277, 82913, 576, 2656, 30407, 235269, 11490, 970, 29871, 235265]
tokenizer.decode(outputs["input_ids"])
['<bos>Sphinx of black quartz, judge my vow.']
```

Înregistrează token-uri speciale cu nume suplimentare folosind argumentul `extra_special_tokens`. Modelele multimodale le folosesc ca placeholders pentru imagini, video sau audio.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-pt",
    extra_special_tokens={"image_token": "<image>"}
)
```

## Procesare în batch

Procesarea în batch tokenizează mai multe secvențe într-un singur apel. [`TokenizersBackend`] gestionează batch-uri mari mai rapid pentru că backend-ul său bazat pe Rust paralelizează munca pe thread-uri.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt"
)
```

Procesarea în batch presupune că toate secvențele au aceeași lungime. Padding-ul și trunchierea sunt strategii ca să gestionezi secvențe de lungimi diferite.

### Padding

Padding-ul adaugă token-uri speciale ca secvențele mai scurte să aibă aceeași lungime cu cea mai lungă din batch. Masca de atenție marchează pozițiile de padding cu `0` pentru ca modelul să le ignore. Setează `padding=True` ca să faci padding până la cea mai lungă secvență sau pasează `max_length` ca să faci padding la o dimensiune fixă.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt",
    padding=True,
)
{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265],
        [     0,      2,   6519,    970,   3741,    675,   4105,  25955,  42184, 225789, 235265],
        [     0,      2,   2299,  73378,  17844,   4320, 224463,   4949,  48977,  9902, 235341]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
}
```

> [!NOTE]
> Modelele lingvistice mari fac padding pe partea *stângă* ca să nu perturbe generarea, care prezice token-ul următor din partea *dreaptă*.

### Trunchiere

Trunchierea taie token-uri ca o secvență să se încadreze într-o lungime maximă. Setează `truncation=True` și specifică `max_length` ca să o activezi.

Padding-ul și trunchierea funcționează împreună. Secvențele scurte primesc token-uri de padding, iar cele lungi pierd token-uri de la coadă. Împreună, produc un tensor dreptunghiular compact.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=5
)
{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656],
        [     2,   6519,    970,   3741,    675],
        [     2,   2299,  73378,  17844,   4320]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
}
```

## Backend-uri

Fiecare tokenizer de model este definit într-un singur fișier și suportă patru backend-uri de tokenization.

| backend | implementare | descriere |
|---|---|---|
| [`TokenizersBackend`] | [Tokenizers](https://huggingface.co/docs/tokenizers) | implicit pentru majoritatea modelelor |
| [`SentencePieceBackend`] | [SentencePiece](https://github.com/google/sentencepiece) | modele care necesită SentencePiece |
| [`PythonBackend`] | Python | modele care necesită tokenizere personalizate specializate |
| [`MistralCommonBackend`] | [mistral-common](https://mistralai.github.io/mistral-common/) | modele Mistral și Pixtral |

Toate backend-urile moștenesc din [`PreTrainedTokenizerBase`] și împart aceleași API-uri pentru encodare, decodare, padding, trunchiere, salvare și încărcare. Diferența constă în pipeline-ul de tokenization care rulează pe dedesubt.

[`AutoTokenizer`] selectează cel mai bun backend disponibil când apelezi [`~AutoTokenizer.from_pretrained`].

1. Citește fișierul `tokenizer_config.json` pentru câmpul `tokenizer_class`.
2. Registrul potrivește `tokenizer_class` cu un nume de clasă. Clasa rezolvată moștenește din unul din cele patru backend-uri. De exemplu, [`GemmaTokenizer`] moștenește din [`TokenizersBackend`], iar [`SiglipTokenizer`] moștenește din [`SentencePieceBackend`].

    Unele modele, ca GLM, se mapează direct la [`TokenizersBackend`] pentru că fișierul `tokenizer.json` descrie complet pipeline-ul. [`GemmaTokenizer`] există ca subclasă deoarece definește setări suplimentare specifice modelului în Python pe care `tokenizer.json` nu le captează.

    ```py
    TOKENIZER_MAPPING_NAMES = OrderedDict([
        ("gemma2", "GemmaTokenizer" if is_tokenizers_available() else None),
        ("glm", "TokenizersBackend" if is_tokenizers_available() else None),
        (
            "mistral",
            "MistralCommonBackend"
            if is_mistral_common_available()
            else ("TokenizersBackend" if is_tokenizers_available() else None),
        ),
        ("siglip", "SiglipTokenizer" if is_sentencepiece_available() else None),
        ...
    ]
    ```
    
    Când un backend ca mistral-common nu este instalat, [`AutoTokenizer`] revine la [`TokenizersBackend`].

Verifică ce backend folosește un tokenizer cu proprietatea `backend`.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.backend
'tokenizers'
```

## Inspectează arhitectura tokenizer-ului

Inspectează componentele interne ale unui tokenizer (normalizator, pre-tokenizator, model, decoder) cu atributul `_tokenizer`.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
print(tokenizer._tokenizer.normalizer)
print(tokenizer._tokenizer.pre_tokenizer)
print(tokenizer._tokenizer.model)
print(tokenizer._tokenizer.decoder)
```

## Resurse

- Postarea [Tokenization in Transformers v5](https://huggingface.co/blog/tokenizers) discută motivația din spatele noilor backend-uri de tokenization.
- Consultă [ghidul de migrare](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md#tokenization) pentru o prezentare a schimbărilor de tokenization.
