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

# Personalizarea tokenizerelor

Tokenizerele sunt decuplate de vocabularele lor învățate. Asta îți permite să inițializezi un tokenizer gol pentru antrenare sau să creezi unul direct cu propriul vocabular. Pipeline-ul de bază pentru tokenization rămâne același (normalizer, pre-tokenizer, algoritmul de tokenization), deci nu trebuie să îl recreezi de la zero.

Acest ghid îți arată cum să antrenezi și să creezi un tokenizer personalizat.

## Antrenarea unui tokenizer

Un tokenizer gol antrenabil înlocuiește vocabularul cu un nou vocabular țintă. Este util pentru adaptarea la un nou domeniu, cum ar fi finanțe, o limbă cu resurse reduse sau cod.

Creează un tokenizer gol și încarcă un dataset.

```py
from datasets import load_dataset
from transformers import GemmaTokenizer

tokenizer = GemmaTokenizer()
dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
```

Folosește metoda [`TokenizersBackend.train_new_from_iterator`] ca să antrenezi tokenizerul. Metoda acceptă o funcție generator ca să returneze bucăți de text din dataset în loc să încarce totul în memorie dintr-o dată. Argumentul `vocab_size` setează dimensiunea vocabularului tokenizer-ului.

```py
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["assistant"]

trained_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=32000,
)
encoded = trained_tokenizer("The stock market rallied today.")
print(encoded["input_ids"])
[5866, 11503, 98, 5885, 8617, 13381, 30]
```

Adaugă token-uri speciale noi cu argumentul `new_special_tokens` sau folosește `special_tokens_map` ca să redenumești token-urile speciale vechi cu cele noi.

Salvează noul tokenizer de finanțe cu [`~PreTrainedTokenizerBase.save_pretrained`] sau salvează-l și încarcă-l pe Hub cu [`~PreTrainedTokenizerBase.push_to_hub`]. Asta creează un fișier `tokenizer.json` care captează vocabularul nou antrenat, regulile de îmbinare și configurația completă a pipeline-ului.

```py
trained_tokenizer.save_pretrained("./finance-gemma-tokenizer")
trained_tokenizer.push_to_hub("finance-gemma-tokenizer")
```

## Vocabular personalizat

Un tokenizer gol suportă vocabular personalizat cu argumentele `vocab` și `merges`.

- `vocab` este setul complet de token-uri pe care un tokenizer le cunoaște, iar fiecare intrare mapează un token la input id-ul său.
- `merges` definește cum ar trebui algoritmul BPE să combine token-urile adiacente.

```py
from transformers import GemmaTokenizer

vocab={
    "<pad>": 0,
    "</s>": 1,
    "<s>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "▁the": 5,
    "▁stock": 6,
    "▁market": 7,
    "▁": 8,
    "r": 9,
    "a": 10,
    "l": 11,
    "i": 12,
    "e": 13,
    "d": 14,
    "ra": 15,
    "li": 16,
    "lie": 17,
    "lied": 18,
    "ral": 19,
    "ralli": 20,
    "rallie": 21,
    "rallied": 22,
}
merges=[
    ("r", "a"),       # r + a → ra
    ("l", "i"),       # l + i → li
    ("li", "e"),      # li + e → lie
    ("lie", "d"),     # lie + d → lied
    ("ra", "l"),      # ra + l → ral
    ("ral", "li"),    # ral + li → ralli
    ("ralli", "e"),   # ralli + e → rallie
    ("rallie", "d"),  # rallie + d → rallied
]

tokenizer = GemmaTokenizer(vocab=vocab, merges=merges)
encoded = tokenizer("the stock market rallied")
print(encoded["input_ids"])
```

## Subclasarea TokenizersBackend

Tokenizers suportă patru [backend-uri](./fast_tokenizers#backend-uri) diferite. În general, ar trebui să folosești [`TokenizersBackend`] ca să definești un tokenizer nou deoarece este mai rapid.

> [!TIP]
> [`PythonBackend`] este un tokenizer pur Python care nu depinde de backend-uri ca Rust, SentencePiece sau mistral-common. Folosește [`PythonBackend`] doar dacă construiești un tokenizer foarte specializat care nu poate fi exprimat de backend-ul Rust.

1. Subclasează [`TokenizersBackend`] cu atribute de clasă precum latura de padding și algoritmul de tokenizare de folosit.
2. Definește pipeline-ul de tokenizare în `__init__`. Asta include algoritmul de tokenizare de folosit, cum să împartă textul brut înaintea algoritmului și cum să decodifice token-urile înapoi în text.

```py
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from transformers import TokenizersBackend

class NewTokenizer(TokenizersBackend):
    padding_side = "left"
    model = BPE

    def __init__(
        self,
        vocab=None,
        merges=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    ):
        self._vocab = vocab or {
            str(unk_token): 0,
            str(bos_token): 1,
            str(eos_token): 2,
            str(pad_token): 3,
        }
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(vocab=self._vocab, merges=self._merges, fuse_unk=True)
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )
```

Antrenează sau salvează noul tokenizer gol.

```py
tokenizer = NewTokenizer()

# antrenează pe corpus nou
tokenizer.train_new_from_iterator()
# salvează tokenizer-ul
tokenizer.save_pretrained("./new-tokenizer")
```
