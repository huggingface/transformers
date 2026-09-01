<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Personaliza tokenizadores

Los tokenizadores están desacoplados de sus vocabularios aprendidos. Esto te permite inicializar un tokenizador vacío para entrenarlo o crear uno directamente con tu propio vocabulario. El pipeline de tokenización se mantiene igual (normalizer, pre-tokenizador, algoritmo de tokenización), así que no necesitas recrearlo desde cero.

Esta guía te muestra cómo entrenar y crear un tokenizador personalizado.

## Entrenar un tokenizador

Un tokenizador vacío y entrenable reemplaza el vocabulario por un vocabulario objetivo nuevo. Esto es útil si quieres adaptarte a un dominio nuevo como finanzas, a un idioma de bajos recursos o a código.

Crea un tokenizador vacío y carga un dataset.

```py
from datasets import load_dataset
from transformers import GemmaTokenizer

tokenizer = GemmaTokenizer()
dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
```

Usa el método [`TokenizersBackend.train_new_from_iterator`] para entrenar el tokenizador. Este método acepta una función generadora que devuelve trozos de texto del dataset en lugar de cargar todo en memoria de una sola vez. El argumento `vocab_size` define el tamaño del vocabulario del tokenizador.

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

Añade tokens especiales nuevos con el argumento `new_special_tokens` o usa `special_tokens_map` para renombrar los tokens especiales antiguos a los tokens especiales nuevos.

Guarda el nuevo tokenizador de finanzas con [`~PreTrainedTokenizerBase.save_pretrained`] o guárdalo y súbelo al Hub con [`~PreTrainedTokenizerBase.push_to_hub`]. Esto crea un archivo `tokenizer.json` que captura el vocabulario recién entrenado, las reglas de fusión y la configuración completa del pipeline.

```py
trained_tokenizer.save_pretrained("./finance-gemma-tokenizer")
trained_tokenizer.push_to_hub("finance-gemma-tokenizer")
```

## Vocabulario personalizado

Un tokenizador vacío admite un vocabulario personalizado con los argumentos `vocab` y `merges`.

- `vocab` es el conjunto completo de tokens que un tokenizador conoce y cada entrada asocia un token con su input id.
- `merges` define cómo el algoritmo BPE debe combinar tokens adyacentes.

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

## Subclasificar TokenizersBackend

Tokenizers admite cuatro [backends](./fast_tokenizers#backends) distintos. En general, deberías usar [`TokenizersBackend`] para definir un tokenizador nuevo porque es más rápido.

> [!TIP]
> El [`PythonBackend`] es un tokenizador en Python puro que no depende de backends como Rust, SentencePiece o mistral-common. Solo deberías usar [`PythonBackend`] si estás construyendo un tokenizador muy especializado que no se puede expresar con el backend de Rust.

1. Subclasifica [`TokenizersBackend`] con atributos de clase como el lado del padding y el algoritmo de tokenización a usar.
2. Define el pipeline de tokenización en el `__init__`. Esto incluye el algoritmo de tokenización a usar, cómo dividir el texto en bruto antes del algoritmo y cómo decodificar los tokens de vuelta a texto.

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

Entrena o guarda el nuevo tokenizador vacío.

```py
tokenizer = NewTokenizer()

# train on new corpus
tokenizer.train_new_from_iterator()
# save tokenizer
tokenizer.save_pretrained("./new-tokenizer")
```
