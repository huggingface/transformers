<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Modelli multilingue per l'inferenza

[[open-in-colab]]

Ci sono diversi modelli multilingue in 🤗 Transformers, e il loro utilizzo per l'inferenza differisce da quello dei modelli monolingua. Non *tutti* gli utilizzi dei modelli multilingue sono però diversi. Alcuni modelli, come [bert-base-multilingual-uncased](https://huggingface.co/bert-base-multilingual-uncased), possono essere usati come un modello monolingua. Questa guida ti mostrerà come utilizzare modelli multilingue che utilizzano un modo diverso per fare l'inferenza.

## XLM

XLM ha dieci diversi checkpoint, di cui solo uno è monolingua. I nove checkpoint rimanenti possono essere suddivisi in due categorie: i checkpoint che utilizzano i language embeddings e quelli che non li utilizzano.

### XLM con language embeddings

I seguenti modelli XLM utilizzano gli embeddings linguistici per specificare la lingua utilizzata per l'inferenza:

- `xlm-mlm-ende-1024` (Modellazione mascherata del linguaggio (Masked language modeling, in inglese), Inglese-Tedesco)
- `xlm-mlm-enfr-1024` (Modellazione mascherata del linguaggio, Inglese-Francese)
- `xlm-mlm-enro-1024` (Modellazione mascherata del linguaggio, Inglese-Rumeno)
- `xlm-mlm-xnli15-1024` (Modellazione mascherata del linguaggio, lingue XNLI)
- `xlm-mlm-tlm-xnli15-1024` (Modellazione mascherata del linguaggio + traduzione, lingue XNLI)
- `xlm-clm-enfr-1024` (Modellazione causale del linguaggio, Inglese-Francese)
- `xlm-clm-ende-1024` (Modellazione causale del linguaggio, Inglese-Tedesco)

Gli embeddings linguistici sono rappresentati come un tensore delle stesse dimensioni dell' `input_ids` passato al modello. I valori in questi tensori dipendono dal linguaggio usato e sono identificati dagli attributi `lang2id` e `id2lang` del tokenizer.

In questo esempio, carica il checkpoint `xlm-clm-enfr-1024` (Modellazione causale del linguaggio, Inglese-Francese):

```py
>>> import torch
>>> from transformers import XLMTokenizer, XLMWithLMHeadModel

>>> tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
>>> model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
```

L'attributo `lang2id` del tokenizer mostra il linguaggio del modello e il suo ids:

```py
>>> print(tokenizer.lang2id)
{'en': 0, 'fr': 1}
```

Poi, crea un esempio di input:

```py
>>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")])  # batch size of 1
```

Imposta l'id del linguaggio a `"en"` e usalo per definire il language embedding. Il language embedding è un tensore riempito con `0` perché questo è il language id per l'inglese. Questo tensore dovrebbe avere la stessa dimensione di `input_ids`.

```py
>>> language_id = tokenizer.lang2id["en"]  # 0
>>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

>>> # We reshape it to be of size (batch_size, sequence_length)
>>> langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)
```

Adesso puoi inserire `input_ids` e language embedding nel modello:

```py
>>> outputs = model(input_ids, langs=langs)
```

Lo script [run_generation.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation/run_generation.py) può generare testo tramite i language embeddings usando i checkpoints `xlm-clm`.

### XLM senza language embeddings

I seguenti modelli XLM non richiedono l'utilizzo dei language embeddings per fare inferenza:

- `xlm-mlm-17-1280` (Modellazione mascherata del linguaggio, 17 lingue)
- `xlm-mlm-100-1280` (Modellazione mascherata del linguaggio, 100 lingue)

Questi modelli sono utilizzati per rappresentazioni generiche di frasi, a differenza dei precedenti checkpoints XML.

## BERT

Il seguente modello BERT può essere usato per compiti multilingue:

- `bert-base-multilingual-uncased` (Modellazione mascherata del linguaggio + Previsione della prossima frase, 102 lingue)
- `bert-base-multilingual-cased` (Modellazione mascherata del linguaggio + Previsione della prossima frase, 104 lingue)

Questi modelli non richiedono language embeddings per fare inferenza. Riescono ad identificare il linguaggio dal contesto e inferire di conseguenza.

## XLM-RoBERTa

Il seguente modello XLM-RoBERTa può essere usato per compiti multilingue:

- `xlm-roberta-base` (Modellazione mascherata del linguaggio, 100 lingue)
- `xlm-roberta-large` (Modellazione mascherata del linguaggio, 100 lingue)

XLM-RoBERTa è stato addestrato su 2.5TB di dati CommonCrawl appena creati e puliti in 100 lingue. Offre notevoli vantaggi rispetto ai modelli multilingue rilasciati in precedenza, come mBERT o XLM, in compiti come la classificazione, l'etichettatura delle sequenze e la risposta alle domande.

## M2M100

Il seguente modello M2M100 può essere usato per compiti multilingue:

- `facebook/m2m100_418M` (Traduzione)
- `facebook/m2m100_1.2B` (Traduzione)

In questo esempio, carica il checkpoint `facebook/m2m100_418M`  per tradurre dal cinese all'inglese. Puoi impostare la lingua di partenza nel tokenizer:

```py
>>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> chinese_text = "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."

>>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="zh")
>>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
```

Applica il tokenizer al testo:

```py
>>> encoded_zh = tokenizer(chinese_text, return_tensors="pt")
```

M2M100 forza l'id della lingua obiettivo come primo token generato per tradurre nella lingua obiettivo. Imposta il parametro `forced_bos_token_id` a `en` nel metodo `generate` per tradurre in inglese:

```py
>>> generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
'Do not interfere with the matters of the witches, because they are delicate and will soon be angry.'
```

## MBart

Il seguente modello MBart può essere usato per compiti multilingue:

- `facebook/mbart-large-50-one-to-many-mmt` (Traduzione automatica multilingue uno-a-molti, 50 lingue)
- `facebook/mbart-large-50-many-to-many-mmt` (Traduzione automatica multilingue molti-a-molti, 50 lingue)
- `facebook/mbart-large-50-many-to-one-mmt` (Traduzione automatica multilingue molti-a-uno, 50 lingue)
- `facebook/mbart-large-50` (Traduzione multilingue, 50 lingue)
- `facebook/mbart-large-cc25`

In questo esempio, carica il checkpoint `facebook/mbart-large-50-many-to-many-mmt` per tradurre dal finlandese all'inglese. Puoi impostare la lingua di partenza nel tokenizer:

```py
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
>>> fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia."

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
```

Applica il tokenizer sul testo:

```py
>>> encoded_en = tokenizer(en_text, return_tensors="pt")
```

MBart forza l'id della lingua obiettivo come primo token generato per tradurre nella lingua obiettivo. Imposta il parametro `forced_bos_token_id` a `en` nel metodo `generate` per tradurre in inglese:

```py
>>> generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id("en_XX"))
>>> tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
"Don't interfere with the wizard's affairs, because they are subtle, will soon get angry."
```

Se stai usando il checkpoint `facebook/mbart-large-50-many-to-one-mmt`, non hai bisogno di forzare l'id della lingua obiettivo come primo token generato altrimenti l'uso è lo stesso.