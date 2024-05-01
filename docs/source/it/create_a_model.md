<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Crea un'architettura personalizzata 

Una [`AutoClass`](model_doc/auto) deduce automaticamente il modello dell'architettura e scarica la configurazione e i pesi pre-allenati. Generalmente, noi consigliamo di usare un `AutoClass` per produrre un codice indipendente dal checkpoint. Ma gli utenti che desiderano un controllo maggiore su parametri specifici del modello possono creare un modello 🤗 Transformers personalizzato da poche classi base. Questo potrebbe essere particolarmente utile per qualunque persona sia interessata nel studiare, allenare o sperimentare con un modello 🤗 Transformers. In questa guida, approfondisci la creazione di un modello personalizzato senza `AutoClass`. Impara come:

- Caricare e personalizzare una configurazione del modello.
- Creare un'architettura modello.
- Creare un tokenizer lento e veloce per il testo.
- Creare un estrattore di caratteristiche per attività riguardanti audio o immagini.
- Creare un processore per attività multimodali.

## Configurazione

Una [configurazione](main_classes/configuration) si riferisce agli attributi specifici di un modello. Ogni configurazione del modello ha attributi diversi; per esempio, tutti i modelli npl hanno questi attributi in comune `hidden_size`, `num_attention_heads`, `num_hidden_layers` e `vocab_size`. Questi attributi specificano il numero di attention heads o strati nascosti con cui costruire un modello.

Dai un'occhiata più da vicino a [DistilBERT](model_doc/distilbert) accedendo a [`DistilBertConfig`] per ispezionare i suoi attributi:

```py
>>> from transformers import DistilBertConfig

>>> config = DistilBertConfig()
>>> print(config)
DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

[`DistilBertConfig`] mostra tutti gli attributi predefiniti usati per costruire una base [`DistilBertModel`]. Tutti gli attributi sono personalizzabili, creando uno spazio per sperimentare. Per esempio, puoi configurare un modello predefinito per:

- Provare un funzione di attivazione diversa con il parametro `activation`.
- Utilizzare tasso di drop out più elevato per le probalità di attention con il parametro `attention_dropout`.

```py
>>> my_config = DistilBertConfig(activation="relu", attention_dropout=0.4)
>>> print(my_config)
DistilBertConfig {
  "activation": "relu",
  "attention_dropout": 0.4,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

Nella funzione [`~PretrainedConfig.from_pretrained`] possono essere modificati gli attributi del modello pre-allenato:

```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

Quando la configurazione del modello ti soddisfa, la puoi salvare con [`~PretrainedConfig.save_pretrained`]. Il file della tua configurazione è memorizzato come file JSON nella save directory specificata:

```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

Per riutilizzare la configurazione del file, caricalo con [`~PretrainedConfig.from_pretrained`]:

```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
```

<Tip>

Puoi anche salvare il file di configurazione come dizionario oppure come la differenza tra gli attributi della tua configurazione personalizzata e gli attributi della configurazione predefinita! Guarda la documentazione [configuration](main_classes/configuration) per più dettagli.

</Tip>

## Modello

Il prossimo passo e di creare [modello](main_classes/models). Il modello - vagamente riferito anche come architettura - definisce cosa ogni strato deve fare e quali operazioni stanno succedendo. Attributi come `num_hidden_layers` provenienti dalla configurazione sono usati per definire l'architettura. Ogni modello condivide la classe base [`PreTrainedModel`] e alcuni metodi comuni come il ridimensionamento degli input embeddings e la soppressione delle self-attention heads . Inoltre, tutti i modelli sono la sottoclasse di [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) o [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html). Cio significa che i modelli sono compatibili con l'uso di ciascun di framework.

<frameworkcontent>
<pt>
Carica gli attributi della tua configurazione personalizzata nel modello:

```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
>>> model = DistilBertModel(my_config)
```

Questo crea modelli con valori casuali invece di pesi pre-allenati. Non sarai in grado di usare questo modello per niente di utile finché non lo alleni. L'allenamento è un processo costoso e che richiede tempo . Generalmente è meglio usare un modello pre-allenato per ottenere risultati migliori velocemente, utilizzando solo una frazione delle risorse neccesarie per l'allenamento.

Crea un modello pre-allenato con [`~PreTrainedModel.from_pretrained`]:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

Quando carichi pesi pre-allenati, la configurazione del modello predefinito è automaticamente caricata se il modello è fornito da 🤗 Transformers. Tuttavia, puoi ancora sostituire gli attributi - alcuni o tutti - di configurazione del modello predefinito con i tuoi se lo desideri:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", config=my_config)
```
</pt>
<tf>
Carica gli attributi di configurazione personalizzati nel modello:

```py
>>> from transformers import TFDistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
>>> tf_model = TFDistilBertModel(my_config)
```


Questo crea modelli con valori casuali invece di pesi pre-allenati. Non sarai in grado di usare questo modello per niente di utile finché non lo alleni. L'allenamento è un processo costoso e che richiede tempo . Generalmente è meglio usare un modello pre-allenato per ottenere risultati migliori velocemente, utilizzando solo una frazione delle risorse neccesarie per l'allenamento.

Crea un modello pre-allenoto con [`~TFPreTrainedModel.from_pretrained`]:

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

Quando carichi pesi pre-allenati, la configurazione del modello predefinito è automaticamente caricato se il modello è fornito da 🤗 Transformers. Tuttavia, puoi ancora sostituire gli attributi - alcuni o tutti - di configurazione del modello predefinito con i tuoi se lo desideri:

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", config=my_config)
```

</tf>
</frameworkcontent>

### Model head

A questo punto, hai un modello DistilBERT base i cui output sono gli *hidden states* (in italiano stati nascosti). Gli stati nascosti sono passati come input a un model head per produrre l'output finale. 🤗 Transformers fornisce un model head diverso per ogni attività fintanto che il modello supporta l'attività  (i.e., non puoi usare DistilBERT per un attività sequence-to-sequence come la traduzione).

<frameworkcontent>
<pt>
Per esempio, [`DistilBertForSequenceClassification`] è un modello DistilBERT base con una testa di classificazione per sequenze. La sequenza di classificazione head è uno strato lineare sopra gli output ragruppati.

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

Riutilizza facilmente questo checkpoint per un'altra attività passando ad un model head differente. Per un attività di risposta alle domande, utilizzerai il model head [`DistilBertForQuestionAnswering`]. La head per compiti di question answering è simile alla classificazione di sequenza head tranne per il fatto che è uno strato lineare sopra l'output degli stati nascosti (hidden states in inglese) 

```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```
</pt>
<tf>
Per esempio, [`TFDistilBertForSequenceClassification`] è un modello DistilBERT base con classificazione di sequenza head. La classificazione di sequenza head è uno strato lineare sopra gli output raggruppati.

```py
>>> from transformers import TFDistilBertForSequenceClassification

>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

Riutilizza facilmente questo checkpoint per un altra attività passando ad un modello head diverso. Per un attività di risposta alle domande, utilizzerai il model head [`TFDistilBertForQuestionAnswering`]. Il head di risposta alle domande è simile alla sequenza di classificazione head tranne per il fatto che è uno strato lineare sopra l'output degli stati nascosti (hidden states in inglese)

```py
>>> from transformers import TFDistilBertForQuestionAnswering

>>> tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```
</tf>
</frameworkcontent>

## Tokenizer

L'ultima classe base di cui hai bisogno prima di utilizzare un modello per i dati testuali è un [tokenizer](main_classes/tokenizer) per convertire il testo grezzo in tensori. Ci sono due tipi di tokenizer che puoi usare con 🤗 Transformers:

- [`PreTrainedTokenizer`]: un'implementazione Python di un tokenizer.
- [`PreTrainedTokenizerFast`]: un tokenizer dalla nostra libreria [🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/) basata su Rust. Questo tipo di tokenizer è significativamente più veloce, specialmente durante la batch tokenization, grazie alla sua implementazione Rust. Il tokenizer veloce offre anche metodi aggiuntivi come *offset mapping* che associa i token alle loro parole o caratteri originali.

Entrambi i tokenizer supportano metodi comuni come la codifica e la decodifica, l'aggiunta di nuovi token e la gestione di token speciali.

<Tip warning={true}>

Non tutti i modelli supportano un tokenizer veloce. Dai un'occhiata a questo [tabella](index#supported-frameworks) per verificare se un modello ha il supporto per tokenizer veloce. 

</Tip>

Se hai addestrato il tuo tokenizer, puoi crearne uno dal tuo file *vocabolario*: 

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt", do_lower_case=False, padding_side="left")
```

È importante ricordare che il vocabolario di un tokenizer personalizzato sarà diverso dal vocabolario generato dal tokenizer di un modello preallenato. È necessario utilizzare il vocabolario di un modello preallenato se si utilizza un modello preallenato, altrimenti gli input non avranno senso. Crea un tokenizer con il vocabolario di un modello preallenato con la classe [`DistilBertTokenizer`]:

```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

Crea un tokenizer veloce con la classe [`DistilBertTokenizerFast`]:

```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip>

Per l'impostazione predefinita, [`AutoTokenizer`] proverà a caricare un tokenizer veloce. Puoi disabilitare questo comportamento impostando `use_fast=False` in `from_pretrained`.

</Tip>

## Estrattore Di Feature

Un estrattore di caratteristiche (feature in inglese) elabora input audio o immagini. Eredita dalla classe [`~feature_extraction_utils.FeatureExtractionMixin`] base e può anche ereditare dalla classe [`ImageFeatureExtractionMixin`] per l'elaborazione delle caratteristiche dell'immagine o dalla classe [`SequenceFeatureExtractor`] per l'elaborazione degli input audio.

A seconda che tu stia lavorando a un'attività audio o visiva, crea un estrattore di caratteristiche associato al modello che stai utilizzando. Ad esempio, crea un [`ViTFeatureExtractor`] predefinito se stai usando [ViT](model_doc/vit) per la classificazione delle immagini:

```py
>>> from transformers import ViTFeatureExtractor

>>> vit_extractor = ViTFeatureExtractor()
>>> print(vit_extractor)
ViTFeatureExtractor {
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

<Tip>

Se non stai cercando alcuna personalizzazione, usa il metodo `from_pretrained` per caricare i parametri di default dell'estrattore di caratteristiche di un modello.

</Tip>

Modifica uno qualsiasi dei parametri [`ViTFeatureExtractor`] per creare il tuo estrattore di caratteristiche personalizzato:

```py
>>> from transformers import ViTFeatureExtractor

>>> my_vit_extractor = ViTFeatureExtractor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTFeatureExtractor {
  "do_normalize": false,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.3,
    0.3,
    0.3
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": "PIL.Image.BOX",
  "size": 224
}
```

Per gli input audio, puoi creare un [`Wav2Vec2FeatureExtractor`] e personalizzare i parametri in modo simile:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor()
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 16000
}
```

## Processore

Per modelli che supportano attività multimodali, 🤗 Transformers offre una classe di processore che racchiude comodamente un estrattore di caratteristiche e un tokenizer in un unico oggetto. Ad esempio, utilizziamo [`Wav2Vec2Processor`] per un'attività di riconoscimento vocale automatico (ASR). ASR trascrive l'audio in testo, quindi avrai bisogno di un estrattore di caratteristiche e di un tokenizer.

Crea un estrattore di feature per gestire gli input audio:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

Crea un tokenizer per gestire gli input di testo:

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

Combinare l'estrattore di caratteristiche e il tokenizer in [`Wav2Vec2Processor`]:

```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

Con due classi di base - configurazione e modello - e una classe di preelaborazione aggiuntiva (tokenizer, estrattore di caratteristiche o processore), puoi creare qualsiasi modello supportato da 🤗 Transformers. Ognuna di queste classi base è configurabile, consentendoti di utilizzare gli attributi specifici che desideri. È possibile impostare facilmente un modello per l'addestramento o modificare un modello preallenato esistente per la messa a punto.