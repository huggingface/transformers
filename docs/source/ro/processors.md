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

# Procesatoare

Modelele multimodale necesită un preprocesator capabil să gestioneze input-uri care combină mai mult de o modalitate. În funcție de modalitatea de input, un procesator trebuie să convertească textul într-un array de tensori, imaginile în valori de pixeli și audio într-un array cu tensori cu rata de eșantionare corectă.

De exemplu, [PaliGemma] este un model viziune-limbaj care folosește procesatorul de imagini [SigLIP] și tokenizer-ul [Llama]. O clasă [`ProcessorMixin`] înfășoară ambele tipuri de preprocesatoare, furnizând o clasă de procesor unică și unificată pentru un model multimodal.

Apelează [`~ProcessorMixin.from_pretrained`] ca să încarci un procesator. Pasează tipul de input procesatorului ca să generezi input-urile așteptate de model, input ids și valori de pixeli.

```py
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "answer en Where is the cat standing?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs
```

Acest ghid descrie clasa de procesor și cum să preprocesezi input-uri multimodale.

## Clasele de procesatoare

Toate procesatoarele moștenesc din clasa [`ProcessorMixin`] care furnizează metode precum [`~ProcessorMixin.from_pretrained`], [`~ProcessorMixin.save_pretrained`] și [`~ProcessorMixin.push_to_hub`] pentru încărcarea, salvarea și partajarea procesatoarelor pe Hub.

Există două moduri de a încărca un procesator: cu un [`AutoProcessor`] sau cu o clasă de procesor specifică modelului.

<hfoptions id="processor-class">
<hfoption id="AutoProcessor">

API-ul [AutoClass] furnizează o interfață simplă ca să încarci procesatoare fără să specifici direct clasa de model specifică căreia îi aparține.

Folosește [`~AutoProcessor.from_pretrained`] ca să încarci un procesator.

```py
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
```

</hfoption>
<hfoption id="model-specific processor">

Procesatoarele sunt și ele asociate cu o clasă specifică de model multimodal preantrenat. Poți încărca un procesator direct din clasa de model cu [`~ProcessorMixin.from_pretrained`].

```py
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
```

Poți și să încarci separat cele două tipuri de preprocesatoare, [`WhisperTokenizerFast`] și [`WhisperFeatureExtractor`].

```py
from transformers import WhisperTokenizerFast, WhisperFeatureExtractor, WhisperProcessor

tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

</hfoption>
</hfoptions>

## Preprocesare

Procesatoarele preprocesează input-urile multimodale în formatul Transformers așteptat. Există câteva combinații de modalități de input pe care un procesator le poate gestiona, cum ar fi text și audio sau text și imagine.

Task-urile de recunoaștere automată a vorbirii (ASR) necesită un procesator care poate gestiona input-uri de text și audio. Încarcă un dataset și uită-te la coloanele `audio` și `text` (poți elimina celelalte coloane care nu sunt necesare).

```py
from datasets import load_dataset

dataset = load_dataset("lj_speech", split="train")
dataset = dataset.map(remove_columns=["file", "id", "normalized_text"])
dataset[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

dataset[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

Nu uita să dai resample ratei de eșantionare ca să se potrivească cu rata de eșantionare cerută de modelul preantrenat.

```py
from datasets import Audio

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

Încarcă un procesator și pasează-i coloanele `array` audio și `text`.

```py
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

def prepare_dataset(example):
    audio = example["audio"]
    example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))
    return example
```

Aplică funcția `prepare_dataset` ca să preprocesezi dataset-ul. Procesatorul returnează `input_features` pentru coloana `audio` și `labels` pentru coloana de text.

```py
prepare_dataset(dataset[0])
```
