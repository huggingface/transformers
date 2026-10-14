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

# Adaugă componente de procesare audio

Modelele audio necesită un feature extractor accesibil prin punctul de intrare [`AutoFeatureExtractor`].

> [!NOTE]
> Pentru pașii de modelare și configurare, urmează mai întâi ghidul [modular](./modular_transformers).

## Feature extractor

Adaugă un feature extractor când modelul consumă audio brut sau features derivate din audio.

Creează `feature_extraction_<model_name>.py` în directorul modelului. Moștenește din [`SequenceFeatureExtractor`] pentru ca noua clasă să obțină comportamentul comun de padding, truncare, salvare și încărcare.

```py
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor


class MyModelFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "attention_mask"]

    def __init__(self, feature_size=80, sampling_rate=16000, padding_value=0.0, **kwargs):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

    def __call__(self, raw_speech, sampling_rate=None, **kwargs):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"`sampling_rate` must be {self.sampling_rate}, but got {sampling_rate}.")

        # Convertește raw_speech în features ale modelului aici.
        ...
```

Ține constructorul mic și serializabil. Stochează fiecare valoare necesară pentru reproducerea preprocesării ca atribut de instanță și evită stocarea valorilor exclusiv de runtime, cum ar fi fișiere deschise, dispozitive sau array-uri audio decodate.

Metoda `__call__` trebuie să valideze sampling rate-ul de input când utilizatorii pasează `sampling_rate`. Dacă rata de input diferă de rata așteptată de model, ridică o eroare în loc să refaci eșantionarea în tăcere.

Salvează feature extractor-ul cu checkpoint-ul instanțiindu-l în scriptul de conversie și apelând [`~FeatureExtractionMixin.save_pretrained`]. Nu crea sau edita manual fișierele de config de preprocesare.

> [!TIP]
> Vezi [`Gemma4AudioFeatureExtractor`] ca referință.

## Înregistrarea claselor

Expune noile clase din `__init__.py`-ul pachetului modelului. Urmează pattern-ul de import lazy folosit de modelele vecine și protejează importurile cu aceleași dependențe opționale necesare clasei.

Mapează noua clasă la config-ul modelului pentru ca [`AutoFeatureExtractor`] să o poată încărca. Adaugă o intrare în `FEATURE_EXTRACTOR_MAPPING_NAMES` din `src/transformers/models/auto/feature_extraction_auto.py`, urmând pattern-ul intrărilor vecine. Apoi verifică dacă tipul de model apare acolo sub `FEATURE_EXTRACTOR_MAPPING_NAMES` pentru [`AutoFeatureExtractor`].

- `FEATURE_EXTRACTOR_MAPPING_NAMES` pentru [`AutoFeatureExtractor`]

## Testare

Adaugă teste pentru fiecare componentă de procesare audio în directorul de teste al modelului. Testele pentru feature extractor se află de obicei în `tests/models/<model_name>/test_feature_extraction_<model_name>.py`.

Pentru feature extractor-ele care moștenesc din [`SequenceFeatureExtractor`], moștenește din [`SequenceFeatureExtractionTestMixin`]. Mixin-ul acoperă comportamentul de salvare și încărcare, padding, truncare, conversia tensorilor și proprietățile comune ale feature extractor-ului. Furnizează un obiect tester cu `prepare_feat_extract_dict()` și `prepare_inputs_for_common()` pentru ca mixin-ul să poată instanția feature extractor-ul și construi input-uri audio dummy scurte.

```py
from ...test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin

class MyModelFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):
    feature_extraction_class = MyModelFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = MyModelFeatureExtractionTester(self)
```

Adaugă teste focalizate pentru comportamentul specific modelului pe care mixin-ul nu îl cunoaște. Pentru feature extractors audio, asta înseamnă de obicei verificarea formei feature-ului returnat de `__call__`, validarea că un `sampling_rate` incorect cauzează o eroare și verificarea oricărei normalizări personalizate sau calcul de features.

Dacă modelul are și un [`ProcessorMixin`] care dă wrap feature extractor-ului, adaugă `tests/models/<model_name>/test_processing_<model_name>.py` și moștenește din [`ProcessorTesterMixin`]. Setează `processor_class` și suprascrie metodele de clasă `_setup_<component>()` pentru componentele care nu pot fi construite fără argumente. Folosește `_setup_test_attributes()` ca să expui token-urile placeholder folosite de testele comune ale procesorului.

```py
from ...test_processing_common import ProcessorTesterMixin

class MyModelProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MyModelProcessor

    @classmethod
    def _setup_feature_extractor(cls):
        return cls._get_component_class_from_processor("feature_extractor")(sampling_rate=16000)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.audio_token = getattr(processor, "audio_token", "")
```

## Pașii următori

- Citește ghidul [Auto-generarea docstring-urilor](./auto_docstring) ca să auto-generezi docstring-uri consistente cu `@auto_docstring`.
- Citește ghidul [Feature extractors] pentru comportamentul de preprocesare orientat către utilizator.
