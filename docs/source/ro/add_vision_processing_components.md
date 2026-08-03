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

# Adaugă componente de procesare vizuală

Adăugarea unui model vizual necesită componente de procesare a imaginilor sau videoclipurilor pe lângă abordarea standard [modulară](./modular_transformers). Modelele exclusiv pentru imagini au nevoie de procesatoare de imagini, iar modelele video au nevoie de un procesator video, ambele accesibile prin punctele de intrare [`AutoImageProcessor`] și [`AutoVideoProcessor`].

> [!NOTE]
> Pentru pașii de modelare și configurare, urmează mai întâi ghidul [modular](./modular_transformers).

## Procesatoare de imagini

Creează procesatoare de imagini când modelul consumă imagini. Backend-ul [torchvision](https://docs.pytorch.org/vision/stable/index.html) este varianta implicită și suportă accelerarea GPU. [PIL](https://pillow.readthedocs.io/en/stable/index.html) este varianta de fallback când torchvision nu este disponibil.

Ambele clase de procesatoare de imagini împart aceeași logică de preprocesare, dar au backend-uri diferite. Semnăturile constructorilor și valorile implicite trebuie să fie identice. [`AutoImageProcessor.from_pretrained`] selectează backend-ul la momentul încărcării și revine la PIL când torchvision nu este disponibil. Semnăturile nepotrivite fac ca același config salvat să se comporte diferit în funcție de mediu.

### torchvision

Creează `image_processing_<model_name>.py` cu o clasă care moștenește din [`TorchvisionBackend`]. Dacă procesatorul tău necesită parametri personalizați dincolo de [ImagesKwargs] standard, definește o clasă kwargs.

```py
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring

class MyModelImageProcessorKwargs(ImagesKwargs, total=False):
    tile_size: int  # orice kwargs specifice modelului

@auto_docstring
class MyModelImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[MyModelImageProcessorKwargs]):
        super().__init__(**kwargs)
```

> [!TIP]
> Vezi [`LlavaOnevisionImageProcessor`] ca referință.

### PIL

Creează `image_processing_pil_<model_name>.py` cu o clasă care moștenește din [`PilBackend`]. Duplică clasa kwargs aici în loc s-o importezi din fișierul torchvision, pentru că poate eșua când torchvision nu este instalat. Adaugă un comentariu `# Adapted from` ca cele două să rămână sincronizate. Pentru procesoarele fără parametri personalizați, folosește [`ImagesKwargs`] direct.

```py
from ...image_processing_backends import PilBackend
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring

# Adaptat din transformers.models.my_model.image_processing_my_model.MyModelImageProcessorKwargs
class MyModelImageProcessorKwargs(ImagesKwargs, total=False):
    tile_size: int  # orice kwargs specifice modelului

@auto_docstring
class MyModelImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[MyModelImageProcessorKwargs]):
        super().__init__(**kwargs)
```

> [!TIP]
> Vezi [`LlavaOnevisionImageProcessorPil`] ca referință.

## Procesator video

Adaugă un procesator video când modelul consumă videoclipuri sau cadre video eșantionate.

Creează `video_processing_<model_name>.py` în folder-ul modelului. [`BaseVideoProcessor`] moștenește din [`TorchvisionBackend`] și furnizează comportament comun de decodare, eșantionare a cadrelor, redimensionare, rescalare, normalizare, salvare și încărcare.

Atributele de clasă sunt valorile implicite de preprocesare. Utilizatorii le pot suprascrie la inițializare sau la apel. Folosește aceleași nume ca [`VideosKwargs`] când e posibil, cum ar fi `size`, `crop_size`, `do_resize`, `do_sample_frames`, `num_frames` și `fps`.

Definește o clasă kwargs dacă procesorul tău video necesită parametri personalizați dincolo de [`VideosKwargs`] standard. Seteaz-o ca `valid_kwargs` și folosește-o ca să adnotezi `__init__` atât pentru validarea la rulare, cât și pentru docstring-ul auto-generat.

```py
from ...processing_utils import Unpack, VideosKwargs
from ...utils import auto_docstring
from ...video_processing_utils import BaseVideoProcessor

class MyModelVideoProcessorKwargs(VideosKwargs, total=False):
    min_frames: int
    max_frames: int

@auto_docstring
class MyModelVideoProcessor(BaseVideoProcessor):
    size = {"shortest_edge": 224}
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_normalize = True
    do_sample_frames = True
    num_frames = 16
    model_input_names = ["pixel_values_videos"]
    valid_kwargs = MyModelVideoProcessorKwargs

    def __init__(self, **kwargs: Unpack[MyModelVideoProcessorKwargs]):
        super().__init__(**kwargs)
```

Suprascrie [`~BaseVideoProcessor.sample_frames`] doar când modelul necesită o regulă de eșantionare pe care eșantionatorul uniform de bază nu o poate exprima. De exemplu, unele modele impun un număr minim sau maxim de cadre sau eșantionează pe baza unor constrângeri specifice modelului.

Dacă metoda forward a modelului așteaptă un nume de input legacy, suprascrie `preprocess` și redenumește cheia după apelarea implementării de bază.

```py
class MyModelVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values"]

    def preprocess(self, videos, **kwargs):
        batch = super().preprocess(videos, **kwargs)
        batch["pixel_values"] = batch.pop("pixel_values_videos")
        return batch
```

Salvează procesatorul video cu checkpoint-ul instanțiindu-l în scriptul de conversie și apelând [`~BaseVideoProcessor.save_pretrained`]. Dacă un [`ProcessorMixin`] dă wrap procesatorului video, apelează [`~ProcessorMixin.save_pretrained`] în schimb. Nu crea sau edita manual fișierele de configurație pentru preprocesare.

> [!TIP]
> Vezi [`Qwen3VLVideoProcessor`] ca referință.

## Înregistrarea claselor

Expune clasele de procesare din `__init__.py`-ul pachetului modelului. Urmează pattern-ul de import lazy folosit de modelele vecine și protejează importurile cu aceleași dependențe opționale necesare fiecărui backend.

Mapează noile clase la configurația modelului pentru ca clasele `Auto` să le poată încărca. Fișierul de mapare auto generat are un avertisment la început. Nu îl edita manual. Adaugă sau actualizează configurația modelului, apoi rulează:

```bash
python utils/check_auto.py --fix_and_overwrite
```

După generarea mapării, verifică că tipul de model apare în mapările relevante din `src/transformers/models/auto/auto_mappings.py`.

- `IMAGE_PROCESSOR_MAPPING_NAMES` pentru [`AutoImageProcessor`]
- `VIDEO_PROCESSOR_MAPPING_NAMES` pentru [`AutoVideoProcessor`]

## Testare

Adaugă teste pentru fiecare componentă de procesare vizuală în directorul de teste al modelului. Testele pentru procesoarele de imagini și video urmează același pattern. Moștenești din mixin-ul comun, indici clasele de procesare fast și slow când descoperirea automată nu este suficientă, furnizezi kwargs de inițializare specifice modelului și suprascrii numele inputului când modelul folosește o cheie de ieșire non-implicită.

### Teste pentru procesoare de imagini

Testele pentru procesoarele de imagini se află de obicei în `tests/models/<model_name>/test_image_processing_<model_name>.py` și moștenesc din [`ImageProcessingTestMixin`].

Mixin-ul de procesare a imaginilor găsește clasele de procesoare de imagini din `IMAGE_PROCESSOR_MAPPING_NAMES`. Expune valorile implicite specifice modelului prin `image_processor_dict`. Adaugă un obiect tester doar când ai nevoie de input-uri dummy reutilizabile sau metode ajutătoare pentru teste focalizate.

```py
from transformers.testing_utils import require_torch, require_vision
from ...test_image_processing_common import ImageProcessingTestMixin

@require_torch
@require_vision
class MyModelImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    @property
    def image_processor_dict(self):
        return {"size": {"shortest_edge": 224}, "do_resize": True}
```

Adaugă teste focalizate pentru comportamentul pe care mixin-ul nu îl poate deduce, cum ar fi regulile de redimensionare personalizate sau kwargs specifice modelului.

### Teste pentru procesoare video

Testele pentru procesoarele video trăiesc de obicei în `tests/models/<model_name>/test_video_processing_<model_name>.py` și moștenesc din [`VideoProcessingTestMixin`]. Setează `fast_video_processing_class`, definește `video_processor_dict` și suprascrie `input_name` dacă modelul folosește o altă cheie decât `pixel_values_videos`.

```py
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available
from ...test_video_processing_common import VideoProcessingTestMixin

@require_torch
@require_vision
class MyModelVideoProcessingTest(VideoProcessingTestMixin, unittest.TestCase):
    fast_video_processing_class = MyModelVideoProcessor if is_torchvision_available() else None
    input_name = "pixel_values_videos"

    @property
    def video_processor_dict(self):
        return {"size": {"shortest_edge": 224}, "num_frames": 16}
```

Adaugă teste video focalizate pentru eșantionarea cadrelor, gestionarea metadatelor, input-urile video decodate, input-urile de tip listă de cadre și formele de ieșire. Dacă procesatorul tău redenumește `pixel_values_videos`, verifică dacă cheia redenumită este returnată.

Dacă modelul are și un [`ProcessorMixin`] care înfășoară procesatorul de imagini sau video, adaugă `tests/models/<model_name>/test_processing_<model_name>.py` și moștenește din [`ProcessorTesterMixin`]. Setează `processor_class` și suprascrie metodele de clasă `_setup_<component>()` pentru componentele care nu pot fi construite fără argumente. Folosește `_setup_test_attributes()` ca să expui token-urile placeholder folosite de testele comune ale procesatorului.

```py
from ...test_processing_common import ProcessorTesterMixin

class MyModelProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MyModelProcessor

    @classmethod
    def _setup_image_processor(cls):
        return cls._get_component_class_from_processor("image_processor")(size={"shortest_edge": 224})

    @classmethod
    def _setup_video_processor(cls):
        return cls._get_component_class_from_processor("video_processor")(num_frames=2)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = getattr(processor, "image_token", "")
        cls.video_token = getattr(processor, "video_token", "")
```

## Pașii următori

- Citește ghidul [Auto-generarea docstring-urilor](./auto_docstring) ca să auto-generezi docstring-uri consistente cu `@auto_docstring`.
- Citește ghidurile [Procesoare de imagini](./image_processors) și [Procesoare video](./video_processors) pentru comportamentul de preprocesare orientat către utilizator.
