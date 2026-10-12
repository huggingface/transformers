<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Procesatoare multimodale

Un procesator combină un tokenizer cu unul sau mai multe procesatoare de modalitate, cum ar fi un procesator de imagini, un procesator video sau un feature extractor. Expune o singură metodă `__call__` care direcționează fiecare input la componenta potrivită și îmbină ieșirile într-un singur dicționar.

Unele modele multimodale intercalează textul cu imagini, videoclipuri sau audio. Pentru aceste modele, [`ProcessorMixin`] poate înlocui token-urile placeholder precum `<image>`, `<video>` și `<audio>` cu pattern-ul de token pe care îl așteaptă modelul.


## Adăugarea unui procesator nou

Definești o clasă de procesator creând `src/transformers/models/<model>/processing_<my_model_name>.py` și subclasând `ProcessorMixin`. Asigură-te că definești un obiect `TypedDict` cu valori implicite și îl atribui ca `cls.valid_processor_kwargs`.

```python
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack

class MyModelProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MyModelImageProcessorKwargs
    _defaults = {
        "text_kwargs": {"padding": True},
        "images_kwargs": {"do_convert_rgb": True},
    }

class MyModelProcessor(ProcessorMixin):
    valid_processor_kwargs = MyModelProcessorKwargs

    def __init__(self, image_processor, tokenizer, chat_template=None, **kwargs):
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )
```

Implementează `replace_<modality>_token` dacă e nevoie. Acesta primește dicționarul complet de ieșire de la subprocesator și indexul inputului curent, returnând șirul de înlocuire expandat pentru acel input. Șirul de înlocuire este ceea ce modelul așteaptă în secvența de input.

Dacă modelul nu folosește deloc repetarea placeholder-ului (fără `image_token` definit), nu trebuie să suprascrii această metodă. Lasă `self.image_token` nesetat și clasa de bază sare peste înlocuire complet.

```python
def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
    num_crops = image_inputs["num_crops"][image_idx]
    return f"{self.boi_token}{self.image_token * self.num_image_tokens * num_crops}{self.eoi_token}"
```

Opțional, suprascrie metodele `prepare_inputs_layout` și `validate_inputs` dacă modelul necesită o structură specifică de input înainte ca procesarea să înceapă, precum reordonarea imaginilor ca o listă imbricată sau o validare specifică modelului pe lângă verificările comune.

```python
def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
    # Apelează `super()` ca să aplici mai întâi pașii comuni de pregătire
    images, text, videos, audio = super().prepare_inputs_layout(images, text, videos, audio)
    if images is not None:
        images = make_nested_list_of_images(images)
    return images, text, videos, audio

def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
    super().validate_inputs(images=images, text=text, **kwargs)
    if text is not None and images is not None:
        n_tokens = [s.count(self.image_token) for s in text]
        n_images = [len(img_list) for img_list in images]
        if n_tokens != n_images:
            raise ValueError(
                f"Number of {self.image_token} tokens in text {n_tokens} does not match "
                f"number of images {n_images}."
            )
```

> [!TIP]
> Vezi [`Gemma4Processor`] și [`Qwen2VLProcessor`] ca referință.

## Testare

Toate procesatoarele multimodale ar trebui să aibă o clasă de test care moștenește din [`ProcessorTesterMixin`]. Mixin-ul acesta oferă o suită standard care acoperă tokenizarea, procesarea imaginilor, batch-urile și codificarea round-trip.

```python
# tests/models/my_model_name/test_processor_<my_model_name>.py

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available
from ...test_processing_common import ProcessorTesterMixin

if is_vision_available():
    from transformers import MyModelProcessor

@require_vision
class MyModelProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MyModelProcessor

    def get_processor(self):
        return MyModelProcessor.from_pretrained("hf-internal-testing/my-model-test")
```
