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

# Procesatoare de imagini

Procesatoarele de imagini convertesc imaginile în valori de pixeli, tensori care reprezintă culorile și dimensiunea imaginilor. Valorile de pixeli sunt input-urile unui model de viziune. Ca să se asigure că un model preantrenat primește input-ul corect, un procesator de imagini poate efectua următoarele operații ca imaginea să fie exact ca imaginile pe care modelul a fost preantrenat.

- decupare centrată sau redimensionare a imaginii
- normalizarea sau rescalarea valorilor de pixeli

Folosește [`~ImageProcessingMixin.from_pretrained`] ca să încarci configurația unui procesator de imagini (dimensiunea imaginii, dacă să normalizeze și rescaleze etc.) de la un model de viziune de pe Hub-ul Hugging Face sau dintr-un director local. Configurația pentru fiecare model preantrenat este salvată într-un fișier [preprocessor_config.json](https://huggingface.co/google/vit-base-patch16-224/blob/main/preprocessor_config.json).

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

Pasează o imagine procesatorului de imagini ca să o transformi în valori de pixeli și setează `return_tensors="pt"` ca să returnezi tensori PyTorch. Poți să printezi input-urile ca să vezi cum arată imaginea ca tensor.

```py
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/image_processor_example.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = image_processor(image, return_tensors="pt")
```

Acest ghid acoperă clasa procesatorului de imagini și cum să preprocesezi imagini pentru modelele de viziune.

## Clasele de procesatoare de imagini

Procesatoarele de imagini folosesc o arhitectură bazată pe două backend-uri:

- [`TorchvisionBackend`] — implementarea implicită bazată pe [torchvision](https://pytorch.org/vision/stable/index.html). Accelerată GPU și de până la 33x mai rapidă decât backend-ul PIL pentru batch-uri de input-uri [torch.Tensor](https://pytorch.org/docs/stable/tensors.html). Toate modelele suportă acest backend; modelele mai noi suportă doar acest backend.
- [`PilBackend`] — alternativa PIL/NumPy. Portabilă și doar pe CPU. Disponibilă doar pentru modelele mai vechi, unde este utilă ca să reproduci ieșirile numerice exacte ale implementării originale.

Backend-ul activ pe un procesator încărcat poate fi inspectat cu atributul `backend` (de ex., `processor.backend == "torchvision"`). Fiecare procesator de imagini subclasează [`ImageProcessingMixin`] care furnizează metodele [`~ImageProcessingMixin.from_pretrained`] și [`~ImageProcessingMixin.save_pretrained`].

Există două moduri în care poți încărca un procesator de imagini: cu [`AutoImageProcessor`] sau direct dintr-o clasă specifică modelului.

<hfoptions id="image-processor-classes">
<hfoption id="AutoImageProcessor">

API-ul [AutoClass] furnizează o metodă convenabilă de a încărca un procesator de imagini fără să specifici direct modelul cu care procesatorul de imagini este asociat.

Folosește [`~AutoImageProcessor.from_pretrained`] cu argumentul `backend` ca să selectezi backend-ul. Când `backend` este omis (implicit), torchvision este ales când este instalat, iar PIL este folosit altfel. Reține că `backend="pil"` este suportat doar pentru modele mai vechi; modelele mai noi expun doar backend-ul torchvision.

> **Notă:** un set mic de modele mai vechi (Chameleon, Flava, Idefics3, SmolVLM) folosesc interpolarea Lanczos pe care torchvision nu o suportă, deci revin mereu la backend-ul PIL indiferent de disponibilitatea torchvision. Pasează `backend="torchvision"` explicit ca să suprascrii asta.

```py
from transformers import AutoImageProcessor

# Implicit: alege torchvision dacă este disponibil, altfel pil
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Solicită explicit backend-ul torchvision
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", backend="torchvision")

# Solicită explicit backend-ul PIL (doar pentru modele care îl suportă)
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", backend="pil")
```

</hfoption>
<hfoption id="model-specific image processor">

Fiecare procesator de imagini este asociat cu un model de viziune preantrenat specific, iar configurația sa conține dimensiunea așteptată a modelului și parametrii de normalizare.

Încarcă procesatorul cu backend-ul torchvision direct din clasa specifică modelului.

```py
from transformers import ViTImageProcessor

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

Pentru modelele care îl suportă, poți încărca backend-ul PIL cu clasa cu sufixul `Pil`. Util când ai nevoie de paritate numerică exactă cu implementarea originală.

```py
from transformers import ViTImageProcessorPil

image_processor = ViTImageProcessorPil.from_pretrained("google/vit-base-patch16-224")
```

</hfoption>
</hfoptions>

## Procesatoare cu backend torchvision

[`TorchvisionBackend`] este backend-ul **implicit**. Asigură-te că [torchvision](https://pytorch.org/get-started/locally/#mac-installation) este instalat, apoi încarcă-l cu `backend="torchvision"` (sau omite pur și simplu `backend`, deoarece torchvision este selectat automat când este disponibil).

```py
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", backend="torchvision")
```

Controlează pe ce device se face procesarea cu argumentul `device`. Procesarea se face implicit pe același device ca input-ul dacă input-urile sunt tensori, altfel revine la CPU. Exemplul de mai jos rulează procesarea pe GPU.

```py
from torchvision.io import read_image
from transformers import DetrImageProcessor

images = read_image("image.jpg")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(images, return_tensors="pt", device="cuda")
```

<details>
<summary>Benchmarks</summary>

Benchmark-urile sunt obținute de pe o instanță [AWS EC2 g5.2xlarge](https://aws.amazon.com/ec2/instance-types/g5/) cu un GPU NVIDIA A10G Tensor Core.

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_padded.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_batched_compiled.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_single.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_batched.png" />
</div>
</details>

## Preprocesare

Modelele de viziune din Transformers se așteaptă la input ca tensori PyTorch de valori de pixeli. Un procesator de imagini gestionează conversia imaginilor în valori de pixeli, reprezentate prin dimensiunea batch-ului, numărul de canale, înălțimea și lățimea. Ca să realizeze asta, o imagine este redimensionată (decupată central) și valorile de pixeli sunt normalizate și rescalate la valorile așteptate de model.

Preprocesarea imaginilor nu este același lucru cu *augmentarea imaginilor*. Augmentarea imaginilor face modificări (luminozitate, culori, rotație etc.) unei imagini cu scopul de a crea exemple de antrenare noi sau de a preveni overfitting-ul. Preprocesarea imaginilor face modificări unei imagini cu scopul de a se potrivi formatului de input așteptat de un model preantrenat.

De obicei, imaginile sunt augmentate (ca să crești performanța) și apoi preprocesate înainte de a fi pasate unui model. Poți folosi orice bibliotecă ([Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb), [Kornia](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)) pentru augmentare și un procesator de imagini pentru preprocesare.

Acest ghid folosește modulul [transforms](https://pytorch.org/vision/stable/transforms.html) din torchvision pentru augmentare.

Începe prin a încărca un eșantion mic din dataset-ul [food101](https://hf.co/datasets/food101).

```py
from datasets import load_dataset

dataset = load_dataset("ethz/food101", split="train[:100]")
```

Din modulul [transforms](https://pytorch.org/vision/stable/transforms.html), folosește API-ul [Compose](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) ca să înlănțuiești [RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) și [ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html). Transformările astea decupează și redimensionează aleatoriu o imagine și ajustează aleatoriu culorile imaginii.

Dimensiunea imaginii la care să decupezi aleatoriu poate fi obținută de la procesatorul de imagini. Pentru unele modele se așteaptă valori exacte pentru înălțime și lățime, iar pentru altele este necesar doar `shortest_edge`.

```py
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

Aplică transformările pe imagini și convertește-le la formatul RGB. Apoi pasează imaginile augmentate procesatorului de imagini ca să returneze valorile de pixeli.

Parametrul `do_resize` este setat la `False` pentru că imaginile au fost deja redimensionate în pasul de augmentare de [RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html). Dacă nu augmentezi imaginile, procesatorul de imagini le redimensionează și normalizează automat cu valorile `image_mean` și `image_std`. Aceste valori se găsesc în fișierul de configurație al preprocesatorului.

```py
def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples
```

Aplică funcția combinată de augmentare și preprocesare întregului dataset din mers cu [`~datasets.Dataset.set_transform`].

```py
dataset.set_transform(transforms)
```

Convertește valorile de pixeli înapoi într-o imagine ca să vezi cum a fost augmentată și preprocesată imaginea.

```py
import numpy as np
import matplotlib.pyplot as plt

img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">înainte</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">după</figcaption>
  </div>
</div>

Pentru alte task-uri de viziune, cum ar fi detecția obiectelor sau segmentarea, procesatorul de imagini include metode de post-procesare ca să convertească ieșirile brute ale modelului în predicții cu sens, cum ar fi bounding box-uri sau hărți de segmentare.

### Padding

Unele modele, ca [DETR], aplică [augmentarea la scară](https://paperswithcode.com/method/image-scale-augmentation) în antrenare, ceea ce poate face ca imaginile dintr-un batch să aibă dimensiuni diferite. Imaginile cu dimensiuni diferite nu pot fi grupate în batch-uri.

Ca să rezolvi asta, faci padding imaginilor cu token-ul special de padding `0`. Folosește metoda [pad](https://github.com/huggingface/transformers/blob/9578c2597e2d88b6f0b304b5a05864fd613ddcc1/src/transformers/models/detr/image_processing_detr.py#L1151) ca să faci padding imaginilor și definește o funcție de collatare personalizată ca să le grupezi în batch-uri.

```py
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
```
