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

# Backbone-uri

Task-urile de computer vision de nivel mai ridicat, cum ar fi detecția obiectelor sau segmentarea imaginilor, folosesc mai multe modele împreună ca să genereze o predicție. Un model separat este folosit pentru *backbone*, neck și head. Backbone-ul extrage feature-uri utile dintr-o imagine de intrare într-un feature map, neck-ul combină și procesează feature map-urile, iar head-ul le folosește ca să facă o predicție.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Backbone.png"/>
</div>

Încarcă un backbone cu [`~PreTrainedConfig.from_pretrained`] și folosește parametrul `out_indices` ca să determini din ce layer, dat prin index, să extragi un feature map.

```py
from transformers import AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

Acest ghid descrie clasa backbone, backbone-urile din biblioteca [timm](https://hf.co/docs/timm/index) și cum să extragi feature-uri cu ele.

## Clasele de backbone

Există două clase de backbone.

- [`~transformers.utils.BackboneMixin`] îți permite să încarci un backbone și include funcții pentru extragerea feature map-urilor și a indicilor din config.
- [`~transformers.utils.BackboneConfigMixin`] îți permite să setezi, aliniezi și verifici feature map-ul și indicii dintr-o configurație de backbone.

Consultă documentația API [Backbone] ca să verifici ce modele suportă un backbone.

Există două moduri de a încărca un backbone Transformers: [`AutoBackbone`] și o clasă de backbone specifică modelului.

<hfoptions id="backbone-classes">
<hfoption id="AutoBackbone">

API-ul [AutoClass] încarcă automat un model de viziune preantrenat cu [`~PreTrainedConfig.from_pretrained`] ca backbone dacă este suportat.

Setează parametrul `out_indices` la layer-ul din care vrei să obții feature map-ul. Dacă știi numele layer-ului, poți folosi și `out_features`. Acești parametri pot fi folosiți alternativ, dar dacă îi folosești pe amândoi, asigură-te că se referă la același layer.

Când `out_indices` sau `out_features` nu este folosit, backbone-ul returnează feature map-ul din ultimul layer. Codul de exemplu de mai jos folosește `out_indices=(1,)` ca să obțină feature map-ul din primul layer.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Swin%20Stage%201.png"/>
</div>

```py
from transformers import AutoImageProcessor, AutoBackbone

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
```

</hfoption>
<hfoption id="model-specific backbone">

Când știi că un model suportă un backbone, poți încărca backbone-ul și neck-ul direct în configurația modelului. Pasează configurația modelului ca să îl inițializezi pentru un task.

Exemplul de mai jos încarcă un backbone [ResNet] și neck pentru un head de segmentare a instanțelor [MaskFormer].

Reține că inițializarea din config creează modelul cu weights aleatorii. Dacă vrei să încarci un model preantrenat, folosește API-ul `from_pretrained`.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

backbone_config = AutoConfig.from_pretrained("microsoft/resnet-50")
config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

O altă opțiune este să încarci configurația backbone-ului separat și apoi să o pasezi la `backbone_config` din configurația modelului.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig

# instanțiază configurația backbone-ului
backbone_config = ResNetConfig()
# încarcă backbone-ul în model
config = MaskFormerConfig(backbone_config=backbone_config)
# atașează backbone-ul la head-ul modelului
model = MaskFormerForInstanceSegmentation(config)
```

</hfoption>
</hfoptions>

## Backbone-uri timm

[timm] este o colecție de modele de viziune pentru antrenare și inferență. Transformers suportă modelele timm ca backbone-uri cu clasele [`TimmBackbone`] și [`TimmBackboneConfig`]. Setează checkpoint-ul de backbone necesar în `backbone` ca să creezi un model cu backbone timm cu weights inițializate aleatoriu.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

backbone_config = TimmBackboneConfig(backbone="resnet50", out_indices=[-1])
config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

Poți și să apelezi explicit clasa [`TimmBackboneConfig`] ca să încarci și să creezi un backbone timm preantrenat.

```py
from transformers import TimmBackboneConfig

backbone_config = TimmBackboneConfig("resnet50")
```

Pasează configurația backbone-ului la configurația modelului și instanțiază head-ul modelului, [`MaskFormerForInstanceSegmentation`], cu backbone-ul.

```py
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation

config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)
```

## Extragerea feature-urilor

Backbone-ul este folosit ca să extragă feature-uri din imagini. Pasează o imagine prin backbone ca să obții feature map-urile.

Încarcă și preprocesează o imagine și pasează-o backbone-ului. Exemplul de mai jos extrage feature map-urile din primul layer.

```py
from transformers import AutoImageProcessor, AutoBackbone
import torch
from PIL import Image
import requests

model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
```

Feature-urile sunt stocate și accesate din atributul `feature_maps` al ieșirilor.

```py
feature_maps = outputs.feature_maps
list(feature_maps[0].shape)
[1, 96, 56, 56]
```
