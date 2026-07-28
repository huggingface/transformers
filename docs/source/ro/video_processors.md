<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Procesatoare video

Un **procesator video** este un utilitar responsabil cu pregătirea feature-urilor de input pentru modelele video, cât și cu post-procesarea output-urilor lor. Furnizează transformări precum redimensionarea, normalizarea și conversia în PyTorch.

Procesatorul video extinde funcționalitatea procesatoarelor de imagini, permițând modelelor să gestioneze videoclipuri cu un set distinct de argumente față de imagini. Servește drept punte între datele video brute și model, asigurând că feature-urile de input sunt optimizate pentru VLM.

Folosește [`~BaseVideoProcessor.from_pretrained`] ca să încarci configurația unui procesator video (dimensiunea imaginii, dacă să normalizeze și rescaleze etc.) de la un model video de pe Hub-ul Hugging Face sau dintr-un director local. Configurația pentru fiecare model preantrenat ar trebui salvată într-un fișier [video_preprocessor_config.json], dar modelele mai vechi pot avea configurația salvată în fișierul [preprocessor_config.json](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/blob/main/preprocessor_config.json). Reține că acesta din urmă este mai puțin preferat și va fi eliminat în viitor.

## Exemplu de utilizare

Iată un exemplu de cum să încarci un procesator video cu modelul [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf):

```python
from transformers import AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
```

Dacă folosești un procesator de imagini de bază pentru videoclipuri, acesta procesează datele video tratând fiecare cadru ca o imagine individuală și aplică transformările cadru cu cadru. Deși funcțională, această abordare nu este foarte eficientă. Folosind `AutoVideoProcessor` poți să profiți de **procesatoarele video rapide**, care valorifică biblioteca [torchvision](https://pytorch.org/vision/stable/index.html). Procesatoarele rapide gestionează întregul batch de videoclipuri dintr-o dată, fără să itereze pe fiecare video sau cadru. Aceste actualizări introduc accelerarea GPU și cresc semnificativ viteza de procesare, mai ales pentru task-uri care necesită un throughput ridicat.

Procesatoarele video rapide sunt disponibile pentru toate modelele și sunt încărcate implicit când se inițializează un `AutoVideoProcessor`. Când folosești un procesator video rapid, poți seta și argumentul `device` ca să specifici device-ul pe care se face procesarea. Implicit, procesarea se face pe același device ca input-urile dacă input-urile sunt tensori, altfel pe CPU. Ca să câștigi și mai multă viteză, poți compila procesatorul când folosești `cuda` ca device.

```python
import torch
from transformers.video_utils import load_video
from transformers import AutoVideoProcessor

video = load_video("video.mp4")
processor = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda")
processor = torch.compile(processor)
processed_video = processor(video, return_tensors="pt")
```
