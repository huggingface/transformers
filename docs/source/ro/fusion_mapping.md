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

# Fusion mapping (funcție experimentală)

Fusion mapping oferă o modalitate opțională de a înlocui sub-modulele modelului la momentul încărcării, păstrând în același timp formatul original al checkpoint-ului.

Se bazează pe:

- [Monkey patching](./monkey_patching) pentru a schimba clasele de module înainte de instanțierea modelului.
- [Încărcarea dinamică de weights](./weightconverter) pentru a mapa weights între layout-ul de rulare original și cel fuzionat.

> [!WARNING]
> Fusion mapping este o funcție experimentală de încărcare. Schimbă structura modulelor la rulare și poți afecta comportamentul modelului. Folosește-o doar când dorești explicit un layout de rulare fuzionat.

## Pornire rapidă

Fusion este activat prin [`~PreTrainedModel.from_pretrained`] cu `fusion_config`:

```python
from transformers import AutoModelForImageTextToText


model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    fusion_config={"patch_embeddings": True},
)
```

În mod implicit, nu se aplică niciun fusion.
Dacă `fusion_config` este stocat în configurația modelului, `from_pretrained()` îl va reutiliza automat.

## Cum funcționează

Înregistrarea fusion are loc înainte de instanțierea modelului:

1. [`~PreTrainedModel.from_pretrained`] folosește argumentul explicit `fusion_config` sau recurge la `config.fusion_config`.
2. Registrul fusion validează numele fusion-urilor solicitate.
3. Fiecare fusion activat meta-inițializează clasa modelului țintă, filtrează opțional modulele candidate după nume și folosește `is_fusable(...)` pentru a descoperi clasele de module compatibile.
4. Clasele de înlocuire fuzionate sunt înregistrate prin [`~transformers.monkey_patching.register_patch_mapping`].
5. Regulile [`~WeightTransform`] corespunzătoare sunt generate din configurație pentru ca încărcarea checkpoint-ului să poată mapa weights în layout-ul de runtime fuzionat.
6. În mod implicit, [`~PreTrainedModel.save_pretrained`] folosește calea de conversie inversă pentru a restaura layout-ul original al checkpoint-ului. Pasează `save_original_format=False` pentru a păstra în schimb layout-ul de runtime convertit.

Aceasta permite unui fusion să folosească o structură de module de runtime diferită, încărcând în continuare din formatul original al checkpoint-ului și salvând înapoi în același format în mod implicit.

Notă: Cu mecanismul actual de monkey-patching, înregistrarea fusion este la nivel de clasă: o clasă de modul compatibilă se mapează la o clasă de înlocuire fuzionată.

## Familii de fusion curente

În prezent, `fusion_config` suportă o familie de fusion:

- `patch_embeddings`
  Activează cu:

  ```python
  fusion_config = {"patch_embeddings": True}
  ```

  Efect:
  Înlocuiește proiecțiile de patch embedding `nn.Conv3d` compatibile cu proiecții `nn.Linear` aplatizate echivalente la runtime.

## Extinderea fusion mapping

Pentru a adăuga o nouă familie de fusion:

1. Adaugă un predicat `is_fusable`.
   Acesta determină dacă un modul descoperit este compatibil cu fusion-ul.
2. Adaugă opțional `target_modules_patterns`.
   Aceasta face pasul de descoperire mai explicit prin pre-filtrarea numelor modulelor candidate înainte de `is_fusable(...)`.
3. Adaugă un factory `make_fused_class`.
   Acesta returnează clasa de înlocuire de runtime pentru o clasă de modul compatibilă.
4. Adaugă un factory `make_transforms` dacă layout-ul fuzionat necesită conversia checkpoint-ului.
   Acesta returnează regulile [`~WeightTransform`] care mapează weights între layout-urile original și fuzionat pentru o configurație dată.
5. Înregistrează noul `ModuleFusionSpec` în [`fusion_mapping.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/fusion_mapping.py).

Odată înregistrat, noul fusion devine disponibil prin `fusion_config`.

## API intern

[[autodoc]] fusion_mapping.ModuleFusionSpec

[[autodoc]] fusion_mapping.PatchEmbeddingsFusionSpec

[[autodoc]] fusion_mapping._register_module_fusion

[[autodoc]] fusion_mapping.register_fusion_patches
