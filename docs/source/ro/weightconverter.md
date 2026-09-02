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

# Încărcarea dinamică de weights

Checkpoint-urile sunt des serializate într-un format diferit de cel pe care un model îl așteaptă la rulare. Scenariile comune includ:

1. **Weights fuzionate**: Checkpoint-urile stochează weights separate `gate_proj` și `up_proj`, dar modelul utilizează un weight fuzionat `gate_up_proj` pentru eficiență.
2. **Consolidare expertă MoE**: Weights experte individuale (`experts.0.weight`, `experts.1.weight`, ...) trebuie să fie combinate într-un singur tensor 3D.
3. **Denumire veche**: Checkpoint-urile vechi utilizează diferite convenții de numire (e.g., `LayerNorm.gamma` vs `LayerNorm.weight`).
4. **Modele compuse**: Un model lingvistic vizual conține două sub-module `PreTrainedModel`, fiecare cu propria convenție pentru checkpoint-uri.
5. **Quantization**: Weights ar putea fi stocate în formate quantized care necesită deserializare.

Încărcarea dinamică de weights rezolvă această problemă prin aplicarea de operații planificate și reversibile asupra tensorilor din checkpoint pe măsură ce sunt încărcați. Transformers expune aceasta prin [`WeightConverter`] și [`WeightRenaming`], care descriu cum una sau mai multe chei de checkpoint se mapează la unul sau mai mulți parametri ai modelului și ce [`ConversionOps`] compozabile ar trebui să ruleze pe tensorii potriviți. Această abordare se adaptează la noi layout-uri de weights, suportă mixture-of-experts (MoE) quantized și se integrează cu tensor parallelism.

Acest ghid demonstrează cum să utilizezi [`WeightConverter`] pentru a converti tensori. Mapările de conversie se află în [`conversion_mapping.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py); o mapare înregistrată este indexată fie printr-un string `model_type` (e.g. `"mixtral"`) fie printr-un nume de clasă (e.g. `"LlavaModel"`).

## Pipeline-ul complet de încărcare

Toate modelele trec prin sistemul de încărcare dinamică de weights. Maparea de conversie este un **pas opțional în cadrul acelui sistem** care se activează doar atunci când modelul are înregistrări pentru numele clasei sale sau `model_type`.

```
Fișier Checkpoint → from_pretrained() → convert_and_load_state_dict_in_model()
                                              ↓
                         ┌───────────────────────────────────────────────────────────┐
                         │  Pentru fiecare weight din checkpoint:                    │
                         │  1. Potrivește cheia sursă redenumită/procesată cu        │
                         │     parametrul modelului                                  │
                         │  2. Fragmentează weight-ul și trimite pe dispozitiv       │
                         │     (asincron)                                            │
                         │  3. Colectează tensorii cu același source_pattern         │
                         │     împreună (e.g. experți MoE, gate_up_proj)             │
                         │  4. Aplică dequantization/deserializarea (dacă pre-quant)  │
                         │  5. Aplică conversia (dacă este definită)                 │
                         │  6. Aplică quantizarea (dacă activată și pasul 4 nu       │
                         │     a fost folosit)                                       │
                         │  7. Setează parametrul pe model                           │
                         └───────────────────────────────────────────────────────────┘
```

| Pas | Când se activează |
|-----|-------------------|
| Încărcare dinamică | Întotdeauna, pentru toate modelele |
| Mapare de conversie | Doar când clasa modelului sau `model_type` este înregistrată în `_MODEL_TO_CONVERSION_PATTERN` |
| Sharding TP | Doar când `tp_plan="auto"` și modelul are `base_model_tp_plan` |
| Dequantization/deserializare | Doar la încărcarea unui checkpoint pre-quantized |
| Quantization | Doar când există o configurație de quantization și weights nu sunt pre-quantized |

### Modele dense (e.g., Llama)

Pentru majoritatea modelelor dense, formatul checkpoint-ului corespunde direct formatului modelului, așadar  nu este necesară nicio mapare de conversie. Unele modele pot necesita, totuși, redenumire (e.g., convenții de numire vechi). Sharding-ul TP se aplică în continuare când este activat.

```
Checkpoint:                             Model:
model.layers.0.self_attn.q_proj.weight  →  model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight  →  model.layers.0.self_attn.k_proj.weight
model.layers.0.mlp.gate_proj.weight     →  model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight       →  model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight     →  model.layers.0.mlp.down_proj.weight
```

Checkpoint-urile vechi pot folosi convenții de numire mai vechi care sunt gestionate de redenumiri integrate aplicate tuturor modelelor:

```
Checkpoint:                          Model:
LayerNorm.gamma              →       LayerNorm.weight
LayerNorm.beta               →       LayerNorm.bias
```

### Modele MoE (e.g., Mixtral)

Pentru modelele MoE, formatul checkpoint-ului diferă de formatul modelului. Maparea de conversie transformă weights separate ale experților în tensori 3D fuzionați, iar sharding-ul TP se aplică după conversie.

```
Checkpoint:                              Model:
experts.0.w1.weight  ─┐
experts.1.w1.weight   │ MergeModulelist
...                   ├───────────────→  experts.gate_up_proj (8, hidden, 2*intermediate)
experts.0.w3.weight   │ + Concatenate
experts.1.w3.weight  ─┘
```

### Modele compuse (e.g., viziune-limbaj)

Un `PreTrainedModel` poate conține alte sub-module `PreTrainedModel`. Fiecare sub-model poate avea propria sa mapare de conversie, înregistrată fie împotriva numelui clasei sale, fie împotriva `model_type`-ului. Când modelul părinte este încărcat, [`get_model_conversion_mapping`] parcurge sub-modelele în ordine depth-first, colectează mapările lor și **aplică automat scope-ul** fiecărei transformări la calea sub-modulului său prin `scope_prefix`.

```
Model compus:                                 Mapări per-submodel (auto-scoped):
LlavaForConditionalGeneration
├── vision_model: SiglipVisionModel    →     mapare SiglipVisionModel (scope="vision_model")
└── language_model: LlamaForCausalLM   →     mapare LlamaForCausalLM  (scope="language_model")
```

`scope_prefix` este calea punctată a sub-modulului (`"vision_model"`, `"language_model.model"`, etc.). O transformare cu scope se declanșează doar pentru cheile care încep cu `f"{scope_prefix}."`; prefixul este eliminat înainte de potrivirea pattern-ului și re-atașat după substituire, astfel că maparea fiecărui sub-model este scrisă *relativ la sub-model*, exact ca și cum ar fi root-ul.

## Arhitectură

Sistemul este construit în jurul mai multor componente cheie definite în [`core_model_loading.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/core_model_loading.py):

**Faza 1 — Procesare per-cheie** (iterează peste cheile checkpoint-ului):

1. **Parcurge lista de transformări** o singură dată. Fiecare [`WeightRenaming`] care se potrivește se declanșează (e.g. `block_sparse_moe` → `mlp`), iar **cel mult un** [`WeightConverter`] poate revendica cheia (e.g. `experts.*.w1.weight`).
2. **Sharding (TP) și trimite pe dispozitiv** asincron prin `ThreadPoolExecutor`.
3. **Colectează** tensorii cu același `source_pattern` împreună (e.g. toate weights ale experților MoE, proiecțiile gate + up).

**Faza 2 — Procesare per-mapare** (iterează peste mapările colectate):

1. **Dequantization/deserializare** (doar pentru checkpoint-uri pre-quantized).
2. **Aplică lanțul [`ConversionOps`]**: `Chunk`, `Concatenate`, `MergeModulelist`, `Transpose`, etc.
3. **Quantization** din mers (dacă nu este pre-quantized).
4. **Setează parametrul** pe model.

### WeightTransform

Clasa de bază care gestionează potrivirea pattern-urilor și colectarea tensorilor:

- **Compilarea pattern-urilor**: Pattern-urile sursă sunt expresii regulate complete potrivite cu `re.search()`. Wildcardul `*` potrivește orice componentă indexabilă și grupează toate potrivirile împreună pentru operații batch.
- **Grupuri captante și referințe inverse**: Grupurile captante din pattern-urile sursă pot fi referențiate ca `\1`, `\2`, ... în pattern-urile țintă pentru a păstra subșiruri (e.g. indici de straturi) în cadrul redenumirilor.
- **Scoping**: `scope_prefix` (setat automat per sub-model de [`get_model_conversion_mapping`]) restricționează transformarea la cheile de sub acea cale. Prefixul este eliminat înainte de potrivire și re-atașat după substituire.
- **Redenumirea cheilor**: `rename_source_key()` aplică regex-ul (cu gestionare de scope) și returnează `(renamed_key, source_pattern)` astfel că loader-ul știe care convertor, dacă există, a revendicat cheia.
- **Colectarea tensorilor**: `add_tensor()` acumulează tensorii rezolvați (sau `Future`-uri) sub `source_pattern`-ul lor astfel că toți tensorii necesari pentru o singură conversie (e.g. toate weights ale experților MoE) sunt grupați împreună înainte de rularea lanțului de operații.
- **Reversibilitate**: `reverse_transform()` inversează pattern-urile sursă ↔ țintă și inversează fiecare operație, astfel că aceeași listă inversată gestionează salvarea.
- **Urmărire**: `was_used()` raportează dacă transformarea a potrivit efectiv vreo cheie la încărcare; aceasta este necesară pentru ca redenumirile non-bijective (e.g. [`PrefixChange`] adăugând un prefix care poate fi deja prezent) să poată fi re-aplicate simetric la salvare.

### WeightRenaming

[`WeightRenaming`] este un [`WeightTransform`] specializat pentru redenumiri pure de chei fără operații asupra tensorilor. Spre deosebire de [`WeightConverter`], un `WeightRenaming` nu **revendică** cheia, astfel că mai multe redenumiri pot fi înlănțuite liber. Pe calea de încărcare, toate redenumirile rulează înainte de convertor; pe calea de salvare (lista inversată), toate redenumirile inversate rulează după convertorul inversat — a se vedea regula de ordonare de mai sus.

```py
# Compatibilitate cu checkpoint-uri vechi
WeightRenaming("LayerNorm.gamma", "LayerNorm.weight")

# Modificări ale path-ului modulului
WeightRenaming(".block_sparse_moe.", ".mlp.")
```

[`PrefixChange`] este un wrapper de nivel înalt în jurul [`WeightRenaming`] pentru cazul comun de eliminare sau adăugare a unei întregi componente de cale. Opționalul `model_prefix` aplică scope-ul operației la cheile din acel namespace:

```py
# "model.layers.bad_prefix.weight" → "model.layers.weight"
PrefixChange(prefix_to_remove="bad_prefix", model_prefix="model.layers")
# "layers.0.weight" → "model.layers.0.weight"
PrefixChange(prefix_to_add="model")
```

### WeightConverter

[`WeightConverter`] extinde [`WeightTransform`] cu un lanț de [`ConversionOps`] care acționează asupra tensorilor colectați. Cele patru cardinalități suportate sunt:

| Cardinalitate | Pattern-uri sursă | Pattern-uri țintă | Operație tipică |
|---------------|-------------------|-------------------|-----------------|
| unu-la-unu | 1 | 1 | [`Transpose`], [`PermuteForRope`] |
| unu-la-mulți | 1 | >1 | [`Chunk`] (e.g. despachetare `qkv_proj`) |
| mulți-la-unu | >1 | 1 | [`Concatenate`], [`MergeModulelist`] (e.g. fuzionarea experților) |
| mulți-la-mulți | >1 | >1 | doar cu operații care îl suportă explicit (e.g. `ErnieFuseAndSplitTextVisionExperts`) |

```python
WeightConverter(
    source_patterns=[".experts.*.w1.weight", ".experts.*.w3.weight"],
    target_patterns=".experts.gate_up_proj",
    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
)
```

Un `WeightConverter` este de asemenea reversibil: `reverse_transform()` inversează sursă ↔ țintă și înlocuiește fiecare operație cu `reverse_op`-ul său, astfel că o mapare de conversie înregistrată este bidirecțională prin construcție (încărcarea folosește lista ca atare, salvarea o folosește inversată).

### Ordonarea redenumirilor și convertoarelor

Listează intrările `WeightRenaming` înaintea intrărilor `WeightConverter` și menține frunzele lor disjuncte: redenumirile normalizează cheile pe care le consumă convertoarele, dar nu țintesc niciodată o frunză pe care un convertor o produce. Calea de salvare se bazează pe aceasta pentru a inversa maparea în două faze (convertoare inverse, apoi redenumiri inverse).

Lista de transformări este parcursă în ordine, o singură dată per cheie de checkpoint. Pentru fiecare cheie:

- Fiecare `WeightRenaming` care se potrivește se declanșează; mai multe redenumiri pot fi înlănțuite.
- Primul `WeightConverter` care se potrivește **revendică** cheia. Convertoarele ulterioare sunt omise — aceasta garantează că tensorul este direcționat către un singur convertor pentru pasul de merge/split și este motivul pentru care două convertoare cu intenție suprapusă ar reprezenta o configurare greșită.

```python
weight_mapping = [
    WeightRenaming("^old_prefix", "encoder"),              # redenumirea rulează mereu
    WeightConverter(                                       # convertorul revendică cheia
        "attn.qkv_proj.weight",
        ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"],
        operations=[Chunk(dim=0)],
    ),
]
# Încărcare:  "old_prefix.attn.qkv_proj.weight"
#   → WeightRenaming  → "encoder.attn.qkv_proj.weight"
#   → WeightConverter → "encoder.attn.{q,k,v}_proj.weight"
#
# Salvare: lista inversată, fiecare transformare inversată:
#   → rev(WeightConverter) reîmpachetează QKV → "encoder.attn.qkv_proj.weight"
#   → rev(WeightRenaming)  corectează prefixul → "old_prefix.attn.qkv_proj.weight"
```

## Operații de conversie

Clasa [`WeightConverter`] are mai multe operații care sunt executate când este apelat [`~PreTrainedModel.from_pretrained`] pentru transformarea tensorilor sursă din checkpoint în tensorii țintă ai modelului.

Operațiile sunt complet reversibile. Salvarea inversează conversiile și returnează checkpoint-ul original, astfel că poți lucra ușor cu diferite framework-uri.

| Operație | Inversă |
|----------|---------|
| [`Chunk(dim)`] | [`Concatenate(dim)`] |
| [`Concatenate(dim)`] | [`Chunk(dim)`] |
| [`MergeModulelist(dim)`] | [`SplitModulelist(dim)`] |
| [`SplitModulelist(dim)`] | [`MergeModulelist(dim)`] |
| [`Transpose(d0, d1)`] | [`Transpose(d1, d0)`] |
| [`PermuteForRope()`] | [`PermuteForRope()`] |
| [`Conv3dToLinear(...)`] | [`LinearToConv3d(...)`] |

### Chunk

Operația [`Chunk`] împarte un tensor în părți egale de-a lungul unei dimensiuni. De exemplu, dacă un model se așteaptă ca Q, K și V să fie trei tensori separați în loc de un singur tensor.

```py
WeightConverter(
    "self_attn.qkv_proj",
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    operations=[Chunk(dim=0)],
)
```

### Concatenate

Operația [`Concatenate`] fuzionează tensori separați într-un singur tensor. De exemplu, dacă un model se așteaptă ca Q, K și V să fie un singur tensor în loc de tensori separați.

```py
WeightConverter(
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "self_attn.qkv_proj",
    operations=[Concatenate(dim=0)],
)
```

### MergeModulelist

[`MergeModulelist`] îmbină o listă de tensori 2D într-un singur tensor 3D. De exemplu, poți compune [`MergeModulelist`] cu [`Concatenate`] pentru a stiva experții dintr-un MoE și a-i împacheta într-un singur tensor.

```py
WeightConverter(
    ["block_sparse_moe.experts.*.w1.weight", "block_sparse_moe.experts.*.w3.weight"],
    "mlp.experts.gate_up_proj",
    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
)
```

### SplitModulelist

[`SplitModulelist`] împarte un tensor 3D înapoi într-o listă de tensori 2D. De exemplu, poți împărți o stivă de experți înapoi în experți individuali.

```py
WeightConverter(
    "mlp.experts.down_proj",
    "block_sparse_moe.experts.*.w2.weight",
    operations=[SplitModulelist(dim=0)],
)
```

### PermuteForRope

[`PermuteForRope`] convertește weights din formatul intercalat pentru a folosi formatul sin/cos. De exemplu, poți compune [`Chunk`] cu [`PermuteForRope`] pentru a împărți un tensor QKV fuzionat și a aplica permutarea sin/cos RoPE la Q și K.

```py
WeightConverter(
    ["model.layers.*.self_attn.qkv_proj.weight"],
    [
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ],
    operations=[Chunk(dim=0), PermuteForRope()],
)
```

### Transpose

[`Transpose`] inversează dimensiunile unui tensor. Util pentru conversia layout-urilor de weights între diferite convenții.

```py
WeightConverter(
    source_patterns="mlp.gate.weight",
    target_patterns="mlp.text_moe.gate.weight",
    operations=[Transpose(dim0=0, dim1=1)],
)
```

## Înlănțuirea operațiilor

Operațiile pot fi înlănțuite pentru a efectua transformări complexe. Operațiile se execută în ordine, ieșirea fiecărei operații devenind intrarea operației următoare.

### Exemplu: Conversia MoE Mixtral

```python
WeightConverter(
    source_patterns=[
        ".experts.*.w1.weight",  # gate_proj per expert
        ".experts.*.w3.weight",  # up_proj per expert
    ],
    target_patterns=".experts.gate_up_proj",
    operations=[
        MergeModulelist(dim=0),  # Stivuiește toți experții: (n_experts, in, out)
        Concatenate(dim=1),      # Fuzionează gate+up: (n_experts, in, 2*out)
    ],
)
```

**Fluxul de date:**
```
Intrare:
  ".experts.*.w1.weight": [tensor_0, tensor_1, ..., tensor_7]  # 8 experți
  ".experts.*.w3.weight": [tensor_0, tensor_1, ..., tensor_7]  # 8 experți

După MergeModulelist(dim=0):
  ".experts.*.w1.weight": (8, 4096, 14336)  # gate stivuit
  ".experts.*.w3.weight": (8, 4096, 14336)  # up stivuit

După Concatenate(dim=1):
  ".experts.gate_up_proj": (8, 4096, 28672)  # gate_up fuzionat
```

### Reguli de potrivire a pattern-urilor

- Pattern-urile sursă sunt regex-uri complete evaluate cu `re.search()`. `^` ancorează la începutul cheii **cu scope** (după eliminarea `scope_prefix`).
- `*` este un wildcard per-index care colectează toți tensorii potriviți sub același pattern sursă (păstrând ordinea checkpoint-ului pentru concatenare corectă).
- Grupurile captante din pattern-urile sursă pot fi referențiate ca `\1`, `\2`, ... în pattern-urile țintă pentru a păstra părți ale cheii originale (e.g. indici de straturi).
- Transformările cu scope (`scope_prefix` setat) potrivesc doar cheile care încep cu `f"{scope_prefix}."`.

## Înregistrarea unei mapări de conversie

O listă de conversie este înregistrată împotriva unei chei string — fie un `model_type` (e.g. `"mixtral"`) fie un nume de clasă (e.g. `"LlavaModel"`):

```python
from transformers.conversion_mapping import register_checkpoint_conversion_mapping

register_checkpoint_conversion_mapping(
    "my_model_type",
    [
        WeightRenaming(".old.", ".new."),
        WeightConverter(
            ".experts.*.w1.weight",
            ".experts.gate_proj",
            operations=[MergeModulelist(dim=0)],
        ),
    ],
)
```

### Reguli de căutare

Când [`get_model_conversion_mapping`] procesează un `PreTrainedModel`, fiecare sub-`PreTrainedModel` este vizitat în ordine DFS (`nn.Module.named_modules()` filtrat la instanțe `PreTrainedModel`). Pentru fiecare:

1. **Căutarea după numele clasei este încercată prima**, apoi `model_type`. Dacă ambele sunt înregistrate pentru același modul, **maparea după numele clasei câștigă** și cea `model_type` este ignorată pentru acel modul — aceasta permite unui cap de sarcină (e.g. `LlavaForConditionalGeneration`) să suprascrie baseline-ul `model_type` partajat (`"llava"`).
2. Maparea selectată are `scope_prefix` setat la calea punctată a sub-modulului (`""` pentru root).
3. **Deduplicarea bazată pe strămoș** decide dacă se păstrează maparea:
   - Dacă un path **ancestor** a revendicat deja același identificator (numele clasei sau `model_type`), sub-modulul este **omis** — maparea fără scope sau cu scope mai înalt a strămoșului acoperă deja acest subarbore.
   - Dacă doar un **sibling** l-a revendicat, sub-modulul este **păstrat** cu propriul `scope_prefix`. Fiecare sibling primește propria mapare cu scope.

Listele văzute pentru numele clasei și `model_type` sunt urmărite separat, cu o subtilitate: când un modul este potrivit **prin numele clasei**, `model_type`-ul său *nu* este adăugat la lista văzută. Aceasta este pentru ca alte module care partajează același `model_type` dar fără o mapare specifică clasei (e.g. `DetrModel` sub `DetrForSegmentation`) să rămână accesibile prin căutarea `model_type`.

### Mapări bazate pe clasă vs aliasuri `model_type`

Ambele stiluri coexistă în `_MODEL_TO_CONVERSION_PATTERN`:

```python
_MODEL_TO_CONVERSION_PATTERN = {
    # aliasuri model_type (căutare după config.model_type)
    "minimax": "mixtral",
    "qwen3_moe": "qwen2_moe",
    "mistral3": "llava",
    # aliasuri după numele clasei (căutare după type(submodule).__name__)
    "PaliGemmaModel": "LlavaModel",
    "MaskFormerDetrDecoder": "DetrModel",
    ...
}
```

Cheile după numele clasei sunt preferate când `model_type`-ul este partajat, dar o clasă specifică necesită comportament diferit.

## Integrarea cu tensor parallelism

Sistemul de încărcare dinamică se integrează cu tensor parallelism (TP) prin ierarhia `TensorParallelLayer` definită în `src/transformers/integrations/tensor_parallel.py`.

Când TP este activat, tensorii sunt fragmentați **în timpul** materializării, nu după. Aceasta înseamnă că fiecare rang încarcă doar porțiunea de tensor de care are nevoie.

```python
def spawn_tp_materialize(thread_pool, tensor, sharding_method, tensor_idx, device, dtype):
    def _job():
        return sharding_method.shard_tensor(tensor, tensor_idx=tensor_idx, device=device, dtype=dtype)
    return thread_pool.submit(_job)
```

### Stiluri de paralelism disponibile

| Stil | Dimensiune Fragment Weight | Descriere |
|------|---------------------------|-----------|
| `colwise` | -2 | Column-wise: caracteristici de ieșire fragmentate |
| `rowwise` | -1 | Row-wise: caracteristici de intrare fragmentate |
| `packed_colwise` | -2 | Pentru weights fuzionate (gate_up_proj) |
| `packed_rowwise` | -1 | Pentru weights fuzionate |
| `embedding_rowwise` | 0 | Paralelism vocabular |
| `grouped_gemm` | 0 | Paralelism expert pentru MoE |
| `sequence_parallel` | None | Fără fragmentare de weights |

### Gestionarea weights împachetate

Pentru weights fuzionate precum `gate_up_proj`, este necesară atenție specială pentru fragmentarea corectă:

```python
def get_packed_weights(param, empty_param, device_mesh, rank, dim):
    """
    Interleaves gate and up shards correctly.

    Packed tensor: [G0 G1 G2 G3 | U0 U1 U2 U3]

    With TP=2:
    - Rank 0 gets: [G0 G1 | U0 U1]
    - Rank 1 gets: [G2 G3 | U2 U3]
    """
```

Operația TP este stocată în [`WeightTransform`] și aplicată după operațiile de conversie:

```python
if matched_tp_pattern := tp_plan_alt.search(renamed_key):
    tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]]
    mapping.distributed_operation = tp_layer(
        device_mesh=device_mesh,
        rank=device_mesh.get_local_rank(),
        empty_param=empty_param.clone()
    )
```

## Integrarea cu quantization

Quantization se integrează în pipeline-ul de încărcare în două moduri, în funcție de dacă checkpoint-ul este deja quantized:

- **Checkpoint-uri pre-quantized**: Quantizer-ul furnizează instanțe [`WeightConverter`] (prin `get_weight_conversions()`) care deserializează tensorii quantizați. Dtype-urile checkpoint-ului sunt păstrate pentru a evita cast-urile nedorite.
- **Quantization din mers**: Quantizer-ul furnizează o operație de quantization care este aplicată după operațiile de conversie, aplicând-o pe weights pe măsură ce sunt încărcate.

Quantizer-ul poate, de asemenea, rescrie întreaga listă de conversie la sfârșitul [`get_model_conversion_mapping`] prin `update_weight_conversions(...)` — de exemplu, dequantizer-ul FP8 adaugă o operație `Fp8Dequantize` la începutul fiecărui convertor existent, astfel că scalele per-bloc sunt aplicate *înainte* ca orice operații de expert-merge/concat să aplatizeze structura per-expert.

## Încărcarea rapidă și eficientă a modelelor

Încărcarea unui model este mai rapidă și utilizează mai puțină memorie deoarece loader-ul știe ce tensori sunt necesari pentru operații și le planifică materializarea leneș.

Loader-ul scanează checkpoint-ul *o singură dată* pentru a descoperi potriviri de pattern și a colecta tensori. Îi stochează ca obiecte `Future` și îi trimite unui pool de thread-uri pentru încărcare asincronă fără a bloca GIL-ul. Un parametru începe să se încarce imediat ce un thread devine disponibil pentru el.

Dacă sistemul tău rulează alte procese grele, mai multe thread-uri pot încetini încărcarea în loc să o accelereze. În acest caz, setează variabila de mediu `HF_DEACTIVATE_ASYNC_LOAD=1` pentru a încărca weights secvențial.

> [!NOTE]
> Implicit sunt 4 thread-uri pentru încărcarea asincronă a parametrilor. Aceasta oferă cel mai bun compromis pentru scenariile de încărcare și hardware. Munca este în mare parte limitată de I/O, dar în funcție de hardware-ul acceleratorului și de `dtype`-ul necesar la încărcare, poate deveni limitată de CPU/GPU dacă `dtype`-ul diferă de cel serializat (aceasta necesită o operație suplimentară de copiere).

### Încărcare asincronă vs sincronă

```python
def spawn_materialize(thread_pool, tensor, device, dtype) -> Future | Callable:
    def _job():
        return _materialize_copy(tensor, device, dtype)

    if thread_pool is not None:
        return thread_pool.submit(_job)  # Async: returnează Future
    else:
        return _job  # Sync: returnează Callable (execuție amânată)
```

Încărcarea sincronă este utilizată când:
- Variabila de mediu `HF_DEACTIVATE_ASYNC_LOAD=1` este setată.
- Descărcarea pe disc este activată (constrângerile de memorie necesită încărcare secvențială).
- Quantization din mers este activat (evită thread-urile worker care depășesc pasul de quantization).

### Fluxul de materializare

```
1. Iterarea checkpoint-ului (Faza 1):
   - Pentru fiecare cheie, parcurge lista de transformări o singură dată
   - Trimite job-ul de materializare la ThreadPoolExecutor
   - Job-ul returnează Future (async) sau Callable (sync)
   - Colectează în WeightConverter / WeightRenaming potrivit

2. Procesarea per-mapare (Faza 2, câte o mapare pe rând):
   - materialize_tensors() așteaptă doar Future-urile acestei mapări
   - Aplică lanțul de operații de conversie (self.operations)
   - Aplică operația de quantizare (dacă din mers)
   - Setează parametrii pe model
   - Șterge tensorii realizați imediat

3. Curățare:
   - Oprirea pool-ului de thread-uri (cu cancel_futures=True pentru întreruperi)
```

### Eficiența memoriei

La conversia unui weight, convertorul așteaptă ca toți tensorii necesari să se materializeze dacă nu s-au încărcat încă. De exemplu, operația [`MergeModulelist`] necesită ca toate weights din `ModuleList` să fie încărcate înainte de îmbinare.

Concatenarea tensorilor necesită o copie temporară, astfel că operații precum [`MergeModulelist`] și [`Concatenate`] necesită de 2 ori memoria tensorilor de bază în timpul conversiei. Odată îmbinați, în memorie rămâne doar tensorul rezultat. Vârful teoretic maxim de memorie este dimensiunea modelului plus tensorii necesari pentru cea mai mare operație [`MergeModulelist`] sau [`Concatenate`].

Acest caz cel mai rău apare doar când toți ceilalți parametri s-au încărcat înainte de rularea conversiei exigente. Două scenarii declanșează aceasta.

1. Toți parametrii s-au încărcat asincron înainte de intrarea în conversia exigentă (pool-ul de thread-uri a fost mai rapid decât coada de conversie).
2. Conversia exigentă este ultima.

De exemplu, pentru un model MoE care utilizează [`MergeModulelist`] pentru experți pe fiecare strat, vârful teoretic maxim de memorie este dimensiunea modelului plus experții de pe un strat.

Aceste scenarii de caz cel mai rău sunt neobișnuite. Vârful real de memorie tinde să rămână aproape de dimensiunea modelului.

## Reversibilitate

Sistemul suportă salvarea modelelor cu transformările inverse, permițând save/load dus-întors. Salvarea rulează în două faze: convertoare inverse mai întâi (fiecare tensor se potrivește cel mult unuia), apoi redenumiri inverse, conform regulii de ordonare de mai sus.

```python
def revert_weight_conversion(model, state_dict):
    """Aplică conversiile inverse pentru salvare."""
    weight_conversions = getattr(model, "_weight_conversions", None)

    # Inversează toate transformările
    reverse_weight_conversion = [
        conversion.reverse_transform() for conversion in weight_conversions
    ]

    # Aplică în ordine inversă
    for first_param_name, reversed_converter in conversion_mapping.items():
        realized_value = reversed_converter.convert(first_param_name, model=model)
```

Lista de transformări utilizate la momentul încărcării este cached pe model ca `_weight_conversions` (se păstrează doar intrările care s-au declanșat efectiv, astfel că redenumirile non-bijective precum [`PrefixChange`] sunt re-aplicate corect simetric). Când modelul a fost instanțiat fără `from_pretrained` (și deci nu are `_weight_conversions`), `revert_weight_conversion` recurge la recalcularea mapării prin [`get_model_conversion_mapping`] și elimină orice [`PrefixChange`] din aceasta (nu putem determina dacă checkpoint-ul original avea prefixul).

Pattern-urile țintă pot conține elemente regex care necesită procesare pentru direcția inversă:

```python
def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    - Removes `^` and `$` anchors
    - Removes negative lookahead/lookbehind
    - Detects capturing groups, replaces with \1
    """
```

## Exemple reale

### MoE în stil Mixtral

**Formatul checkpoint-ului:**
```
model.layers.0.block_sparse_moe.experts.0.w1.weight  # gate per expert
model.layers.0.block_sparse_moe.experts.0.w2.weight  # down per expert
model.layers.0.block_sparse_moe.experts.0.w3.weight  # up per expert
...
model.layers.0.block_sparse_moe.experts.7.w1.weight
```

**Formatul modelului:**
```
model.layers.0.mlp.experts.gate_up_proj  # (8, 4096, 28672)
model.layers.0.mlp.experts.down_proj     # (8, 14336, 4096)
```

**Maparea de conversie** (din `conversion_mapping.py`):
```python
"mixtral": [
    WeightRenaming(".block_sparse_moe.", ".mlp."),
    WeightConverter(
        source_patterns=[".experts.*.w1.weight", ".experts.*.w3.weight"],
        target_patterns=".experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    ),
    WeightConverter(
        source_patterns=[".experts.*.w2.weight"],
        target_patterns=".experts.down_proj",
        operations=[MergeModulelist(dim=0)],
    ),
],
```

### Model compus viziune-limbaj (Gemma3)

Gemma3 are configurarea canonică "logică de prefix diferită pentru modelul cap față de modelul de bază". Două intrări sunt înregistrate (aici prin aliasuri care indică listele partajate `"llava"` / `"LlavaModel"`):

```python
# model_type: se aplică la Gemma3ForConditionalGeneration / Gemma3ForSequenceClassification
"gemma3": "llava"           # → adaugă "model." în fața language_model / vision_tower / ...

# clasă:    se aplică la inner Gemma3Model
"Gemma3Model": "LlavaModel"  # → redenumire minimă în namespace-ul deja prefixat
```

Parcurgere DFS pentru `Gemma3ForConditionalGeneration`:

1. Root — căutarea după clasă eșuează (`Gemma3ForConditionalGeneration` nu este înregistrată), căutarea după model_type reușește (`"gemma3"`) → transformările de rescriere a prefixului `"llava"` sunt adăugate **fără scope**, punând `language_model`, `vision_tower`, `multi_modal_projector` sub `model.*`.
2. `model: Gemma3Model` interior — **căutarea după clasă reușește** (`Gemma3Model` → `LlavaModel`) → doar maparea `LlavaModel` este aplicată, cu scope la `"model"`. Redenumirile de prefix mult mai largi `"gemma3"` (=`"llava"`) **nu** sunt re-aplicate aici, ceea ce este exact ce vrei: modelul interior trăiește într-un namespace unde prefixul este deja corect.

Aceeași formă de pereche este utilizată de fiecare VLM din familia Llava (`PaliGemma`, `InternVL`, `Mistral3`, ...). Intrarea model_type gestionează chirurgia prefixului modelului cap; intrarea de clasă menține maparea modelului de bază interior minimă.

### Imbricare mai profundă cu suprascrierea specifică capului (DETR)

`DetrForSegmentation` arată o suprascrierea specifică capului stratificată pe deasupra a două niveluri imbricate:

```
DetrForSegmentation                  (clasă înregistrată: redenumiri specifice segmentării)
├── detr: DetrForObjectDetection     (fără mapare; doar parcurs)
│   └── model: DetrModel             (clasă înregistrată: transformări de bază partajate)
│       └── backbone, encoder, ...
└── mask_head, bbox_attention        (weights specifice capului)
```

Sunt implicate două mapări (una per cheie înregistrată):

```python
"DetrModel":           [WeightRenaming("backbone.conv_encoder", "backbone"), ...]      # bază partajată
"DetrForSegmentation": [WeightRenaming("mask_head.lay1", "mask_head.conv1.conv"), ...] # specific capului
```

Parcurgere DFS:

1. Root `DetrForSegmentation` se potrivește după clasă → redenumiri de segmentare adăugate **fără scope**.
2. `detr: DetrForObjectDetection` nu este înregistrat → nu se adaugă transformări; DFS continuă în interiorul său.
3. `detr.model: DetrModel` se potrivește după clasă → transformări de bază adăugate cu `scope_prefix="detr.model"`.

De aceea `DetrForObjectDetection` nu necesită propria mapare: singura mapare înregistrată în subarborele său (`DetrModel`) este automat aplicată cu scope la calea corectă.

Aliasurile cheiate după clasă reutilizează maparea de bază fără înregistrare suplimentară: `"MaskFormerDetrDecoder": "DetrModel"` face ca un decoder `MaskFormer` să preia aceleași transformări sub propriul nume de clasă.

### Operații personalizate (ERNIE 4.5 VL MoE)

Când operațiile integrate nu sunt suficiente, poți crea o subclasă [`ConversionOps`] personalizată. De exemplu, ERNIE 4.5 VL MoE trebuie să împartă o listă de experți partajată între modalitățile text și viziune — ceva ce nicio operație integrată singulară nu gestionează. Operația personalizată `ErnieFuseAndSplitTextVisionExperts` împarte și re-stivuiește experții pe două chei țintă:

```python
"ernie4_5_vl_moe": [
    WeightRenaming("vision_model", "vision_tower"),
    WeightConverter(
        source_patterns=["experts.*.down_proj.weight"],
        target_patterns=[
            "text_moe.experts.down_proj",
            "vision_moe.experts.down_proj",
        ],
        operations=[ErnieFuseAndSplitTextVisionExperts(stack_dim=0, concat_dim=1)],
    ),
],
```

Operațiile personalizate trebuie să implementeze `convert()` și proprietatea `reverse_op` pentru a suporta save/load dus-întors.

### Aliasuri de tip model

Multe modele partajează pattern-uri de conversie:

```python
_MODEL_TO_CONVERSION_PATTERN = {
    "mixtral": "mixtral",
    "minimax": "mixtral",
    "qwen2_moe": "qwen2_moe",
    "deepseek_v2": "qwen2_moe",
    "deepseek_v3": "qwen2_moe",
    "qwen3_moe": "qwen2_moe",
    "olmoe": "qwen2_moe",
    ...
}
```

## Reutilizarea blocurilor de construcție ale încărcării dinamice

Încărcarea dinamică de weights nu se limitează la checkpoint-uri complete ale modelului. Aceleași blocuri de construcție îți permit să încarci *orice* set de weights atâta timp cât poți descrie cum se mapează cheile checkpoint-ului la parametri și te asiguri că modulele țintă există.

La un nivel înalt, contractul arată astfel:

1. **Pregătește namespace-ul modelului.** Asigură-te că modulele/parametrii pe care dorești să îi încarci sunt prezenți și numiți în modul în care maparea ta îi va targeta. Pentru adaptori, aceasta înseamnă apelarea `inject_adapter_in_model(...)` pentru ca modulele adaptor să existe înainte de încărcare. Pentru capete personalizate sau module suplimentare, instanțiază-le mai întâi pe model.
2. **Descrie cum să mapezi weights.** Construiește o listă de conversie/redenumire (de exemplu, într-un helper precum `_build_peft_weight_mapping(...)`) utilizând [`WeightConverter`] sau [`WeightRenaming`]. Acesta este locul unde exprimi cum trebuie convertite, împărțite, îmbinate sau redenumite cheile checkpoint-ului pentru a se potrivi namespace-ului modelului tău.
   Poți face în principal 3 lucruri:
    - adaugă operații la lista de convertoare: acestea vor fi aplicate pe toate weights, cu excepția celor colectate în oricare din `WeightConverter`. În general, acestea ar trebui să fie operații `WeightRenaming`
    - adaugă operații la lista de operații a fiecărui convertor: aceasta este ceea ce se întâmplă pentru `Quantization`, unde adăugăm o operație de quantization după lista de operații a oricărui `WeightConverter`.
    - înlocuiește/mapează operații cu operațiile tale personalizate: aceasta este ceea ce se întâmplă cu `peft`. Înlocuim operația `Concatenate` din, să zicem, `mixtral`, cu `PeftConcatenate` (care este definită în PEFT). În acest fel, când checkpoint-ul adaptor este citit, weights de concatenat sunt colectate și formatate corespunzător pentru `peft`
3. **Încarcă + finalizează + raportează.** Folosește loader-ul de bază pentru a efectua conversia și a popula tensorii, apoi finalizează și înregistrează rezultatele. Concret, acest flux este:
   - `LoadStateDictConfig(...)` + `_load_pretrained_model(...)` pentru a încărca și converti.
   - `_finalize_load_state_dict(...)` pentru a muta tensorii lipsă/nepotriviți de pe `meta`, a-i inițializa și a lega weights.
   - `log_state_dict_report(...)` pentru a raporta cheile lipsă/neașteptate/nepotrivite (și erorile de conversie).

Aceste API-uri sunt expuse pentru a-ți permite să gestionezi cod personalizat, formate de weights personalizate, dar și pentru a te asigura că beneficiezi de cea mai înaltă și mai eficientă încărcare de weights, fragmentare și calitate a vieții oferite de API-ul `transformers`!

## Referință fișiere cheie

| Fișier | Scop |
|--------|------|
| `src/transformers/core_model_loading.py` | Logica de bază de încărcare, `WeightConverter`, `WeightRenaming`, `ConversionOps` |
| `src/transformers/conversion_mapping.py` | Mapări integrate și compoziție per-submodel (`get_model_conversion_mapping`) |
| `src/transformers/integrations/tensor_parallel.py` | Clase și utilitare pentru sharding TP |
| `src/transformers/quantizers/base.py` | Hook-uri de quantization și clasa de bază |
