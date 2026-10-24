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

# Monkey patching (funcție experimentală)

Monkey patching îți permite să înlocuiești componente ale modelului la nivel global fără a modifica codul original al modelului. Odată înregistrate, patch-urile sunt aplicate automat la încărcarea oricărui model cu [`~PreTrainedModel.from_pretrained`] sau [`~PreTrainedModel.from_config`]. Aceasta îți permite să restructurezi modele pentru cerințe specifice precum compatibilitatea cu quantization, să aplici optimizări sau să experimentezi cu variante arhitecturale.

> [!WARNING]
> **Monkey patching ar trebui folosit ca ultimă soluție** atunci când trebuie să schimbi layout-ul și structura unui modul și/sau weights asociate acestuia. Pentru nevoile de personalizare și optimizare, încearcă să folosești în schimb [interfața Attention], [interfața Experts] sau [registrul Kernels]. Folosește monkey patching doar când ai nevoie de schimbări structurale care nu pot fi realizate doar prin implementări forward personalizate (e.g., pentru compatibilitatea cu biblioteci de quantization, fuzionarea layers, sau experimente arhitecturale).

## Pornire rapidă

Iată un exemplu simplu care arată cum să înlocuiești o componentă a modelului:

```python
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.monkey_patching import register_patch_mapping


# Definește clasa ta de înlocuire (trebuie să moștenească din nn.Module)
class CustomLlamaAttention(LlamaAttention):
    def forward(self, *args, **kwargs):
        # Implementarea ta personalizată
        print("Using custom attention!")
        return super().forward(*args, **kwargs)


# Înregistrează patch-ul global (se aplică doar modulelor de modelare transformers)
register_patch_mapping(mapping={"LlamaAttention": CustomLlamaAttention})

# Încarcă un model - patch-ul este aplicat automat în timpul inițializării
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Toate layers LlamaAttention din model sunt acum instanțe CustomLlamaAttention
print(type(model.model.layers[0].self_attn))  # <class '__main__.CustomLlamaAttention'>
```

## Cum funcționează

Monkey patches funcționează printr-un proces în două etape:

1. **Înregistrare**: Apelează [`register_patch_mapping`] pentru a adăuga mapări la un registru global.

2. **Aplicare**: Patch-urile sunt aplicate automat în timpul inițializării modelului:
   - **`from_pretrained` / `from_config`**: Patch-urile sunt aplicate **automat** printr-un context manager intern. Nu este necesară nicio acțiune suplimentară!
   - **Construcție manuală** (e.g., `Model(config)`): Trebuie să utilizezi manual context manager-ul [`apply_patches`].

Odată ce patch-urile sunt înregistrate, persistă și afectează toate încărcările ulterioare de modele până când le ștergi cu [`clear_patch_mapping`].

**Limitări importante**:

- Doar clasele din modulele de modelare `transformers` pot fi patched (e.g., `LlamaAttention`, `LlamaMLP`).
- Cheile mapării pot fi fie nume exacte de clase, fie pattern-uri de expresii regulate (vezi [Potrivirea pattern-urilor](#potrivirea-pattern-urilor) mai jos).

## Înregistrare globală

Folosește [`register_patch_mapping`] pentru a înregistra mapări global:

```python
from transformers.monkey_patching import register_patch_mapping

# Înregistrează un singur patch
register_patch_mapping(
    mapping={"Qwen2MoeExperts": SequentialExperts}
)

# Înregistrează mai multe patch-uri simultan
register_patch_mapping(
    mapping={
        "Qwen2MoeExperts": SequentialExperts,
        "Qwen2MoeAttention": CustomAttention,
    },
    # Suprascrie patch-urile existente dacă există
    overwrite=True,
)
```

## Potrivirea pattern-urilor

Poți folosi expresii regulate pentru a potrivi mai multe clase cu un singur pattern:

```python
from transformers.monkey_patching import register_patch_mapping

# Potrivește toate clasele care conțin "Attention"
register_patch_mapping(
    mapping={".*Attention": CustomAttention}
)

# Mai multe exemple
register_patch_mapping(
    mapping={
        ".*MoeExperts$": CustomExperts,           # Se termină cu "MoeExperts"
        "^Llama\\d+Attention$": CustomAttention,  # Llama2Attention, Llama3Attention, etc.
    }
)
```

**Important**: Potrivirile exacte au prioritate față de pattern-uri. Dacă înregistrezi atât `"LlamaAttention"` cât și `".*Attention"`, clasele numite `LlamaAttention` vor folosi înlocuirea prin potrivire exactă, în timp ce alte clase potrivite vor folosi înlocuirea prin potrivire după pattern.

> [!WARNING]
> **Pattern-urile regex pot strica modelele în tăcere.** Un pattern larg precum `".*Attention"` va potrivi *fiecare* clasă al cărei nume conține "Attention" — inclusiv clasele container care învelesc attention-ul pe care vrei să îl înlocuiești. De exemplu, BERT are trei clase legate de attention: `BertSelfAttention` și `BertCrossAttention` (implementările interioare de attention) și `BertAttention` (un modul exterior care *conține* una dintre acele clase interioare). Patch-uirea tuturor celor trei cu același layer de attention personalizat produce un model stricat deoarece `BertAttention`-ul exterior nu mai învelește cel interior — *este* unul, eliminând sub-modulele așteptate precum `self` și `output`. Preferă pattern-uri înguste (e.g., `".*SelfAttention$"`) sau nume exacte de clase pentru a evita potrivirile neintenționate.

Pentru a dezînregistra patch-uri, folosește [`unregister_patch_mapping`]:

```python
from transformers.monkey_patching import unregister_patch_mapping

# Dezînregistrează un singur patch (folosește numele exact sau pattern-ul din înregistrare)
unregister_patch_mapping(keys=["Qwen2MoeExperts"])

# Dezînregistrează mai multe patch-uri simultan
unregister_patch_mapping(keys=["Qwen2MoeExperts", "Qwen2MoeAttention"])

# Dezînregistrează un pattern
unregister_patch_mapping(keys=[".*Attention"])
```

Pentru a șterge toate patch-urile înregistrate, folosește [`clear_patch_mapping`]:

```python
from transformers.monkey_patching import clear_patch_mapping

clear_patch_mapping()
```

Pentru a vizualiza patch-urile înregistrate curent, folosește [`get_patch_mapping`]:

```python
from transformers.monkey_patching import get_patch_mapping

current_patches = get_patch_mapping()
print(current_patches)
```

## Construcția manuală a modelului

Context manager-ul [`apply_patches`] este necesar doar atunci când construiești modele **manual** (e.g., `Model(config)`) fără a folosi `from_pretrained` sau `from_config`:

```python
from transformers import LlamaModel, LlamaConfig
from transformers.monkey_patching import register_patch_mapping, apply_patches

# Înregistrează patch-ul global
register_patch_mapping(mapping={"LlamaAttention": CustomAttention})

# Pentru construcția manuală, ai nevoie de context manager
with apply_patches():
    model = LlamaModel(LlamaConfig())  # Utilizează CustomAttention

# Fără context manager, construcția manuală utilizează clasele originale
model = LlamaModel(LlamaConfig())  # Utilizează LlamaAttention

# Dar from_pretrained și from_config vor aplica întotdeauna patch-urile înregistrate
model = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")  # Utilizează CustomAttention
```

## Note importante

- **Gestionarea weights**: Monkey patching înlocuiește doar clasele, nu și weights. Dacă clasa ta patched are un layout de weights diferit, va trebui să gestionezi [conversiile de weights](./weightconverter) separat pentru a asigura compatibilitatea cu weights pre-antrenate. Vezi [Exemplul complet](#exemplu-complet) de mai jos pentru a combina monkey patches cu mapări de conversie a weights.

- **Efect global**: Patch-urile înregistrate cu [`register_patch_mapping`] sunt aplicate global tuturor modelelor încărcate după înregistrare. Folosește întotdeauna [`clear_patch_mapping`] pentru a face curățenie când termini, mai ales în teste, notebooks sau aplicații de lungă durată.

- **Validarea claselor**: API-ul validează automat că clasele de înlocuire sunt subclase `nn.Module`. Dacă pasezi o clasă invalidă, vei primi un mesaj de eroare clar.

- **Thread safety**: Toate operațiile de patching sunt thread-safe. Poți înregistra, dezînregistra și aplica patch-uri în siguranță din mai multe thread-uri.

- **Comportamentul de potrivire**: Când folosești nume exacte de clase, acestea trebuie să corespundă exact cu numele claselor originale din codul sursă al modelului (cu distincție între majuscule și minuscule). Când folosești pattern-uri regex, acestea sunt potrivite față de numele claselor folosind `re.search()`.

## Depanare

### Patch-ul meu nu se aplică

**Verifică numele clasei sau pattern-ul**: Asigură-te că numele clasei sau pattern-ul din maparea ta este corect:

```python
# Pentru nume exacte - trebuie să corespundă exact (cu distincție între majuscule și minuscule)
register_patch_mapping(mapping={"LlamaAttention": CustomAttention})

# Pentru pattern-uri - folosește regex valid
register_patch_mapping(mapping={".*Attention": CustomAttention})
```

**Verifică înregistrarea**: Folosește [`get_patch_mapping`] pentru a confirma că maparea ta este înregistrată:

```python
print(get_patch_mapping())
# Afișează toate mapările înregistrate: {'LlamaAttention': <class 'CustomAttention'>, '.*MLP': <class 'CustomMLP'>}
```

**Verifică sursa modelului**: Găsește numele exact al clasei în sursa modelului:

```python
from transformers.models.llama import modeling_llama
print(dir(modeling_llama))  # Caută numele clasei
```

### Cum știu dacă patch-ul meu funcționează?

Inspectează modelul încărcat pentru a verifica patch-ul:

```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Verifică tipul unui modul specific
print(type(model.model.layers[0].self_attn))  # Ar trebui să afișeze clasa ta personalizată

# Sau iterează prin toate modulele
for name, module in model.named_modules():
    if 'attention' in name.lower():
        print(f"{name}: {type(module)}")
```

### Erori de nepotrivire a dimensiunilor weights

Dacă clasa ta patch-uită are dimensiuni de weights diferite, înregistrează o conversie de weights:

```python
from transformers.conversion_mapping import register_checkpoint_conversion_mapping, WeightConverter
from transformers.monkey_patching import register_patch_mapping

register_patch_mapping(
    mapping={
        "LlamaAttention": LlamaFusedAttention,
    }
)

register_checkpoint_conversion_mapping(
    model_type="llama",
    mapping=[
        WeightConverter(
            source_patterns=["q_proj", "k_proj", "v_proj"],
            target_patterns=["qkv_proj"],
            operations=[
                Concatenate(dim=0),
            ],
        )
    ],
    overwrite=True,
)
```

### Curățarea patch-urilor

Curăță întotdeauna patch-urile când termini pentru a evita afectarea altui cod:

```python
from transformers.monkey_patching import register_patch_mapping, clear_patch_mapping

try:
    register_patch_mapping(mapping={"LlamaAttention": CustomAttention})
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # ... folosește modelul ...
finally:
    clear_patch_mapping()  # Curăță întotdeauna
```

## Exemplu complet

Iată un exemplu cuprinzător care arată cum să restructurezi atât modulele experts cât și cele de attention dintr-un model Mixture-of-Experts (`qwen2_moe`) pentru optimizare și compatibilitate cu quantization. Acesta demonstrează:

1. Crearea de clase de înlocuire personalizate care mențin aceeași interfață
2. Înregistrarea de monkey patches pentru mai multe componente
3. Gestionarea conversiilor de weights pentru noua structură

```python
from typing import Unpack

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, Concatenate, WeightConverter
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2_moe.modeling_qwen2_moe import apply_rotary_pos_emb
from transformers.monkey_patching import register_patch_mapping
from transformers.utils.generic import TransformersKwargs


class MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Adaptat din Qwen2MoeExperts original
class ModuleListExperts(nn.ModuleList):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(MoeMLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# Adaptat din Qwen2MoeAttention original
class FusedQKVAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        if self.config.layer_types[layer_idx] == "sliding_attention":
            self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self.qkv_proj(hidden_states).chunk(3, dim=-1)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Înregistrează monkey patches pentru noile module de attention și experts.
register_patch_mapping(
    mapping={
        "Qwen2MoeExperts": ModuleListExperts,
        "Qwen2MoeAttention": FusedQKVAttention,
    }
)

# Înregistrează mapările de conversie a weights adaptate pentru noile module. Această înregistrare va:
# - Suprascrie maparea de conversie originală pentru qwen2_moe care concatena experții într-un format de parametru unic.
# - Concatenează weights/biases q_proj, k_proj, v_proj într-un singur weight/bias qkv_proj pentru noul modul de attention fuzionat.
register_checkpoint_conversion_mapping(
    model_type="qwen2_moe",
    mapping=[
        WeightConverter(
            source_patterns=["q_proj", "k_proj", "v_proj"],
            target_patterns=["qkv_proj"],
            operations=[Concatenate(dim=0)],
        ),
    ],
    overwrite=True,
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
```

## Înregistrarea și reluarea rutării experților MoE

Fluxurile de antrenare Mixture-of-Experts precum RLHF trebuie să înregistreze la ce experți a fost dirijat fiecare token în timpul generării, apoi să repete exact acea rutare într-un forward pass de antrenare separat. Poți construi aceasta end-to-end cu mecanismele existente de monkey patching și captare a output-urilor — nu sunt necesare modificări ale fișierelor de modelare.

Pattern-ul are trei componente:

1. O **subclasă de router reluabilă** care poate citi opțional indici de experți forțați dintr-un atribut de instanță.
2. Un **context manager** care setează acele atribute pe fiecare router înainte de un forward pass și le șterge după.
3. O intrare în registrul de captare a output-urilor modelului astfel că `output_<name>=True` expune indicii prin calea standard `@capture_outputs`.

```python
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
from transformers.monkey_patching import apply_patches, register_patch_mapping
from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder


class ReplayableQwen3MoeTopKRouter(Qwen3MoeTopKRouter):
    _forced_indices: torch.Tensor | None = None

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)

        if self._forced_indices is not None:
            router_indices = self._forced_indices.to(router_logits.device).long()
            # Replay în stil Megatron: păstrează calea expertului, recalculează scorurile curente
            router_top_value = router_logits.gather(-1, router_indices)
        else:
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)

        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value.to(router_logits.dtype), router_indices


@contextmanager
def replay_moe_routing(model, selected_experts_per_layer):
    routers = [m for m in model.modules() if isinstance(m, ReplayableQwen3MoeTopKRouter)]
    if len(routers) != len(selected_experts_per_layer):
        raise ValueError(f"Got {len(routers)} routers but {len(selected_experts_per_layer)} tensors")
    for r, t in zip(routers, selected_experts_per_layer):
        r._forced_indices = t
    try:
        yield
    finally:
        for r in routers:
            r._forced_indices = None


# Înlocuiește clasa router și construiește modelul
register_patch_mapping({"Qwen3MoeTopKRouter": ReplayableQwen3MoeTopKRouter})
with apply_patches():
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(...)).eval()

# Expune `output_selected_experts=True` pe modelul de bază adăugând un OutputRecorder
# la runtime. Indexul 2 din output-ul tuple al router-ului reprezintă indicii experților.
inner = model.model
existing = _CAN_RECORD_REGISTRY.get(str(inner.__class__), {}) or {}
_CAN_RECORD_REGISTRY[str(inner.__class__)] = {
    **existing,
    "selected_experts": OutputRecorder(ReplayableQwen3MoeTopKRouter, index=2),
}

# Înregistrează
captured = inner(input_ids=input_ids, output_selected_experts=True)
selected_experts = captured.selected_experts  # tuple de LongTensors (num_tokens, top_k)

# Replay — aceeași cale a expertului indiferent de weights-urile curente ale router-ului
with replay_moe_routing(inner, list(selected_experts)):
    outputs = inner(input_ids=input_ids)
```

Replay-ul păstrează indicii exacte ai experților și recalculează scorurile de rutare cu weights curente ale router-ului, astfel că gradients curg prin parametrii activi în timp ce selecția experților rămâne fixă. Acesta este contractul minimal de replay utilizat în antrenarea MoE în stil Megatron.

### Interoperabilitate cu vLLM

Opțiunea `enable_return_routed_experts=True` din vLLM populează `CompletionOutput.routed_experts` ca un array `np.int32` de forma `(seq_len, num_layers, top_k)`. Convertește-l în lista per-layer pe care o utilizează acest pattern cu o singură expresie:

```python
selected = [
    torch.from_numpy(routed_experts[:, layer, :].copy()).long()
    for layer in range(routed_experts.shape[1])
]
with replay_moe_routing(model, selected):
    loss = model(input_ids=input_ids, labels=labels).loss
```

Aceeași rețetă se aplică și altor familii MoE — subclasează `*TopKRouter`-ul familiei, potrivește contractul de return original (de obicei `(router_logits, router_scores, router_indices)`) și înregistrează patch-ul. Consultă clasa router a fiecărui model pentru semnătura exactă.

## Referință API

[[autodoc]] transformers.monkey_patching.register_patch_mapping

[[autodoc]] transformers.monkey_patching.unregister_patch_mapping

[[autodoc]] transformers.monkey_patching.clear_patch_mapping

[[autodoc]] transformers.monkey_patching.get_patch_mapping

[[autodoc]] transformers.monkey_patching.apply_patches
