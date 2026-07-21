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

# Adaugă un model cu transformers modulare

Transformers modulare reduce codul necesar pentru a adăuga un model, permițând importuri și moștenire, spre deosebire de politica [un model, un fișier](https://huggingface.co/blog/transformers-design-philosophy). În loc să repeți componentele modelului în mai multe fișiere, adaugi un fișier *modular* în folder-ul modelului și moștenește din clasele existente.

Un convertor generează fișiere standalone din fișierul modular. Utilizatorii primesc aceeași interfață single-file cu care sunt obișnuiți.

> [!NOTE]
> Transformers modulare nu e menit să înlocuiască [codul de modelare legacy](./add_new_model). Dacă modelul tău nu se bazează pe un model existent, adaugă manual un fișier `modeling.py`. Același lucru se aplică fișierelor de configurare, de tokenization sau procesare care nu pot moșteni dintr-un fișier similar.
>
> Nu există nicio ordine unică corectă. Unii contributori scriu mai întâi fișierul modular și generează din el. Alții încep cu un fișier `modeling.py` scris manual și îl refactorizează ulterior într-un fișier modular. Ambele abordări funcționează.

## Implementarea unui fișier modular

Începe prin a găsi un model în Transformers similar cu al tău. Puncte de plecare bune sunt [Mistral], [Qwen2], [Cohere], [Cohere2] și [Llama]. Tabelul de mai jos mapează componentele comune la modele din care poți moșteni.

| Componentă | Model |
|---|---|
| Mixture of experts | Mixtral sau Qwen2-MoE |
| Interleaved (și/sau parțial) rotary embedding | GLM, Phi |
| State space models | Jamba, Bamba, Zamba, Mamba2 |
| Recurrent hidden states | Gemma2 |
| Sliding window attention/full attention patterns per layer | Gemma2, Cohere2 |
| QKV clipping | Olmo |
| QK normalization | Olmo2, Cohere |
| Fused QKV (nu se recomandă) | Phi3 |

> [!TIP]
> Folosește instrumentul [modular-detector-v2](https://huggingface.co/spaces/Molbap/modular-detector-v2) pentru a găsi implementări existente din care să moștenești. Introdu un snippet de cod și îți returnează cele mai similare metode deja existente în Transformers, ca să identifici cea mai bună clasă părinte înainte să începi să scrii.

Nu modifica un model existent doar ca să faci moștenirea să funcționeze pentru cel nou. Dacă redenumirea sau subclasarea unei clase părinte este prea incomodă, copiază direct codul relevant.

Creează `src/transformers/models/<name>/modular_<name>.py`, unde `<name>` corespunde numelui directorului modelului în snake_case. Această secțiune te ghidează prin implementarea [Olmo2] din [Olmo] cu abordarea modulară (consultă fișierul original [modular_olmo2.py](../../../src/transformers/models/olmo2/modular_olmo2)).

### Config

Există două puncte în care [`Olmo2Config`] diferă de [`OlmoConfig`].

1. Există un argument nou, `rms_norm_eps`.
2. Argumentul `clip_qkv` nu mai este folosit.

Declară argumentele noi ca adnotări de tip la nivel de clasă cu o valoare implicită. Pentru argumentele eliminate, atribuie `AttributeError()` ca să suprimi atributul moștenit în fișierul generat (vezi [Eliminarea atributelor](#eliminarea-atributelor)).

```diff
- @auto_docstring(checkpoint="allenai/OLMo-7B-hf")
+ @auto_docstring(checkpoint="allenai/Olmo2-7B-1124-hf")
+ @strict
- class OlmoConfig(PreTrainedConfig):
+ class Olmo2Config(OlmoConfig):
      ...
-     model_type = "olmo"
+     model_type = "olmo2"
      ...
+     rms_norm_eps: float = 1e-5
-     clip_qkv: float | None = None
+     clip_qkv = AttributeError()
```

`@auto_docstring` generează automat documentația standard pentru argumente (vezi ghidul [@auto_docstring](./auto_docstring)). `@strict` respinge kwargs necunoscute la momentul instanțierii, prinde greșelile de scriere și argumentele depășite din timp. Adaugă ambele la fiecare clasă de config pentru că decoratorii nu sunt moșteniți din părinte. Declară-i explicit chiar dacă config-ul părinte îi are deja.

Pentru a seta un atribut derivat sau a gestiona logica de compatibilitate înapoi, folosește `__post_init__` în loc de `__init__`. De exemplu, Cohere2 calculează `head_dim` și derivă `layer_types` la momentul inițializării.

```py
def __post_init__(self, **kwargs):
    if self.num_key_value_heads is None:
        self.num_key_value_heads = self.num_attention_heads
    self.head_dim = self.hidden_size // self.num_attention_heads
    super().__post_init__(**kwargs)
```

Pentru modelele cu suport de tensor sau pipeline parallelism, definește `base_model_tp_plan` și `base_model_pp_plan` ca dicționare la nivel de clasă pe config. Ambele dicționare definesc cum să se facă sharding-ul modelului pe dispozitive. Vezi config-urile existente precum [Olmo2](../../../src/transformers/models/olmo2/modular_olmo2) sau [Cohere2](../../../src/transformers/models/cohere2/modular_cohere2) pentru exemple.

```py
class MyNewModelConfig(PreTrainedConfig):
    model_type = "my_new_model"

    # Tensor parallelism: mapează pattern-uri de nume de layers la strategii de sharding.
    # Folosește "colwise" / "rowwise" pentru sharding standard, sau variantele "gather_output" /
    # "split_input" când o operație suplimentară (e.g. un QK norm) împiedică fuzionarea.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    # Pipeline parallelism: mapează numele submodulelor la numele tensorilor (input, output).
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```

### Norm

Ca să copiezi o clasă părinte fără modificări, moștenește cu `pass`. Linter-ul copiază conținutul părintelui și redenumește toate referințele să corespundă noului model.

```py
from ..olmo.modeling_olmo import OlmoRotaryEmbedding

class Olmo2RotaryEmbedding(OlmoRotaryEmbedding):
    pass
```

Ca să schimbi un comportament specific, moștenește și suprascrie doar ce diferă. [`Olmo2RMSNorm`] diferă de [`LlamaRMSNorm`] pe o singură linie. Înmulțirea se face *înainte* de a converti înapoi la input dtype, nu după.

```diff
  from ..llama.modeling_llama import LlamaRMSNorm

  class Olmo2RMSNorm(LlamaRMSNorm):
      def forward(self, hidden_states):
          input_dtype = hidden_states.dtype
          hidden_states = hidden_states.to(torch.float32)
          variance = hidden_states.pow(2).mean(-1, keepdim=True)
          hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
-         return self.weight * hidden_states.to(input_dtype)
+         return (self.weight * hidden_states).to(input_dtype)
```

### Attention

Attention-ul lui Olmo2 este identic cu cel al lui Olmo, cu excepția că aplică [`RMSNorm`] la queries și keys și elimină qkv clipping. `super().__init__(...)` copiază corpul părintelui și adaugă cele două linii noi de normalizare. `forward`-ul este redefinit complet pentru că queries și keys trec acum prin norme înainte de proiecție. Linter-ul trage și orice funcții importate în fișierul generat, inclusiv `apply_rotary_pos_emb`, `eager_attention_forward` și dependecies.

```diff
  class Olmo2Attention(OlmoAttention):
      def __init__(self, config: Olmo2Config, layer_idx: int | None = None):
          super().__init__(config, layer_idx=layer_idx)
+         self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
+         self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)

      def forward(self, ...):
          ...
-         query_states = self.q_proj(hidden_states)
-         key_states = self.k_proj(hidden_states)
+         query_states = self.q_norm(self.q_proj(hidden_states))
+         key_states = self.k_norm(self.k_proj(hidden_states))
          value_states = self.v_proj(hidden_states)

-         if self.config.clip_qkv is not None:
-             query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
-             key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
-             value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
-
          ...
```

### DecoderLayer

După `super().__init__(...)`, suprascrie atributele de normalizare cu instanțe `Olmo2RMSNorm` și reatribuie `self.self_attn` la noua clasă `Olmo2Attention`. `del self.input_layernorm` elimină atribuirea `input_layernorm` a părintelui deoarece Olmo2 aplică norma *după* attention, nu înainte. Vezi [Eliminarea atributelor](#eliminarea-atributelor) pentru detalii despre ce face și ce nu face `del`.

`forward`-ul este rescris pentru a reflecta plasarea normei post-attention. O rescriere a `forward`-ului este necesară doar când un atribut este redenumit, nu când îi schimbi doar tipul.

```diff
  class Olmo2DecoderLayer(OlmoDecoderLayer):
      def __init__(self, config: Olmo2Config, layer_idx: int):
          super().__init__(config, layer_idx=layer_idx)
-         self.self_attn = OlmoAttention(config=config, layer_idx=layer_idx)
-         self.input_layernorm = OlmoLayerNorm(config.hidden_size)
-         self.post_attention_layernorm = OlmoLayerNorm(config.hidden_size)
+         self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx)
+         del self.input_layernorm
+         self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+         self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

      def forward(self, ...):
          residual = hidden_states
-         hidden_states = self.input_layernorm(hidden_states)
          # Self Attention
          hidden_states, _ = self.self_attn(...)
-         hidden_states = residual + hidden_states
+         hidden_states = self.post_attention_layernorm(hidden_states)
+         hidden_states = residual + hidden_states

          # Fully Connected
          residual = hidden_states
-         hidden_states = self.post_attention_layernorm(hidden_states)
          hidden_states = self.mlp(hidden_states)
-         hidden_states = residual + hidden_states
+         hidden_states = self.post_feedforward_layernorm(hidden_states)
+         hidden_states = residual + hidden_states
          return hidden_states
```

### Model

Doar tipul lui `self.norm` se schimbă aici. Metoda `forward` este identică cu cea a părintelui, deci linter-ul o preia automat.

```diff
  class Olmo2Model(OlmoModel):
      def __init__(self, config: Olmo2Config):
          super().__init__(config)
-         self.layers = nn.ModuleList(
-             [OlmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
-         )
-         self.norm = OlmoLayerNorm(config.hidden_size)
+         self.norm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+         self.layers = nn.ModuleList(
+             [Olmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
+         )
```

### Model head

Logica este identică cu [`OlmoForCausalLM`], deci nu sunt necesare modificări.

```py
from ..olmo.modeling_olmo import OlmoForCausalLM

class Olmo2ForCausalLM(OlmoForCausalLM):
    pass
```

### Alte clase

Fișierul [modeling_olmo2.py](../../../src/transformers/models/olmo2/modeling_olmo2) generat de linter conține și clase ([`Olmo2MLP`], [`Olmo2RotaryEmbedding`], [`Olmo2PreTrainedModel`]) care nu au fost definite explicit în `modular_olmo2.py`.

Linter-ul trage orice clasă de care depinde o clasă moștenită, dacă nu o redefinești explicit. Funcțiile importate precum `apply_rotary_pos_emb` urmează aceeași regulă.

De exemplu, [`OlmoDecoderLayer`] are `self.mlp = OlmoMLP(config)`. [`Olmo2MLP`] n-a fost definit niciodată în fișierul modular, deci linter-ul îl creează automat, echivalent cu folosirea `pass`.

```py
from ..olmo.modeling_olmo import OlmoMLP

class Olmo2MLP(OlmoMLP):
    pass
```

Dacă vrei ca [`Olmo2MLP`] să moștenească dintr-un model diferit, fii explicit.

```py
# trece la definiția Mistral
from ..mistral.modeling_mistral import MistralMLP

class Olmo2MLP(MistralMLP):
    pass
```

### Finalizarea fișierului

Fiecare fișier modular trebuie să declare un `logger` și o listă `__all__` la nivel de modul.

```py
logger = logging.get_logger(__name__)

__all__ = [
    "Olmo2Config",
    "Olmo2ForCausalLM",
    "Olmo2Model",
    "Olmo2PreTrainedModel",
]
```

`__all__` trebuie să listeze fiecare clasă publică din fișier. Convertorul și importurile din aval depind de aceasta. O clasă lipsă din `__all__` nu va fi exportată corect.

## Generarea fișierelor de modelare

Script-ul `modular_model_converter.py` generează fișiere standalone `modeling.py`, `configuration.py` și altele din fișierul tău modular. Pentru fiecare clasă moștenită, copiază corpul părintelui în copil, redenumește toate referințele să corespundă noului model și trage orice funcții sau clase ajutătoare de care depind acei părinți.

Fișierele de ieșire nu conțin importuri cross-model și nu moștenesc din alte directoare de modele. Linter-ul aplatizează moștenirea la un singur nivel. Dacă [`Olmo2Attention`] moștenește din [`OlmoAttention`], `Olmo2Attention`-ul generat este complet autonom. Dar dacă `OlmoAttention` însuși a moștenit din altceva, linter-ul nu inlinează acel strămoș.

Rulează comanda de mai jos pentru a genera fișiere dintr-un fișier modular.

```bash
python utils/modular_model_converter.py your_model
```

Nu edita niciodată direct fișierele generate, pentru că orice modificări vor fi suprascrise la următoarea rulare.

## Tipare pentru fișierele modulare

Secțiunile de mai jos documentează tipare comune de utilizare, cum ar fi eliminarea atributelor sau suprascrierea metodelor decorate, când lucrezi cu un fișier modular.

### Eliminarea atributelor

Eliminarea unui atribut moștenit depinde de dacă lucrezi cu o clasă de config sau o subclasă `nn.Module`.

Pentru o clasă de config, atribuie `AttributeError()` la atribut la nivel de clasă.

```py
class MyNewConfig(ParentConfig):
    removed_attr = AttributeError()
```

Linter-ul elimină complet declarația atributului din fișierul de config generat. Clasele de config folosesc un layout de tip dataclass fără `__init__`, deci atribuirea `AttributeError()` la nivel de clasă este abordarea corectă.

Pentru o subclasă `nn.Module`, folosește `del self.attribute` după `super().__init__(...)`.

```py
class MyNewModel(ParentModel):
    def __init__(self, config: MyNewConfig):
        super().__init__(config)
        del self.attribute
```

`del self.attribute` elimină doar linia de atribuire `self.attribute = ...` din corpul copiat al părintelui. Nu elimină alte linii care fac referire la `self.attribute`. Dacă `forward`-ul părintelui sau alte metode fac referire la atribut, suprascrie și acele metode.

```py
class DummyModel(nn.Module):
    def __init__(self, config: DummyConfig):
        super().__init__()
        self.attribute = config.attribute
        if self.attribute:
            # fă mai multe lucruri cu `self.attribute` aici
            ...

class MyNewDummyModel(DummyModel):
    def __init__(self, config: MyNewDummyConfig):
        super().__init__(config)
        del self.attribute
        # 'self.attribute = config.attribute' este eliminat, dar blocul 'if self.attribute:' rămâne.
        # Suprascrie forward() sau orice altă metodă care face referire la self.attribute.
```

### Utilizarea `super()`

`super().__init__(config)` îi spune convertorului să copieze corpul părintelui în copil. Două tipare îți permit să suprascrii acest comportament.

- Apelează direct o clasă părinte specifică când ai nevoie ca ieșirea generată să apeleze un strămoș (`nn.Module.__init__`) în loc de părintele modular.
- Folosește `**super_kwargs` pentru a moșteni semnătura completă a metodei unui părinte, adăugând în același timp un docstring personalizat sau schimbând un decorator.

#### Apelarea directă a unui strămoș

Fii explicit despre ce clasă apelezi când ai nevoie ca `super()` să țintească părintele clasei generate în loc de părintele modular. Exemplul de mai jos apelează direct `nn.Module.__init__(self)`. `DummyModule` este el însuși un `nn.Module`, deci convertorul îl scrie ca `super().__init__()` în `MyNewDummyModule`-ul generat.

```py
class MyNewDummyModule(DummyModule):                   |     class MyNewDummyModule(nn.Module):
                                                       |
  def __init__(self):                                  |       def __init__(self):
    nn.Module.__init__(self)                           |         super().__init__()
    self.foo = config.foo                              |         self.foo = config.foo
    ...                                                |         ...
```

#### super_kwargs

Folosește `**super_kwargs` pentru a moșteni semnătura completă a metodei unui părinte, adăugând în același timp un docstring personalizat sau schimbând un decorator. În semnătura suprascrisă, îi spune linter-ului să expandeze toate argumentele părintelui în ieșirea generată.

Cel mai comun use case este adăugarea unui docstring specific modelului, cum ar fi documentarea argumentului `labels`, fără a rescrie semnătura completă.

```py
# modular_gemma.py
class GemmaForCausalLM(LlamaForCausalLM):
    def forward(**super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM
        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        ...
        ```"""
        return super().forward(**super_kwargs)
```

`GemmaForCausalLM.forward`-ul generat are semnătura completă a lui `LlamaForCausalLM` fără copiere manuală.

`**super_kwargs` este o scurtătură pentru cazuri de nișă. Dacă schimbi comportamentul, scrie semnătura completă.

### Ștergerea metodelor neutilizate

Elimină o metodă a părintelui suprascriind-o cu o instrucțiune `raise AttributeError("")`. Linter-ul elimină metoda din fișierul generat.

```py
class GemmaTokenizer(LlamaTokenizer):
    ...

    def get_spm_processor(self):
        raise AttributeError("Not needed for Gemma")

    def unk_token_length(self):
        raise AttributeError("Not needed for Gemma")
```

### Suprascrierea metodelor decorate

Când suprascrieți o metodă decorată a părintelui, decoratorul părintelui se transferă automat. Dacă adaugi propriul decorator, acesta îl înlocuiește pe cel al părintelui.

Două decoratoare apar frecvent în librărie: unul pentru [capturarea ieșirilor intermediare ale modelului](./model_output_tracing) și unul pentru [auto-generarea docstring-urilor](./auto_docstring).

În exemplul de mai jos, o subclasă suprascrie o metodă decorată a părintelui fără a adăuga propriul decorator. Decoratorul părintelui se transferă.

```py
class NewModel(DummyModel):       |   class NewModel(nn.Module):
  ...                             |     ...
                                  |
  def forward(...):               |     @decorator(...)
    ...                           |     def forward(...):
                                  |       ...
```

Dacă adaugi un decorator nou, decoratorul tău îl înlocuiește pe cel al părintelui.

```py
class NewModel(DummyModel):       |   class NewModel(nn.Module):
  ...                             |     ...
                                  |
  @my_new_decorator(...)          |     @my_new_decorator(...)
  def forward(...):               |     def forward(...):
    ...                           |       ...
```

### Denumire specială

Linter-ul redenumește automat totul când moștenești dintr-o clasă. Folosește același prefix de nume de clasă în toate clasele din același fișier.

Evită amestecarea prefixelor ca în exemplul de mai jos. `MyModelIncredibleMLP` încalcă convențiile de denumire și linter-ul nu va ști dacă să folosească `MyModelIncredible` sau `MyModel` la redenumirea dependențelor de ordin superior.

```py
class MyModelIncredibleMLP(LlamaMLP):
    ...

class MyModelDecoderLayer(LlamaDecoderLayer):
    ...
```

Fără [dependențe implicite](#alte-clase), poți redenumi o singură clasă local. Redefinește explicit fiecare altă mențiune a acelei clase cu noul tipar de nume. Altfel, linter-ul adaugă o clasă `MyModelMLP` nedorită alături de `MyModelIncredibleMLP`.

Linter-ul ridică un avertisment când detectează un prefix ambiguu.

```text
We detected multiple prefix names when inheriting from transformers.models.llama.modeling_llama: ('Emu3Text', 'Emu3'). We will only use the most used 'Emu3' prefix when grabbing args and dependencies. Make sure to subclass the intermediate classes with the prefix you want (if different from 'Emu3') or use a single prefix in all the modular (best).
```

Prefixele ambigue sunt cele mai comune în modelele multimodale unde numele claselor includ un calificativ de modalitate precum `Text`. Ca să dai unei dependențe un prefix specific, redenumește-o explicit cu un `pass`.

```py
class Emu3TextMLP(LlamaMLP):
    pass
```

### Docstring-uri pentru config

Linter-ul nu suportă încă moștenirea parțială a docstring-urilor. Când adaugi sau elimini atribute de config, adaugă docstring-ul complet direct în fișierul modular sub definiția clasei.

## Conversia checkpoint-urilor

Odată ce ai generat fișierele de modelare, verifică că weights-urile reale se încarcă corect. Scrie un script de conversie pentru a traduce formatul checkpoint-ului upstream într-unul compatibil cu Transformers, apoi salvează-l pe Hub.

### Scrie un script de conversie

Adaugă un fișier `convert_<model>_to_hf.py` în `src/transformers/models/<model>/`. Scriptul încarcă weights-urile upstream, redenumește și remodelează cheile să corespundă numelor parametrilor modulului tău și salvează rezultatul cu [`~PreTrainedModel.save_pretrained`].

> [!TIP]
> Caută un script existent pe care să îl copiezi și adaptezi. Modelele din `src/transformers/models/` includ un `convert_*_to_hf.py` pe care îl poți folosi ca punct de plecare.

După rularea scriptului, încarcă checkpoint-ul salvat cu [`~PreTrainedModel.from_pretrained`] și confirmă că fiecare weight s-a încărcat corect. Cheile de checkpoint neutilizate indică nume nepotrivite, deci printează-le ca să detectezi problemele din timp.

```py
model = YourModelForTask.from_pretrained("path/to/output/")
```

Verifică potrivirile de formă și nume când iterezi peste chei. Nepotrivirile de formă indică de obicei că un parametru din config este greșit, că arhitectura diferă de cea originală sau că un weight trebuie transpus.

```py
for key, tensor in original_state_dict.items():
    hf_tensor = hf_model.state_dict().get(mapped_key)
    assert hf_tensor.shape == tensor.shape, (
        f"Shape mismatch for {key}: expected {tensor.shape}, got {hf_tensor.shape}"
    )
```

Rezolvă orice problemă iterând între fișierul modular, fișierul de modelare generat și scriptul de conversie până când toate weights-urile se încarcă curat.

Odată ce checkpoint-ul se încarcă curat, publicați-l pe Hub cu [`~PreTrainedModel.push_to_hub`]. Consultă ghidul [distribuirea modelelor](./model_sharing) pentru mai multe detalii.

```py
model.push_to_hub("username/your-model-name")
```

### Maparea conversiei la runtime

Adaugă o mapare runtime în `src/transformers/conversion_mapping.py` când weights publicate nu corespund layout-ului de parametri al modulului tău. Cazurile comune includ weights fuzionate stocate separat și tensorii de experți MoE care trebuie stivuiți. Maparea permite ca [`~PreTrainedModel.from_pretrained`] să încarce checkpoint-ul de pe Hub fără un pas de export separat.

Consultă ghidul [încărcarea dinamică de weights](./weightconverter) pentru cum să scrii reguli [`WeightRenaming`] și [`WeightConverter`] și să le înregistrezi pentru `model_type`-ul tău.

## Pașii următori

- [Regulile de structură a modelului](./modeling_rules) sunt reguli statice aplicate pe toate fișierele `modeling_*.py`, `modular_*.py` și `configuration_*.py`. Rulează `make typing` să le verifici înainte de a deschide un PR.
- [Adaugă componente de procesare vizuală](./add_vision_processing_components) te ghidează prin adăugarea unui procesor de imagini, procesor video și procesor pentru un model multimodal.
- [Auto-generarea docstring-urilor](./auto_docstring) arată cum să folosești `@auto_docstring` ca să nu scrii manual documentația pentru argumente la API-uri comune.
- [Scrierea testelor pentru modele](./testing) acoperă cum să scrii teste de integrare pentru noul tău model și să le rulezi local.
- [Verificările pentru pull request](./pr_checks) explică verificările CI pe care PR-ul tău trebuie să le treacă înainte de a fi integrat și cum să le reproduci și rezolvi local.
