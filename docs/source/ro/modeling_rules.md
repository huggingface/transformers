<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Regulile de structură a modelului

Transformers aplică un set de reguli statice pe fiecare fișier `modeling_*.py`, `modular_*.py` și `configuration_*.py`. Pachetul [mlinter](https://github.com/huggingface/transformers-mlinter) furnizează motorul de verificare, iar repository-ul păstrează setul activ de reguli în `utils/rules.toml`. Acel fișier TOML local ne permite să activăm, dezactivăm sau ajustăm rapid regulile fără să așteptăm o nouă versiune de `transformers-mlinter`.

Acestea sunt convențiile pentru modele când adaugi sau modifici cod de modelare. Mențin codul consistent și asigură compatibilitatea cu features precum pipeline parallelism, device maps și weight tying.

## Rularea checker-ului

`make typing` rulează `mlinter` alături de type checker-ul `ty` prin wrapper-ul repository-ului, deci preia `utils/rules.toml`. Poți rula același wrapper direct cu comenzile de mai jos.

```bash
python utils/check_modeling_structure.py                 # verifică toate fișierele de modelare
python utils/check_modeling_structure.py --changed-only  # verifică doar fișierele modificate față de origin/main
python utils/check_modeling_structure.py --list-rules    # listează toate regulile și statusul lor de activare
python utils/check_modeling_structure.py --rule TRF001   # arată documentația built-in pentru o regulă specifică
```

Flag-ul `--changed-only` este cea mai rapidă opțiune în timpul dezvoltării. Verifică doar fișierele pe care le-ai modificat față de branch-ul main. Dacă invoci `mlinter` direct în loc de wrapper, pasează `--rules-toml utils/rules.toml` ca să fie aplicate override-urile locale.

## Remedierea unei violări

Când se detectează o violare a unei reguli, eroarea arată astfel:

```
src/transformers/models/acme/modeling_acme.py:18: TRF013: AcmeModel.__init__ does not call self.post_init().
```

Folosește ID-ul regulii ca să cauți remedierea în [referința regulilor](#referința-regulilor). TRF013 se declanșează când o subclasă [`PreTrainedModel`] nu apelează `self.post_init()`. Metoda respectivă efectuează pași esențiali de finalizare, iar omiterea ei cauzează bug-uri la rulare.

```diff
 class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
         self.layers = nn.ModuleList(
             [AcmeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
         )
+        self.post_init()
```

## Referința regulilor

Fiecare regulă de mai jos listează ce impune și un diff care arată remedierea. Rulează `python utils/check_modeling_structure.py --rule TRF001` ca să vezi documentația built-in pentru orice regulă cu setul curent de reguli al repository-ului.

<!-- BEGIN RULES REFERENCE -->

### TRF001

Verifică consistența denumirii între <Model>PreTrainedModel și config_class. Un config_class nepotrivit poate strica încărcarea, clasele auto și așteptările developerilor.

```diff
class AcmePreTrainedModel(PreTrainedModel):
-    config_class = WileConfig
+    config_class = AcmeConfig
```

### TRF002

Verifică dacă base_model_prefix, când este setat, este un string literal non-gol, fără spații albe. Prefixele invalide pot strica maparea cheilor la încărcarea de weights și pattern-urile de acces la modelul de bază.

```diff
class AcmePreTrainedModel(PreTrainedModel):
-    base_model_prefix = ""
+    base_model_prefix = "model"
```

### TRF003

Detectează metodele forward care folosesc vechiul pattern 'if not return_dict: return (x,)'. Vechiul pattern de branching return_dict este predispus la erori și verbose. Folosește în schimb decoratorii capture_output sau can_return_tuple.

```diff
-def forward(self, x, return_dict=None):
-    if not return_dict:
-        return (x,)
-    return AcmeModelOutput(last_hidden_state=x)
+@can_return_tuple
+def forward(self, x):
+    return AcmeModelOutput(last_hidden_state=x)
```

### TRF004

Verifică dacă nicio clasă de model nu definește o metodă tie_weights. Suprascrierea tie_weights duce la consecințe grave pentru încărcare, calculul device_map și salvare. Folosește atributul de clasă _tied_weights_keys ca să declari weights legate.

```diff
-def tie_weights(self):
-    self.lm_head.weight = self.emb.weight
+class AcmeForCausalLM(AcmePreTrainedModel):
+    _tied_weights_keys = ["lm_head.weight"]
```

### TRF005

Verifică forma lui _no_split_modules când este prezent. Valorile malformate pot strica partiționarea device-map și comportamentul de sharding.

```diff
-_no_split_modules = [SomeLayerClass, ""]
+_no_split_modules = ["AcmeDecoderLayer", "AcmeAttention"]
```

### TRF006

Verifică forward signatures care expun argumente de cache pentru utilizarea acelor argumente în corpul metodei. Argumentele de cache neutilizate pot indica suport de caching incomplet și comportament inconsistent al API-ului.

```diff
def forward(self, x, past_key_values=None, use_cache=False):
+    if use_cache:
+        ...
     return x
```

### TRF007

Verifică atribuirile de atribute self după self.post_init() în __init__. Mutarea structurii modelului după post_init poate ocoli logica intenționată de inițializare/finalizare.

```diff
def __init__(self, config):
     ...
-    self.post_init()
-    self.proj = nn.Linear(...)
+    self.proj = nn.Linear(...)
+    self.post_init()
```

### TRF008

Verifică utilizarea add_start_docstrings pe clasele de modele pentru argumente de docstring non-goale. Utilizarea cu decorator gol produce documentație neclară și slăbește calitatea documentației API generate.

```diff
-@add_start_docstrings("")
+@add_start_docstrings("The Acme model.")
 class AcmeModel(AcmePreTrainedModel):
     ...
```

### TRF009

Verifică fișierele de modelare pentru importuri cross-model precum transformers.models.other_model.* sau importuri from ..other_model.*. Importurile de implementare cross-model violează politica single-file și fac comportamentul modelului mai greu de inspectat și întreținut.

```diff
-from transformers.models.llama.modeling_llama import LlamaAttention
+# Păstrează implementarea locală în acest fișier.
+# Dacă refolosești cod, copiază-l cu un comentariu # Copied from.
```

### TRF010

Verifică subclasele directe PreTrainedConfig/PretrainedConfig din configuration_*.py și modular_*.py pentru un decorator explicit @strict(accept_kwargs=True). Fără strict, noile clase de config ratează contractul de validare de tip runtime al repository-ului și se abat de la standardul de config bazat pe dataclass.

```diff
+@strict(accept_kwargs=True)
 class AcmeConfig(PreTrainedConfig):
     ...
```

### TRF011

În metodele forward() ale subclaselor PreTrainedModel, verifică accesele la atribute ale submodulelor care nu ar exista pe torch.nn.Identity. Aceasta include accesele la atribute pe variabilele de loop care iterează peste self.layers și lanțurile self.<submodule>.<attr> unde <attr> nu este un atribut standard al nn.Module. Pipeline parallelism poate înlocui orice submodul cu torch.nn.Identity. Accesarea atributelor personalizate (e.g. decoder_layer.attention_type) pe un modul înlocuit ridică AttributeError la rulare. Metadata per-layer trebuie citită din self.config.

```diff
def forward(self, ...):
-    for decoder_layer in self.layers:
+    for i, decoder_layer in enumerate(self.layers):
         hidden_states = decoder_layer(
             hidden_states,
-            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
+            attention_mask=causal_mask_mapping[self.config.layer_types[i]],
         )
```

### TRF012

Verifică dacă _init_weights(self, module) nu folosește operații in-place (e.g. .normal_(), .zero_()) direct pe module weights. Ne bazăm pe flag-uri interne setate pe parametri ca să urmărim dacă au nevoie de re-inițializare. Operațiile in-place ocolesc acest mecanism. Folosește în schimb primitivele `init`.

```diff
+from transformers import initialization as init
+
 def _init_weights(self, module):
-    module.weight.normal_(mean=0.0, std=0.02)
+    init.normal_(module.weight, mean=0.0, std=0.02)
```

### TRF013

Verifică dacă fiecare subclasă PreTrainedModel cu o metodă __init__ apelează self.post_init(). În fișierele modulare, apelarea super().__init__() este de asemenea acceptată deoarece propagă post_init din părinte. post_init efectuează finalizarea esențială (inițializarea weights, configurarea gradient checkpointing-ului etc.). Omiterea lui cauzează bug-uri subtile la runtime.

```diff
class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
         self.layers = nn.ModuleList(...)
+        self.post_init()
```

### TRF014

Verifică dacă `trust_remote_code` este pasat sau folosit în cod (e.g. ca kwarg) în fișierele de integrare native ale modelului. `trust_remote_code` permite încărcarea arbitrară, inclusiv a binarelor, ceea ce ar trebui să fie o funcționalitate pentru utilizatori avansați, nu un use case standard. Integrările native nu trebuie să depindă de el, deoarece codul remote nu poate fi revizuit sau întreținut în transformers.

```diff
class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
-        self.model = AutoModel.from_pretrained(..., trust_remote_code=True)
+        self.model = AutoModel.from_pretrained(...)
```

### TRF015

Când o subclasă PreTrainedModel definește _tied_weights_keys ca o colecție non-goală, verifică dacă fișierul de configurare corespunzător declară un câmp tie_word_embeddings. Fără tie_word_embeddings în config, utilizatorii nu pot controla comportamentul de legare a weights-urilor. Modelul leagă weights-urile necondiționat, stricând serialization round-trip-urile și împiedicând fine-tuning-ul cu heads nelegate.

```diff
# configuration_foo.py
 @strict(accept_kwargs=True)
 class FooConfig(PreTrainedConfig):
     hidden_size: int = 768
+    tie_word_embeddings: bool = True
```

<!-- END RULES REFERENCE -->

## Suprimarea violărilor

Dacă trebuie să suprimezi o violare a unei reguli, folosește una din cele două opțiuni de mai jos.

### Suprimare inline

Adaugă un comentariu `# trf-ignore: RULE_ID` pe linia violatoare. Include o explicație ca să înțeleagă reviewerii de ce suprimarea este justificată.

```py
# trf-ignore: TRF011 — masca este derivată din self.config, nu din layer
hidden_states = layer(hidden_states, attention_mask=mask_from_config)
```

Nu folosi `trf-ignore` ca să ascunzi violări care ar trebui remediate în cod.

### `allowlist_models`

Pentru modelele cu cod legacy care nu poate fi remediat imediat, adaugă numele directorului modelului în lista `allowlist_models` a regulii relevante din [mlinter rules.toml](https://github.com/huggingface/transformers-mlinter/blob/main/mlinter/rules.toml).

```toml
[rules.TRF004]
allowlist_models = ["existing_model", "your_model_name"]
```
