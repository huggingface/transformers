# Urmărirea output-urilor intermediare ale modelului

Metoda `forward()` a fiecărui model obișnuia să rezolve manual flag-uri de tip `None` precum `output_attentions` din valorile implicite din config, să acumuleze attention weights și hidden states per layer în tuples și să convertească dataclasses [`ModelOutput`] în tuple-uri plain când `return_dict=False`. Doi decoratori înlocuiesc tot acel cod boilerplate.

- `@capture_outputs` rezolvă flag-urile de ieșire, colectează valorile intermediare și gestionează conversia `return_dict`.
- `@merge_with_config_defaults` rezolvă `use_cache` din config. Omite-l pentru modelele care nu fac cache, precum [`CLIPModel`].

Vei întâlni acești decoratori mai ales când integrezi un model nou. Vezi [adăugarea unui model în 🤗 Transformers](./modular_transformers) pentru un ghid pas cu pas.

## Declară ce submodule să captezi

Aplică `@capture_outputs` pe metoda `forward()` a modelului de bază. Atașează forward hooks care:

- Interceptează ieșirile din clasele de submodule specificate în timpul forward pass-ului, fără ca acele submodule să știe că sunt observate.
- Colectează attention weights și hidden states per layer în tuple-uri.
- Injectează valorile colectate în dataclass-ul [`ModelOutput`] returnat.
- Convertesc dataclass-ul într-un tuple plain când `return_dict=False`.
- Rezolvă `output_attentions` și `output_hidden_states` din kwargs sau `self.config` când sunt `None`.

## Mapează câmpurile de ieșire la submodule

`@capture_outputs` trebuie să știe ce submodul produce ce ieșire. Declară un dicționar `_can_record_outputs` la nivel de clasă pe subclasa ta `PreTrainedModel`. Fiecare cheie este un nume de câmp de ieșire (`"hidden_states"`, `"attentions"`, `"cross_attentions"`), iar fiecare valoare este o clasă de modul, un șir cu numele clasei, o instanță `OutputRecorder` sau o listă cu acestea pentru a înregistra mai multe tipuri de module sub aceeași cheie.

## Control detaliat cu OutputRecorder

`OutputRecorder` acceptă un `target_class` (o subclasă `nn.Module` ale cărei output-uri să le colectezi) și un `index` opțional pentru a selecta ce element din tuple-ul de ieșire al modulului să ia. Pasează `layer_name` ca să atașezi hook-uri doar la modulele cu un nume de atribut specific. Folosește `layer_name` când două layers partajează aceeași clasă, de exemplu self-attention vs. cross-attention.

Exemplul de mai jos arată ambii decoratori în practică cu diferite niveluri de control al ieșirii. Vezi [LlamaModel](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) pentru o referință din lumea reală.

```python
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs, OutputRecorder

class MyPreTrainedModel(PreTrainedModel):
    _can_record_outputs = {
        # Capturează hidden_states: hook-ul se declanșează după fiecare forward MyDecoderBlock,
        # luând prima ieșire (index 0 implicit).
        "hidden_states": MyDecoderBlock,

        # Capturează self-attention weights: hook-ul se declanșează după fiecare forward MyAttention,
        # luând a doua ieșire (index=1 implicit).
        "attentions": MyAttention,

        # Capturează cross-attention weights: aceeași clasă, submodul diferit.
        # layer_name țintește atributul `self.crossattention` din interiorul block-ului.
        # Capturează a doua ieșire cum s-a cerut (index=1)
        "cross_attentions": OutputRecorder(
            MyAttention, layer_name="crossattention", index=1
        ),
    }

# Acum în forward-ul modelului de bază avem nevoie de decoratori și `Unpack` `kwargs`
class MyModel(MyPreTrainedModel):

    @merge_with_config_defaults # ← rezolvă use_cache
    @capture_outputs            # ← gestionează colectarea ieșirilor + return_dict
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        # Nu e nevoie de colectare manuală. Rulează layers normal.
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)

        # Returnează ieșirile primare. Decoratorul va completa automat
        # hidden_states/attentions/cross_attentions.
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values
        )
```

## Patch-uiește clasele de layers

Urmărirea output-urilor depinde de `_can_record_outputs` care pointează la clasele *exacte* pe care layer-urile modelului le instanțiază. Dacă înlocuiești o implementare de layer cu un kernel de attention personalizat, un layer de expert quantized sau o variantă arhitecturală, acei pointeri trebuie să rămână sincronizați. [API-ul de patching](./monkey_patching) oferă un registru global curat pentru a menține `_can_record_outputs` consistent.

`register_patch_mapping` mapează numele originale ale claselor la subclase `nn.Module` de înlocuire. Cheile pot fi nume exacte de clase sau pattern-uri regex. Potrivirile exacte au prioritate. Pattern-urile sunt testate cu `re.search()`, deci pattern-urile neancorate se potrivesc oriunde în numele clasei. Înregistrarea aceleiași chei de două ori ridică `ValueError` dacă nu pasezi `overwrite=True`.

Elimină intrările cu `unregister_patch_mapping`.

```python
from transformers.monkey_patching import register_patch_mapping, unregister_patch_mapping

# Nume exact – înlocuiește doar Qwen2MoeExperts
register_patch_mapping({"Qwen2MoeExperts": SequentialExperts})

# Regex – înlocuiește orice clasă al cărei nume se termină cu "Attention"
register_patch_mapping({".*Attention$": FusedAttention})

# Versiunea ancorată – se potrivește doar cu Llama2Attention, Llama3Attention, …
register_patch_mapping({"^Llama\\d+Attention$": CustomLlamaAttention})

# La fel, cheile personalizate pot fi eliminate din registru pasând numele care a fost înregistrat
unregister_patch_mapping(["Qwen2MoeExperts", ".*Attention$"])
```

Odată ce mapările sunt înregistrate, `patch_output_recorders` parcurge fiecare submodul și actualizează fiecare `OutputRecorder.target_class` la înlocuitorul înregistrat.

> [!TIP]
> Metoda [`~PreTrainedModel.from_pretrained`] apelează `patch_output_recorders` automat. Trebuie să o apelezi tu doar când construiești un model direct.

```python
from transformers.monkey_patching import patch_output_recorders
# Construit manual, în afara from_pretrained
model = Qwen2MoeModel(config)

# Fără asta, _can_record_outputs tot pointează la clasa originală Qwen2MoeExperts
# și hook-urile nu se vor declanșa niciodată pe instanțele CustomExperts.
patch_output_recorders(model)
```
