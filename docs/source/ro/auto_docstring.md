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

# Auto-generarea docstring-urilor

Decoratorul `@auto_docstring` generează docstring-uri consistente pentru clasele și metodele de model. Trage automat descrierile standard ale argumentelor, deci scrii documentație doar pentru argumentele noi sau personalizate. Când [adaugi un model nou](./modular_transformers), sari peste boilerplate și concentrează-te pe ce e nou.

## @auto_docstring

Importă decoratorul în fișierul tău `modular_model.py` (sau `modeling_model.py` pentru modele mai vechi).

```python
from ...utils import auto_docstring
```

Dacă modelul tău moștenește dintr-un alt model din librărie într-un fișier modular, `@auto_docstring` este deja aplicat în părinte. `make fix-repo` îl copiază în fișierul `modeling_model.py` generat. Aplică decoratorul explicit doar ca să personalizezi comportamentul (modele standalone, intro-uri personalizate sau argumente suprascrise).

> [!WARNING]
> Când suprascrii orice decorator într-un fișier modular, include **toți** decoratorii din funcția sau clasa părinte. Dacă suprascrii doar unii, restul nu vor apărea în fișierul de modelare generat.

Decoratorul acceptă următoarele argumente opționale:

| argument | descriere |
|---|---|
| `custom_intro` | O descriere a clasei sau metodei, inserată înainte de secțiunea Args. Necesară pentru clasele al căror nume nu se termină cu un [sufix recunoscut](#cum-funcționează) precum `ForCausalLM` sau `ForTokenClassification`. |
| `custom_args` | Text de docstring pentru parametri specifici. Util când aceleași argumente personalizate apar în mai multe locuri din fișierul de modelare. |
| `checkpoint` | Un identificator de checkpoint (`"org/my-model"`) folosit ca să genereze exemple de utilizare. Suprascrie checkpoint-ul dedus automat din clasa de config. Se setează de obicei pe clasele de config. |

## Utilizare

Cum funcționează `@auto_docstring` depinde de ce decorezi. Clasele de model trag documentația parametrilor din `__init__`, clasele de config trag din adnotările la nivel de clasă, clasele de procesor auto-generează intro-uri din componentele lor, iar metodele precum `forward` primesc tipuri de returnare și exemple de utilizare.

### Clase de model

Pune `@auto_docstring` direct deasupra definiției clasei. Decoratorul derivă descrierile parametrilor din semnătura și docstring-ul metodei `__init__`.

```python
from transformers.modeling_utils import PreTrainedModel
from ...utils import auto_docstring

@auto_docstring
class MyAwesomeModel(PreTrainedModel):
    def __init__(self, config, custom_parameter: int = 10, another_custom_arg: str = "default"):
        r"""
        custom_parameter (`int`, *optional*, defaults to 10):
            Descrierea custom_parameter pentru MyAwesomeModel.
        another_custom_arg (`str`, *optional*, defaults to "default"):
            Documentație pentru un alt argument unic.
        """
        super().__init__(config)
        self.custom_parameter = custom_parameter
        self.another_custom_arg = another_custom_arg
        # ... restul init-ului tău

    # ... alte metode
```

Pasează `custom_intro` și `custom_args` pentru mai mult control. Argumentele personalizate pot merge în `custom_args` sau în docstring-ul `__init__`. Folosește `custom_args` când aceleași argumente se repetă în mai multe metode.

```python
@auto_docstring(
    custom_intro="""Acest model efectuează operații sinergice specifice.
    Se construiește pe arhitectura standard Transformer cu modificări unice.""",
    custom_args="""
    custom_parameter (`type`, *optional*, defaults to `default_value`):
        O descriere concisă pentru custom_parameter dacă nu este definit sau suprascrie descrierea din `auto_docstring.py`.
    internal_helper_arg (`type`, *optional*, defaults to `default_value`):
        O descriere concisă pentru internal_helper_arg dacă nu este definit sau suprascrie descrierea din `auto_docstring.py`.
    """
)
class MySpecialModel(PreTrainedModel):
    def __init__(self, config: ConfigType, custom_parameter: "type" = "default_value", internal_helper_arg=None):
        # ...
```

Aplică `@auto_docstring` și la clasele care moștenesc din [`~utils.ModelOutput`].

```python
@auto_docstring(
    custom_intro="""
    Ieșiri personalizate ale modelului cu câmpuri suplimentare.
    """
)
@dataclass
class MyModelOutput(ImageClassifierOutput):
    r"""
    loss (`torch.FloatTensor`, *optional*):
        Loss-ul modelului.
    custom_field (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
        Un câmp de ieșire personalizat specific acestui model.
    """

    # Câmpurile standard (hidden_states, logits, attentions, etc.) sunt documentate automat când
    # descrierea corespunde textului standard. Loss-ul variază de obicei per model, deci documentează-l mai sus.
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    # Câmpurile personalizate trebuie documentate în docstring-ul de mai sus
    custom_field: Optional[torch.FloatTensor] = None
```

### Clase de config

Pune `@auto_docstring` direct deasupra unei subclase [`PreTrainedConfig`], alături de decoratorul `@strict`. `@strict` adaugă validarea de tip la runtime și transformă clasa într-un dataclass validat. Parametrii de config sunt *adnotări la nivel de clasă* (nu argumente `__init__`), iar `@auto_docstring` îi citește din corpul clasei ca să genereze docs.

[`ConfigArgs`] furnizează parametri standard precum `vocab_size`, `hidden_size` și `num_hidden_layers`, deci nu au nevoie de descriere dacă comportamentul nu diferă. Parametrii de bază [`PreTrainedConfig`] sunt excluși automat. Argumentul `checkpoint` generează exemplul de utilizare.

```python
from huggingface_hub.dataclasses import strict
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring

@strict
@auto_docstring(checkpoint="org/my-model-checkpoint")
class MyModelConfig(PreTrainedConfig):
    r"""
    custom_param (`int`, *optional*, defaults to 64):
        Descrierea unui parametru specific acestui model.
    another_param (`str`, *optional*, defaults to `"gelu"`):
        Descrierea unui alt parametru specific modelului.

    ```python
    >>> from transformers import MyModelConfig, MyModel

    >>> configuration = MyModelConfig()
    >>> model = MyModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "my_model"

    # Parametrii standard (vocab_size, hidden_size, etc.) sunt auto-documentați din ConfigArgs.
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    # Parametrii specifici modelului trebuie documentați în docstring-ul clasei de mai sus.
    custom_param: int = 64
    another_param: str = "gelu"
```

### Clase de procesator

Procesoarele multimodale (subclase [`ProcessorMixin`], `processing_*.py`) folosesc întotdeauna `@auto_docstring` simplu. Intro-ul clasei este auto-generat. Documentează doar parametrii `__init__` care nu sunt deja acoperiți de [`ProcessorArgs`] (`image_processor`, `tokenizer`, `chat_template` și alții).

Dacă fiecare parametru este standard, omite docstring-ul. Decorează și `__call__` cu `@auto_docstring`. Docstring-ul corpului său conține doar o secțiune `Returns:` plus orice argumente de apel suplimentare specifice modelului. `return_tensors` este adăugat automat.

```python
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack
from ...utils import auto_docstring

class MyModelProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}

@auto_docstring
class MyModelProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, custom_param: int = 4, **kwargs):
        r"""
        custom_param (`int`, *optional*, defaults to 4):
            Un parametru specific acestui procesor, neacoperit de ProcessorArgs standard.
        """
        super().__init__(image_processor, tokenizer)
        self.custom_param = custom_param

    @auto_docstring
    def __call__(self, images=None, text=None, **kwargs: Unpack[MyModelProcessorKwargs]):
        r"""
        Returns:
            [`BatchFeature`]: Un [`BatchFeature`] cu următoarele câmpuri:

            - **input_ids** -- Token ids de pasat modelului.
            - **pixel_values** -- Valori pixel de pasat modelului.
        """
        # ...
```

#### Procesatoare de imagini și video

Procesatoarele de imagini și video (subclase `BaseImageProcessor`, `image_processing_*.py`) urmează unul din două pattern-uri.

Dacă procesorul are parametri specifici modelului, definești un `TypedDict` `XxxImageProcessorKwargs(ImagesKwargs, total=False)` cu un docstring pentru acei parametri, setezi `valid_kwargs` pe clasă și folosești `@auto_docstring` simplu. `__init__` nu are docstring.

```python
class MyModelImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    custom_threshold (`float`, *optional*, defaults to `self.custom_threshold`):
        Un parametru specific acestui procesor de imagini.
    """
    custom_threshold: float | None

@auto_docstring
class MyModelImageProcessor(TorchvisionBackend):
    valid_kwargs = MyModelImageProcessorKwargs
    custom_threshold: float = 0.5

    def __init__(self, **kwargs: Unpack[MyModelImageProcessorKwargs]):
        super().__init__(**kwargs)
```

Dacă clasa setează doar atribute standard la nivel de clasă (`size`, `resample`, `image_mean` etc.) fără kwargs personalizate, folosește `@auto_docstring(custom_intro="Constructs a MyModel image processor.")`.

```python
@auto_docstring(custom_intro="Constructs a MyModel image processor.")
class MyModelImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
```

Când suprascrii `preprocess`, decorează-l cu `@auto_docstring` și documentează doar argumentele care nu sunt în [`ImageProcessorArgs`]. Argumentele standard și `return_tensors` sunt incluse automat.

### Funcții

Pune `@auto_docstring` direct deasupra definiției funcției. Decoratorul derivă descrierile parametrilor din semnătura funcției.

Decoratorul generează textul valorii de returnare din docstring-ul clasei [`ModelOutput`].

```python
class MyModel(PreTrainedModel):
    # ...
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        new_custom_argument: Optional[torch.Tensor] = None,
        # ... alte argumente
    ) -> Union[Tuple, ModelOutput]:
        r"""
        new_custom_argument (`torch.Tensor`, *optional*):
            Descrierea acestui argument personalizat nou și forma sau tipul așteptat.
        """
        # ...
```

Pasează `custom_intro` și `custom_args` pentru mai mult control. Folosește `custom_args` ca să definești documentația argumentelor partajate o singură dată când aceiași parametri apar în mai multe metode.

```python
MODEL_COMMON_CUSTOM_ARGS = r"""
    common_arg_1 (`torch.Tensor`, *optional*, defaults to `default_value`):
        Descrierea lui common_arg_1
    common_arg_2 (`torch.Tensor`, *optional*, defaults to `default_value`):
        Descrierea lui common_arg_2
"""

class MyModel(PreTrainedModel):
    # ...
    @auto_docstring(
        custom_intro="""Acesta este un intro personalizat pentru funcție.""",
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(self, input_ids=None, common_arg_1=None, common_arg_2=None) -> ModelOutput:
        r"""argumente specifice metodei merg aici"""
        # ...
```

Scrie secțiunile `Returns` și `Examples` manual în docstring ca să suprascrii versiunile auto-generate.

```python
    def forward(self, input_ids=None) -> torch.Tensor:
        r"""
        Returns:
            `torch.Tensor`: O secțiune Returns personalizată pentru tipuri de returnare non-ModelOutput.

        Example:

        ```python
        >>> model = MyModel.from_pretrained("org/my-model")
        >>> output = model(input_ids)
        ```
        """
        # ...
```

### Documentarea argumentelor

Urmează aceste reguli când documentezi tipuri diferite de argumente.

- `auto_docstring.py` definește argumentele standard (`input_ids`, `attention_mask`, `pixel_values` etc.) și le include automat. Nu le redefini local dacă argumentul se comportă la fel ca în modelul tău.

    Dacă un argument standard se comportă diferit în modelul tău, suprascrie-l local într-un bloc `r""" """`. Definiția locală are prioritate. Argumentul `labels`, de exemplu, este personalizat des per model și necesită să fie suprascris.

- Argumentele standard de config (`vocab_size`, `hidden_size`, `num_hidden_layers` etc.) urmează același principiu, dar vin din [`ConfigArgs`]. Argumentele standard de procesor (`image_processor`, `tokenizer`, `do_resize`, `return_tensors` etc.) vin din [`ProcessorArgs`] și [`ImageProcessorArgs`]. Documentează un parametru doar dacă este specific modelului sau se comportă diferit față de descrierea standard.

- Documentează argumentele noi sau personalizate într-un bloc `r""" """`. Pune-le după semnătură pentru funcții, în docstring-ul `__init__` pentru clase de model sau procesor, în docstring-ul corpului clasei pentru clase de config, sau în corpul `TypedDict`-ului `XxxImageProcessorKwargs` pentru procesoare de imagini.

    ```py
    argument_name (`type`, *optional*, defaults to `X`):
        Descrierea argumentului.
        Explică scopul, forma/tipul așteptat dacă e complex și comportamentul implicit.
        Poate fi pe mai multe linii.
    ```

  * Include `type` în backticks.
  * Adaugă *optional* dacă argumentul nu este obligatoriu sau are o valoare implicită.
  * Adaugă "defaults to X" dacă are o valoare implicită. Nu trebuie să adaugi "defaults to `None`" dacă valoarea implicită este `None`.
  * Pasează același bloc în `custom_args` când aceleași argumente se repetă în mai multe metode (vezi [exemplul cu Funcții de mai sus](#funcții)).

- Decoratorul extrage tipurile din semnăturile funcțiilor automat. Dacă un parametru are o adnotare de tip, nu trebuie să repeți tipul în format string în docstring. Când ambele sunt prezente, tipul semnăturii are prioritate. Tipul din docstring acționează ca fallback pentru parametrii fără adnotare.

## Verificarea docstring-urilor

Un script utilitar validează docstring-urile când deschizi un pull request. CI rulează scriptul și verifică următoarele.

> [!TIP]
> Dacă vezi `[ERROR]` în output, adaugă descrierea parametrului în docstring sau în clasa Args corespunzătoare din `auto_docstring.py`.

* Verifică că `@auto_docstring` este aplicat la clasele de model relevante și metodele publice.
* Validează completitudinea și consistența argumentelor: argumentele documentate trebuie să existe în semnătură, iar tipurile și valorile implicite trebuie să corespundă. Argumentele necunoscute fără o descriere locală sunt semnalate.
* Semnalează placeholder-urile incomplete precum `<fill_type>` și `<fill_docstring>`.
* Verifică că docstring-urile urmează stilul de formatare așteptat.

Rulează verificarea local înainte de commit.

```bash
make fix-repo
```

`make fix-repo` rulează și alte verificări. Ca să rulezi doar verificările de docstring și auto-docstring, folosește comanda de mai jos.

```bash
# verifică doar fișierele incluse în diff fără să le remedieze
python utils/check_docstrings.py
# remediază și suprascrie fișierele din diff
# python utils/check_docstrings.py --fix_and_overwrite
# remediază și suprascrie toate fișierele
# python utils/check_docstrings.py --fix_and_overwrite --check_all
```

## Checklist de referință rapidă

| Fă | Nu face |
|---|---|
| Aplică `@auto_docstring` la clasele de model, config și procesor și metodele lor principale (`forward`, `__call__`, `preprocess`). | Adaugă `@auto_docstring` la modelele moștenite în fișierele modulare pentru că se transferă automat. |
| Documentează doar argumentele noi sau specifice modelului. | Redefinește argumentele standard (`input_ids`, `attention_mask`, `vocab_size` etc.) care se comportă la fel cu descrierile lor implicite. |
| Pune parametrii de config în docstring-ul corpului clasei ca adnotări la nivel de clasă. | Pune parametrii de config în `__init__`. |
| Pune parametrii procesorului de imagini într-un `TypedDict` `XxxImageProcessorKwargs`. | Pune parametrii procesorului de imagini în `__init__`. |
| Rulează `python utils/check_docstrings.py --fix_and_overwrite` înainte de commit. | Ignoră ieșirile `[ERROR]` pentru că înseamnă că un parametru este nedocumentat. |

## Cum funcționează

Decoratorul `@auto_docstring` generează docstring-uri prin pașii următori.

1. Decoratorul inspectează semnătura ca să citească argumentele, tipurile și valorile implicite din `__init__`-ul clasei decorate sau din funcția decorată. Pentru clasele de config, parcurge adnotările la nivel de clasă în sus pe lanțul de moștenire și se oprește înainte de [`PreTrainedConfig`], excluzând câmpurile clasei de bază.

    Filtrează automat parametri precum `self`, `kwargs`, `args`, `deprecated_arguments` și nume cu prefix `_`. Câțiva parametri privați sunt redenumiți la echivalentele lor publice (`_out_features` → `out_features` pentru modele backbone).

2. Descrierile comune ale argumentelor vin din `auto_docstring.py`: [`ModelArgs`] (inputuri model), [`ModelOutputArgs`] (câmpuri de ieșire precum `hidden_states`), [`ImageProcessorArgs`] (preprocesare imagini), [`ProcessorArgs`] (componente procesor multimodal) și [`ConfigArgs`] (hyperparameteri de config).

3. Descrierea fiecărui parametru urmează acest lanț de prioritate:

    - Un docstring manual (bloc `r""" """` sau `custom_args`) are prioritate.
    - Dicționarul de surse predefinit ([`ModelArgs`], [`ConfigArgs`], [`ImageProcessorArgs`], [`ProcessorArgs`], [`ModelOutputArgs`]) este fallback-ul.
    - Dacă nicio sursă nu are o descriere, parametrul este semnalat cu `[ERROR]` în ieșirea de build.

4. Pentru clasele de model cu nume standard precum `ModelForCausalLM`, sau clasele care mapează la un pipeline, `@auto_docstring` generează intro-ul. Pentru procesoarele multimodale, intro-ul listează ce componente (tokenizer, procesor de imagini etc.) dau wrap clasei. Vezi [ClassDocstring](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/auto_docstring.py#L2437) pentru lista completă.

    Dacă numele clasei nu este în `ClassDocstring`, setează `custom_intro`.

5. Docstring-urile predefinite pot face referire la valori dinamice din [auto_modules](https://github.com/huggingface/transformers/tree/main/src/transformers/models/auto) Transformers, cum ar fi `{processor_class}`, `{image_processor_class}` și `{config_class}`. Placeholder-urile se rezolvă automat.

6. Decoratorul alege exemple de utilizare bazat pe task-ul sau compatibilitatea cu pipeline a modelului. Citește metadatele checkpoint-ului din clasa de configurare ca exemplele să folosească ID-uri reale de model. Argumentul `checkpoint` suprascrie checkpoint-ul dedus din docstring-ul clasei de config. Setează `checkpoint` pe clasele de config sau când deducerea checkpoint-ului eșuează. Dacă vezi o eroare precum `"Config not found for <model_name>"`, adaugă o intrare în `HARDCODED_CONFIG_FOR_MODELS` din `auto_docstring.py`.

7. Pentru metode precum `forward`, decoratorul scrie secțiunea `Returns` din tipul de returnare al metodei. Când tipul de returnare este o subclasă [`~transformers.utils.ModelOutput`], `@auto_docstring` trage descrierile câmpurilor din docstring-ul acelei clase. Un bloc `Returns` personalizat în docstring-ul funcției are prioritate.

8. Pentru metodele din `UNROLL_KWARGS_METHODS` și clasele din `UNROLL_KWARGS_CLASSES`, decoratorul expandează `**kwargs` tipizat cu `Unpack[KwargsTypedDict]`. Fiecare cheie din `TypedDict` devine un parametru documentat.

    Aceeași expandare se aplică metodelor `__call__` și `preprocess` pe subclasele [`BaseImageProcessor`] și [`ProcessorMixin`]. Tipurile de bază generice (`TextKwargs`, `ImagesKwargs`, `VideosKwargs`, `AudioKwargs`) sunt sărite. Doar subclasele specifice modelului sunt expandate.
