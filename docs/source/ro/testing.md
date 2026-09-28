<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Scrierea testelor pentru modele

Suita de teste Transformers folosește o arhitectură bazată pe mixin-uri ca să auto-genereze 100+ teste din cod minimal. Scrii o cantitate mică de cod specific modelului, iar mixin-urile se ocupă de save/load, generare, pipeline-uri, antrenare și tensor parallelism.

Rulează testele modelului tău cu comenzile de mai jos.

```bash
# rulează testele modelului tău
pytest tests/models/mymodel/test_modeling_mymodel.py -v

# rulează un test specific
pytest tests/models/mymodel/test_modeling_mymodel.py::MyModelTest::test_model

# rulează testele care se potrivesc cu un pattern de cuvinte cheie (util ca să rulezi toate testele de integrare)
pytest tests/models/mymodel/ -k integration -v

# include testele de integrare slow
RUN_SLOW=1 pytest tests/models/mymodel/ -v
```

CI-ul Hugging Face rulează testele de model fără `@slow` la fiecare pull request, iar testele slow rulează pe un program nightly (vezi [Verificările pentru pull request](./pr_checks) pentru ce validează CI-ul).

## Alege o clasă de test de bază

Trei clase de bază acoperă cele mai comune familii de modele. Alege-o pe cea care se potrivește cu modalitatea modelului tău.

| Clasă de bază | Folosit pentru | Mixin-uri |
|---|---|---|
| `CausalLMModelTest` | Modele cauzale de limbaj | `ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin`, `TrainingTesterMixin`, `TensorParallelTesterMixin` |
| `VLMModelTest` | Modele vizual-lingvistice | `ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin` |
| `ALMModelTest` | Modele audio-lingvistice | `ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin` |

`VLMModelTest` și `ALMModelTest` partajează un părinte comun `MultiModalModelTest` care îmbricuiește sub-config-uri într-un config compus de nivel superior și plasează token-uri placeholder de modalitate în `input_ids` alături de features-urile brute de modalitate (audio sau vizuale). `CausalLMModelTest` nu folosește părintele multimodal. Se bazează pe cele trei mixin-uri partajate și adaugă `TrainingTesterMixin` și `TensorParallelTesterMixin` pentru acoperire de antrenare și tensor-parallel.

Pentru arhitecturi care nu se potrivesc cu niciunul din cele trei (encoder-only, encoder-decoder etc.), construiește infrastructura de test direct din [pattern-ul cu două clase](#modeltester-și-modeltest) și [mixin-urile de test](#mixin-uri-de-test) descrise mai jos.

## CausalLMModelTest

`CausalLMModelTest` este clasa de bază recomandată pentru testarea modelelor cauzale de limbaj. Moștenește din cinci [mixin-uri de test](#mixin-uri-de-test) și auto-generează teste pentru save/load, generare, pipeline-uri, antrenare și tensor parallelism.

```py
import unittest

from transformers.testing_utils import require_torch
from transformers import is_torch_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester

if is_torch_available():
    from transformers import MyModel


class MyModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MyModel


@require_torch
class MyModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MyModelTester
```

Aceste două clase oferă acoperire completă de teste pentru `MyModel` și toate clasele sale head (`MyModelForCausalLM`, `MyModelForSequenceClassification` etc.). Vezi [tests/models/llama/test_modeling_llama.py](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py) pentru un exemplu real.

`CausalLMModelTester` necesită doar `base_model_class`. Tester-ul elimină sufixul `Model` ca să obțină un nume de bază (`LlamaModel` devine `Llama`), apoi adaugă sufixe precum `Config` sau `ForCausalLM` ca să descopere clasele conexe. Dacă o clasă nu există în modul, atributul rămâne `None` și tester-ul sare peste testele corespunzătoare.

### Suprascrierea valorilor implicite în CausalLMTester

Dacă modelul tău nu urmează denumirea standard, sau trebuie să personalizezi comportamentul, suprascrie atributele pe tester sau pe clasa de test.

```py
class MyModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MyModel
        # suprascrie dacă numele clasei nu urmează convenția
        causal_lm_class = MyCustomCausalLM


@require_torch
class MyModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MyModelTester
    # dezactivează testele de resize ale embedding-urilor
    test_resize_embeddings = False
```

Pentru modele care necesită parametri personalizați de constructor pe tester, suprascrie `__init__` și apelează `super().__init__(parent=parent)` înainte de a seta atribute suplimentare. Vezi [tests/models/youtu/test_modeling_youtu.py](https://github.com/huggingface/transformers/blob/main/tests/models/youtu/test_modeling_youtu.py) pentru un exemplu real.

```py
class YoutuModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = YoutuModel

    def __init__(self, parent, kv_lora_rank=16, q_lora_rank=32):
        super().__init__(parent=parent)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
```

## VLMModelTest

`VLMModelTest` este clasa de bază pentru modelele vizual-lingvistice. Moștenește din trei mixin-uri (`ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin`) și setează `_is_composite = True` ca să gestioneze mai multe sub-modele.

```py
import unittest

from transformers.testing_utils import require_torch
from transformers import is_torch_available

from ...vlm_tester import VLMModelTest, VLMModelTester

if is_torch_available():
    from transformers import (
        MyVLMConfig,
        MyVLMModel,
        MyVLMTextConfig,
        MyVLMVisionConfig,
        MyVLMForConditionalGeneration,
    )


class MyVLMTester(VLMModelTester):
    if is_torch_available():
        base_model_class = MyVLMModel
        config_class = MyVLMConfig
        text_config_class = MyVLMTextConfig
        vision_config_class = MyVLMVisionConfig
        conditional_generation_class = MyVLMForConditionalGeneration


@require_torch
class MyVLMTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MyVLMTester
```

### Suprascrierea valorilor implicite în VLMModelTester

Când VLM-ul necesită parametri de viziune personalizați sau valori de config non-implicite, suprascrie `__init__`. Setează valorile implicite cu `setdefault` înainte de a apela `super().__init__(parent, **kwargs)`. Exemplul de mai jos arată primele câteva valori implicite din [tests/models/qianfan_ocr/test_modeling_qianfan_ocr.py](https://github.com/huggingface/transformers/blob/main/tests/models/qianfan_ocr/test_modeling_qianfan_ocr.py).

```py
class QianfanOCRVisionText2TextModelTester(VLMModelTester):
    base_model_class = QianfanOCRModel
    config_class = QianfanOCRConfig
    text_config_class = Qwen3Config
    vision_config_class = QianfanOCRVisionConfig
    conditional_generation_class = QianfanOCRForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 1)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_channels", 3)
        # ... mai multe valori implicite
        super().__init__(parent, **kwargs)
```

Testele VLM diferă de `CausalLMModelTest` în câteva moduri.

- Trebuie să setezi `config_class`, `text_config_class`, `vision_config_class` și `conditional_generation_class` pe tester.
- `VLMModelTest` nu include `TrainingTesterMixin` sau `TensorParallelTesterMixin`.
- `__init__`-ul tester-ului acceptă parametri de viziune (`image_size`, `patch_size`, `num_channels`, `num_image_tokens`) din `**kwargs` și `setdefault()`.
- `ConfigTester` folosește `has_text_modality=False` pentru că config-ul de nivel superior este un config compus, nu un config de model text.

## ALMModelTest

`ALMModelTest` este clasa de bază pentru modelele audio-lingvistice (ALM) precum Qwen2Audio, AudioFlamingo3 și GraniteSpeech. Oglindește pattern-ul VLM cu același părinte `MultiModalModelTest` și auto-descoperirea claselor head. Mecanismul vizual este înlocuit cu features audio, un sub-config audio și o strategie de plasare a token-urilor audio.

```py
class MyALMTester(ALMModelTester):
    config_class = MyALMConfig
    text_config_class = MyALMTextConfig
    audio_config_class = MyALMAudioConfig
    conditional_generation_class = MyALMForConditionalGeneration
    audio_mask_key = "feature_attention_mask"


class MyALMTest(ALMModelTest, unittest.TestCase):
    model_tester_class = MyALMTester
```

### Suprascrierea valorilor implicite în ALMModelTester

`__init__`-ul tester-ului setează valori implicite specifice ALM (`feat_seq_length=128`, `num_mel_bins=80`, `audio_token_id=0`). Suprascrie-le cu `setdefault` înainte de a apela `super().__init__(parent, **kwargs)`.

Două atribute de clasă îi spun tester-ului cum sunt denumite lucrurile în modelul tău.

- `audio_mask_key`: numele kwarg-ului pe care modelul tău îl așteaptă pentru masca audio (`"feature_attention_mask"`, `"input_features_mask"` etc.). Lasă-l `None` dacă modelul tău nu consumă o mască audio separată.
- `audio_config_key`: numele atributului pe care config-ul tău de nivel superior îl folosește ca să îmbricuieze sub-config-ul audio. Implicit `"audio_config"`, dar modele precum GraniteSpeech folosesc `"encoder_config"`.

```py
class Qwen2AudioModelTester(ALMModelTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("feat_seq_length", 60)
        kwargs.setdefault("max_source_positions", kwargs["feat_seq_length"] // 2)
        super().__init__(parent, **kwargs)
```

`ALMModelTester` îți cere să suprascrii un hook, `get_audio_embeds_mask(audio_mask)`, și expune câteva opțiuni pentru personalizare.

- `get_audio_embeds_mask(audio_mask)`: returnează masca per-batch a pozițiilor de embedding audio după downsampling-ul encoder-ului. Tester-ul folosește suma sa row-wise ca să decidă câte placeholder-uri `audio_token_id` să insereze în `input_ids`, deci numărul trebuie să corespundă cu ce emite encoder-ul tău.
- `create_audio_features()`: returnează tensorul de features audio. Forma implicită este `[batch_size, num_mel_bins, feat_seq_length]`. Suprascrie când modelul tău, precum GraniteSpeech, așteaptă features time-first (`[batch_size, feat_seq_length, num_mel_bins]`).
- `create_audio_mask()`: returnează masca de attention la nivel audio. Implicit construiește o regiune validă contiguă aleatorie per rând în batch. Suprascrie cu o mască deterministă full-length dacă testele tale compară două invocații `prepare_config_and_inputs_for_common()` una față de alta, sau dacă encoder-ul tău audio se duce la un backend care respinge măști non-null.
- `place_audio_tokens(input_ids, config, num_audio_tokens)`: plasează token-urile placeholder audio contiguu după `BOS`. Suprascrie doar dacă modelul tău are nevoie de un layout diferit.
- `get_audio_feature_key()`: returnează cheia din dict-ul de inputuri pentru features audio (`"input_features"` implicit).

Pe lângă testele multimodale moștenite, `ALMModelTest` adaugă `test_mismatching_num_audio_tokens`. Testul afirmă că modelul ridică un `ValueError` clar când numărul de features audio nu corespunde cu numărul de token-uri placeholder audio din `input_ids` și verifică că un prompt cu mai multe segmente audio se face forward cu succes.

## Scrie teste pentru alte arhitecturi

Pentru arhitecturi encoder-only, encoder-decoder, audio sau alte arhitecturi non-standard, construiește infrastructura de test direct din pattern-ul cu două clase și mixin-urile de test descrise mai jos.

### ModelTester și ModelTest

Fiecare fișier de test pentru modele urmează aceeași structură.

1. `ModelTester` (clasă simplă) creează config-uri mici și inputuri dummy pentru testare și poate conține și teste mici de regresie specifice modelului.
2. `ModelTest` (`unittest.TestCase` + mixin-uri) moștenește testele auto-generate și le rulează pe fiecare variantă de model.

`ModelTest` apelează `prepare_config_and_inputs_for_common()` pe tester ca să obțină un tuple `(config, inputs_dict)`. Toate mixin-urile se bazează pe `prepare_config_and_inputs_for_common()` pentru datele de test.

### Mixin-uri de test

Alege mixin-urile de care are nevoie modelul tău.

| Mixin | Fișier sursă | Ce testează |
|---|---|---|
| `ModelTesterMixin` | `tests/test_modeling_common.py` | Save/load, gradient checkpointing, semnătura forward, atribute comune |
| `GenerationTesterMixin` | `tests/generation/test_utils.py` | Greedy, sampling, beam search, assisted decoding |
| `PipelineTesterMixin` | `tests/test_pipeline_mixin.py` | Un test per task de pipeline |
| `TrainingTesterMixin` | `tests/test_training_mixin.py` | Overfitting pe un batch mic |
| `TensorParallelTesterMixin` | `tests/test_tensor_parallel_mixin.py` | Tensor parallelism distribuit |

### Scrierea unui test de model

Vezi [tests/models/modernbert/test_modeling_modernbert.py](https://github.com/huggingface/transformers/blob/main/tests/models/modernbert/test_modeling_modernbert.py) pentru un exemplu complet funcțional. Pașii cheie sunt subliniați mai jos.

1. Clasa `ModelTester` construiește config-uri mici și inputuri dummy. Ține dimensiunile mici ca testele să se termine în secunde pe CPU. Folosește cei trei helpers de tensor de mai jos ca să construiești inputuri.

    - `ids_tensor(shape, vocab_size)`: tensor de întregi aleatori în `[0, vocab_size)`. Folosește pentru `input_ids`, `token_type_ids` și tensori de label.
    - `random_attention_mask(shape)`: tensor binar (0 și 1) unde primul token este întotdeauna 1. Folosește pentru `attention_mask`.
    - `floats_tensor(shape, scale=1.0)`: tensor de float-uri aleatorii. Folosește pentru inputuri continue precum `pixel_values` sau `inputs_embeds`.

    Tester-ul trebuie să implementeze `get_config()`, `prepare_config_and_inputs()` și `prepare_config_and_inputs_for_common()`. Adaugă metode `create_and_check_*` pentru fiecare head de task (model de bază, clasificare de secvențe, clasificare de token-uri etc.).

2. Moștenești din mixin-urile de care are nevoie modelul tău, setezi `all_model_classes` și `pipeline_model_mapping` și definești `setUp()`. Scrii metode `test_*` care deleghează la metodele `create_and_check_*` ale tester-ului.

3. Pentru fiecare head de task, adaugi o metodă `create_and_check_*` pe tester care instanțiază modelul, rulează un forward pass și afirmă formele de ieșire. Apoi adaugi o metodă `test_*` corespunzătoare pe clasa de test.

### Organizarea fișierelor

Fișierele de test trăiesc în `tests/models/mymodel/` urmând structura de mai jos.

```text
tests/models/mymodel/
├── __init__.py
├── test_modeling_mymodel.py            # teste de model (obligatoriu)
├── test_tokenization_mymodel.py        # teste de tokenizer (dacă ai tokenizer personalizat)
├── test_image_processing_mymodel.py    # teste pentru procesorul de imagini (dacă e model vizual)
├── test_feature_extraction_mymodel.py  # teste pentru feature extractor (dacă e model audio/speech)
└── test_processing_mymodel.py          # teste de procesor (dacă e multimodal)
```

Testele pentru tokenizer urmează același pattern. Moștenești `TokenizerTesterMixin` din `tests/test_tokenization_common.py`, setezi câteva atribute și primești teste auto-generate. Vezi [tests/models/llama/test_tokenization_llama.py](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_tokenization_llama.py) pentru un exemplu.

## Teste de config

`ConfigTester` verifică că o clasă de config gestionează serializarea, save/load și proprietățile standard corect. `CausalLMModelTest` și `VLMModelTest` includ testele de config automat. Pentru calea generală cu `ModelTester` și `ModelTest`, definești tester-ul de config manual în `setUp()`.

```py
from tests.test_configuration_common import ConfigTester

def setUp(self):
    self.config_tester = ConfigTester(self, config_class=MyModelConfig, hidden_size=32)

def test_config(self):
    self.config_tester.run_common_tests()
```

`run_common_tests()` rulează mai multe verificări.

- Verifică că proprietăți comune precum `hidden_size`, `num_attention_heads` și `num_hidden_layers` există (și `vocab_size` dacă `has_text_modality=True`).
- Testează serializarea JSON cu `to_json_string()` și `to_json_file()`.
- Face round-trip cu `save_pretrained()` și `from_pretrained()`.
- Confirmă consistența `id2label` și `label2id`.
- Creează un config fără argumente ca să valideze inițializarea cu valori implicite.
- Setează kwargs comune precum `output_hidden_states` și confirmă că sunt stocate corect.

Pasează `has_text_modality=False` pentru modelele exclusiv vizuale care nu au `vocab_size` și pasează `**kwargs` suplimentar ca să suprascrieți valorile implicite ale config-ului.

```py
self.config_tester = ConfigTester(
    self, config_class=MyVisionConfig, has_text_modality=False, hidden_size=64
)
```

## Teste de integrare și modele mici

Testele bazate pe mixin-uri folosesc config-uri mici cu weights aleatorii ca să verifice rapid comportamentul modelului. Testele de integrare rulează inferența cu weights preantrenate reale ca să valideze corectitudinea ieșirii. Modelele mici de pe Hub sunt suficient de mici pentru CI rapid, dar structurate ca checkpoint-uri reale.

### Scrierea testelor de integrare

Pune testele de integrare într-o clasă de test separată și marcheaz-le cu `@slow`. Fiecare test descarcă weights reale, rulează inferența și verifică ieșirile față de valorile așteptate. Apelează `cleanup(torch_device, gc_collect=False)` în `setUp` și `tearDown` ca să eviți memory leaks.

```py
import torch
from transformers import AutoTokenizer
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

class MyModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=False)

    def tearDown(self):
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_torch
    def test_inference(self):
        model = MyModelForCausalLM.from_pretrained("myorg/mymodel-base").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("myorg/mymodel-base")
        inputs = tokenizer("Hello, world", return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        # verifică față de valorile așteptate
        expected_slice = torch.tensor([[-0.1234, 0.5678, -0.9012]])
        torch.testing.assert_close(outputs.logits[0, :1, :3], expected_slice, atol=1e-4, rtol=1e-4)
```

Marchează orice test cu `@slow` dacă descarcă weights, încarcă un dataset mare sau durează mai mult de câteva secunde. [CI-ul pentru pull request](./pr_checks) sare peste testele slow, dar programul nightly le rulează.

#### Teste de integrare pentru generare

Folosește `do_sample=False` în testele de generare ca ieșirea să fie deterministă la rulări și hardware diferite. Pentru modelele Mixture-of-Experts, apelează și `model.set_experts_implementation("eager")` înainte de generare ca să forțezi o cale de dispatching stabilă a experților. Fără asta, diferențe numerice mici în router pot schimba ce expert gestionează un token și pot schimba ieșirea.

```py
@slow
@require_torch
def test_generate(self):
    model = MyModelForCausalLM.from_pretrained("myorg/mymodel-base").to(torch_device)
    tokenizer = AutoTokenizer.from_pretrained("myorg/mymodel-base")
    inputs = tokenizer("Hello, world", return_tensors="pt").to(torch_device)

    # model.set_experts_implementation("eager")  # decomentează pentru modelele MoE
    generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    self.assertEqual(output, ["Hello, world! This is the expected continuation..."])
```

#### Așteptări specifice hardware-ului

CI-ul Transformers rulează testele slow pe un NVIDIA A10. Rezultatele numerice pot varia ușor între generații de GPU, deci testele de integrare folosesc clasa [Expectations](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py#L3247) ca să înregistreze valori așteptate per dispozitiv. `Expectations` alege cea mai bună potrivire pentru hardware-ul curent pe baza cheilor SM `(device_type, (major, minor))` și revine la un default când nu se potrivește nimic.

Rulează `torch.cuda.get_device_capability()` ca să printezi versiunea SM locală (e.g. `(8, 6)` pentru A10, `(9, 0)` pentru H100).

```py
from transformers.testing_utils import Expectations

expected_texts = Expectations(
    {
        ("cuda", (8, 6)): ["Hello, world! This is the A10 continuation..."],
        ("cuda", (9, 0)): ["Hello, world! This is the H100 continuation..."],
    }
).get_expectation()

self.assertEqual(output, expected_texts)
```

### Crearea modelelor mici

Modelele mici cu weights aleatorii trăiesc pe Hub sub organizația [hf-internal-testing](https://huggingface.co/hf-internal-testing). Testele de pipeline se bazează pe modele mici când au nevoie de un checkpoint găzduit pe Hub, dar nu le pasă de calitatea ieșirii. Testele fast smoke încarcă și ele modele mici ca să verifice formele forward pass-ului fără să descarce checkpoint-uri mari.

Modelele mici sunt ultimul resort pentru testele de integrare. Folosește-le doar când cel mai mic checkpoint disponibil depășește ~24 GB de VRAM. Folosește weights preantrenate originale când e posibil, ca să prinzi regresii numerice reale.

Scriptul `utils/create_dummy_models.py` generează modele mici din `ModelTester.get_config()`. Scriptul extrage hyperparametri mici din tester-ul tău, construiește un model cu weights aleatorii și încarcă rezultatul pe Hub.

Generează modele mici local.

```bash
python utils/create_dummy_models.py output_dir -m your_model_type
```

Încarcă-le pe Hub.

```bash
python utils/create_dummy_models.py output_dir -m your_model_type --upload --organization hf-internal-testing
```

Fiecare model folosește numele `hf-internal-testing/tiny-random-{ModelClassName}` și este înregistrat în `tests/utils/tiny_model_summary.json`. Un workflow CI (`.github/workflows/check_tiny_models.yml`) regenerează modelele mici zilnic.

## Controlează ce se testează

Flag-urile booleene pe `ModelTesterMixin` comută testele auto-generate. Suprascrie orice flag pe clasa ta de test ca să activezi sau dezactivezi verificări specifice.

```py
class MyModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MyModelTester
    test_resize_embeddings = False
    test_all_params_have_gradient = False  # când nu toți parametrii sunt activați în fiecare forward pass
```

| Flag | Implicit | Ce controlează |
|---|---|---|
| `test_resize_embeddings` | `True` | Redimensionarea layer-ului de embedding |
| `test_resize_position_embeddings` | `False` | Redimensionarea embedding-urilor de poziție |
| `test_mismatched_shapes` | `True` | Gestionarea formelor nepotrivite input/output |
| `test_missing_keys` | `True` | Avertismente pentru chei lipsă la încărcare |
| `test_torch_exportable` | `True` | Compatibilitate `torch.export` |
| `test_all_params_have_gradient` | `True` | Toți parametrii primesc gradienți (setează `False` când nu toți parametrii sunt activați în fiecare forward pass, cum ar fi experții MoE) |
| `is_encoder_decoder` | `False` | Teste specifice encoder-decoder |
| `has_attentions` | `True` | Teste de ieșire attention |
| `_is_composite` | `False` | Gestionarea modelelor compuse/multimodale |
| `model_split_percents` | `[0.5, 0.7, 0.9]` | Procentaje de split pentru testele de model parallelism |

## Pașii următori

- Răsfoiește documentația [pytest](https://docs.pytest.org/en/latest/getting-started.html) pentru mai mult despre selecția testelor, fixtures, logging și altele.
