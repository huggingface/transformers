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

# Writing model tests

The Transformers test suite uses a mixin-based architecture to auto-generate 100+ tests from minimal code. You write a small amount of model-specific code, and the mixins handle save/load, generation, pipelines, training, and tensor parallelism.

Run your model's tests with the following commands.

```bash
# run your model's tests
pytest tests/models/mymodel/test_modeling_mymodel.py -v

# run a specific test
pytest tests/models/mymodel/test_modeling_mymodel.py::MyModelTest::test_model

# run tests matching a keyword pattern (useful to run all integration tests)
pytest tests/models/mymodel/ -k integration -v

# include slow integration tests
RUN_SLOW=1 pytest tests/models/mymodel/ -v
```

The Hugging Face CI runs model tests without `@slow` on every pull request, and slow tests run on a nightly schedule (see [Pull request checks](./pr_checks) for what the CI validates).

## Pick a base test class

Three base classes cover the most common model families. Pick the one that matches your model's modality.

| Base class | Use for | Mixins |
|---|---|---|
| `CausalLMModelTest` | Causal language models | `ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin`, `TrainingTesterMixin`, `TensorParallelTesterMixin` |
| `VLMModelTest` | Vision-language models | `ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin` |
| `ALMModelTest` | Audio-language models | `ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin` |

`VLMModelTest` and `ALMModelTest` share a common `MultiModalModelTest` parent that nests sub-configs into a composite top-level config and places modality placeholder tokens in `input_ids` alongside the raw modality features (audio or vision). `CausalLMModelTest` doesn't use the multimodal parent. It builds on the three shared mixins and adds `TrainingTesterMixin` and `TensorParallelTesterMixin` for training and tensor-parallel coverage.

For architectures that don't fit any of the three (encoder-only, encoder-decoder, etc.), build the test infrastructure directly from the [two-class pattern](#modeltester-and-modeltest) and [test mixins](#test-mixins) described below.

## CausalLMModelTest

`CausalLMModelTest` is the recommended base class for testing causal language models. It inherits from five [test mixins](#test-mixins) and auto-generates tests for save/load, generation, pipelines, training, and tensor parallelism.

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

These two classes give full test coverage for `MyModel` and all its head classes (`MyModelForCausalLM`, `MyModelForSequenceClassification`, etc.). See [tests/models/llama/test_modeling_llama.py](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py) for a real example.

`CausalLMModelTester` only requires `base_model_class`. The tester strips the `Model` suffix to get a base name (`LlamaModel` becomes `Llama`), then appends suffixes like `Config` or `ForCausalLM` to discover related classes. If a class doesn't exist in the module, the attribute stays `None` and the tester skips the corresponding tests.

### Overriding defaults in the CausalLMTester

If your model doesn't follow standard naming, or you need to customize behavior, override attributes on the tester or test class.

```py
class MyModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MyModel
        # override if the class name doesn't follow the convention
        causal_lm_class = MyCustomCausalLM


@require_torch
class MyModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MyModelTester
    # disable embedding resize tests
    test_resize_embeddings = False
```

For models that need custom constructor parameters on the tester, override `__init__` and call `super().__init__(parent=parent)` before setting extra attributes. See [tests/models/youtu/test_modeling_youtu.py](https://github.com/huggingface/transformers/blob/main/tests/models/youtu/test_modeling_youtu.py) for a real example.

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

`VLMModelTest` is the base class for vision-language models. It inherits from three mixins (`ModelTesterMixin`, `GenerationTesterMixin`, `PipelineTesterMixin`) and sets `_is_composite = True` to handle multiple sub-models.

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

### Overriding defaults in the VLMModelTester

When the VLM needs custom vision parameters or non-default config values, override `__init__`. Set defaults with `setdefault` before calling `super().__init__(parent, **kwargs)`. The example below shows the first few defaults from [tests/models/qianfan_ocr/test_modeling_qianfan_ocr.py](https://github.com/huggingface/transformers/blob/main/tests/models/qianfan_ocr/test_modeling_qianfan_ocr.py).

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
        # ... more defaults
        super().__init__(parent, **kwargs)
```

VLM tests differ from `CausalLMModelTest` in a few ways.

- You must set `config_class`, `text_config_class`, `vision_config_class`, and `conditional_generation_class` on the tester.
- `VLMModelTest` doesn't include `TrainingTesterMixin` or `TensorParallelTesterMixin`.
- The tester's `__init__` accepts vision parameters (`image_size`, `patch_size`, `num_channels`, `num_image_tokens`) from `**kwargs` and `setdefault()`.
- `ConfigTester` uses `has_text_modality=False` because the top-level config is a composite config rather than a text model config.

## ALMModelTest

`ALMModelTest` is the base class for audio-language models (ALMs) like Qwen2Audio, AudioFlamingo3, and GraniteSpeech. It mirrors the VLM pattern with the same `MultiModalModelTest` parent and auto-discovery of head classes. The vision-side machinery is swapped for audio features, an audio sub-config, and an audio-token placement strategy.

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

### Overriding defaults in the ALMModelTester

The tester's `__init__` sets ALM-specific defaults (`feat_seq_length=128`, `num_mel_bins=80`, `audio_token_id=0`). Override them with `setdefault` before calling `super().__init__(parent, **kwargs)`.

Two class attributes tell the tester how your model names things.

- `audio_mask_key`: the kwarg name your model expects for the audio mask (`"feature_attention_mask"`, `"input_features_mask"`, etc.). Leave it `None` if your model doesn't consume a separate audio mask.
- `audio_config_key`: the attribute name your top-level config uses to nest the audio sub-config. Defaults to `"audio_config"` but models like GraniteSpeech use `"encoder_config"`.

```py
class Qwen2AudioModelTester(ALMModelTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("feat_seq_length", 60)
        kwargs.setdefault("max_source_positions", kwargs["feat_seq_length"] // 2)
        super().__init__(parent, **kwargs)
```

`ALMModelTester` requires you to override one hook, `get_audio_embeds_mask(audio_mask)`, and exposes a few more optional ones for customization.

- `get_audio_embeds_mask(audio_mask)`: returns the per-batch mask of audio embedding positions after the encoder's downsampling. The tester uses its row-wise sum to decide how many `audio_token_id` placeholders to insert into `input_ids`, so the count must match what your encoder emits.
- `create_audio_features()`: returns the audio feature tensor. Default shape is `[batch_size, num_mel_bins, feat_seq_length]`. Override when your model, like GraniteSpeech, expects time-first features (`[batch_size, feat_seq_length, num_mel_bins]`).
- `create_audio_mask()`: returns the audio-level attention mask. The default builds a random contiguous valid region per row in the batch. Override with a deterministic full-length mask if your tests compare two `prepare_config_and_inputs_for_common()` invocations against each other, or if your audio encoder dispatches to a backend that rejects non-null masks.
- `place_audio_tokens(input_ids, config, num_audio_tokens)`: places audio placeholder tokens contiguously after `BOS`. Override only if your model needs a different layout.
- `get_audio_feature_key()`: returns the inputs-dict key for audio features (`"input_features"` by default).

In addition to the inherited multimodal tests, `ALMModelTest` adds `test_mismatching_num_audio_tokens`. The test asserts the model raises a clear `ValueError` when the number of audio features doesn't match the number of audio placeholder tokens in `input_ids`, and verifies that a prompt with multiple audio segments still forwards successfully.

## Write tests for other architectures

For encoder-only, encoder-decoder, audio, or other non-standard architectures, build the test infrastructure directly from the two-class pattern and test mixins described below.

### ModelTester and ModelTest

Every model test file follows the same structure.

1. `ModelTester` (plain class) creates tiny configs and dummy inputs for testing, and can also hold small regression tests specific to the model.
2. `ModelTest` (`unittest.TestCase` + mixins) inherits auto-generated tests and runs them against every model variant.

`ModelTest` calls `prepare_config_and_inputs_for_common()` on the tester to get a `(config, inputs_dict)` tuple. All mixins rely on `prepare_config_and_inputs_for_common()` for test data.

### Test mixins

Pick the mixins your model needs.

| Mixin | Source file | What it tests |
|---|---|---|
| `ModelTesterMixin` | `tests/test_modeling_common.py` | Save/load, gradient checkpointing, forward signature, common attributes |
| `GenerationTesterMixin` | `tests/generation/test_utils.py` | Greedy, sampling, beam search, assisted decoding |
| `PipelineTesterMixin` | `tests/test_pipeline_mixin.py` | One test per pipeline task |
| `TrainingTesterMixin` | `tests/test_training_mixin.py` | Overfitting on a small batch |
| `TensorParallelTesterMixin` | `tests/test_tensor_parallel_mixin.py` | Distributed tensor parallelism |

### Writing a model test

See [tests/models/modernbert/test_modeling_modernbert.py](https://github.com/huggingface/transformers/blob/main/tests/models/modernbert/test_modeling_modernbert.py) for a complete working example. The key steps are outlined below.

1. The `ModelTester` class builds tiny configs and dummy inputs. Keep dimensions small so tests finish in seconds on CPU. Use the three tensor helpers below to build inputs.

    - `ids_tensor(shape, vocab_size)`: Random integer tensor in `[0, vocab_size)`. Use for `input_ids`, `token_type_ids`, and label tensors.
    - `random_attention_mask(shape)`: Binary tensor (0s and 1s) where the first token is always 1. Use for `attention_mask`.
    - `floats_tensor(shape, scale=1.0)`: Random float tensor. Use for continuous inputs like `pixel_values` or `inputs_embeds`.

    The tester must implement `get_config()`, `prepare_config_and_inputs()`, and `prepare_config_and_inputs_for_common()`. Add `create_and_check_*` methods for each task head (base model, sequence classification, token classification, etc.).

2. Inherit from the mixins your model needs, set `all_model_classes` and `pipeline_model_mapping`, and define `setUp()`. Write `test_*` methods that delegate to the tester's `create_and_check_*` methods.

3. For each task head, add a `create_and_check_*` method on the tester that instantiates the model, runs a forward pass, and asserts output shapes. Then add a corresponding `test_*` method on the test class.

### File organization

Test files live in `tests/models/mymodel/` following the structure shown below.

```text
tests/models/mymodel/
├── __init__.py
├── test_modeling_mymodel.py            # model tests (required)
├── test_tokenization_mymodel.py        # tokenizer tests (if custom tokenizer)
├── test_image_processing_mymodel.py    # image processor tests (if vision model)
├── test_feature_extraction_mymodel.py  # feature extractor tests (if audio/speech model)
└── test_processing_mymodel.py          # processor tests (if multimodal)
```

Tokenizer tests follow the same pattern. Inherit `TokenizerTesterMixin` from `tests/test_tokenization_common.py`, set a few attributes, and get auto-generated tests. See [tests/models/llama/test_tokenization_llama.py](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_tokenization_llama.py) for an example.

## Config tests

`ConfigTester` verifies that a config class handles serialization, save/load, and standard properties correctly. `CausalLMModelTest` and `VLMModelTest` include config tests automatically. For the general path with `ModelTester` and `ModelTest`, define the config tester manually in `setUp()`.

```py
from tests.test_configuration_common import ConfigTester

def setUp(self):
    self.config_tester = ConfigTester(self, config_class=MyModelConfig, hidden_size=32)

def test_config(self):
    self.config_tester.run_common_tests()
```

`run_common_tests()` runs several checks.

- Checks that common properties like `hidden_size`, `num_attention_heads`, and `num_hidden_layers` exist (and `vocab_size` if `has_text_modality=True`).
- Tests JSON serialization with `to_json_string()` and `to_json_file()`.
- Round-trips `save_pretrained()` and `from_pretrained()`.
- Confirms `id2label` and `label2id` consistency.
- Creates a config with no arguments to validate default initialization.
- Sets common kwargs like `output_hidden_states` and confirms they're stored correctly.

Pass `has_text_modality=False` for vision-only models that lack `vocab_size`, and pass extra `**kwargs` to override config defaults.

```py
self.config_tester = ConfigTester(
    self, config_class=MyVisionConfig, has_text_modality=False, hidden_size=64
)
```

## Integration tests and tiny models

Mixin tests use tiny configs with random weights to verify model behavior quickly. Integration tests run inference with real pretrained weights to validate output correctness. Tiny models on the Hub are small enough for fast CI, but structured like real checkpoints.

### Writing integration tests

Place integration tests in a separate test class and mark them with `@slow`. Each test downloads real weights, runs inference, and checks outputs against expected values. Call `cleanup(torch_device, gc_collect=False)` in `setUp` and `tearDown` to avoid memory residuals.

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

        # check against expected values
        expected_slice = torch.tensor([[-0.1234, 0.5678, -0.9012]])
        torch.testing.assert_close(outputs.logits[0, :1, :3], expected_slice, atol=1e-4, rtol=1e-4)
```

Mark any test with `@slow` if it downloads weights, loads a large dataset, or takes more than a few seconds. The [pull request CI](./pr_checks) skips slow tests, but the nightly schedule runs them.

#### Generation integration tests

Use `do_sample=False` in generation tests so the output is deterministic across runs and hardware. For Mixture-of-Experts models, also call `model.set_experts_implementation("eager")` before generating to force a stable expert dispatch path. Without it, small numerical differences in the router can flip which expert handles a token and change the output.

```py
@slow
@require_torch
def test_generate(self):
    model = MyModelForCausalLM.from_pretrained("myorg/mymodel-base").to(torch_device)
    tokenizer = AutoTokenizer.from_pretrained("myorg/mymodel-base")
    inputs = tokenizer("Hello, world", return_tensors="pt").to(torch_device)

    # model.set_experts_implementation("eager")  # uncomment for MoE models
    generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    self.assertEqual(output, ["Hello, world! This is the expected continuation..."])
```

#### Hardware-specific expectations

Transformers CI runs slow tests on an NVIDIA A10. Numerical results can vary slightly across GPU generations, so integration tests use the [Expectations](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py#L3247) class to register per-device expected values. `Expectations` picks the best match for the current hardware based on `(device_type, (major, minor))` SM keys, and falls back to a default when nothing matches.

Run `torch.cuda.get_device_capability()` to print your local SM version (e.g. `(8, 6)` for an A10, `(9, 0)` for H100).

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

### Creating tiny models

Tiny models with random weights live on the Hub under the [hf-internal-testing](https://huggingface.co/hf-internal-testing) organization. Pipeline tests rely on tiny models when they need a Hub-hosted checkpoint but don't care about output quality. Fast smoke tests also load tiny models to verify forward pass shapes without downloading large checkpoints.

Tiny models are a last resort for integration tests. Only use them when the smallest available checkpoint exceeds ~24 GB of VRAM. Use original pretrained weights, when possible, to catch real numerical regressions.

The `utils/create_dummy_models.py` script generates tiny models from `ModelTester.get_config()`. The script extracts tiny hyperparameters from your tester, builds a model with random weights, and uploads the result to the Hub.

Generate tiny models locally.

```bash
python utils/create_dummy_models.py output_dir -m your_model_type
```

Upload them to the Hub.

```bash
python utils/create_dummy_models.py output_dir -m your_model_type --upload --organization hf-internal-testing
```

Each model uses the name `hf-internal-testing/tiny-random-{ModelClassName}` and gets recorded in `tests/utils/tiny_model_summary.json`. A CI workflow (`.github/workflows/check_tiny_models.yml`) regenerates tiny models daily.

## Control what gets tested

Boolean flags on `ModelTesterMixin` toggle auto-generated tests. Override any flag on your test class to enable or disable specific checks.

```py
class MyModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MyModelTester
    test_resize_embeddings = False
    test_all_params_have_gradient = False  # when not all parameters are activated in every forward pass
```

| Flag | Default | What it controls |
|---|---|---|
| `test_resize_embeddings` | `True` | Embedding layer resizing |
| `test_resize_position_embeddings` | `False` | Position embedding resizing |
| `test_mismatched_shapes` | `True` | Mismatched input/output shape handling |
| `test_missing_keys` | `True` | Missing key warnings on load |
| `test_torch_exportable` | `True` | `torch.export` compatibility |
| `test_all_params_have_gradient` | `True` | All parameters receive gradients (set `False` when not all parameters are activated in every forward pass, such as MoE experts) |
| `is_encoder_decoder` | `False` | Encoder-decoder specific tests |
| `has_attentions` | `True` | Attention output tests |
| `_is_composite` | `False` | Composite/multimodal model handling |
| `model_split_percents` | `[0.5, 0.7, 0.9]` | Split percentages for model parallelism tests |

## Next steps

- Browse the [pytest](https://docs.pytest.org/en/latest/getting-started.html) docs for more about test selection, fixtures, logging, and more.
