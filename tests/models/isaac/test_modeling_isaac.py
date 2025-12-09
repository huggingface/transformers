import base64
import hashlib
import io
import json
import os
import unittest
from functools import lru_cache
from pathlib import Path

import pytest
from huggingface_hub import is_offline_mode

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    IsaacConfig,
    IsaacForConditionalGeneration,
    IsaacModel,
    PythonBackend,
    is_torch_available,
)
from transformers.models.isaac.configuration_isaac import IsaacVisionConfig
from transformers.models.isaac.image_processing_isaac_fast import IsaacImageProcessorFast
from transformers.models.isaac.modeling_isaac import IsaacVisionAttention
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_vision_available
from transformers.utils.import_utils import is_perceptron_available


if is_vision_available():
    from PIL import Image
else:
    Image = None

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

if is_perceptron_available():
    from perceptron.tensorstream.ops import modality_mask, role_mask, tensor_stream_token_view
    from perceptron.tensorstream.tensorstream import TensorStream
else:
    TensorStream = None


require_tensorstream = pytest.mark.skipif(TensorStream is None, reason="TensorStream backend is not available")

MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")
HASH_FILE = Path(__file__).with_name("isaac_checkpoint_hashes.json")
GENERATION_GOLDEN_FILE = Path(__file__).with_name("isaac_generation_golden.json")
HASH_FILTERS = {
    "full_model": {"include": None, "exclude": None},
    "core_model": {"include": None, "exclude": {"vision_embedding", "audio_embedding", "inv_freq"}},
    "vision_modules": {"include": {"vision_embedding"}, "exclude": None},
}
RED_DOT_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="


def tensor_stream_snapshot(ts: TensorStream) -> dict[str, object]:
    """Summarize TensorStream tokens/modalities using public utilities."""

    token_view = tensor_stream_token_view(ts).cpu().tolist()
    modality = modality_mask(ts).cpu().tolist()
    roles = role_mask(ts).cpu().tolist()

    return {
        "shape": list(ts.shape),
        "token_view": token_view,
        "modality_mask": modality,
        "role_mask": roles,
    }


def _assert_tensor_stream_snapshot_equal(actual: dict[str, object], expected: dict[str, object]) -> None:
    assert actual["shape"] == expected["shape"], "TensorStream shape changed"
    assert actual["token_view"] == expected["token_view"], "TensorStream token view changed"
    assert actual["modality_mask"] == expected["modality_mask"], "TensorStream modality mask changed"
    assert actual["role_mask"] == expected["role_mask"], "TensorStream role mask changed"


def _tensor_to_bytes(tensor):
    cpu_tensor = tensor.detach().cpu().contiguous()
    if cpu_tensor.is_floating_point():
        cpu_tensor = cpu_tensor.to(dtype=torch.float32)
    return cpu_tensor.numpy().tobytes()


def _iter_filtered_items(state_dict, include=None, exclude=None):
    for name, tensor in state_dict.items():
        if include and not any(token in name for token in include):
            continue
        if exclude and any(token in name for token in exclude):
            continue
        yield name, tensor


def _hash_state_dict(state_dict, *, include=None, exclude=None):
    hasher = hashlib.sha256()
    items = sorted(_iter_filtered_items(state_dict, include=include, exclude=exclude), key=lambda kv: kv[0])
    for name, tensor in items:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_to_bytes(tensor))
    return hasher.hexdigest()


def compute_logits_statistics(tensor: torch.Tensor) -> dict[str, object]:
    """
    Summarize logits with simple statistics that are stable across minor
    implementation changes yet still sensitive to behavioral regressions.
    """

    float_tensor = tensor.detach().to(torch.float32).cpu()
    flat = float_tensor.reshape(-1).to(torch.float64)

    def _rounded(value: torch.Tensor | float) -> float:
        return round(float(value), 10)

    return {
        "shape": list(float_tensor.shape),
        "numel": flat.numel(),
        "mean": _rounded(flat.mean()),
        "std": _rounded(flat.std(unbiased=False)),
        "min": _rounded(flat.min()),
        "max": _rounded(flat.max()),
        "sum": _rounded(flat.sum()),
        "l2_norm": _rounded(torch.linalg.vector_norm(flat, ord=2)),
    }


def _assert_logits_statistics_close(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    rel: float = 1e-5,
    abs_tol: float = 1e-6,
) -> None:
    assert actual["shape"] == expected["shape"], "Logits shape changed"
    assert actual["numel"] == expected["numel"], "Logits numel changed"
    for key in ("mean", "std", "min", "max", "sum", "l2_norm"):
        assert actual[key] == pytest.approx(
            expected[key],
            rel=rel,
            abs=abs_tol,
        ), f"Logits statistic '{key}' drifted"


def _hf_from_pretrained(cls, pretrained_id, **kwargs):
    """
    Wrapper around `cls.from_pretrained` that automatically injects
    the test revision (if any) from MODEL_REVISION.
    """
    if MODEL_REVISION is not None:
        kwargs.setdefault("revision", MODEL_REVISION)
    return cls.from_pretrained(pretrained_id, **kwargs)


@pytest.fixture(scope="session")
def tokenizer(isaac_reference_checkpoint):
    """Load the tokenizer from the converted Perceptron HF checkpoint."""
    return _hf_from_pretrained(
        AutoTokenizer,
        isaac_reference_checkpoint,
        trust_remote_code=True,
    )


@require_torch
def test_isaac_sdpa_attention_backend():
    config = IsaacVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_channels=3,
        num_patches=16,
        patch_size=4,
    )
    config._attn_implementation = "sdpa"

    attn_module = IsaacVisionAttention(config).eval()
    seq_len = 8
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

    with torch.no_grad():
        outputs, attn_weights = attn_module(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
        )

    assert outputs.shape == hidden_states.shape
    assert attn_weights is None


def _hash_tensor(tensor):
    hasher = hashlib.sha256()
    hasher.update(_tensor_to_bytes(tensor))
    return hasher.hexdigest()


@lru_cache(maxsize=1)
def _load_expected_hashes():
    if not HASH_FILE.exists():
        return None
    with HASH_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_generation_golden():
    if not GENERATION_GOLDEN_FILE.exists():
        return None
    with GENERATION_GOLDEN_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_decode(tokenizer, token_ids):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        tokens = [tok for tok in tokens if tok is not None]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text.strip() if isinstance(text, str) else text


@lru_cache(maxsize=1)
def _load_red_dot_image():
    if Image is None:
        return None
    data = base64.b64decode(RED_DOT_B64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _reference_checkpoint_or_skip():
    if TensorStream is None:
        pytest.skip("TensorStream dependency is required for Isaac integration tests.")
    if LOCAL_CHECKPOINT:
        resolved = Path(LOCAL_CHECKPOINT).expanduser()
        if not resolved.exists():
            pytest.skip(f"Local checkpoint path {resolved} does not exist.")
        return str(resolved)
    if is_offline_mode():
        pytest.skip("Offline mode: set ISAAC_TEST_MODEL_PATH to a local checkpoint to run these tests.")
    return MODEL_ID


class SimpleIsaacTokenizer(PythonBackend):
    vocab_files_names = {}
    model_input_names = ["input_ids"]

    def __init__(self):
        self._vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<image>": 4,
        }
        self._ids_to_tokens = {idx: tok for tok, idx in self._vocab.items()}
        super().__init__(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            extra_special_tokens=["<image>"],
            model_max_length=512,
        )
        self.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] | trim }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        clean = text.replace("\n", " ").strip()
        if not clean:
            return []
        return [token for token in clean.split(" ") if token]

    def _convert_token_to_id(self, token):
        if token not in self._vocab:
            next_id = len(self._vocab)
            self._vocab[token] = next_id
            self._ids_to_tokens[next_id] = token
        return self._vocab[token]

    def _convert_id_to_token(self, index):
        return self._ids_to_tokens.get(index, self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1
        return [self.bos_token_id] + list(token_ids_0) + [self.eos_token_id]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


def _make_dummy_image(size=(32, 32), color=(255, 0, 0)):
    if Image is None:
        raise RuntimeError("PIL.Image is not available in this environment.")
    return Image.new("RGB", size, color=color)


@pytest.fixture
def isaac_tiny_config():
    tester = IsaacModelTester(parent=None)
    return tester.get_config()


@pytest.fixture
def isaac_tokenizer():
    return SimpleIsaacTokenizer()


@pytest.fixture
def isaac_processor(isaac_tokenizer, isaac_tiny_config):
    vision_config = isaac_tiny_config.vision_config
    image_processor = IsaacImageProcessorFast(
        patch_size=vision_config.patch_size,
        max_num_patches=vision_config.num_patches,
        pixel_shuffle_scale=vision_config.pixel_shuffle_scale_factor,
        rescale_factor=isaac_tiny_config.vision_rescale_factor,
    )
    return IsaacProcessor(
        image_processor=image_processor,
        tokenizer=isaac_tokenizer,
        config=isaac_tiny_config,
    )


@pytest.fixture(scope="session")
def isaac_reference_checkpoint():
    return _reference_checkpoint_or_skip()


@pytest.fixture(scope="session")
def isaac_config(isaac_reference_checkpoint):
    """Load IsaacConfig from the converted checkpoint."""
    # Load the config directly from the converted checkpoint
    config = _hf_from_pretrained(IsaacConfig, isaac_reference_checkpoint)
    # Most tests assume flash attention in vision unless they explicitly override it.
    config.vision_attn_implementation = "flash_attention_2"
    return config


@pytest.fixture(scope="session")
def isaac_reference_model(isaac_reference_checkpoint, isaac_config):
    model_config = IsaacConfig.from_dict(isaac_config.to_dict())
    model_config.vision_attn_implementation = isaac_config.vision_attn_implementation
    model = _hf_from_pretrained(
        IsaacForConditionalGeneration,
        isaac_reference_checkpoint,
        config=model_config,
        attn_implementation="sdpa",
    )
    return model


@pytest.fixture(scope="session")
def isaac_reference_processor(isaac_reference_checkpoint):
    try:
        processor = _hf_from_pretrained(AutoProcessor, isaac_reference_checkpoint)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unable to load reference Isaac processor from {isaac_reference_checkpoint}") from error
    print(f"[Isaac tests] Loaded processor type: {type(processor)} from {isaac_reference_checkpoint}")
    if not isinstance(processor, IsaacProcessor):
        pytest.skip("Loaded processor is not an IsaacProcessor instance.")
    return processor


class IsaacModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.text_config = {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": hidden_size // num_attention_heads,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "intermediate_size": hidden_size * 3,
            "max_position_embeddings": 128,
            "model_type": "qwen3",
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_attention_heads,
            # Keep the same multi-RoPE setup as the reference checkpoints but shrink the
            # sections so they sum to the rotary half-dimension (4) of this tiny test model.
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            "tie_word_embeddings": True,
        }

        self.vision_config = {
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 2,
            "num_hidden_layers": 1,
            "num_attention_heads": num_attention_heads,
            "num_channels": 3,
            "num_patches": 64,
            "patch_size": 4,
            "pixel_shuffle_scale_factor": 1,
            "attention_dropout": 0.0,
            "layer_norm_eps": 1e-6,
        }

    def get_config(self):
        config = IsaacConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
        )
        # Rely on vanilla SDPA so the tests do not need flash attention.
        config._attn_implementation = "sdpa"
        config.text_config._attn_implementation = "sdpa"
        config.vision_attn_implementation = "sdpa"
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(
            (self.batch_size, self.seq_length),
            dtype=torch.long,
            device=torch_device,
        )
        labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return config, input_ids, attention_mask, labels


@require_torch
class IsaacModelTest(unittest.TestCase):
    all_model_classes = (IsaacModel, IsaacForConditionalGeneration) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = IsaacModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=IsaacConfig,
            has_text_modality=True,
            common_properties=["hidden_size"],
            text_config=self.model_tester.text_config,
            vision_config=self.model_tester.vision_config,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @require_tensorstream
    def test_model_forward(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = IsaacModel(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(
            result.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
        )

    @require_tensorstream
    def test_for_conditional_generation(self):
        config, input_ids, attention_mask, labels = self.model_tester.prepare_config_and_inputs()
        model = IsaacForConditionalGeneration(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        self.assertIsNotNone(result.loss)

    def test_prepare_inputs_for_generation(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = IsaacForConditionalGeneration(config)
        model.to(torch_device)

        prepared_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIn("input_ids", prepared_inputs)
        self.assertIn("position_ids", prepared_inputs)
        self.assertIsNone(prepared_inputs["position_ids"])


def test_isaac_config_extends_qwen3_defaults(isaac_tiny_config):
    assert isaac_tiny_config.hidden_size == isaac_tiny_config.text_config.hidden_size
    assert isaac_tiny_config.num_attention_heads == isaac_tiny_config.text_config.num_attention_heads
    assert isaac_tiny_config.model_type == "isaac"
    assert isaac_tiny_config.vision_config is not None
    assert isaac_tiny_config.vision_config.patch_size == 4
    assert isaac_tiny_config.vision_config.num_patches == 64
    assert isaac_tiny_config.max_sequence_length == 16384
    assert isaac_tiny_config.vision_rescale_factor == pytest.approx(1 / 255)
    assert isaac_tiny_config.vision_token == "<image>"


def test_isaac_config_migrates_legacy_rope_theta():
    cfg = IsaacConfig(text_config={"rope_theta": 12345})
    assert cfg.rope_parameters.get("rope_theta") == 12345
    assert cfg.rope_parameters.get("rope_type") == "default"
    serialized = cfg.to_dict()
    assert "rope_theta" not in serialized
    assert "rope_theta" not in serialized.get("text_config", {})
    assert serialized["rope_parameters"].get("rope_theta") == 12345


@require_torch
@require_tensorstream
def test_isaac_for_conditional_generation_initialization(isaac_tiny_config):
    model = IsaacForConditionalGeneration(isaac_tiny_config)
    model.to(torch_device)
    assert hasattr(model, "model")
    assert hasattr(model, "lm_head")
    assert hasattr(model.model, "vision_embedding")
    assert hasattr(model.model, "embed_fns")

    input_ids = torch.randint(0, isaac_tiny_config.vocab_size, (1, 10), device=torch_device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
    assert outputs.logits.shape == (1, 10, isaac_tiny_config.vocab_size)


@require_torch
@require_tensorstream
def test_isaac_for_conditional_generation_loss_and_generate_flag(isaac_tiny_config):
    model = IsaacForConditionalGeneration(isaac_tiny_config).to(torch_device)
    assert model.can_generate()

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, isaac_tiny_config.vocab_size, (batch_size, seq_len), device=torch_device)
    labels = torch.randint(0, isaac_tiny_config.vocab_size, (batch_size, seq_len), device=torch_device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert outputs.logits.shape == (batch_size, seq_len, isaac_tiny_config.vocab_size)


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_matches_config_defaults(isaac_processor, isaac_tiny_config):
    assert isaac_processor.vision_token == isaac_tiny_config.vision_token
    assert isaac_processor.max_sequence_length == isaac_tiny_config.max_sequence_length
    assert isaac_processor.config is isaac_tiny_config
    assert isinstance(isaac_processor.image_processor, IsaacImageProcessorFast)
    assert isaac_processor.image_processor.rescale_factor == pytest.approx(isaac_tiny_config.vision_rescale_factor)


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_text_only_round_trip(isaac_processor):
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    prompt = isaac_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = isaac_processor(text=prompt, images=None, return_tensors="pt")

    assert "input_ids" in outputs
    assert "tensor_stream" in outputs
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].shape[0] == 1


@require_torch
@require_tensorstream
def test_isaac_processor_accepts_batchencoding_chat_template(isaac_processor):
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    batch_encoding = isaac_processor.apply_chat_template(messages, add_generation_prompt=True)

    outputs = isaac_processor(text=batch_encoding, images=None, return_tensors="pt")

    assert "input_ids" in outputs
    assert "tensor_stream" in outputs
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].shape[0] == 1


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_with_single_image(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"Look at this {vision_token} and describe it."
    image = _make_dummy_image()

    outputs = isaac_processor(text=text, images=[image], return_tensors="pt")
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].ndim == 2


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_with_multiple_images(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"First {vision_token} then {vision_token}"
    images = [_make_dummy_image(color=(255, 0, 0)), _make_dummy_image(color=(0, 255, 0))]

    outputs = isaac_processor(text=text, images=images, return_tensors="pt")
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].shape[0] == 1


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_error_on_image_mismatch(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"{vision_token} {vision_token}"
    image = _make_dummy_image()

    with pytest.raises(ValueError, match="must match number of images"):
        isaac_processor(text=text, images=[image], return_tensors="pt")


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_consistent_tensor_stream_types(isaac_processor):
    text_only = "Simple question?"
    text_with_image = f"Describe this {isaac_processor.vision_token}"
    image = _make_dummy_image()

    outputs_text = isaac_processor(text=text_only, images=None, return_tensors="pt")
    outputs_image = isaac_processor(text=text_with_image, images=[image], return_tensors="pt")

    assert isinstance(outputs_text["tensor_stream"], TensorStream)
    assert isinstance(outputs_image["tensor_stream"], TensorStream)
    assert outputs_text["input_ids"].shape[0] == outputs_image["input_ids"].shape[0] == 1


@require_torch
@require_vision
@require_tensorstream
def test_isaac_generation_with_tensor_stream(isaac_processor, isaac_tiny_config):
    model = IsaacForConditionalGeneration(isaac_tiny_config).to(torch_device)
    model.eval()

    messages = [{"role": "user", "content": "Hello there!"}]
    prompt = isaac_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = isaac_processor(text=prompt, images=None, return_tensors="pt")

    input_ids = processed["input_ids"].to(torch_device)
    tensor_stream = processed["tensor_stream"]
    tensor_stream = tensor_stream.to(torch_device)
    generated = model.generate(
        input_ids=input_ids,
        tensor_stream=tensor_stream,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=isaac_processor.tokenizer.pad_token_id,
        eos_token_id=isaac_processor.tokenizer.eos_token_id,
    )

    assert generated.shape[0] == 1
    assert generated.shape[1] >= input_ids.shape[1]
    decoded_prompt = isaac_processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    assert isinstance(decoded_prompt, str)
    assert decoded_prompt.strip() != ""


@require_torch
@slow
@require_tensorstream
def test_isaac_checkpoint_hashes(isaac_reference_model):
    isaac_reference_model = isaac_reference_model.to("cpu")
    expected_hashes = _load_expected_hashes()
    if not expected_hashes:
        pytest.skip(f"Missing golden hashes file at {HASH_FILE}.")

    missing = [subset for subset in HASH_FILTERS if subset not in expected_hashes]
    if missing:
        pytest.skip(f"Golden hashes missing entries for: {', '.join(missing)}")

    isaac_reference_model.to("cpu")
    state_dict = isaac_reference_model.state_dict()
    for subset, filters in HASH_FILTERS.items():
        current_hash = _hash_state_dict(state_dict, include=filters["include"], exclude=filters["exclude"])
        assert current_hash == expected_hashes[subset], f"Hash mismatch for subset '{subset}'"


def create_isaac_processor(
    tokenizer,
    isaac_config,
    *,
    image_processor=None,
    **overrides,
):
    """Helper to construct IsaacProcessor without requiring an IsaacConfig instance."""
    params = {
        "vision_token": isaac_config.vision_token,
        "max_sequence_length": isaac_config.max_sequence_length,
        "vision_patch_size": isaac_config.vision_patch_size,
        "vision_max_num_patches": isaac_config.vision_max_num_patches,
        "vision_min_num_patches": isaac_config.vision_min_num_patches,
        "pixel_shuffle_scale": isaac_config.pixel_shuffle_scale,
        "rescale_factor": isaac_config.vision_rescale_factor,
        "image_mean": tuple(isaac_config.vision_mean),
        "image_std": tuple(isaac_config.vision_std),
        "vision_attn_implementation": isaac_config.vision_attn_implementation,
    }
    params.update(overrides)

    processor_image = image_processor
    if processor_image is None:
        processor_image = IsaacImageProcessorFast(
            patch_size=params["vision_patch_size"],
            max_num_patches=params["vision_max_num_patches"],
            min_num_patches=params["vision_min_num_patches"],
            pixel_shuffle_scale=params["pixel_shuffle_scale"],
            rescale_factor=params["rescale_factor"],
            image_mean=params["image_mean"],
            image_std=params["image_std"],
        )
    processor_params = {
        "vision_token": isaac_config.vision_token,
        "max_sequence_length": isaac_config.max_sequence_length,
        "rescale_factor": isaac_config.vision_rescale_factor,
    }

    return IsaacProcessor(
        image_processor=processor_image,
        tokenizer=tokenizer,
        config=isaac_config,
        **processor_params,
    )


@require_torch
@require_vision
@slow
@require_tensorstream
def test_hf_generate_vs_training_generate_logits(isaac_reference_model, isaac_reference_processor):
    device = "cuda"
    dtype = torch.bfloat16
    isaac_reference_model = isaac_reference_model.to(device=device, dtype=dtype)
    isaac_reference_model.eval()
    golden = _load_generation_golden()
    if not golden:
        pytest.skip(f"Missing generation golden file at {GENERATION_GOLDEN_FILE}.")

    image = _load_red_dot_image()
    if image is None:
        pytest.skip("PIL.Image is required for Isaac generation tests.")

    messages = [
        {
            "role": "user",
            "content": "Describe this image:",
        },
        {
            "role": "user",
            "content": "<image>",
        },
    ]
    prompt = isaac_reference_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).strip()
    batch = isaac_reference_processor(text=prompt, images=[image], return_tensors="pt")

    input_ids = batch["input_ids"]
    tensor_stream = batch["tensor_stream"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isaac_reference_model.to(device)
    input_ids = input_ids.to(device)
    if tensor_stream is not None and hasattr(tensor_stream, "to"):
        tensor_stream = tensor_stream.to(device)

    torch.manual_seed(0)
    with torch.no_grad():
        outputs = isaac_reference_model.generate(
            input_ids=input_ids,
            tensor_stream=tensor_stream,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=isaac_reference_processor.tokenizer.eos_token_id,
            eos_token_id=isaac_reference_processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )

    logits = torch.cat(outputs.logits, dim=0).to(torch.float32).cpu()
    logits_stats = compute_logits_statistics(logits)
    generated_ids = outputs.sequences[0].tolist()

    assert generated_ids == golden["token_ids"], "Generated token ids changed"
    if "logits_statistics" in golden:
        _assert_logits_statistics_close(logits_stats, golden["logits_statistics"])
    else:
        pytest.fail(
            "Golden file missing both logits_statistics and logits_hash. "
            f"Regenerate {GENERATION_GOLDEN_FILE} via scripts/update_isaac_hashes.py."
        )

    isaac_reference_model.to("cpu")
