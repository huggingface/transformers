"""Local workarounds for known Transformers bugs, applied by the runner before contracts run.

A framework's fleet lock lists the patches it needs under `local_patches`, with
the reason. The runner applies them in its own process and in every isolated
contract process, and records their names in each result's environment, so a
baseline recorded with a workaround says so. Remove a patch, and re-record the
affected baselines, once the fix is in the pinned Transformers build.
"""

import importlib
import inspect

# Audio models whose `masked_spec_embed` (SpecAugment's mask vector) has no
# `_init_weights` branch. Checkpoints that lack the tensor get uninitialized
# memory on load; HuBERT, SEW-D, SpeechT5, and Wav2Vec2-BERT already handle it.
MASKED_SPEC_EMBED_MODELS = {
    "wav2vec2": "Wav2Vec2PreTrainedModel",
    "wav2vec2_conformer": "Wav2Vec2ConformerPreTrainedModel",
    "data2vec": "Data2VecAudioPreTrainedModel",
    "wavlm": "WavLMPreTrainedModel",
    "unispeech": "UniSpeechPreTrainedModel",
    "unispeech_sat": "UniSpeechSatPreTrainedModel",
    "sew": "SEWPreTrainedModel",
}


def masked_spec_embed_init():
    """Initialize masked_spec_embed like HuBERT does: init.uniform_ in _init_weights."""
    import torch
    from transformers import initialization as init

    patched = []
    for model_type, class_name in MASKED_SPEC_EMBED_MODELS.items():
        module_name = f"transformers.models.{model_type}.modeling_{'data2vec_audio' if model_type == 'data2vec' else model_type}"
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        original = cls._init_weights
        if getattr(original, "_mic_patched", False):
            continue
        if "masked_spec_embed" in inspect.getsource(original):
            continue  # fixed upstream (#49386): a second init.uniform_ would draw different values

        def _init_weights(self, module, original=original):
            original(self, module)
            if isinstance(getattr(module, "masked_spec_embed", None), torch.nn.Parameter):
                init.uniform_(module.masked_spec_embed)

        _init_weights._mic_patched = True
        cls._init_weights = _init_weights
        patched.append(class_name)
    return patched


def whisper_decode_tensor_batch():
    """Decode a 2-D tensor with timestamps per sequence, as decode already does for a list of lists (#49403).

    WhisperTokenizer.decode only treats its input as a batch when it is a list of
    lists, so batch_decode(model.generate(...), decode_with_timestamps=True) merges
    the rows into one string (one token per row) or raises (longer rows).
    """
    try:
        from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
    except ImportError:
        return []
    original = WhisperTokenizer.decode
    if getattr(original, "_mic_patched", False):
        return []

    def decode(self, token_ids, *args, original=original, **kwargs):
        if kwargs.get("decode_with_timestamps") and getattr(token_ids, "ndim", 0) == 2:
            token_ids = token_ids.tolist()
        return original(self, token_ids, *args, **kwargs)

    decode._mic_patched = True
    WhisperTokenizer.decode = decode
    return ["WhisperTokenizer"]


def config_field_generation_params():
    """Do not count a config's own declared fields as user-set generation parameters (#49456, from #48282).

    generate() raises when model.config holds generation parameters. It used to
    skip every field of the config class; since #48282 it skips only fields
    whose default is not None, so WhisperConfig.suppress_tokens (default None,
    set in the OpenAI checkpoints' config.json) makes every generate() raise.
    Restore the skip for every declared field.
    """
    import dataclasses

    from transformers import PreTrainedConfig

    original = PreTrainedConfig._get_generation_parameters
    if getattr(original, "_mic_patched", False) or not hasattr(PreTrainedConfig, "default_config_fields"):
        return []  # already patched, or a build before the regression

    def _get_generation_parameters(self, original=original):
        params = original(self)
        if dataclasses.is_dataclass(self):
            declared = {f.name for f in dataclasses.fields(self)}
            params = {k: v for k, v in params.items() if k not in declared}
        return params

    _get_generation_parameters._mic_patched = True
    PreTrainedConfig._get_generation_parameters = _get_generation_parameters
    return ["PreTrainedConfig"]

PATCHES = {
    "masked-spec-embed-init": masked_spec_embed_init,
    "whisper-decode-tensor-batch": whisper_decode_tensor_batch,
    "config-field-generation-params": config_field_generation_params,
}


def apply(names):
    """Apply the named patches; return {name: what was patched}."""
    unknown = sorted(set(names) - set(PATCHES))
    if unknown:
        raise SystemExit(f"unknown local patches: {unknown}")
    return {name: PATCHES[name]() for name in names}
