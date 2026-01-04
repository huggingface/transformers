from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


def test_qwen3_omni_moe_configs_have_initializer_range():
    talker_config = Qwen3OmniMoeTalkerConfig()
    assert hasattr(talker_config, "initializer_range")

    code2wav_config = Qwen3OmniMoeCode2WavConfig()
    assert hasattr(code2wav_config, "initializer_range")

    main_config = Qwen3OmniMoeConfig()
    assert hasattr(main_config, "initializer_range")


def test_qwen3_omni_moe_talker_has_no_tied_weights():
    tied_keys = Qwen3OmniMoeTalkerForConditionalGeneration._tied_weights_keys
    assert tied_keys in (None, {})
