from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    GPT2Config,
    GPT2LMHeadModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    MistralConfig,
    MistralForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.models.mistral.modeling_mistral import MistralModel


def test_causal_lm_get_decoder_returns_underlying_model():
    cfg = MistralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = MistralForCausalLM(cfg)
    dec = model.get_decoder()

    assert dec is model.model, f"Expected get_decoder() to return model.model, got {type(dec)}"


def test_seq2seq_get_decoder_still_returns_decoder_module():
    cfg = BartConfig(
        vocab_size=128,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
    )
    model = BartForConditionalGeneration(cfg)
    dec = model.get_decoder()

    assert dec is model.model.decoder, "Seq2seq get_decoder() should return the decoder submodule"


def test_base_model_returns_self():
    """Test that base transformer models (no decoder/model attributes) return self."""
    cfg = MistralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    base_model = MistralModel(cfg)
    dec = base_model.get_decoder()

    assert dec is base_model, f"Base model get_decoder() should return self, got {type(dec)}"


def test_explicit_decoder_attribute_opt():
    """Test models with explicit decoder attribute (OPT style)."""
    cfg = OPTConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
    )
    model = OPTForCausalLM(cfg)
    dec = model.get_decoder()

    assert dec is model.model.decoder, f"OPT get_decoder() should return model.decoder, got {type(dec)}"


def test_explicit_decoder_attribute_t5():
    """Test encoder-decoder models with explicit decoder attribute."""
    cfg = T5Config(
        vocab_size=128,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_heads=4,
    )
    model = T5ForConditionalGeneration(cfg)
    dec = model.get_decoder()

    assert dec is model.decoder, f"T5 get_decoder() should return decoder attribute, got {type(dec)}"


def test_same_type_recursion_prevention():
    """Test that same-type recursion is prevented (see issue #40815)."""
    cfg = MistralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = MistralForCausalLM(cfg)

    assert type(model) is not type(model.model), "Types should be different to prevent recursion"

    dec = model.get_decoder()
    assert dec is model.model, f"Should return model.model without infinite recursion, got {type(dec)}"

    inner_dec = model.model.get_decoder()
    assert inner_dec is model.model, f"Inner model should return itself, got {type(inner_dec)}"


def test_nested_wrapper_recursion():
    """Test models that don't have model/decoder attributes return self."""
    # GPT2 model which has "transformer" attribute (not "model")
    cfg = GPT2Config(
        vocab_size=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=512,
    )
    model = GPT2LMHeadModel(cfg)
    dec = model.get_decoder()

    # GPT2LMHeadModel has no 'decoder' or 'model' attributes, so should return self
    assert dec is model, f"GPT2 get_decoder() should return self (fallback), got {type(dec)}"


def test_model_without_get_decoder():
    """Test edge case where model has model attribute but no get_decoder method."""

    class MockInnerModel:
        """Mock model without get_decoder method."""

        pass

    class MockWrapperModel:
        """Mock wrapper with model attribute but inner has no get_decoder."""

        def __init__(self):
            self.model = MockInnerModel()

        def get_decoder(self):
            # Use the same logic as modeling_utils.py
            if hasattr(self, "decoder"):
                return self.decoder
            if hasattr(self, "model"):
                inner = self.model
                if hasattr(inner, "get_decoder") and type(inner) is not type(self):
                    return inner.get_decoder()
                return inner
            return self

    wrapper = MockWrapperModel()
    dec = wrapper.get_decoder()

    # Should return the inner model since it doesn't have get_decoder
    assert dec is wrapper.model, f"Should return inner model when no get_decoder, got {type(dec)}"


def test_vision_language_model():
    """Test vision-language models like LLaVA that delegate to language_model."""
    text_config = MistralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )

    vision_config = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_channels": 3,
        "image_size": 224,
        "patch_size": 16,
    }

    cfg = LlavaConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config,
        vocab_size=128,
    )

    model = LlavaForConditionalGeneration(cfg)
    dec = model.get_decoder()

    assert dec is model.language_model, f"LLaVA get_decoder() should return language_model, got {type(dec)}"
