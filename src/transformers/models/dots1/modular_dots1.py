from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3TopkRouter,
)
from ..llama.modeling_llama import (
    KwargsForCausalLM,
    LlamaRMSNorm,
)
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3ForCausalLM, Qwen3Model, Qwen3RotaryEmbedding
from .configuration_dots1 import Dots1Config


logger = logging.get_logger(__name__)


class Dots1RMSNorm(LlamaRMSNorm):
    pass


class Dots1RotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Dots1Attention(Qwen3Attention):
    pass


class Dots1MLP(DeepseekV3MLP):
    pass


class Dots1MoE(DeepseekV3MoE):
    pass


class Dots1TopkRouter(DeepseekV3TopkRouter):
    pass


class Dots1DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config: Dots1Config, layer_idx: int):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]


class Dots1PreTrainedModel(DeepseekV3PreTrainedModel):
    pass


class Dots1Model(Qwen3Model):
    pass


class Dots1ForCausalLM(Qwen3ForCausalLM):
    def forward(
        self,
        **super_kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Dots1ForCausalLM

        >>> model = Dots1ForCausalLM.from_pretrained("rednote-hilab/dots1.llm1.inst")
        >>> tokenizer = AutoTokenizer.from_pretrained("rednote-hilab/dots1.llm1.inst")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Dots1PreTrainedModel",
    "Dots1Model",
    "Dots1ForCausalLM",
]
