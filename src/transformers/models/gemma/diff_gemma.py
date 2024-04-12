from transformers.models.llama.modeling_llama import *
import torch.nn as nn
from transformers.utils import ModelConverter

GemmaConverter = ModelConverter(__file__)

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

GemmaConverter.register("GemmaRotaryEmbedding", LlamaRotaryEmbedding)
GemmaConverter.register("GemmaMLP", LlamaMLP) 

GemmaAttention = GemmaConverter.register("GemmaAttention", LlamaAttention) 
GemmaSdpaAttention = GemmaConverter.register("GemmaSdpaAttention", LlamaAttention) 
GemmaFlashAttention2 = GemmaConverter.register("GemmaFlashAttention2", LlamaAttention) 

COHERE_ATTENTION_CLASSES = {"eager": GemmaAttention, "flash_attention_2": GemmaFlashAttention2, "sdpa": GemmaSdpaAttention}

GemmaConverter.register("GemmaDecoderLayer", LlamaDecoderLayer) 
GemmaConverter.register("GemmaPreTrainedModel", LlamaPreTrainedModel)

class GemmaModel(LlamaModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        return super().forward(None, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict, cache_position)


GemmaConverter.register("GemmaForCausalLM", LlamaForCausalLM)