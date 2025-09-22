import torch
import torch.nn as nn

from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding


class MistralForCausalLMWithCache(MistralForCausalLM):
    """
    Wrapper class for MistralForCausalLM, to be extended with cached rotary embedding.
    """

    def __init__(self, config):
        super().__init__(config)
        print("####### Moreh MistralForCausalLMWithCache initialized #######")
        head_dim = config.hidden_size // config.num_attention_heads
        for layer in self.model.layers:
            layer.self_attn.rotary_emb = MistralRotaryEmbeddingCached(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,   
            )


class MistralRotaryEmbeddingCached(MistralRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base, device=device)
        self.use_cos_sin_cache = True
        if self.use_cos_sin_cache:
            print("Use cos/sin modeling_mistral_cached")
            self._set_cos_sin_cache(max_position_embeddings, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device="cpu")
        freqs = torch.outer(t, self.inv_freq.cpu())   
        emb = torch.cat((freqs, freqs), dim=-1)       
        cos = emb.cos().to(device="cuda", dtype=dtype)
        sin = emb.sin().to(device="cuda", dtype=dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.use_cos_sin_cache:
            seq_len = position_ids.shape[-1]
            assert seq_len <= self.max_position_embeddings
            cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device).unsqueeze(0)
            sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device).unsqueeze(0)
            return cos, sin

        # fallback
        return super().forward(x, position_ids)
