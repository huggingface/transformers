import unittest

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
)


if is_torch_available():
    import torch


@require_torch
class CrossLayerAttentionTest(unittest.TestCase):
    
    test_parameters = [
        # Model, attention implementation, fp8 kv enabled
        ("Qwen/Qwen2.5-32B-Instruct", "eager", True),
        ("Qwen/Qwen2.5-32B-Instruct", "sdpa", True),
        ("Qwen/Qwen2.5-32B-Instruct", "eager", False),
        ("Qwen/Qwen2.5-32B-Instruct", "sdpa", False),   
    ]
        
    def test_naive_cla(self):
        "Compare a base model without CLA with a model with CLA where each layer computes its own KV cache."
        
        for model_name, attn_impl, fp8_kv_enabled in self.test_parameters:
            with self.subTest(model_id=model_name, attn_impl=attn_impl, fp8_kv_enabled=fp8_kv_enabled):
                cfg = AutoConfig.from_pretrained(model_name)
                cfg.num_hidden_layers = 4
                cfg.hidden_size //= 8
                cfg.intermediate_size //= 8
                cfg.num_attention_heads //= 2
                cfg.num_key_value_heads //= 2
                cfg._attn_implementation = attn_impl
                cfg.use_fp8_kv_scale = fp8_kv_enabled
                cfg.palu_kv_compression_enabled = False
                cfg.use_cache = False
                cfg.debug_kv_sharing = False
                cfg.output_attentions = False
                
                x = torch.arange(32, device="cuda").view(1,-1)

                cfg.cla_kv_cache_map = {0:0, 1:1, 2:2, 3:3}                
                model = AutoModelForCausalLM.from_config(cfg)
                model.to(device="cuda", dtype=torch.bfloat16)
                test_output = model(x)
                
                model_state_dict = model.state_dict()

                cfg.cla_kv_cache_map = None
                model = AutoModelForCausalLM.from_config(cfg)
                model.to(device="cuda", dtype=torch.bfloat16)
                model.load_state_dict(model_state_dict);
                base_output = model(x)

                assert torch.equal(test_output.logits, base_output.logits)            
    
    def test_cla_2(self):
        """
        Test CLA with a custom KV cache map.
        
        LLM with 4 layers:
        layer 3: oooooo <---| 
        layer 2: oooooo <-| |
        layer 1: oooooo --| |
        layer 0: oooooo ----|
        """

        for model_name, attn_impl, fp8_kv_enabled in self.test_parameters:
            with self.subTest(model_id=model_name, attn_impl=attn_impl, fp8_kv_enabled=fp8_kv_enabled):
                cfg = AutoConfig.from_pretrained(model_name)
                cfg.num_hidden_layers = 4
                cfg.hidden_size //= 8
                cfg.intermediate_size //= 8
                cfg.num_attention_heads //= 2
                cfg.num_key_value_heads //= 2
                cfg._attn_implementation = attn_impl
                cfg.use_fp8_kv_scale = fp8_kv_enabled
                cfg.cla_kv_cache_map = {0:0, 1:1, 2:1, 3:0}
                cfg.palu_kv_compression_enabled = False
                cfg.use_cache = False
                cfg.debug_kv_sharing = True
                cfg.output_attentions = attn_impl == "eager"
                
                x = torch.arange(32, device="cuda").view(1,-1)
                
                model = AutoModelForCausalLM.from_config(cfg)
                model.to(device="cuda", dtype=torch.bfloat16);
                
                assert model.config.use_fp8_kv_scale == fp8_kv_enabled
                assert model.config.cla_kv_cache_map == {0:0, 1:1, 2:1, 3:0}
                assert model.config.use_cache == False
                
                out_eager = model(x)
                                
                if attn_impl == "eager":
                    assert torch.equal(out_eager.attentions[0], out_eager.attentions[3])
                    assert torch.equal(out_eager.attentions[1], out_eager.attentions[2])
                    assert not torch.equal(out_eager.attentions[0], out_eager.attentions[1])
                    
                attn_outputs = [l.self_attn.debug_cla_attn_output for l in model.model.layers]
                assert len(attn_outputs) == 4
                assert torch.equal(attn_outputs[0], attn_outputs[3])
                assert torch.equal(attn_outputs[1], attn_outputs[2])
                assert not torch.equal(attn_outputs[0], attn_outputs[1])