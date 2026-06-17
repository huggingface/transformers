from torch import nn

from transformers.distributed.fsdp import (
    _resolve_tied_embed_lm_head_plan,
    expand_fsdp_plan,
    is_fsdp_managed_module,
    is_norm_and_head_pair,
    verify_fsdp_plan,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class TestDistributedFSDP:
    def test_verify_fsdp_plan_warns_on_unknown_strategy(self, caplog):
        verify_fsdp_plan(list(TinyModel().named_modules()), {"layers.*": "bad_strategy"})
        assert "unknown strategies" in caplog.text

    def test_verify_fsdp_plan_warns_on_unused_key(self, caplog):
        verify_fsdp_plan(list(TinyModel().named_modules()), {"missing.*": "free_full_weight"})
        assert "not applied to any module" in caplog.text

    def test_expand_fsdp_plan_wildcard(self):
        model = TinyModel()
        targets = expand_fsdp_plan(model, {"layers.*": "free_full_weight"})
        assert len(targets) == 2
        assert all(strategy == "free_full_weight" for _, _, strategy in targets)

    def test_expand_fsdp_plan_exact_key(self):
        model = TinyModel()
        targets = expand_fsdp_plan(model, {"norm": "keep_full_weight"})
        assert targets == [("norm", model.norm, "keep_full_weight")]

    def test_resolve_tied_embed_lm_head_plan(self):
        plan = {
            "model.embed_tokens": "free_full_weight",
            "model.layers.*": "free_full_weight",
            "model.norm": "keep_full_weight",
            "lm_head": "keep_full_weight",
        }
        adapted = _resolve_tied_embed_lm_head_plan(plan, tie_word_embeddings=True)
        assert "lm_head" not in adapted
        assert adapted["model.embed_tokens"] == "keep_full_weight"
        assert adapted["model.layers.*"] == "free_full_weight"

    def test_is_norm_and_head_pair(self):
        model = TinyModel()
        assert is_norm_and_head_pair([("norm", model.norm), ("lm_head", model.lm_head)])
        assert not is_norm_and_head_pair([("norm", model.norm)])

    def test_is_fsdp_managed_module_flag(self):
        model = TinyModel()
        assert not is_fsdp_managed_module(model)
        model._is_fsdp_managed_module = True
        assert is_fsdp_managed_module(model)
