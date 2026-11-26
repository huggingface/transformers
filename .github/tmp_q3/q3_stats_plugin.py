# pytest plugin: record statistics of `test_cached_decode_matches_cacheless` (Qwen3-VL-MoE) without changing it.
# One JSON line per test run in $Q3_STATS_DIR/<pid>.jsonl:
#   - steps: per decode step, max |diff|, mean |diff| and the allclose ratio max(|a-b| / (atol + rtol*|b|)) (> 1 fails)
#   - ids_match: whether the id-first check saw identical sequences
#   - router_min_gap: smallest gap between the k-th and (k+1)-th expert probability over all router calls
#   - xpath: the router logits of each (layer, batch row, position) in the cached generate vs the cacheless generate:
#       delta = max over experts |logit_cached - logit_cacheless|, gap = k-th minus (k+1)-th logit (smaller of the two
#       paths), ratio = delta / gap (> 1 can flip the top-k choice), flips = top-k expert sets differ.
#       "rel" = positions that matter (not left padding); "pad" = masked padding positions.
import json
import os
import re

import pytest
import torch


STATS_DIR = os.environ.get("Q3_STATS_DIR", "q3_stats")
os.makedirs(STATS_DIR, exist_ok=True)
_rec = {}
_st = {"phase": None, "ctx": None, "layers": {}, "logits": {}, "mask": None}


def _reset():
    _rec.clear()
    _st.update(phase=None, ctx=None, layers={}, logits={"cached": {}, "cacheless": {}}, mask=None)


def _install():
    import tests.generation.test_utils as tu
    from transformers.generation.utils import GenerationMixin
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as m

    orig_close = torch.testing.assert_close
    orig_similar = tu.assert_similar_generate_outputs
    orig_router = m.Qwen3VLMoeTextTopKRouter.forward
    orig_block = m.Qwen3VLMoeTextSparseMoeBlock.forward
    orig_generate = GenerationMixin.generate

    def assert_close(actual, expected, *args, **kwargs):
        msg = kwargs.get("msg")
        if isinstance(msg, str) and msg.startswith("step "):
            a, b = actual.detach().float(), expected.detach().float()
            atol, rtol = kwargs.get("atol", 0.0), kwargs.get("rtol", 0.0)
            diff = (a - b).abs()
            _rec.setdefault("steps", []).append(
                {
                    "step": int(re.match(r"step (\d+)", msg).group(1)),
                    "max_diff": diff.max().item(),
                    "mean_diff": diff.mean().item(),
                    "ratio": (diff / (atol + rtol * b.abs())).max().item(),
                    "max_abs_ref": b.abs().max().item(),
                }
            )
        return orig_close(actual, expected, *args, **kwargs)

    def assert_similar(output_1, output_2, *args, **kwargs):
        _rec["ids_match"] = bool((output_1.sequences == output_2.sequences).all())
        return orig_similar(output_1, output_2, *args, **kwargs)

    def generate(self, *args, **kwargs):
        _st["phase"] = "cached" if kwargs.get("use_cache", True) else "cacheless"
        if kwargs.get("attention_mask") is not None:
            _st["mask"] = kwargs["attention_mask"].detach().cpu()
        try:
            return orig_generate(self, *args, **kwargs)
        finally:
            _st["phase"] = None

    def block_forward(self, hidden_states):
        layer = _st["layers"].setdefault(id(self), len(_st["layers"]))
        _st["ctx"] = (layer, hidden_states.shape[0], hidden_states.shape[1])
        try:
            return orig_block(self, hidden_states)
        finally:
            _st["ctx"] = None

    def router_forward(self, hidden_states):
        out = orig_router(self, hidden_states)
        probs = torch.softmax(out[0].float(), dim=-1)
        srt = probs.sort(dim=-1, descending=True).values
        gap = (srt[:, self.top_k - 1] - srt[:, self.top_k]).min().item()
        _rec["router_min_gap"] = min(_rec.get("router_min_gap", 1.0), gap)
        _rec["router_calls"] = _rec.get("router_calls", 0) + 1
        if _st["phase"] and _st["ctx"]:
            layer, b, s = _st["ctx"]
            _st["top_k"] = self.top_k
            _st["logits"][_st["phase"]].setdefault(layer, []).append(out[0].detach().float().cpu().view(b, s, -1))
        return out

    torch.testing.assert_close = assert_close
    tu.assert_similar_generate_outputs = assert_similar
    GenerationMixin.generate = generate
    m.Qwen3VLMoeTextSparseMoeBlock.forward = block_forward
    m.Qwen3VLMoeTextTopKRouter.forward = router_forward


def _cross_path():
    """Compare the router logits of the cached and cacheless generate per (layer, batch row, position)."""
    cached, cacheless, k = _st["logits"]["cached"], _st["logits"]["cacheless"], _st.get("top_k")
    if not cached or not cacheless or k is None:
        return None
    out = {"rel": {"n": 0, "max_delta": 0.0, "min_gap": float("inf"), "max_ratio": 0.0, "flips": 0},
           "pad": {"n": 0, "max_ratio": 0.0, "flips": 0}}
    for layer, calls in cached.items():
        full = torch.cat(calls, dim=1)  # cached: prefill + one position per decode step -> all positions
        for step_logits in cacheless.get(layer, []):  # cacheless: every step recomputes positions 0..L-1
            n = min(full.shape[1], step_logits.shape[1])
            a, b = full[:, :n], step_logits[:, :n]
            delta = (a - b).abs().amax(-1)
            sa, sb = a.sort(-1, descending=True).values, b.sort(-1, descending=True).values
            gap = torch.minimum(sa[..., k - 1] - sa[..., k], sb[..., k - 1] - sb[..., k])
            ratio = delta / gap.clamp_min(1e-30)
            ta = a.topk(k, -1).indices.sort(-1).values
            tb = b.topk(k, -1).indices.sort(-1).values
            flip = (ta != tb).any(-1)
            valid = torch.ones_like(flip)
            mask = _st["mask"]
            if mask is not None and mask.shape[0] == a.shape[0]:
                m_len = min(mask.shape[1], n)
                valid[:, :m_len] = mask[:, :m_len].bool()
            for key, sel in (("rel", valid), ("pad", ~valid)):
                if sel.any():
                    o = out[key]
                    o["n"] += int(sel.sum())
                    o["max_ratio"] = max(o["max_ratio"], ratio[sel].max().item())
                    o["flips"] += int(flip[sel].sum())
                    if key == "rel":
                        o["max_delta"] = max(o["max_delta"], delta[sel].max().item())
                        o["min_gap"] = min(o["min_gap"], gap[sel].min().item())
    return out


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _reset()


def pytest_collection_finish(session):
    _reset()
    _install()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        lines = str(rep.longrepr).splitlines() if rep.failed else []
        msg = next((ln for ln in lines if "max_diff" in ln or "AssertionError" in ln), "")[:300]
        rec = dict(_rec, outcome=rep.outcome, msg=msg)
        try:
            rec["xpath"] = _cross_path()
        except Exception as e:  # never break the run because of the instrumentation
            rec["xpath_error"] = repr(e)[:200]
        with open(os.path.join(STATS_DIR, f"{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
