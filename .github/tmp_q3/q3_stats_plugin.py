# pytest plugin: record statistics of `test_cached_decode_matches_cacheless` (Qwen3-VL-MoE) without changing it.
# One JSON line per test run in $Q3_STATS_DIR/<pid>.jsonl:
#   - steps: per decode step, max |diff|, mean |diff| and the allclose ratio max(|a-b| / (atol + rtol*|b|)) (> 1 fails)
#   - ids_match: whether the id-first check saw identical sequences
#   - router: smallest gap between the k-th and (k+1)-th expert probability (how close a routing choice came to flipping)
#   - outcome: passed / failed (+ message)
import json
import os
import re

import pytest
import torch


STATS_DIR = os.environ.get("Q3_STATS_DIR", "q3_stats")
os.makedirs(STATS_DIR, exist_ok=True)
_rec = {}


def _install():
    import tests.generation.test_utils as tu
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as m

    orig_close = torch.testing.assert_close
    orig_similar = tu.assert_similar_generate_outputs
    orig_router = m.Qwen3VLMoeTextTopKRouter.forward

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

    def router_forward(self, hidden_states):
        out = orig_router(self, hidden_states)
        probs = torch.softmax(out[0].float(), dim=-1)
        srt = probs.sort(dim=-1, descending=True).values
        gap = (srt[:, self.top_k - 1] - srt[:, self.top_k]).min().item()
        _rec["router_min_gap"] = min(_rec.get("router_min_gap", 1.0), gap)
        _rec["router_calls"] = _rec.get("router_calls", 0) + 1
        return out

    torch.testing.assert_close = assert_close
    tu.assert_similar_generate_outputs = assert_similar
    m.Qwen3VLMoeTextTopKRouter.forward = router_forward


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _rec.clear()


def pytest_collection_finish(session):
    _install()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        lines = str(rep.longrepr).splitlines() if rep.failed else []
        msg = next((ln for ln in lines if "max_diff" in ln or "AssertionError" in ln), "")[:300]
        rec = dict(_rec, outcome=rep.outcome, msg=msg)
        with open(os.path.join(STATS_DIR, f"{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
