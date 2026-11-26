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
CLOSEST_MIN_RATIO = float(os.environ.get("Q3_CLOSEST_MIN_RATIO", "1e-3"))
# entries (layer, row, position, step) with delta/gap above this are all listed (compact)
ENTRY_MIN_RATIO = float(os.environ.get("Q3_ENTRY_MIN_RATIO", "1e-2"))
# Q3_RANDOM_SEEDS=1: replace the test's `set_seed(42)` by a random (recorded) weights seed, and seed the testers'
# `global_rng` (input ids / pixel values) with a random (recorded) input seed -> any run can be replayed exactly
RANDOM_SEEDS = os.environ.get("Q3_RANDOM_SEEDS") == "1"
os.makedirs(STATS_DIR, exist_ok=True)
_rec = {}
_st = {"phase": None, "ctx": None, "layers": {}, "logits": {}, "mask": None}


def _reset():
    _rec.clear()
    _st.update(
        phase=None, ctx=None, layers={}, logits={"cached": {}, "cacheless": {}},
        extra={"cached": {}, "cacheless": {}}, mask=None,
    )


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
            # The same softmax call as the router, on the same `router_logits` tensor -> bit-identical to what `topk` saw
            probs = torch.nn.functional.softmax(out[0], dtype=torch.float, dim=-1)
            _st["extra"][_st["phase"]].setdefault(layer, []).append(
                {
                    "probs": probs.detach().cpu().view(b, s, -1),
                    "idx": out[2].detach().cpu().view(b, s, -1),  # the experts the router actually selected
                    "w": out[1].detach().float().cpu().view(b, s, -1),  # their normalized routing weights
                    "h": hidden_states.detach().float().cpu().reshape(b, s, -1),  # the router's input
                }
            )
        return out

    orig_set_seed = tu.set_seed

    def set_seed(seed, *args, **kwargs):
        if RANDOM_SEEDS and "weights_seed" in _rec:
            seed = _rec["weights_seed"]
        _rec["set_seed_used"] = seed
        return orig_set_seed(seed, *args, **kwargs)

    torch.testing.assert_close = assert_close
    tu.assert_similar_generate_outputs = assert_similar
    tu.set_seed = set_seed
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
    closest = None  # the single (layer, row, position, step) entry with the largest delta/gap among relevant positions
    entries, flips = [], []  # entries: delta/gap > ENTRY_MIN_RATIO (compact); flips: every top-k flip (full detail)
    for layer, calls in cached.items():
        full = torch.cat(calls, dim=1)  # cached: prefill + one position per decode step -> all positions
        out["prompt_len"] = calls[0].shape[1]  # the prefill call covers the prompt; later positions are generated
        out["n_positions"] = full.shape[1]
        for step_idx, step_logits in enumerate(cacheless.get(layer, [])):  # cacheless: recomputes 0..L-1 each step
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
            for row, pos in ((ratio > ENTRY_MIN_RATIO) | flip).nonzero().tolist():
                entry = {"layer": layer, "row": row, "position": pos, "cacheless_step": step_idx,
                         "ratio": ratio[row, pos].item(), "delta": delta[row, pos].item(),
                         "gap": gap[row, pos].item(), "flip": bool(flip[row, pos]), "padding": not bool(valid[row, pos])}
                entries.append(entry)
                if flip[row, pos]:
                    detail = dict(entry)
                    for name, t in (("cached", a), ("cacheless", b)):
                        top = t[row, pos].topk(k + 1)
                        detail[f"top{k + 1}_{name}"] = [round(v, 9) for v in top.values.tolist()]
                        detail[f"top{k + 1}_experts_{name}"] = top.indices.tolist()
                    flips.append(detail)
            for key, sel in (("rel", valid), ("pad", ~valid)):
                if sel.any():
                    o = out[key]
                    o["n"] += int(sel.sum())
                    o["max_ratio"] = max(o["max_ratio"], ratio[sel].max().item())
                    o["flips"] += int(flip[sel].sum())
                    if key == "rel":
                        o["max_delta"] = max(o["max_delta"], delta[sel].max().item())
                        o["min_gap"] = min(o["min_gap"], gap[sel].min().item())
                        masked = torch.where(sel, ratio, torch.full_like(ratio, -1.0))
                        idx = int(masked.argmax())
                        row, pos = divmod(idx, masked.shape[1])
                        if closest is None or masked[row, pos].item() > closest["ratio"]:
                            closest = {
                                "ratio": masked[row, pos].item(), "layer": layer, "row": row, "position": pos,
                                "cacheless_step": step_idx, "delta": delta[row, pos].item(),
                                "gap_cached": (sa[row, pos, k - 1] - sa[row, pos, k]).item(),
                                "gap_cacheless": (sb[row, pos, k - 1] - sb[row, pos, k]).item(),
                                "flip": bool(flip[row, pos]),
                            }
                            for name, t in (("cached", a), ("cacheless", b)):
                                top = t[row, pos].topk(k + 1)
                                closest[f"top{k + 1}_{name}"] = [round(v, 9) for v in top.values.tolist()]
                                closest[f"top{k + 1}_experts_{name}"] = top.indices.tolist()
    # Only keep the details of near-misses: they are what we want to look at, and it keeps the files small
    if closest is not None and closest["ratio"] > CLOSEST_MIN_RATIO:
        out["closest"] = closest
    if entries:
        out["entries"] = entries
    if flips:
        out["flip_details"] = flips
    return out


def _ulps_gap(sorted_probs, k):
    """4th-5th probability gap in float32 steps (ULPs) at the 4th value: 0 = exact tie."""
    hi, lo = sorted_probs[..., k - 1], sorted_probs[..., k]
    step = torch.nextafter(hi, torch.full_like(hi, float("inf"))) - hi
    return (hi - lo) / step


def _prob_path():
    """Compare what `topk` actually saw (softmax probabilities) and selected, cached vs cacheless generate."""
    cached, cacheless, k = _st["extra"]["cached"], _st["extra"]["cacheless"], _st.get("top_k")
    if not cached or not cacheless or k is None:
        return None
    res = {
        "n": 0, "ties_cached": 0, "ties_cacheless": 0, "ties_both": 0, "ties_both_same_selection": 0,
        "tie_entries": 0, "tie_lowest_index_selected": 0, "actual_flips": 0,
        "hist_min_gap_ulps": {"0": 0, "1": 0, "2-4": 0, "5-16": 0, ">16": 0},
        "max_weight_diff_same_selection": 0.0, "max_hidden_delta": 0.0, "max_abs_hidden": 0.0,
    }
    closest, close_entries, flip_details = None, [], []
    mask = _st["mask"]
    for layer, calls in cached.items():
        P, I, W, H = (torch.cat([c[key] for c in calls], 1) for key in ("probs", "idx", "w", "h"))
        for step_idx, c2 in enumerate(cacheless.get(layer, [])):
            n = min(P.shape[1], c2["probs"].shape[1])
            pa, pb, ia, ib = P[:, :n], c2["probs"][:, :n], I[:, :n], c2["idx"][:, :n]
            wa, wb, ha, hb = W[:, :n], c2["w"][:, :n], H[:, :n], c2["h"][:, :n]
            valid = torch.ones(pa.shape[:2], dtype=torch.bool)
            if mask is not None and mask.shape[0] == pa.shape[0]:
                m_len = min(mask.shape[1], n)
                valid[:, :m_len] = mask[:, :m_len].bool()
            spa, spb = pa.sort(-1, descending=True).values, pb.sort(-1, descending=True).values
            ua, ub = _ulps_gap(spa, k), _ulps_gap(spb, k)
            umin = torch.minimum(ua, ub)
            tie_a, tie_b = spa[..., k - 1] == spa[..., k], spb[..., k - 1] == spb[..., k]
            flip = (ia.sort(-1).values != ib.sort(-1).values).any(-1)  # the experts actually used differ
            # sanity: the recomputed probabilities must reproduce the router's own selection everywhere
            for p, i in ((pa, ia), (pb, ib)):
                res["recompute_mismatch"] = res.get("recompute_mismatch", 0) + int(
                    (p.topk(k, -1).indices.sort(-1).values != i.sort(-1).values).any(-1).sum()
                )
            wdiff = (wa.gather(-1, ia.argsort(-1)) - wb.gather(-1, ib.argsort(-1))).abs().amax(-1)  # by expert id
            hdelta, habs = (ha - hb).abs().amax(-1), ha.abs().amax(-1)
            sel = valid
            res["n"] += int(sel.sum())
            res["ties_cached"] += int((tie_a & sel).sum())
            res["ties_cacheless"] += int((tie_b & sel).sum())
            res["ties_both"] += int((tie_a & tie_b & sel).sum())
            res["ties_both_same_selection"] += int((tie_a & tie_b & ~flip & sel).sum())
            res["actual_flips"] += int((flip & sel).sum())
            h = res["hist_min_gap_ulps"]
            u = umin[sel]
            h["0"] += int((u == 0).sum())
            h["1"] += int(((u > 0) & (u <= 1.5)).sum())
            h["2-4"] += int(((u > 1.5) & (u <= 4.5)).sum())
            h["5-16"] += int(((u > 4.5) & (u <= 16.5)).sum())
            h[">16"] += int((u > 16.5).sum())
            if (~flip & sel).any():
                res["max_weight_diff_same_selection"] = max(
                    res["max_weight_diff_same_selection"], wdiff[~flip & sel].max().item()
                )
            res["max_hidden_delta"] = max(res["max_hidden_delta"], hdelta[sel].max().item())
            res["max_abs_hidden"] = max(res["max_abs_hidden"], habs[sel].max().item())

            def detail(row, pos):
                d = {
                    "layer": layer, "row": row, "position": pos, "cacheless_step": step_idx,
                    "gap_ulps_cached": ua[row, pos].item(), "gap_ulps_cacheless": ub[row, pos].item(),
                    "tie_cached": bool(tie_a[row, pos]), "tie_cacheless": bool(tie_b[row, pos]),
                    "flip": bool(flip[row, pos]), "padding": not bool(valid[row, pos]),
                    "weight_diff": wdiff[row, pos].item(), "hidden_delta": hdelta[row, pos].item(),
                }
                for name, p, i in (("cached", pa, ia), ("cacheless", pb, ib)):
                    top = p[row, pos].topk(k + 1)
                    d[f"top{k + 1}_probs_{name}"] = top.values.tolist()  # exact float32 values
                    d[f"top{k + 1}_experts_{name}"] = top.indices.tolist()
                    d[f"selected_{name}"] = sorted(i[row, pos].tolist())
                return d

            for row, pos in ((umin <= 16.5) | flip).nonzero().tolist():
                d = detail(row, pos)
                close_entries.append({key: d[key] for key in (
                    "layer", "row", "position", "cacheless_step", "gap_ulps_cached", "gap_ulps_cacheless",
                    "tie_cached", "tie_cacheless", "flip", "padding")})
                if d["flip"]:
                    flip_details.append(d)
                for name, p, i, tie in (("cached", pa, ia, tie_a), ("cacheless", pb, ib, tie_b)):
                    if tie[row, pos] and valid[row, pos]:  # which of the tied experts did `topk` keep?
                        boundary = p[row, pos].sort(descending=True).values[k - 1]
                        tied = (p[row, pos] == boundary).nonzero().flatten().tolist()
                        res["tie_entries"] += 1
                        res["tie_lowest_index_selected"] += int(min(tied) in i[row, pos].tolist())
            masked = torch.where(sel, umin, torch.full_like(umin, float("inf")))
            idx = int(masked.argmin())
            row, pos = divmod(idx, masked.shape[1])
            if masked[row, pos].isfinite() and (closest is None or masked[row, pos].item() < closest["min_ulps"]):
                closest = dict(detail(row, pos), min_ulps=masked[row, pos].item())
    if closest is not None and closest["min_ulps"] <= 64:
        res["closest"] = closest
    if close_entries:
        res["close_entries"] = close_entries
    if flip_details:
        res["flip_details"] = flip_details
    return res


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _reset()
    if RANDOM_SEEDS:
        import tests.test_modeling_common as tcm

        forced = os.environ.get("Q3_FORCE_SEEDS")  # "<weights_seed>,<input_seed>": replay one recorded run
        if forced:
            _rec["weights_seed"], _rec["input_seed"] = (int(x) for x in forced.split(","))
        else:
            _rec["weights_seed"] = int.from_bytes(os.urandom(4), "little")
            _rec["input_seed"] = int.from_bytes(os.urandom(4), "little")
        tcm.global_rng.seed(_rec["input_seed"])


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
        try:
            rec["prob"] = _prob_path()
        except Exception as e:
            rec["prob_error"] = repr(e)[:200]
        with open(os.path.join(STATS_DIR, f"{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
