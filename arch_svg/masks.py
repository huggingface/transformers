"""Attention-mask *pattern* per layer type, generated offline.

The goal is to make the per-layer attention pattern **visible** (full-causal vs sliding vs
chunked vs compressed vs bidirectional). Real masks built from config are technically exact,
but production sliding windows (128, 4096, ...) are far larger than any sequence we can draw,
so they collapse to a plain causal triangle and become indistinguishable. We therefore draw
**schematic** patterns on a small ``seq`` grid using an illustrative window/chunk, and label
the *true* window size elsewhere. Where a real pattern is small enough to be faithful (window
< seq) the schematic coincides with it. Never raises.
"""

from __future__ import annotations


_WIN = 6  # illustrative sliding window / chunk size on the schematic grid


def _causal(seq: int) -> list[list[int]]:
    return [[1 if j <= i else 0 for j in range(seq)] for i in range(seq)]


def _full(seq: int) -> list[list[int]]:
    return [[1] * seq for _ in range(seq)]


def _sliding(seq: int, win: int) -> list[list[int]]:
    return [[1 if (0 <= i - j < win) else 0 for j in range(seq)] for i in range(seq)]


def _chunked(seq: int, chunk: int) -> list[list[int]]:
    # causal within the current chunk only
    return [[1 if (j <= i and j // chunk == i // chunk) else 0 for j in range(seq)] for i in range(seq)]


def _compressed(seq: int, win: int, stride: int = 2) -> list[list[int]]:
    # local sliding window + sparse/strided long-range keys (schematic of compressed attention)
    g = _sliding(seq, win)
    for i in range(seq):
        for j in range(0, i - win + 1, stride):
            g[i][j] = 1
    return g


def _compressed_kv(seq: int, real_rate: int, topk: int | None = None) -> list[list[int]]:
    """Schematic of attention over a *compressed KV cache*: queries (rows) attend to a
    compressed key sequence (cols). Columns = seq / display-rate (the real rate is usually too
    large to draw, so we cap it for visibility and label the true rate elsewhere). Causal over
    the compressed positions; for the sparse variant only the most-recent ``topk`` compressed
    keys are kept (a band)."""
    disp = max(2, min(real_rate, 6))
    cols = max(2, -(-seq // disp))  # ceil
    grid = []
    for i in range(seq):
        ci = i // disp  # compressed position the query has reached
        row = [1 if c <= ci else 0 for c in range(cols)]
        if topk:  # sparse: keep only the most-recent compressed keys (schematic band/top-k)
            for c in range(cols):
                if row[c] and (ci - c) >= max(2, cols // 2):
                    row[c] = 0
        grid.append(row)
    return grid


# a fixed example sentence so mask figures read like the reference (token-labelled axes)
EXAMPLE_WORDS = [
    "The",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
    "ate",
    "a",
    "small",
    "red",
    "fish",
    "this",
    "morning",
]


def concat_compressed_mask(seq: int, m: int, topk: int | None = None):
    """Reproduce the *actual* mask DeepSeek-V4 passes to attention: the sliding K/V cache
    causal mask (seq×seq) concatenated with the compressed-cache block bias (seq×n_comp).

    Mirrors ``DeepseekV4HCACompressor.forward`` (block_bias) + the local sliding window:
    query t attends key j iff ``0 ≤ t-j < m`` (recent window), and compressed entry w iff
    ``w < (t+1)//m`` (the window is closed/ready). Returns ``(grid, split_col, n_comp)`` where
    columns ``[0:seq)`` are the sliding cache and ``[seq:seq+n_comp)`` the compressed cache."""
    n_comp = max(seq // m, 0)
    grid = []
    for t in range(seq):
        left = [1 if (j <= t and t - j < m) else 0 for j in range(seq)]
        thr = (t + 1) // m
        right = [1 if w < thr else 0 for w in range(n_comp)]
        grid.append(left + right)
    return grid, seq, n_comp


def image_text_mask(n_img: int = 6, n_text: int = 8, image_bidirectional: bool = True):
    """The cross-modal attention mask for a VLM: ``n_img`` image (prefix) tokens followed by
    ``n_text`` text tokens. Text attends to ALL image tokens + causal text; image tokens are
    bidirectional among themselves (prefix-LM, e.g. PaliGemma) or causal if not. Returns
    ``(grid, split)`` with ``split = n_img`` (the image|text divider)."""
    n = n_img + n_text
    grid = []
    for i in range(n):
        row = []
        for j in range(n):
            if i < n_img:  # image query
                attend = (j < n_img) if image_bidirectional else (j <= i)
            else:  # text query: all image (prefix) + causal text
                attend = (j < n_img) or (n_img <= j <= i)
            row.append(1 if attend else 0)
        grid.append(row)
    return grid, n_img


def attention_pattern_grids(tcfg, layer_types: list[str], seq: int = 24) -> dict[str, list[list[int]]]:
    """Return ``{layer_type: grid}`` (schematic). grid[i][j]==1 ⇒ query i attends to key j.

    Sliding/chunked are square (q×q). Compressed/heavily-compressed are RECTANGULAR (q ×
    compressed-kv) to show that the KV cache is compressed -- the key point for DeepSeek-V4."""
    types = list(dict.fromkeys(layer_types or ["full_attention"]))
    rates = getattr(tcfg, "compress_rates", None) or {}
    topk = getattr(tcfg, "index_topk", None)
    out: dict[str, list[list[int]]] = {}
    for lt in types:
        n = lt.lower()
        try:
            if "compress" in n:  # compressed / heavily_compressed -> compressed KV cache
                r = rates.get(lt, 4) if isinstance(rates, dict) else 4
                out[lt] = _compressed_kv(seq, int(r), topk if "sparse" in n else None)
            elif "chunk" in n:
                out[lt] = _chunked(seq, _WIN)
            elif "sliding" in n:
                out[lt] = _sliding(seq, _WIN)
            elif "linear" in n or "delta" in n or "mamba" in n or "recurrent" in n:
                out[lt] = _causal(seq)
            elif "bidirectional" in n:
                out[lt] = _full(seq)
            else:
                out[lt] = _causal(seq)
        except Exception:
            continue
    return out
