"""Tiny self-contained layout primitives — our own engine, no external deps.

The point of this module is **text-aware sizing**: every other part of the package can ask
"how wide is this string at this font size?" and "make this string fit in this width", so
boxes are sized to their content and text never overflows or runs together. Kept dependency-
free and deterministic (a fixed proportional-width table) so output stays reproducible.
"""

from __future__ import annotations


# per-character width as a fraction of the font size, for a generic sans-serif.
# Approximate but stable; good enough to size/ellipsize boxes so text fits.
_NARROW = set("iIl.,:;'!|()[]{}ftj ")
_WIDE = set("mMW@%—–→⊙⊕")
_MID = set("rtszc")


def char_em(ch: str) -> float:
    if ch in _NARROW:
        return 0.30
    if ch in _WIDE:
        return 0.92
    if ch in _MID:
        return 0.48
    if ch.isupper() or ch.isdigit():
        return 0.62
    return 0.54


def text_width(s: str, px: float) -> float:
    """Estimated rendered width (in px) of ``s`` at font size ``px``."""
    return sum(char_em(c) for c in s) * px


def fit_text(s: str, max_px: float, px: float) -> str:
    """Return ``s`` unchanged if it fits in ``max_px``, else truncated with a trailing ellipsis."""
    if max_px <= 0 or text_width(s, px) <= max_px:
        return s
    ell = "…"
    budget = max_px - text_width(ell, px)
    out = []
    w = 0.0
    for c in s:
        cw = char_em(c) * px
        if w + cw > budget:
            break
        out.append(c)
        w += cw
    return ("".join(out).rstrip() + ell) if out else ell


def wrap_text(s: str, max_px: float, px: float, max_lines: int = 2) -> list[str]:
    """Greedily wrap ``s`` to ``max_px`` over up to ``max_lines`` lines (last line ellipsized)."""
    if text_width(s, px) <= max_px:
        return [s]
    words = s.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_width(trial, px) <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines - 1:
                break
    rest = " ".join(words[sum(len(line.split(" ")) for line in lines) :]) if lines else cur
    lines.append(fit_text(rest or cur, max_px, px))
    return lines[:max_lines]
