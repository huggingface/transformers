import re
import shutil
import logging
import sys
from typing import Iterable, Optional


class ANSI:
    palette = {
        'reset': '[0m', 'red':'[31m','yellow':'[33m','orange':'[38;5;208m','bold':'[1m','italic':'[3m','dim':'[2m'}
    def __init__(self, enable): self.enable=enable
    def __getitem__(self,key): return self.palette[key] if self.enable else ''

_ansi_re = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ansi_re.sub("", str(s))


def _pad(text, width):
    t = str(text)
    pad = max(0, width - len(_strip_ansi(t)))
    return t + " " * pad


def _make_table(rows, headers):
    # compute display widths while ignoring ANSI codes
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(_strip_ansi(x)) for x in col) for col in cols]
    header_line = " | ".join(_pad(h, w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    body = [" | ".join(_pad(c, w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join([header_line, sep_line] + body)


def _color(s, color, ansi):
    # ansi returns empty strings when disabled, so safe to interpolate
    return f"{ansi[color]}{s}{ansi['reset']}"


def _chunk(items, limit=200):
    it = list(items)
    if len(it) <= limit:
        return it, 0
    return it[:limit], len(it) - limit


def _get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


def _build_compact_table(rows, max_width):
    """Build a compact 2-column table: Key | Status
    Truncate keys if they're too long for the terminal width.
    """
    headers = ["Key", "Status"]
    # compute max status width (strip ANSI)
    status_width = max(len(_strip_ansi(r[1])) for r in rows) if rows else len(headers[1])
    # allocate remaining space to key (allow 3 chars for separator and padding)
    key_max = max(10, max_width - status_width - 3)

    compact_rows = []
    for r in rows:
        key = r[0]
        key_plain = _strip_ansi(key)
        if len(key_plain) > key_max:
            # keep start and end for readability
            keep = max(0, key_max - 3)
            key = key_plain[: keep // 2] + "..." + key_plain[-(keep - keep // 2) :]
        compact_rows.append([key, r[1]])

    return _make_table(compact_rows, headers)


def log_state_dict_report(
    *,
    model,
    pretrained_model_name_or_path,
    logger: Optional[logging.Logger] = None,
    error_msgs: Optional[Iterable[str]] = None,
    unexpected_keys=None,
    missing_keys=None,
    mismatched_keys=None,
    mismatched_shapes=None,
    misc=None,
    update_key_name=lambda x: x,  # keep your mapper
    limit_rows=200,  # safety for huge checkpoints
    color=True,  # allow disabling for plain logs
    min_width_full_table=60,  # terminal min width to attempt full table
):
    """Log a readable report about state_dict loading issues.

    This version is terminal-size aware: for very small terminals it falls back to a compact
    Key | Status view so output doesn't wrap badly.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    error_msgs = error_msgs or []
    unexpected_keys = unexpected_keys or []
    missing_keys = missing_keys or []
    mismatched_keys = mismatched_keys or []
    mismatched_shapes = mismatched_shapes or []
    misc = misc or {}

    # Detect whether the current stdout supports ANSI colors; allow callers to pass `color=False` to force no color
    color_enabled = bool(color and sys.stdout.isatty())
    # instantiate simple ANSI accessor that returns empty strings when disabled
    ansi = ANSI(color_enabled)

    if error_msgs:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    rows = []
    if unexpected_keys:
        keys, extra = _chunk(list(update_key_name(unexpected_keys)), limit_rows)
        for k in keys:
            status = "UNEXPECTED"
            status = _color(status, "orange", ansi)
            rows.append([k, status, "", ""])

    if missing_keys:
        keys, extra = _chunk(list(update_key_name(missing_keys)), max(0, limit_rows - len(rows)))
        for k in keys:
            status = "MISSING"
            status = _color(status, "red", ansi)
            rows.append([k, status, "", ""])

    if mismatched_keys:
        remaining = max(0, limit_rows - len(rows))
        pairs = list(zip(mismatched_keys, mismatched_shapes))
        pairs, extra = _chunk(pairs, remaining if remaining else len(pairs))
        for key, (shape_ckpt, shape_model) in pairs:
            status = "MISMATCH"
            status = _color(status, "yellow", ansi)
            rows.append(
                [key, status, "Reinit due to size mismatch", f"ckpt: {str(shape_ckpt)} vs model:{str(shape_model)}"]
            )

    if misc:
        for k in misc:
            status = "MISC"
            status = _color(status, "red", ansi)
            rows.append([k, status, misc[k], ""])

    if not rows:
        print(
            f"No key issues when initializing {model.__class__.__name__} from {pretrained_model_name_or_path}."
        )
        return

    # Determine terminal width and whether to print full table
    term_w = _get_terminal_width()

    headers = ["Key", "Status", "Checkpoint shape", "Details"]

    # if terminal is very tiny, use a simple one-line-per-entry format
    if term_w < 20:
        # Extremely small terminals: print `key: status` per line, truncating keys to fit.
        lines = []
        for r in rows:
            key_plain = _strip_ansi(r[0])
            status_plain = _strip_ansi(r[1])
            # reserved space for ": " and at least 4 chars of status
            allowed_key = max(1, term_w - len(status_plain) - 3)
            if len(key_plain) > allowed_key:
                key_plain = key_plain[: max(0, allowed_key - 3)] + "..."
            lines.append(f"{key_plain}: {r[1]}")
        table = "".join(lines)

    # if terminal is narrow, fall back to a compact Key | Status table
    if term_w < min_width_full_table:
        # Build compact rows with only first two columns
        compact_rows = [[r[0], r[1]] for r in rows]
        table = _build_compact_table(compact_rows, max_width=term_w)

    else:
        # attempt full table; but if it would exceed terminal width, fall back to compact
        table = _make_table([[r[0], r[1], r[2] if len(r) > 2 else "", r[3] if len(r) > 3 else ""] for r in rows], headers)
        # quick width check: the first line length (header) must fit
        first_line = table.splitlines()[0]
        if len(_strip_ansi(first_line)) > term_w:
            compact_rows = [[r[0], r[1]] for r in rows]
            table = _build_compact_table(compact_rows, max_width=term_w)

    prelude = (
        f"{ansi['bold']}{model.__class__.__name__} LOAD REPORT{ansi['reset']} from: {pretrained_model_name_or_path}\n"
    )
    tips = (
        f"{ansi['italic']}Notes:\n"
        f"- {_color('UNEXPECTED', 'orange', ansi) + ansi['italic']}: can be ignored when loading from different task/architecture; not ok if you expect identical arch.\n"
        f"- {_color('MISSING', 'red', ansi) + ansi['italic']}: those params were newly initialized; consider training on your downstream task.\n"
        f"- {_color('MISMATCH', 'yellow', ansi) + ansi['italic']}: if intentional, use appropriate reinit/resize logic.\n"
        f"- {_color('MISC', 'yellow', ansi) + ansi['italic']}: originate from the conversion scheme\n"
        f"{ansi['reset']}"
    )

    print(prelude + table + "" + tips)