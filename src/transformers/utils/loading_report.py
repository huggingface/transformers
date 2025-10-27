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


def _get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default



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
    ansi = ANSI(color_enabled)

    if error_msgs:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    term_w = _get_terminal_width()
    rows = []
    if unexpected_keys:
        for k in update_key_name(unexpected_keys):
            status = "UNEXPECTED"
            status = _color(status, "orange", ansi)
            rows.append([k, status, "", ""])

    if missing_keys:
        for k in update_key_name(missing_keys):
            status = "MISSING"
            status = _color(status, "red", ansi)
            rows.append([k, status, "", ""])

    if mismatched_keys:
        for key, shape_ckpt, shape_model in mismatched_shapes:
            status = "MISMATCH"
            status = _color(status, "yellow", ansi)
            data = [key, status]
            if term_w > 200:
                data.append(
                    [ "Reinit due to size mismatch", f"ckpt: {str(shape_ckpt)} vs model:{str(shape_model)}"]
                )
            rows.append(data)

    if misc:
        for k in misc:
            status = "MISC"
            status = _color(status, "red", ansi)
            if term_w > 200:
                _details = misc[k]
            else:
                _details = None
            rows.append([k, status, _details, ""])

    if not rows:
        print(
            f"No key issues when initializing {model.__class__.__name__} from {pretrained_model_name_or_path}."
        )
        return


    headers = ["Key", "Status"]
    if term_w > 200:
        headers += ["Checkpoint shape", "Details"]
    else:
        headers += ["", ""]
    table = _make_table(rows, headers=headers)

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
