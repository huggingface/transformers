# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
import shutil
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from typing import Any, Optional


_DIGIT_RX = re.compile(r"(?<=\.)(\d+)(?=\.|$)")  # numbers between dots or at the end


def _pattern_of(key: str) -> str:
    """Replace every dot-delimited integer with '*' to get the structure."""
    return _DIGIT_RX.sub("*", key)


def _fmt_indices(values: list[int], cutoff=10) -> str:
    """Format a list of ints as single number, {a, ..., b}, or first...last."""
    if len(values) == 1:
        return str(values[0])
    values = sorted(values)
    if len(values) > cutoff:
        return f"{values[0]}...{values[-1]}"
    return ", ".join(map(str, values))


def update_key_name(mapping: dict[str, Any]) -> dict[str, Any]:
    """
    Merge keys like 'layers.0.x', 'layers.1.x' into 'layers.{0, 1}.x'
    BUT only merge together keys that have the exact same value.
    Returns a new dict {merged_key: value}.
    """
    # (pattern, value) -> list[set[int]] (per-star index values)
    not_mapping = False
    if not isinstance(mapping, dict):
        mapping = {k: k for k in mapping}
        not_mapping = True

    bucket: dict[tuple[str, Any], list[set[int]]] = defaultdict(list)
    for key, val in mapping.items():
        digs = _DIGIT_RX.findall(key)
        patt = _pattern_of(key)
        for i, d in enumerate(digs):
            if len(bucket[patt]) <= i:
                bucket[patt].append(set())
            bucket[patt][i].add(int(d))
        bucket[patt].append(val)

    out_items = {}
    for patt, values in bucket.items():
        sets, val = values[:-1], values[-1]
        parts = patt.split("*")  # stars are between parts
        final = parts[0]
        for i in range(1, len(parts)):
            if i - 1 < len(sets) and sets[i - 1]:
                insert = _fmt_indices(sorted(sets[i - 1]))
                if len(sets[i - 1]) > 1:
                    final += "{" + insert + "}"
                else:
                    final += insert
            else:
                final += "*"
            final += parts[i]

        out_items[final] = val
    out = OrderedDict(out_items)
    if not_mapping:
        return out.keys()
    return out


# We have a class to simplify disabling ANSI colors
class ANSI:
    palette = {
        "reset": "[0m",
        "red": "[31m",
        "yellow": "[33m",
        "orange": "[38;5;208m",
        "purple": "[35m",
        "bold": "[1m",
        "italic": "[3m",
        "dim": "[2m",
    }

    def __init__(self, enable):
        self.enable = enable

    def __getitem__(self, key):
        return self.palette[key] if self.enable else ""


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
    ignore_mismatched_sizes=True,
    misc=None,
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
            rows.append([k, status, ""])

    if mismatched_keys:
        iterator = {a: (b, c) for a, b, c in mismatched_shapes}
        for key, (shape_ckpt, shape_model) in update_key_name(iterator).items():
            status = "MISMATCH"
            status = _color(status, "yellow", ansi)
            data = [key, status]
            data.append(
                " ".join(["Reinit due to size mismatch", f"ckpt: {str(shape_ckpt)} vs model:{str(shape_model)}"])
            )
            rows.append(data)

    if misc:
        for k, v in update_key_name(misc).items():
            status = "MISC"
            status = _color(status, "purple", ansi)
            _details = v[:term_w]
            rows.append([k, status, _details])

    if not rows:
        return

    headers = ["Key", "Status"]
    if term_w > 200:
        headers += ["Details"]
    else:
        headers += ["", ""]
    table = _make_table(rows, headers=headers)

    prelude = (
        f"{ansi['bold']}{model.__class__.__name__} LOAD REPORT{ansi['reset']} from: {pretrained_model_name_or_path}\n"
    )
    tips = f"\n\n{ansi['italic']}Notes:"
    if unexpected_keys:
        tips += f"\n- {_color('UNEXPECTED', 'orange', ansi) + ansi['italic']}\t:can be ignored when loading from different task/architecture; not ok if you expect identical arch."
    if missing_keys:
        tips += f"\n- {_color('MISSING', 'red', ansi) + ansi['italic']}\t:those params were newly initialized because missing form the checkpoint. Consider training on your downstream task."
    if mismatched_keys:
        tips += f"\n- {_color('MISMATCH', 'yellow', ansi) + ansi['italic']}\t:ckpt weights were loaded, but they did not match the original empty weight."
    if misc:
        tips += f"\n- {_color('MISC', 'purple', ansi) + ansi['italic']}\t:originate from the conversion scheme"
    tips += f"{ansi['reset']}"

    logger.warning(prelude + table + tips)
    if not ignore_mismatched_sizes and mismatched_keys:
        raise RuntimeError(
            "You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!"
        )
    return prelude + table + tips
