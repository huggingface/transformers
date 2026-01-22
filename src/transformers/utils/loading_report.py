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
from dataclasses import dataclass
from typing import Any


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

    bucket: dict[str, list[set[int] | Any]] = defaultdict(list)
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


PALETTE = {
    "reset": "[0m",
    "red": "[31m",
    "yellow": "[33m",
    "orange": "[38;5;208m",
    "purple": "[35m",
    "bold": "[1m",
    "italic": "[3m",
    "dim": "[2m",
}


def _color(s, color):
    """Return color-formatted input `s` if `sys.stdout` is interactive, e.g. connected to a terminal."""
    if sys.stdout.isatty():
        return f"{PALETTE[color]}{s}{PALETTE['reset']}"
    else:
        return s


def _get_terminal_width(default=80):
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default


@dataclass
class LoadStateDictInfo:
    """
    Mutable container for state-dict loading results and diagnostics. Each entry in this structure is mutable,
    and will usually be mutated in-place during the loading pipeline.

    Attributes:
        missing_keys (`set[str]`):
            Keys that are missing from the loaded checkpoints but expected in the model's architecture.
        unexpected_keys (`set[str]`):
            Keys that are found in the checkpoints, but not expected in the model's architecture.
        mismatched_keys (`set[tuple[str, tuple[int], tuple[int]]]`):
            Keys that are found in the checkpoints and are expected in the model's architecture, but with a different shape.
        error_msgs ( `list[str]`):
            Some potential error messages.
        conversion_errors (`dict[str, str]`):
            Errors happening during the on-the-fly weight conversion process.
    """

    missing_keys: set[str]
    unexpected_keys: set[str]
    mismatched_keys: set[tuple[str, tuple[int], tuple[int]]]
    error_msgs: list[str]
    conversion_errors: dict[str, str]

    def missing_and_mismatched(self):
        """Return all effective missing keys, including `missing` and `mismatched` keys."""
        return self.missing_keys | {k[0] for k in self.mismatched_keys}

    def to_dict(self):
        # Does not include the `conversion_errors` to be coherent with legacy reporting in the tests
        return {
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "mismatched_keys": self.mismatched_keys,
            "error_msgs": self.error_msgs,
        }

    def create_loading_report(self) -> str | None:
        """Generate the minimal table of a loading report."""
        term_w = _get_terminal_width()

        rows = []
        tips = ""
        if self.unexpected_keys:
            tips += (
                f"\n- {_color('UNEXPECTED', 'orange') + PALETTE['italic']}\t:can be ignored when loading from different "
                "task/architecture; not ok if you expect identical arch."
            )
            for k in update_key_name(self.unexpected_keys):
                status = _color("UNEXPECTED", "orange")
                rows.append([k, status, "", ""])

        if self.missing_keys:
            tips += (
                f"\n- {_color('MISSING', 'red') + PALETTE['italic']}\t:those params were newly initialized because missing "
                "from the checkpoint. Consider training on your downstream task."
            )
            for k in update_key_name(self.missing_keys):
                status = _color("MISSING", "red")
                rows.append([k, status, ""])

        if self.mismatched_keys:
            tips += (
                f"\n- {_color('MISMATCH', 'yellow') + PALETTE['italic']}\t:ckpt weights were loaded, but they did not match "
                "the original empty weight shapes."
            )
            iterator = {a: (b, c) for a, b, c in self.mismatched_keys}
            for key, (shape_ckpt, shape_model) in update_key_name(iterator).items():
                status = _color("MISMATCH", "yellow")
                data = [
                    key,
                    status,
                    f"Reinit due to size mismatch - ckpt: {str(shape_ckpt)} vs model:{str(shape_model)}",
                ]
                rows.append(data)

        if self.conversion_errors:
            tips += f"\n- {_color('CONVERSION', 'purple') + PALETTE['italic']}\t:originate from the conversion scheme"
            for k, v in update_key_name(self.conversion_errors).items():
                status = _color("CONVERSION", "purple")
                _details = f"\n\n{v}\n\n"
                rows.append([k, status, _details])

        # If nothing is wrong, return None
        if len(rows) == 0:
            return None

        headers = ["Key", "Status"]
        if term_w > 200:
            headers += ["Details"]
        else:
            headers += ["", ""]
        table = _make_table(rows, headers=headers)
        tips = f"\n\n{PALETTE['italic']}Notes:{tips}{PALETTE['reset']}"
        report = table + tips

        return report


def log_state_dict_report(
    model,
    pretrained_model_name_or_path: str,
    ignore_mismatched_sizes: bool,
    loading_info: LoadStateDictInfo,
    logger: logging.Logger | None = None,
):
    """
    Log a readable report about state_dict loading issues.

    This version is terminal-size aware: for very small terminals it falls back to a compact
    Key | Status view so output doesn't wrap badly.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Re-raise errors early if needed
    if loading_info.error_msgs:
        error_msg = "\n\t".join(loading_info.error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` to `from_pretrained(...)` if appropriate."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    # Create the report table
    report = loading_info.create_loading_report()
    if report is None:
        return

    prelude = f"{PALETTE['bold']}{model.__class__.__name__} LOAD REPORT{PALETTE['reset']} from: {pretrained_model_name_or_path}\n"

    # Log the report as warning
    logger.warning(prelude + report)

    # Re-raise in those case, after the report
    if loading_info.conversion_errors:
        raise RuntimeError(
            "We encountered some issues during automatic conversion of the weights. For details look at the `CONVERSION` entries of "
            "the above report!"
        )
    if not ignore_mismatched_sizes and loading_info.mismatched_keys:
        raise RuntimeError(
            "You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!"
        )
