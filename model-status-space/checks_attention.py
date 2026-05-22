import os
import re
import copy
import torch
import transformers
from transformers import AutoConfig
from helpers import HF_TOKEN, get_config_and_class

__all__ = ["check_attention_support"]

# Attention types to probe, in display order
ATTN_TYPES = ["eager", "sdpa", "flash_attention_2", "flex_attention"]
# Map each type to the model-class flag that signals support
ATTN_FLAG = {
    "eager": None,               # always supported, no flag needed
    "sdpa": "_supports_sdpa",
    "flash_attention_2": "_supports_flash_attn",   # v5 name (v4 used _supports_flash_attn_2)
    "flex_attention": "_supports_flex_attn",
}


def _attn_flags_from_source(model_type: str) -> dict[str, bool]:
    """
    Read _supports_* flags by grepping the model's source file — no import needed.
    Falls back to an empty dict if the file can't be read.
    """
    flag_names = [
        "_supports_sdpa", "_supports_flash_attn",
        "_supports_flash_attn_2", "_supports_flex_attn",
    ]
    model_file = os.path.join(
        os.path.dirname(transformers.__file__),
        "models", model_type, f"modeling_{model_type}.py",
    )
    try:
        source = open(model_file).read()
        return {
            f: bool(re.search(rf"\b{re.escape(f)}\s*=\s*True\b", source))
            for f in flag_names
        }
    except Exception:
        return {}


def check_attention_support(model_id: str) -> tuple[str, str]:
    """
    For each attention type (eager / sdpa / flash_attention_2 / flex_attention):
      1. Static: flag declared in the model class (or source file if import fails).
      2. Dynamic: construct on torch.device("meta") with that attn_implementation —
         runs the same validation as from_pretrained without loading weights.
         Skipped gracefully when the model class cannot be imported.
    """
    summary, details = [], []

    try:
        config, model_cls, arch = get_config_and_class(model_id)
    except Exception as e:
        return f"❌ Config: `{e}`", ""

    details.append(f"**Architecture:** `{arch}`\n")

    # Source-file flags are the authoritative fallback when the class can't be imported
    src_flags = _attn_flags_from_source(config.model_type)
    has_dynamic = model_cls is not None

    if not has_dynamic:
        details.append("_Model class could not be imported — showing static flags only._\n")

    for attn_type in ATTN_TYPES:
        flag_name = ATTN_FLAG[attn_type]

        # ── 1. Static flag ──────────────────────────────────────────────────
        if flag_name is None:
            static_ok = True  # eager is always supported
        elif model_cls is not None:
            static_ok = bool(
                getattr(model_cls, flag_name, False)
                or (attn_type == "flash_attention_2" and getattr(model_cls, "_supports_flash_attn_2", False))
            )
        else:
            static_ok = bool(
                src_flags.get(flag_name, False)
                or (attn_type == "flash_attention_2" and src_flags.get("_supports_flash_attn_2", False))
            )

        if not static_ok:
            summary.append(f"⚫ {attn_type}")
            details.append(f"**`{attn_type}`:** ⚫ not declared")
            continue

        # ── 2. Dynamic: meta-device construction ────────────────────────────
        if not has_dynamic:
            summary.append(f"✅ {attn_type}")
            details.append(f"**`{attn_type}`:** ✅ declared (dynamic test skipped — import unavailable)")
            continue

        try:
            cfg = copy.deepcopy(config)
            cfg._attn_implementation = attn_type
            with torch.device("meta"):
                model_cls(cfg)
            summary.append(f"✅ {attn_type}")
            details.append(f"**`{attn_type}`:** ✅ model construction OK")
        except (ImportError, ModuleNotFoundError) as e:
            summary.append(f"⚠️ {attn_type}")
            details.append(f"**`{attn_type}`:** ⚠️ missing dependency — `{e}`")
        except ValueError as e:
            summary.append(f"❌ {attn_type}")
            details.append(f"**`{attn_type}`:** ❌ `{str(e)[:300]}`")
        except Exception as e:
            summary.append(f"❌ {attn_type}")
            details.append(f"**`{attn_type}`:** ❌ unexpected — `{str(e)[:300]}`")

    return " &nbsp; ".join(summary), "\n\n".join(details)
