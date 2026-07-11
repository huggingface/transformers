"""arch_svg - introspect any `transformers` model and emit an architecture diagram as SVG.

The headline feature is exploiting *modular* transformers: a model defined as a
``modular_<name>.py`` that inherits from another model can be rendered in ``diff`` mode,
showing only what it changes relative to its parent(s). A clean modular model produces a
tiny, legible diff -- a direct measure of how modular the library really is.

See ``python -m arch_svg --help``.
"""

from .introspect import ArchModel, LayerBlock, introspect
from .modular import ArchDiff, ClassChange, diff, resolve_parent


__all__ = [
    "ArchModel",
    "LayerBlock",
    "introspect",
    "ArchDiff",
    "ClassChange",
    "diff",
    "resolve_parent",
]
