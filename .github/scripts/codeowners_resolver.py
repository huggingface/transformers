# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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

"""A shared module to determine who should review a given file, using `.github/scripts/codeowners_for_review_action`.

Shared by the reviewer-assignment workflow (`assign_reviewers.py`, which resolves against the PR
being opened) and by the repository check (`utils/check_reviewers.py`, which resolves against the
whole tree). Reading files is injected so both can use their own source: the workflow prefers the
PR head for files it touches, the check reads the working tree.

Resolution order for a file, most specific first:

1. `# Reviewers: @a @b` in the leading comment block of the model's `modular_*.py`/`modeling_*.py`
2. a path rule in the codeowners file
3. the model's modality, from its section in the doc toctree
4. the `*` catch-all
"""

import re
from pathlib import Path


TOCTREE_PATH = "docs/source/en/_toctree.yml"
CODEOWNERS_PATH = ".github/scripts/codeowners_for_review_action"
MODELS_DIR = "src/transformers/models"

CATCH_ALL_PATTERN = "*"
MODEL_PATH_RE = re.compile(r"^/?src/transformers/models/([^/]+)/")
REVIEWERS_TAG_RE = re.compile(r"^#\s*Reviewers?\s*:\s*(.+)$", re.IGNORECASE)

# A model file with no rule of its own is routed by modality, so a new model does not need a line
# in the codeowners file to reach someone. The modality is the model's section in the doc toctree,
# which every model must appear in (`utils/check_doc_toc.py::ensure_all_models_in_toctree`).
MODALITY_PREFIX = "@@modality/"
MODALITY_SECTIONS = {
    "Text models": "text",
    "Vision models": "vision",
    "Audio models": "audio",
    "Video models": "video",
    "Multimodal models": "multimodal",
    "Reinforcement learning models": "reinforcement-learning",
    "Time series models": "time-series",
    "Graph models": "graph",
}
# A directory whose name extends a doc page's name is a variant of that model (`dinov3_vit` ->
# `dinov3`, `wav2vec2_with_lm` -> `wav2vec2`) and shares its modality. Only used when there is no
# exact match, and only for a prefix long enough not to collide by accident.
MIN_DOC_PREFIX = 4

_TOCTREE_CACHE = {}


def read_local_file(path):
    # Contents of `path` in the working tree, or "" when it does not exist.
    local = Path(path)
    return local.read_text(encoding="utf-8") if local.exists() else ""


def pattern_to_regex(pattern):
    if pattern.startswith("/"):
        start_anchor = True
        pattern = re.escape(pattern[1:])
    else:
        start_anchor = False
        pattern = re.escape(pattern)
    # Replace `*` with "any number of non-slash characters"
    pattern = pattern.replace(r"\*", "[^/]*")
    if start_anchor:
        pattern = r"^\/?" + pattern  # Allow an optional leading slash after the start of the string
    return pattern


def iter_rules(codeowners_lines):
    # (pattern, owners) for every rule, in file order. Owners can be empty: a rule with no owner
    # marks a path as deliberately unowned (generated files, deprecated models).
    for line in codeowners_lines:
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        yield parts[0], [owner.removeprefix("@") for owner in parts[1:]]


def match_codeowners(file_path, codeowners_lines):
    # The matching rule as `(pattern, owners)`, or `(None, [])` if nothing matched. The pattern
    # comes back so the caller can tell a real rule from the `*` catch-all.
    # Process rules in reverse (last matching pattern takes precedence)
    for pattern, owners in reversed(list(iter_rules(codeowners_lines))):
        # Modality rules are not path patterns; `modality_owners_table` reads them instead.
        if pattern.startswith(MODALITY_PREFIX):
            continue
        if re.search(pattern_to_regex(pattern), file_path) is not None:
            return pattern, owners
    return None, []


def get_file_owners(file_path, codeowners_lines):
    return match_codeowners(file_path, codeowners_lines)[1]


def pr_author_is_in_hf(pr_author, codeowners_lines):
    # Whether the PR author owns anything themselves, in which case they pick their own reviewer.
    author = pr_author.casefold()
    return any(author in {owner.casefold() for owner in owners} for _, owners in iter_rules(codeowners_lines))


def modality_owners_table(codeowners_lines):
    # The `@@modality/<slug> @owner ...` rules, as {slug: [owners]}. Unknown slugs are dropped;
    # `unknown_modality_slugs` reports them.
    known = set(MODALITY_SECTIONS.values())
    return {
        pattern.removeprefix(MODALITY_PREFIX): owners
        for pattern, owners in iter_rules(codeowners_lines)
        if pattern.startswith(MODALITY_PREFIX) and pattern.removeprefix(MODALITY_PREFIX) in known
    }


def unknown_modality_slugs(codeowners_lines):
    known = set(MODALITY_SECTIONS.values())
    return [
        pattern.removeprefix(MODALITY_PREFIX)
        for pattern, _ in iter_rules(codeowners_lines)
        if pattern.startswith(MODALITY_PREFIX) and pattern.removeprefix(MODALITY_PREFIX) not in known
    ]


def normalize_model_name(name):
    # `kosmos2` and `kosmos-2` name the same model in `models/` and in the toctree.
    return re.sub(r"[-_.]", "", name).lower()


def toctree_modalities(read_file=read_local_file):
    # {normalized doc page name: modality slug}, read from the doc toctree's own sections.
    import yaml

    text = read_file(TOCTREE_PATH)
    if text not in _TOCTREE_CACHE:
        try:
            toctree = yaml.safe_load(text) or []
        except yaml.YAMLError:
            toctree = []
        index = {}
        collect_toctree_modalities(toctree, None, index)
        _TOCTREE_CACHE[text] = index
    return _TOCTREE_CACHE[text]


def collect_toctree_modalities(node, modality, index):
    if isinstance(node, list):
        for item in node:
            collect_toctree_modalities(item, modality, index)
        return
    if not isinstance(node, dict):
        return
    modality = MODALITY_SECTIONS.get(node.get("title"), modality)
    page = node.get("local") or ""
    if modality and page.startswith("model_doc/"):
        index[normalize_model_name(page.split("/", 1)[1])] = modality
    if "sections" in node:
        collect_toctree_modalities(node["sections"], modality, index)


def modality_of_model(model, read_file=read_local_file):
    # The model's modality slug, or None when the toctree does not place it.
    index = toctree_modalities(read_file)
    key = normalize_model_name(model)
    if key in index:
        return index[key]
    # No page of its own: a variant directory inherits the modality of the model it extends.
    candidates = [(len(page), modality) for page, modality in index.items() if len(page) >= MIN_DOC_PREFIX and key.startswith(page)]
    if not candidates:
        return None
    longest = max(length for length, _ in candidates)
    modalities = {modality for length, modality in candidates if length == longest}
    return modalities.pop() if len(modalities) == 1 else None


def model_of_path(file_path):
    match = MODEL_PATH_RE.match(file_path)
    return match.group(1) if match else None


def modality_owners(file_path, codeowners_lines, read_file=read_local_file):
    model = model_of_path(file_path)
    if model is None:
        return []
    modality = modality_of_model(model, read_file)
    if modality is None:
        return []
    return modality_owners_table(codeowners_lines).get(modality, [])


def tagged_reviewers(file_path, read_file=read_local_file):
    # `# Reviewers: @a @b` from the leading comment block of the model's modular/modeling file. The
    # modular converter copies that block verbatim into the generated modeling file
    # (`utils/modular_model_converter.py`, `header=modular_mapper.python_module.header`), so the tag
    # survives regeneration and either file can carry it.
    model = model_of_path(file_path)
    if model is None:
        return []
    for name in (f"modular_{model}.py", f"modeling_{model}.py"):
        for line in read_file(f"{MODELS_DIR}/{model}/{name}").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                break  # the header ends at the first statement
            tag = REVIEWERS_TAG_RE.match(line)
            if tag:
                return [owner.removeprefix("@") for owner in tag.group(1).split()]
    return []


def owners_for_file(file_path, codeowners_lines, read_file=read_local_file):
    # Most specific wins: an in-file tag, then a codeowners rule, then the model's modality, then
    # whatever the catch-all says.
    tagged = tagged_reviewers(file_path, read_file)
    if tagged:
        return tagged
    pattern, owners = match_codeowners(file_path, codeowners_lines)
    if pattern is not None and pattern != CATCH_ALL_PATTERN:
        return owners
    return modality_owners(file_path, codeowners_lines, read_file) or owners


def resolution_source(file_path, codeowners_lines, read_file=read_local_file):
    # Which of the four mechanisms answered for `file_path`: "tag", "rule", "modality" or
    # "catch-all". `utils/check_reviewers.py` uses it to find files nothing claims.
    if tagged_reviewers(file_path, read_file):
        return "tag"
    pattern, _ = match_codeowners(file_path, codeowners_lines)
    if pattern is not None and pattern != CATCH_ALL_PATTERN:
        return "rule"
    if modality_owners(file_path, codeowners_lines, read_file):
        return "modality"
    return "catch-all"
