# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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

import json
import os
import re
from collections import Counter
from pathlib import Path

import github
import yaml
from github import Github


MAX_REVIEWERS = 2

# A model file with no rule of its own is routed by modality instead, so a new model does not
# need a line in the codeowners file to reach someone. The modality comes from the model's
# section in the doc toctree, which `make check-repository-consistency` already requires every
# model to appear in (`utils/check_doc_toc.py::ensure_all_models_in_toctree`).
TOCTREE_PATH = "docs/source/en/_toctree.yml"
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
CATCH_ALL_PATTERN = "*"
MODEL_PATH_RE = re.compile(r"^/?src/transformers/models/([^/]+)/")
# `# Reviewers: @a @b` in a model file's leading comment block wins over everything else. The
# modular converter copies that block verbatim into the generated modeling file
# (`utils/modular_model_converter.py`, `header=modular_mapper.python_module.header`), so the tag
# survives regeneration and can be read from whichever of the two files exists.
REVIEWERS_TAG_RE = re.compile(r"^#\s*Reviewers?\s*:\s*(.+)$", re.IGNORECASE)

# Set once by `main`; only needed for a file this PR adds or edits, which the checked-out base
# either lacks or has in its pre-PR state.
_REPO = None
_HEAD_SHA = None
_CHANGED_PATHS = frozenset()
_FILE_CACHE = {}
_TOCTREE_CACHE = {}


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

def match_codeowners(file_path, codeowners_lines):
    # The matching rule as `(pattern, owners)`, or `(None, [])` if nothing matched. The pattern
    # comes back so the caller can tell a real rule from the `*` catch-all.
    # Process lines in reverse (last matching pattern takes precedence)
    for line in reversed(codeowners_lines):
        # Skip comments and empty lines, strip inline comments
        line = line.split('#')[0].strip()
        if not line:
            continue

        # Split into pattern and owners
        parts = line.split()
        pattern = parts[0]
        # Modality rules are not path patterns; `modality_owners` reads them instead.
        if pattern.startswith(MODALITY_PREFIX):
            continue
        # Can be empty, e.g. for dummy files with explicitly no owner!
        owners = [owner.removeprefix("@") for owner in parts[1:]]

        # Check if file matches pattern
        file_regex = pattern_to_regex(pattern)
        if re.search(file_regex, file_path) is not None:
            return pattern, owners  # Remember, owners can still be empty!
    return None, []  # Should never happen, but just in case

def get_file_owners(file_path, codeowners_lines):
    return match_codeowners(file_path, codeowners_lines)[1]

def set_pr_context(repo, head_sha, changed_paths):
    # Lets the readers below fall back to the PR head. `pull_request_target` checks out the base,
    # so a model this PR adds is missing there, and a `# Reviewers:` tag it adds is not yet visible.
    global _REPO, _HEAD_SHA, _CHANGED_PATHS
    _REPO, _HEAD_SHA, _CHANGED_PATHS = repo, head_sha, frozenset(changed_paths)

def read_repo_file(path):
    # Contents of `path`, preferring the PR's own version for a file it touches. Only ever parsed
    # as data below, never executed. "" when the file exists in neither place.
    if path in _FILE_CACHE:
        return _FILE_CACHE[path]
    text = ""
    if path in _CHANGED_PATHS:
        text = read_head_file(path)
    if not text:
        local = Path(path)
        text = local.read_text(encoding="utf-8") if local.exists() else read_head_file(path)
    _FILE_CACHE[path] = text
    return text

def read_head_file(path):
    if _REPO is None or _HEAD_SHA is None:
        return ""
    try:
        return _REPO.get_contents(path, ref=_HEAD_SHA).decoded_content.decode("utf-8")
    except github.GithubException:
        return ""  # a file that exists in neither the base nor the head is a normal outcome

def modality_owners_table(codeowners_lines):
    # The `@@modality/<slug> @owner ...` rules, as {slug: [owners]}.
    table = {}
    known = set(MODALITY_SECTIONS.values())
    for line in codeowners_lines:
        line = line.split('#')[0].strip()
        if not line.startswith(MODALITY_PREFIX):
            continue
        parts = line.split()
        slug = parts[0].removeprefix(MODALITY_PREFIX)
        if slug not in known:
            warn(f"Unknown modality {slug!r} in codeowners; expected one of {sorted(known)}")
            continue
        table[slug] = [owner.removeprefix("@") for owner in parts[1:]]
    return table

def normalize_model_name(name):
    # `kosmos2` and `kosmos-2` name the same model in `models/` and in the toctree.
    return re.sub(r"[-_.]", "", name).lower()

def toctree_modalities():
    # {normalized model name: modality slug}, read from the doc toctree's own sections.
    if "index" not in _TOCTREE_CACHE:
        index = {}
        try:
            toctree = yaml.safe_load(read_repo_file(TOCTREE_PATH)) or []
        except yaml.YAMLError as e:
            warn(f"Could not parse {TOCTREE_PATH}: {e}")
            toctree = []
        collect_toctree_modalities(toctree, None, index)
        _TOCTREE_CACHE["index"] = index
    return _TOCTREE_CACHE["index"]

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

def model_of_path(file_path):
    match = MODEL_PATH_RE.match(file_path)
    return match.group(1) if match else None

def modality_owners(file_path, codeowners_lines):
    model = model_of_path(file_path)
    if model is None:
        return []
    modality = toctree_modalities().get(normalize_model_name(model))
    if modality is None:
        return []
    return modality_owners_table(codeowners_lines).get(modality, [])

def tagged_reviewers(file_path):
    # `# Reviewers: @a @b` from the leading comment block of the model's modular/modeling file.
    model = model_of_path(file_path)
    if model is None:
        return []
    for name in (f"modular_{model}.py", f"modeling_{model}.py"):
        for line in read_repo_file(f"src/transformers/models/{model}/{name}").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                break  # the header ends at the first statement
            tag = REVIEWERS_TAG_RE.match(line)
            if tag:
                return [owner.removeprefix("@") for owner in tag.group(1).split()]
    return []

def owners_for_file(file_path, codeowners_lines):
    # Most specific wins: an in-file tag, then a codeowners rule, then the model's modality, then
    # whatever the catch-all says.
    tagged = tagged_reviewers(file_path)
    if tagged:
        return tagged
    pattern, owners = match_codeowners(file_path, codeowners_lines)
    if pattern is not None and pattern != CATCH_ALL_PATTERN:
        return owners
    return modality_owners(file_path, codeowners_lines) or owners

def pr_author_is_in_hf(pr_author, codeowners_lines):
    # Check if the PR author is in the codeowners file
    for line in codeowners_lines:
        line = line.split('#')[0].strip()
        if not line:
            continue

        # Split into pattern and owners
        parts = line.split()
        owners = [owner.removeprefix("@") for owner in parts[1:]]

        if pr_author.casefold() in {owner.casefold() for owner in owners}:
            return True
    return False

def warn(message):
    # Surfaced on the workflow run itself. Without it an unassignable reviewer is a
    # line in a step log on a green job, which is how #48070 ended up with none.
    print(f"::warning::{message}")

def is_collaborator(repo, login):
    # A review can only be requested from a collaborator, and GitHub rejects the
    # WHOLE request rather than the offending name -- so one codeowner who has left
    # takes their valid co-owners down with them. Check before asking.
    try:
        return repo.has_in_collaborators(login)
    except github.GithubException as e:
        warn(f"Could not check whether {login} is a collaborator: {e}")
        return False

def request_reviews(repo, pr, candidates, limit=MAX_REVIEWERS):
    # Request up to `limit` reviews, in ranked order, one name per call so a
    # rejection cannot cancel the others. Owners who cannot be asked are skipped
    # and the next-ranked owner takes the slot. Returns the logins requested.
    requested = []
    for login in candidates:
        if len(requested) == limit:
            break
        if not is_collaborator(repo, login):
            warn(f"Skipping {login}: not a collaborator of {repo.full_name} (stale codeowners entry?)")
            continue
        try:
            pr.create_review_request([login])
        except github.GithubException as e:
            warn(f"Failed to request review from {login}: {e}")
            continue
        requested.append(login)
    return requested

def main():
    script_dir = Path(__file__).parent.absolute()
    with open(script_dir / "codeowners_for_review_action") as f:
        codeowners_lines = f.readlines()

    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo("huggingface/transformers")
    with open(os.environ['GITHUB_EVENT_PATH']) as f:
        event = json.load(f)

    # The PR number is available in the event payload
    pr_number = event['pull_request']['number']
    pr = repo.get_pull(pr_number)
    pr_author = pr.user.login
    if pr_author_is_in_hf(pr_author, codeowners_lines):
        print(f"PR author {pr_author} is in codeowners, skipping review request.")
        return

    existing_reviews = list(pr.get_reviews())
    if existing_reviews:
        print(f"Already has reviews: {[r.user.login for r in existing_reviews]}")
        return

    changed_files = list(pr.get_files())
    set_pr_context(repo, pr.head.sha, [f.filename for f in changed_files])

    users_requested, teams_requested = pr.get_review_requests()
    users_requested = list(users_requested)
    if users_requested:
        print(f"Reviewers already requested: {users_requested}")
        return

    # Tally per person, not per spelling: a login is case-insensitive on GitHub, and the same
    # owner appearing as `@ArthurZucker` on one line and `@arthurzucker` on another would
    # otherwise split their total across two entries (and be requested twice).
    locs_per_owner = Counter()
    spelling = {}
    for file in changed_files:
        owners = owners_for_file(file.filename, codeowners_lines)
        for owner in owners:
            key = owner.casefold()
            spelling.setdefault(key, owner)
            locs_per_owner[key] += file.changes

    # Assign the top 2 based on locs changed as reviewers, but skip the owner if present
    locs_per_owner.pop(pr_author.casefold(), None)
    ranked_owners = [spelling[key] for key, _ in locs_per_owner.most_common()]
    print("Top owners", [(spelling[key], locs) for key, locs in locs_per_owner.most_common(MAX_REVIEWERS)])
    requested = request_reviews(repo, pr, ranked_owners)
    if requested:
        print(f"Requested review from {requested}")
    elif ranked_owners:
        warn(f"No reviewer could be requested for #{pr_number} out of {ranked_owners}")
    else:
        warn(f"No codeowner matched the files changed in #{pr_number}")



if __name__ == "__main__":
    main()
