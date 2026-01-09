import argparse
import os
import re
import subprocess
from datetime import date, datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from huggingface_hub import paper_info


ROOT = os.getcwd().split("utils")[0]
DOCS_PATH = os.path.join(ROOT, "docs/source/en/model_doc")
MODELS_PATH = os.path.join(ROOT, "src/transformers/models")
GITHUB_REPO_URL = "https://github.com/huggingface/transformers"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/huggingface/transformers/main"

COPYRIGHT_DISCLAIMER = """<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->"""

ARXIV_PAPERS_NOT_IN_HF_PAPERS = {
    "gemma3n.md": "2506.06644",
    "xmod.md": "2205.06266",
}


def check_file_exists_on_github(file_path: str) -> bool:
    """Check if a file exists on the main branch of the GitHub repository.

    Args:
        file_path: Relative path from repository root

    Returns:
        True if file exists on GitHub main branch (or if check failed), False only if confirmed 404

    Note:
        On network errors or other issues, returns True (assumes file exists) with a warning.
        This prevents the script from failing due to temporary network issues.
    """
    # Convert absolute path to relative path from repository root if needed
    if file_path.startswith(ROOT):
        file_path = file_path[len(ROOT) :].lstrip("/")

    # Construct the raw GitHub URL for the file
    url = f"{GITHUB_RAW_URL}/{file_path}"

    try:
        # Make a HEAD request to check if file exists (more efficient than GET)
        request = Request(url, method="HEAD")
        request.add_header("User-Agent", "transformers-add-dates-script")

        with urlopen(request, timeout=10) as response:
            return response.status == 200
    except HTTPError as e:
        if e.code == 404:
            # File doesn't exist on GitHub
            return False
        # HTTP error (non-404): assume file exists and continue with local git history
        return True
    except Exception:
        # Network/timeout error: assume file exists and continue with local git history
        return True


def get_modified_cards() -> list[str]:
    """Get the list of model names from modified files in docs/source/en/model_doc/"""

    result = subprocess.check_output(["git", "diff", "--name-only", "upstream/main"], text=True)

    model_names = []
    for line in result.strip().split("\n"):
        if line:
            # Check if the file is in the model_doc directory
            if line.startswith("docs/source/en/model_doc/") and line.endswith(".md"):
                file_path = os.path.join(ROOT, line)
                if os.path.exists(file_path):
                    model_name = os.path.splitext(os.path.basename(line))[0]
                    if model_name not in ["auto", "timm_wrapper"]:
                        model_names.append(model_name)

    return model_names


def get_paper_link(model_card: str | None, path: str | None) -> str:
    """Get the first paper link from the model card content."""

    if model_card is not None and not model_card.endswith(".md"):
        model_card = f"{model_card}.md"
    file_path = path or os.path.join(DOCS_PATH, f"{model_card}")
    model_card = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find known paper links
    paper_ids = re.findall(r"https://huggingface\.co/papers/\d+\.\d+", content)
    paper_ids += re.findall(r"https://arxiv\.org/abs/\d+\.\d+", content)
    paper_ids += re.findall(r"https://arxiv\.org/pdf/\d+\.\d+", content)

    if len(paper_ids) == 0:
        return "No_paper"

    return paper_ids[0]


def get_first_commit_date(model_name: str | None) -> str:
    """Get the first commit date of the model's init file or model.md. This date is considered as the date the model was added to HF transformers"""

    if model_name.endswith(".md"):
        model_name = f"{model_name[:-3]}"

    model_name_src = model_name
    if "-" in model_name:
        model_name_src = model_name.replace("-", "_")
    file_path = os.path.join(MODELS_PATH, model_name_src, "__init__.py")

    # If the init file is not found (only true for legacy models), the doc's first commit date is used
    if not os.path.exists(file_path):
        file_path = os.path.join(DOCS_PATH, f"{model_name}.md")

    # Check if file exists on GitHub main branch
    file_exists_on_github = check_file_exists_on_github(file_path)

    if not file_exists_on_github:
        # File does not exist on GitHub main branch (new model), use today's date
        final_date = date.today().isoformat()
    else:
        # File exists on GitHub main branch, get the first commit date from local git history
        final_date = subprocess.check_output(
            ["git", "log", "--reverse", "--pretty=format:%ad", "--date=iso", file_path], text=True
        )
    return final_date.strip().split("\n")[0][:10]


def get_release_date(link: str) -> str:
    if link.startswith("https://huggingface.co/papers/"):
        link = link.replace("https://huggingface.co/papers/", "")

        try:
            info = paper_info(link)
            return info.published_at.date().isoformat()
        except Exception:
            # Error fetching release date, function returns None (will use placeholder)
            pass

    elif link.startswith("https://arxiv.org/abs/") or link.startswith("https://arxiv.org/pdf/"):
        return r"{release_date}"


def replace_paper_links(file_path: str) -> bool:
    """Replace arxiv links with huggingface links if valid, and replace hf.co with huggingface.co"""

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Replace hf.co with huggingface.co
    content = content.replace("https://hf.co/", "https://huggingface.co/")

    # Find all arxiv links
    arxiv_links = re.findall(r"https://arxiv\.org/abs/(\d+\.\d+)", content)
    arxiv_links += re.findall(r"https://arxiv\.org/pdf/(\d+\.\d+)", content)

    for paper_id in arxiv_links:
        try:
            # Check if paper exists on huggingface
            paper_info(paper_id)
            # If no exception, replace the link
            old_link = f"https://arxiv.org/abs/{paper_id}"
            if old_link not in content:
                old_link = f"https://arxiv.org/pdf/{paper_id}"
            new_link = f"https://huggingface.co/papers/{paper_id}"
            content = content.replace(old_link, new_link)

        except Exception:
            # Paper not available on huggingface, keep arxiv link
            continue

    # Write back only if content changed
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def _normalize_model_card_name(model_card: str) -> str:
    """Ensure model card has .md extension"""
    return model_card if model_card.endswith(".md") else f"{model_card}.md"


def _should_skip_model_card(model_card: str) -> bool:
    """Check if model card should be skipped"""
    return model_card in ("auto.md", "timm_wrapper.md")


def _read_model_card_content(model_card: str) -> str:
    """Read and return the content of a model card"""
    file_path = os.path.join(DOCS_PATH, model_card)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _get_dates_pattern_match(content: str):
    """Search for the dates pattern in content and return match object"""
    pattern = r"\n\*This model was released on (.*) and added to Hugging Face Transformers on (\d{4}-\d{2}-\d{2})\.\*"
    return re.search(pattern, content)


def _dates_differ_significantly(date1: str, date2: str) -> bool:
    """Check if two dates differ by more than 1 day"""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        return abs((d1 - d2).days) > 1
    except Exception:
        return True  # If dates can't be parsed, consider them different


def check_missing_dates(model_card_list: list[str]) -> list[str]:
    """Check which model cards are missing release dates and return their names"""
    missing_dates = []

    for model_card in model_card_list:
        model_card = _normalize_model_card_name(model_card)
        if _should_skip_model_card(model_card):
            continue

        content = _read_model_card_content(model_card)
        if not _get_dates_pattern_match(content):
            missing_dates.append(model_card)

    return missing_dates


def check_incorrect_dates(model_card_list: list[str]) -> list[str]:
    """Check which model cards have incorrect HF commit dates and return their names"""
    incorrect_dates = []

    for model_card in model_card_list:
        model_card = _normalize_model_card_name(model_card)
        if _should_skip_model_card(model_card):
            continue

        content = _read_model_card_content(model_card)
        match = _get_dates_pattern_match(content)

        if match:
            existing_hf_date = match.group(2)
            actual_hf_date = get_first_commit_date(model_name=model_card)

            if _dates_differ_significantly(existing_hf_date, actual_hf_date):
                incorrect_dates.append(model_card)

    return incorrect_dates


def insert_dates(model_card_list: list[str]):
    """Insert or update release and commit dates in model cards"""
    for model_card in model_card_list:
        model_card = _normalize_model_card_name(model_card)
        if _should_skip_model_card(model_card):
            continue

        file_path = os.path.join(DOCS_PATH, model_card)

        # First replace arxiv paper links with hf paper link if possible
        replace_paper_links(file_path)

        # Read content and ensure copyright disclaimer exists
        content = _read_model_card_content(model_card)
        markers = list(re.finditer(r"-->", content))

        if len(markers) == 0:
            # No copyright marker found, adding disclaimer to the top
            content = COPYRIGHT_DISCLAIMER + "\n\n" + content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            markers = list(re.finditer(r"-->", content))

        # Get dates
        hf_commit_date = get_first_commit_date(model_name=model_card)
        paper_link = get_paper_link(model_card=model_card, path=file_path)

        if paper_link in ("No_paper", "blog"):
            release_date = r"{release_date}"
        else:
            release_date = get_release_date(paper_link)

        match = _get_dates_pattern_match(content)

        # Update or insert the dates line
        if match:
            # Preserve existing release date unless it's a placeholder
            existing_release_date = match.group(1)
            existing_hf_date = match.group(2)

            if existing_release_date not in (r"{release_date}", "None"):
                release_date = existing_release_date

            if _dates_differ_significantly(existing_hf_date, hf_commit_date) or existing_release_date != release_date:
                old_line = match.group(0)
                new_line = f"\n*This model was released on {release_date} and added to Hugging Face Transformers on {hf_commit_date}.*"
                content = content.replace(old_line, new_line)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
        else:
            # Insert new dates line after copyright marker
            insert_index = markers[0].end()
            date_info = f"\n*This model was released on {release_date} and added to Hugging Face Transformers on {hf_commit_date}.*"
            content = content[:insert_index] + date_info + content[insert_index:]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)


def get_all_model_cards():
    """Get all model cards from the docs path"""

    all_files = os.listdir(DOCS_PATH)
    model_cards = []
    for file in all_files:
        if file.endswith(".md"):
            model_name = os.path.splitext(file)[0]
            if model_name not in ["auto", "timm_wrapper"]:
                model_cards.append(model_name)
    return sorted(model_cards)


def main(all=False, models=None, check_only=False):
    if check_only:
        # Check all model cards for missing dates
        all_model_cards = get_all_model_cards()
        print(f"Checking all {len(all_model_cards)} model cards for missing dates...")
        missing_dates = check_missing_dates(all_model_cards)

        # Check modified model cards for incorrect dates
        modified_cards = get_modified_cards()
        print(f"Checking {len(modified_cards)} modified model cards for incorrect dates...")
        incorrect_dates = check_incorrect_dates(modified_cards)

        if missing_dates or incorrect_dates:
            problematic_cards = missing_dates + incorrect_dates
            model_names = [card.replace(".md", "") for card in problematic_cards]
            raise ValueError(
                f"Missing or incorrect dates in the following model cards: {' '.join(problematic_cards)}\n"
                f"Run `python utils/add_dates.py --models {' '.join(model_names)}` to fix them."
            )
        print("All dates are present and correct!")
        return

    # Determine which model cards to process
    if all:
        model_cards = get_all_model_cards()
        print(f"Processing all {len(model_cards)} model cards from docs directory")
    elif models:
        model_cards = models
        print(f"Processing specified model cards: {model_cards}")
    else:
        model_cards = get_modified_cards()
        if not model_cards:
            print("No modified model cards found.")
            return
        print(f"Processing modified model cards: {model_cards}")

    insert_dates(model_cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add release and commit dates to model cards")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--models", nargs="+", help="Specify model cards to process (without .md extension)")
    group.add_argument("--all", action="store_true", help="Process all model cards in the docs directory")
    group.add_argument("--check-only", action="store_true", help="Check if the dates are already present")

    args = parser.parse_args()
    try:
        main(args.all, args.models, args.check_only)
    except subprocess.CalledProcessError as e:
        print(
            f"An error occurred while executing git commands but it can be ignored (git issue) most probably local: {e}"
        )
