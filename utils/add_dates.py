import argparse
import os
import re
import subprocess
from typing import Optional

from huggingface_hub import paper_info


ROOT = os.getcwd().split("utils")[0]
DOCS_PATH = os.path.join(ROOT, "docs/source/en/model_doc")
MODELS_PATH = os.path.join(ROOT, "src/transformers/models")

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


def get_modified_cards() -> list[str]:
    """Get the list of model names from modified files in docs/source/en/model_doc/"""

    result = subprocess.check_output(["git", "status", "--porcelain"], text=True)

    model_names = []
    for line in result.strip().split("\n"):
        if line:
            # Split on whitespace and take the last part (filename)
            filename = line.split()[-1]
            if filename.startswith("docs/source/en/model_doc/") and filename.endswith(".md"):
                model_name = os.path.splitext(os.path.basename(filename))[0]
                if model_name not in ["auto", "timm_wrapper"]:
                    model_names.append(model_name)

    return model_names


def get_paper_link(model_card: Optional[str], path: Optional[str]) -> str:
    """Get the first paper link from the model card content."""

    if model_card is not None and not model_card.endswith(".md"):
        model_card = f"{model_card}.md"
    file_path = path or os.path.join(DOCS_PATH, f"{model_card}")
    model_card = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "blog" in content or "report" in content or "post" in content:
        print(f"Insert the release date of the blog post or technical report at the top of {model_card}")
        return "blog"

    # Find known paper links
    paper_ids = re.findall(r"https://huggingface\.co/papers/\d+\.\d+", content)
    paper_ids += re.findall(r"https://arxiv\.org/abs/\d+\.\d+", content)

    # If no known paper links are found, look for other potential paper links
    if len(paper_ids) == 0:
        # Find all https links
        all_https_links = re.findall(r"https://[^\s\)]+", content)

        # Filter out huggingface.co and github links
        other_paper_links = []
        for link in all_https_links:
            link = link.rstrip(".,;!?)")
            if "huggingface.co" not in link and "github.com" not in link:
                other_paper_links.append(link)

        # Remove duplicates while preserving order
        other_paper_links = list(dict.fromkeys(other_paper_links))

        if other_paper_links:
            print(f"No Hugging Face or Arxiv papers found. The possible paper links found in {model_card}:")
            for link in other_paper_links:
                print(f"  - {link}")

        return "No_paper"

    return paper_ids[0]


def get_first_commit_date(model_name: Optional[str]) -> str:
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

    result = subprocess.check_output(
        ["git", "log", "--reverse", "--pretty=format:%ad", "--date=iso", file_path], text=True
    )
    return result.strip().split("\n")[0][:10]


def get_release_date(link: str) -> str:
    if link.startswith("https://huggingface.co/papers/"):
        link = link.replace("https://huggingface.co/papers/", "")

        try:
            info = paper_info(link)
            return info.published_at.date().isoformat()
        except Exception as e:
            print(f"Error fetching release date for the paper https://huggingface.co/papers/{link}: {e}")

    elif link.startswith("https://arxiv.org/abs/"):
        print(f"This paper {link} is not yet available in Hugging Face papers, skipping the release date attachment.")
        return r"{release_date}"


def replace_paper_links(file_path: str) -> bool:
    """Replace arxiv links with huggingface links if valid, and replace hf.co with huggingface.co"""

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    model_card = os.path.basename(file_path)
    original_content = content

    # Replace hf.co with huggingface.co
    content = content.replace("https://hf.co/", "https://huggingface.co/")

    # Find all arxiv links
    arxiv_links = re.findall(r"https://arxiv\.org/abs/(\d+\.\d+)", content)

    for paper_id in arxiv_links:
        try:
            # Check if paper exists on huggingface
            paper_info(paper_id)
            # If no exception, replace the link
            old_link = f"https://arxiv.org/abs/{paper_id}"
            new_link = f"https://huggingface.co/papers/{paper_id}"
            content = content.replace(old_link, new_link)
            print(f"Replaced {old_link} with {new_link}")

        except Exception:
            # Paper not available on huggingface, keep arxiv link
            print(f"Paper {paper_id} for {model_card} is not available on huggingface, keeping the arxiv link")
            continue

    # Write back only if content changed
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def insert_dates(model_card_list: list[str]):
    """Insert release and commit dates into model cards"""

    for model_card in model_card_list:
        if not model_card.endswith(".md"):
            model_card = f"{model_card}.md"

        if model_card == "auto.md" or model_card == "timm_wrapper.md":
            continue

        file_path = os.path.join(DOCS_PATH, model_card)

        # First replace arxiv paper links with hf paper link if possible
        links_replaced = replace_paper_links(file_path)
        if links_replaced:
            print(f"Updated paper links in {model_card}")

        pattern = (
            r"\n\*This model was released on (.*) and added to Hugging Face Transformers on (\d{4}-\d{2}-\d{2})\.\*"
        )

        # Check if the copyright disclaimer sections exists, if not, add one with 2025
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        markers = list(re.finditer(r"-->", content))  # Dates info is placed right below this marker
        if len(markers) == 0:
            print(f"No marker found in {model_card}. Adding copyright disclaimer to the top.")

            # Add copyright disclaimer to the very top of the file
            content = COPYRIGHT_DISCLAIMER + "\n\n" + content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            markers = list(re.finditer(r"-->", content))

        hf_commit_date = get_first_commit_date(model_name=model_card)

        match = re.search(pattern, content)

        # If the dates info line already exists, only check and update the hf_commit_date, don't modify the existing release date
        if match:
            release_date = match.group(1)  # The release date part
            existing_hf_date = match.group(2)  # The existing HF date part
            if existing_hf_date != hf_commit_date:
                old_line = match.group(0)  # Full matched line
                new_line = f"\n*This model was released on {release_date} and added to Hugging Face Transformers on {hf_commit_date}.*"

                content = content.replace(old_line, new_line)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

        # If the dates info line does not exist, add it
        else:
            paper_link = get_paper_link(model_card=model_card, path=file_path)
            release_date = ""

            if not (paper_link == "No_paper" or paper_link == "blog"):
                release_date = get_release_date(paper_link)
            else:
                release_date = r"{release_date}"

            insert_index = markers[0].end()

            date_info = f"\n*This model was released on {release_date} and added to Hugging Face Transformers on {hf_commit_date}.*"
            content = content[:insert_index] + date_info + content[insert_index:]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Added {model_card} release and commit dates.")


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


def main(all=False, auto=True, models=None):
    if all:
        model_cards = get_all_model_cards()
        print(f"Processing all {len(model_cards)} model cards from docs directory")
    elif auto:
        model_cards = get_modified_cards()
        if not model_cards:
            print("No modified model cards found.")
            return
        print(f"Processing modified model cards: {model_cards}")
    else:
        model_cards = models
        print(f"Processing specified model cards: {model_cards}")

    insert_dates(model_cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add release and commit dates to model cards")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--auto", action="store_true", help="Automatically process modified model cards from git status"
    )
    group.add_argument("--models", nargs="+", help="Specify model cards to process (without .md extension)")
    group.add_argument("--all", action="store_true", help="Process all model cards in the docs directory")

    parser.set_defaults(auto=True)
    args = parser.parse_args()

    main(args.all, args.auto, args.models)
