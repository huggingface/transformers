import re
import os
import subprocess
import argparse
from arxiv import Search, Client


root = os.getcwd().split('utils')[0]
DOCS_PATH = os.path.join(root, "docs", "source", "en", "model_doc")
MODELS_PATH = os.path.join(root, "src/transformers/models")


def get_modified_model_names():
    """Get list of model names from modified files in docs/source/en/model_doc/"""
    result = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True
    )
    
    model_names = []
    for line in result.strip().split('\n'):
        if line:
            # Split on whitespace and take the last part (filename)
            filename = line.split()[-1]
            if filename.startswith("docs/source/en/model_doc/") and filename.endswith(".md"):
                model_name = os.path.splitext(os.path.basename(filename))[0]
                model_names.append(model_name)
    
    return model_names



def paper_link(model_name=None, path=None):
    
    if model_name != None and not model_name.endswith(".md"):
        model_name = f"{model_name}.md"
    file_path = path or os.path.join(DOCS_PATH, f"{model_name}")

    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read()
    paper_ids = re.findall(r"https://huggingface\.co/papers/(\d+\.\d+)", content)
    paper_ids += re.findall(r"https://hf\.co/papers/(\d+\.\d+)", content)
    paper_ids += re.findall(r"https://arxiv\.org/abs/(\d+\.\d+)", content)
    if len(paper_ids) == 0:
        return None, 0
    return paper_ids[0], len(set(paper_ids))

# info = {}

# docs_list = os.listdir(DOCS_PATH)
# docs_list.sort()
# #print(docs_list)
# for docs in docs_list:
#     if docs.endswith(".md"):
#         paper_ids, count = paper_link(docs)
#         # if count > 0:
#         info[docs] = {"paper_id": paper_ids, "count": count}

# no_papers = [k for k, v in info.items() if v["paper_id"] is None]
# many_papers = [k for k, v in info.items() if v["count"] > 1]
# single_paper = [k for k, v in info.items() if v["count"] == 1]


def get_first_commit_date(model_name=None):
    
    if model_name.endswith(".md"):
        model_name = f"{model_name[:-3]}"
        
    model_name_src = model_name
    if "-" in model_name:
        model_name_src = model_name.replace("-", "_")
    file_path = os.path.join(MODELS_PATH, model_name_src, "__init__.py")
    if not os.path.exists(file_path):
        file_path = os.path.join(DOCS_PATH, f"{model_name}.md")
    result = subprocess.check_output(
        ["git", "log", "--reverse", "--pretty=format:%ad", "--date=iso", file_path],
        text=True
    )
    return result.strip().split('\n')[0][:10]


def get_release_date(link):
    if link.startswith("https://huggingface.co/papers/"):
        link = link.replace("https://huggingface.co/papers/", "")
    client = Client()
    search = Search(id_list=[link])
    results = list(client.results(search))
    if len(results) != 0:
        return results[0].published.date().isoformat()
    else:
        return f"No info found for the paper https://arxiv.org/abs/{link}"

no_marker = []

def insert_dates(docs_list):
    for doc in docs_list:
        if not doc.endswith(".md"):
            doc = f"{doc}.md"
        file_path = os.path.join(DOCS_PATH, doc)
        paper_id, count = paper_link(path=file_path)
        if paper_id is not None:
            release_date = get_release_date(paper_id)
        else:
            print("no huggingface/arxiv paper link found in", doc)
            release_date = "{release_date}"
        hf_commit_date = get_first_commit_date(model_name=doc)

        with open(file_path, 'r', encoding="utf-8") as f:
            content = f.read()
        markers = list(re.finditer(r"-->", content))
        if len(markers) == 0:
            print(f"No marker found in {doc}. Skipping.")
            no_marker.append(doc)
            continue

        # if paper_id is None:
        #     continue
        if doc == "auto.md" or doc == "timm_wrapper":
            continue

        insert_index = markers[0].end()
        date_info = f"\n*This model was released on {release_date} and added to Hugging Face Transformers on {hf_commit_date}.*"

        if date_info not in content:
            pattern = r"\n\*This model was released on .* and added to Hugging Face Transformers on .*\.\*"
            
            if re.search(pattern, content):
                content = re.sub(pattern, "", content)
                content = content[:insert_index] + date_info + content[insert_index:]
                with open(file_path, 'w', encoding="utf-8") as f:
                    f.write(content)
                print(f"Updated {doc} release and commit dates.")

            else:
                content = content[:insert_index] + date_info + content[insert_index:]
                with open(file_path, 'w', encoding="utf-8") as f:
                    f.write(content)
                print(f"Added {doc} release and commit dates.")   # This if else block can be shortened if diff messages are not needed for update and add

        else:
            print(f"{doc} already has the release and commit dates.")

def main():
    parser = argparse.ArgumentParser(description="Add release and commit dates to model documentation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--auto", 
        action="store_true", 
        help="Automatically process modified files from git status"
    )
    group.add_argument(
        "--models", 
        nargs="+", 
        help="Specify model names to process (without .md extension)"
    )
    
    args = parser.parse_args()
    
    if args.auto:
        modified_model_names = get_modified_model_names()
        if not modified_model_names:
            print("No modified model documentation files found.")
            return
        print(f"Processing modified models: {modified_model_names}")
    else:
        modified_model_names = args.models
        print(f"Processing specified models: {modified_model_names}")
    
    insert_dates(modified_model_names)

if __name__ == "__main__":
    main()

