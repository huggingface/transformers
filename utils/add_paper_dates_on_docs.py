from arxiv import Search, Client
import os
import re
import subprocess

"""
Plan
1. Go thru all the md docs inside the folder "transformers/docs/source/en/model_doc"
2. check if the doc has a paper_id
3. if yes, then get the published and updated date from arxiv and save in variables, if no add the model to a list of models without paper_id
4. get the date of creation of the doc
5. modify the md file to add the dates

"""


def add_dates_to_docs():
    """
    Add paper published dates and update dates to model documentation files, if there is an associated paper.
    Add the date of creation of the doc as well.
    """
    #step1
    root = os.getcwd().split('utils')[0]
    docs_path = os.path.join(root, "docs", "source", "en", "model_doc")
    docs_list = os.listdir(docs_path)

    paper_exists = False
    no_paper_models = []
    err_in_search = []
    published_date = None
    updated_date = None
    client = Client()

    for doc in docs_list:
        print(f"Processing {doc}...")
        path = os.path.join(docs_path, doc)
        with open(path, 'r', encoding="utf-8") as f:
            content = f.read()
        paper_ids = re.findall(r"https://huggingface\.co/papers/(\d+\.\d+)", content)
        if paper_ids:
            paper_id = paper_ids[0]
            try:
                search = Search(id_list=[paper_id])
                results = client.results(search)
                result = next(results)
                published_date = result.published.date()
                updated_date = result.updated.date()
            except:
                print(f"No results found for paper ID: {paper_id}")
                err_in_search.append(paper_id)
                continue
        else:
            no_paper_models.append(doc)
        
        first_commit_date = subprocess.check_output(["git", "log", "--reverse", "--pretty=format:%ad", "--date=iso", path],text=True).strip().split('\n')[0][:10]  # Get the first commit date in YYYY-MM-DD format
        
        paper_date = (
            f"*Paper published on {published_date}, updated on {updated_date}*\n"
            "\n"
        )
        paper_published_date = (
            f"*Paper published on {published_date}",
            "\n"
        )
        hf_date = (
            f"*Added to Hugging Face Transformers on {first_commit_date}*\n"
            "\n"
        )

        if "Added to Hugging Face Transformers on" not in content:
        
            matches = list(re.finditer(r"</div>", content))

            if matches:
                last_match = matches[-1]
                insert_pos = last_match.end()
                
                if published_date and updated_date:
                    date_info = f"\n{paper_date}{hf_date}"
                elif published_date and not updated_date:
                    date_info = f"\n{paper_published_date}{hf_date}"
                else:
                    date_info = f"\n{hf_date}"

                new_content = content[:insert_pos] + "\n" + date_info + content[insert_pos:]
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)

    with open(os.path.join(docs_path, "utils", "no_paper_models.txt"), "w") as f:
        f.write("\n".join(no_paper_models))
    with open(os.path.join(docs_path, "utils", "err_in_search.txt"), "w") as f:
        f.write("\n".join(err_in_search))
            


if __name__ == "__main__":
    add_dates_to_docs()