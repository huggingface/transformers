"""An internal script to process `new_model_failures_with_bad_commit.json` produced by `utils/check_bad_commit.py`.

This is used by `.github/workflows/check_failed_model_tests.yml` to produce a slack report of the following form

```
<{url}|New failed tests>
{
   "GH_ydshieh": {
       "vit": 1
   }
}
```
"""

import datetime
import json
import os
from collections import Counter
from copy import deepcopy

from huggingface_hub import HfApi


if __name__ == "__main__":
    api = HfApi()

    with open("new_model_failures_with_bad_commit.json") as fp:
        data = json.load(fp)

    # TODO: extend
    team_members = [
        "ydshieh",
        "zucchini-nlp",
        "ArthurZucker",
        "gante",
        "LysandreJik",
        "molbap",
        "qubvel",
        "Rocketknight1",
        "muellerzr",
        "SunMarc",
    ]

    # Counting the number of failures grouped by authors
    new_data = {}
    for model, model_result in data.items():
        for device, failed_tests in model_result.items():
            for failed_test in failed_tests:
                author = failed_test["author"]

                if author not in team_members:
                    author = failed_test["merged_by"]

                if author not in new_data:
                    new_data[author] = Counter()
                new_data[author].update([model])
    for author in new_data:
        new_data[author] = dict(new_data[author])

    # Group by author
    new_data_full = {author: deepcopy(data) for author in new_data}
    for author, _data in new_data_full.items():
        for model, model_result in _data.items():
            for device, failed_tests in model_result.items():
                failed_tests = [x for x in failed_tests if x["author"] == author or x["merged_by"] == author]
                model_result[device] = failed_tests
            _data[model] = {k: v for k, v in model_result.items() if len(v) > 0}
        new_data_full[author] = {k: v for k, v in _data.items() if len(v) > 0}

    # Upload to Hub and get the url
    with open("new_model_failures_with_bad_commit_grouped_by_authors.json", "w") as fp:
        json.dump(new_data_full, fp, ensure_ascii=False, indent=4)
    commit_info = api.upload_file(
        path_or_fileobj="new_model_failures_with_bad_commit_grouped_by_authors.json",
        path_in_repo=f"{datetime.datetime.today().strftime('%Y-%m-%d')}/ci_results_run_models_gpu/new_model_failures_with_bad_commit_grouped_by_authors.json",
        repo_id="hf-internal-testing/transformers_daily_ci",
        repo_type="dataset",
        token=os.environ.get("TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN", None),
    )
    url = f"https://huggingface.co/datasets/hf-internal-testing/transformers_daily_ci/raw/{commit_info.oid}/{datetime.datetime.today().strftime('%Y-%m-%d')}/ci_results_run_models_gpu/new_model_failures_with_bad_commit_grouped_by_authors.json"

    # Add `GH_` prefix as keyword mention
    output = {}
    for author, item in new_data.items():
        author = f"GH_{author}"
        output[author] = item

    report = f"<{url}|New failed tests>\\n\\n"
    report += json.dumps(output, indent=4).replace('"', '\\"').replace("\n", "\\n")
    print(report)
