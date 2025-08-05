"""An internal script to process `new_failures_with_bad_commit.json` produced by `utils/check_bad_commit.py`.

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

import json
import os
from collections import Counter
from copy import deepcopy




if __name__ == "__main__":
    with open("new_failures_with_bad_commit.json") as fp:
        data = json.load(fp)

    with open(f"job_links.json") as fp:
        job_links = json.load(fp)

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
                # prepare job_link and add it to each entry of new failed test information.
                # need to change from `single-gpu` to `single` and same for `multi-gpu` to match `job_link`.
                key = model
                job_link = job_links[key][device.replace("-gpu", "")]

                failed_tests = [x for x in failed_tests if x["author"] == author or x["merged_by"] == author]
                for x in failed_tests:
                    x.update({"job_link": job_link})
                model_result[device] = failed_tests
            _data[model] = {k: v for k, v in model_result.items() if len(v) > 0}
        new_data_full[author] = {k: v for k, v in _data.items() if len(v) > 0}

    with open("new_failures_with_bad_commit_grouped_by_authors.json", "w") as fp:
        json.dump(new_data_full, fp, ensure_ascii=False, indent=4)

